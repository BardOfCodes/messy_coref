
from collections import defaultdict
import time
import torch as th
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Type, Union
from imitation.algorithms import base as algo_base
from stable_baselines3.common import policies, utils, vec_env
from imitation.data import rollout, types
import math
import numpy as np
import os
from pathlib import Path
import torch.multiprocessing as mp
from stable_baselines3.common.utils import obs_as_tensor
from imitation.algorithms.bc import BC, EpochOrBatchIteratorWithProgress
from stable_baselines3.common.vec_env import DummyVecEnv
import CSG.env as csg_env
import copy
from .bc_trainer import AdaptedBC, BCEnvDataset
from .rewrite_engines.rewriters import DifferentiableOptimizer, GraphSweeper, CodeSplicer, NoisyRewriter
from .rewrite_engines.generators import BeamSearcher, WakeSleeper
import _pickle as cPickle
import random
from CSG.utils.train_utils import load_all_weights, save_all_weights
from .rewrite_engines.utils import format_data, format_rl_data
from .rewrite_engines.dataset import MultipleProgramBCEnvDataset
from .rewrite_engines.train_state import PladTrainState
from .best_programs_ds import SRTBestProgramsDS


class PLADBC(AdaptedBC):
    """
    In PLAD style - Find the best programs for each data point before running optimization. 
    """

    def __init__(self, bc_config, save_dir, seed, config, action_space, observation_space,
                 model_info, train_env, custom_logger,
                 device: Union[str, th.device] = "auto",
                 *args, **kwargs):

        super(PLADBC, self).__init__(bc_config, save_dir, seed, config, action_space,
                                     observation_space, model_info, train_env, custom_logger,
                                     device, *args, **kwargs)
        self.model_info['train_state'] = PladTrainState
        # Now for other stats:
        self.latent_execution = bc_config.PLAD.LATENT_EXECUTION
        self.le_only_origins = bc_config.PLAD.LE_ONLY_ORIGINS
        self.init_ler = bc_config.PLAD.INIT_LER
        self.final_ler = bc_config.PLAD.FINAL_LER
        self.le_add_noise = bc_config.PLAD.LE_ADD_NOISE

        self.score_threshold = bc_config.PLAD.SEARCH_EVAL_THRESHOLD
        self.search_patience = bc_config.PLAD.SEARCH_PATIENCE
        self.max_search_wait = bc_config.PLAD.MAX_SEARCH_WAIT

        self.load_best_before_search = bc_config.PLAD.LOAD_BEST_BEFORE_SEARCH
        self.load_best_training_weights = bc_config.PLAD.LOAD_BEST_TRAINING_WEIGHTS
        self.loss_based_poor_epoch_reset = bc_config.PLAD.LOSS_BASED_POOR_EPOCH_RESET
        # Reward Weighting
        self.reward_weighting = bc_config.PLAD.REWARD_WEIGHTING
        self.optimizer_reload = bc_config.PLAD.OPTIMIZER_RELOAD
        self.randomize_rewriters = True
        self.reset_train_state = bc_config.PLAD.RESET_TRAIN_STATE
        self.eval_epoch = bc_config.PLAD.EVAL_EPOCHS
        
        self.rl_mode = bc_config.PLAD.RL_MODE
        self.rl_reward_pow = bc_config.PLAD.RL_REWARD_POW
        self.rl_moving_baseline_alpha = bc_config.PLAD.RL_MOVING_BASELINE_ALPHA
        
        self.max_length = observation_space['previous_steps'].nvec.shape[0]
        self.bpds = SRTBestProgramsDS(bc_config.PLAD.BPDS, self.logger, 
                                      self.rl_mode, self.rl_moving_baseline_alpha,
                                      self.max_length)
        

        # HACK
        self.max_grad_norm = 1.0
        self.perform_search = True
        self.best_program_init = True
        self.perform_rewrite = True
        self.rewrite_frequency = 7
        self.length_alpha = bc_config.PLAD.LENGTH_ALPHA
        self.length_alpha_curriculum = bc_config.PLAD.LENGTH_CURRICULUM
        self.init_length_alpha = bc_config.PLAD.INIT_LENGTH_ALPHA
        self.final_length_alpha = bc_config.PLAD.FINAL_LENGTH_ALPHA

        self.beam_search_generator = BeamSearcher(
            bc_config.BS, self.max_length, self.save_dir, self.logger, self.model_info, self.device, self.init_model_path, self.length_alpha)

        self.diff_opt_rewriter = DifferentiableOptimizer(
            bc_config.DO, self.max_length, self.save_dir, self.logger, self.length_alpha)
        self.graph_sweep_rewriter = GraphSweeper(
            bc_config.GS, self.max_length, self.save_dir, self.logger, self.length_alpha)
        self.code_splice_rewriter = CodeSplicer(
            bc_config.CS, self.max_length, self.save_dir, self.logger, self.model_info, self.device, self.init_model_path, self.length_alpha)

        self.wake_sleep_generator = WakeSleeper(
            bc_config.WS, self.config, bc_config, self.seed, self.model_info, self.save_dir, self.logger, self.device, 
            self.reset_train_state)
        
        self.noisy_rewriter = NoisyRewriter(
            bc_config.NR, self.max_length, self.save_dir, self.logger, self.length_alpha)
        

    def init_data_loader(self):

        bc_env_class = getattr(csg_env, self.bc_config.ENV.TYPE)
        # bc_env = bc_env_class(config, config.BC, seed=seed)
        bc_env = DummyVecEnv([lambda ind=i: bc_env_class(config=self.config, phase_config=self.bc_config,
                                                         seed=self.seed + ind, n_proc=self.bc_config.N_ENVS, proc_id=ind) for i in range(self.bc_config.N_ENVS)])
        for env in bc_env.envs:
            env.program_generator.set_execution_mode(th.device("cuda"), th.float16)
        bc_env.reset()
        batch_iters = math.ceil(self.batch_size/self.bc_config.N_ENVS)
        if self.rl_mode:
            fetch_reward = True
            format_func = format_rl_data
        else:
            fetch_reward = False
            format_func = format_data

        bc_dataset = MultipleProgramBCEnvDataset(bc_env, 
                                                n_iters=self.n_iters_per_epoch * batch_iters,
                                                latent_execution_rate=self.init_ler, 
                                                le_add_noise=self.le_add_noise,
                                                le_only_origins=self.le_only_origins, 
                                                fetch_reward=fetch_reward)
        dataset = th.utils.data.DataLoader(bc_dataset, batch_size=batch_iters, pin_memory=False, collate_fn=format_func,
                                        num_workers=self.num_workers, shuffle=False)
        return dataset
    def update_best_programs(self, save_path, train_state, quantize=True, log_interval=100):
        st = time.time()
        # Enumerate on the programs till you find what you want:
        print("Updating/Seaching Best Programs")
        bc_env_class = getattr(csg_env, self.bc_config.ENV.TYPE)
        temp_env = bc_env_class(
            config=self.config, phase_config=self.bc_config, seed=self.seed, n_proc=1, proc_id=0)
        temp_env.mode = "EVAL"

        if self.best_program_init:
            self.best_program_init = False
            self.bpds.initialize_bpd()
            
        if self.beam_search_generator.enable:
            all_prog_objs = self.beam_search_generator.generate_programs(
                save_path, temp_env, train_state.tensorboard_step)
            self.bpds.set_best_programs(all_prog_objs, temp_env)
            # file_name = "temp_fcsg3d.pkl"
            # self.bpds.bpd = cPickle.load(open(file_name, "rb"))

        if self.perform_rewrite:
            if self.randomize_rewriters:
                rewriters = random.sample(
                    [self.diff_opt_rewriter, self.graph_sweep_rewriter, self.code_splice_rewriter], 3)
            else:
                rewriters = [self.diff_opt_rewriter, self.graph_sweep_rewriter, self.code_splice_rewriter]
                # rewriters = [self.graph_sweep_rewriter, self.code_splice_rewriter]
                # rewriters = [self.code_splice_rewriter]
                # save_loc = "../logs/final_evaluation/ablations/mcsg3d_icr_do_cs_2.pkl"
            for rewriter in rewriters:
                if rewriter.enable:
                    # Slow version first:
                    all_prog_objs, failed_keys = rewriter.rewrite_programs(
                        temp_env, self.bpds.bpd, train_state.tensorboard_step, quantize, train_state.cur_epoch, save_path)
                    if failed_keys:
                        self.bpds.mark_failure_cases(failed_keys, rewriter)
                    self.bpds.set_best_programs(all_prog_objs, temp_env)
                    th.cuda.empty_cache()
                    # if not self.randomize_rewriters:
                        # Eval mode:
                        # program_list = self.construct_training_data(train_state)
                        # rewards = [x['reward'] for x in program_list]
                        # print("++++Final Reward++++", np.mean(rewards))

        if self.wake_sleep_generator.enable:
            all_prog_objs = self.wake_sleep_generator.generate_programs(
                self.bpds, train_state.tensorboard_step, log_interval)
            # remove previous WS programs:
            self.bpds.remove_selected_programs(remove_origin="WS")
            self.bpds.set_best_programs(all_prog_objs, temp_env)
        
        et = time.time()
        self.logger.record("Training data/Overall Search Time", et - st)
        self.bpds.log_training_data_details(train_state)

    def construct_training_data(self, train_state):
        bc_env_class = getattr(csg_env, self.bc_config.ENV.TYPE)
        temp_env = bc_env_class(
            config=self.config, phase_config=self.bc_config, seed=self.seed, n_proc=1, proc_id=0)
        temp_env.mode = "EVAL"
        program_list = self.bpds.construct_training_data(self.noisy_rewriter, temp_env, train_state)
        
        return program_list
        
    def train(
        self,
        train_eval,
        val_eval,
        log_interval,
        on_epoch_end=None,
        on_batch_end=None,
    ):

        dataset = self.init_data_loader()
        train_env = dataset.dataset.env.envs[0]

        train_eval.logger = self.logger
        val_eval.logger = self.logger
        if self.load_best_training_weights:
            best_weights_path = os.path.join(
                train_eval.best_model_save_path, "best_model.ptpkl")
        else:
            best_weights_path = os.path.join(
                val_eval.best_model_save_path, "best_model.ptpkl")
        _ = None

        _, _, train_state, _ = load_all_weights(load_path=self.init_model_path, train_env=train_env, model_info=self.model_info,
                                                    device=self.device, instantiate_model=True)
        # print("Reinitializing the entire net")
        if self.reset_train_state:
            train_state = PladTrainState(0, self.n_epochs, self.n_iters_per_epoch)
        start_epoch = train_state.cur_epoch

        for epoch in range(start_epoch, self.n_epochs):
            # epoch = start_epoch
            # Do search~ here if required!
            train_state.cur_epoch = epoch
            cur_epoch_ler = self.init_ler + train_state.cur_epoch / \
                float(self.n_epochs) * (self.final_ler - self.init_ler)
            
            if self.length_alpha_curriculum:
                self.length_alpha_curriculum_update(train_state)
            dataset.dataset.update_ler(cur_epoch_ler)
            epoch_loss_list = []
            if self.perform_search:
                train_state.n_search += 1
                if self.rl_mode:
                    if train_state.n_search % self.rewrite_frequency != 1:
                        self.perform_rewrite = False
                    else:
                        self.perform_rewrite = True
                th.cuda.empty_cache()
                self.update_best_programs(save_path=best_weights_path, train_state=train_state,
                                          log_interval=log_interval)
                th.cuda.empty_cache()
                self.perform_search = False
                if train_state.cur_epoch == 0:
                    load_path = self.init_model_path
                else:
                    if os.path.exists(best_weights_path):
                        load_path = best_weights_path
                    else:
                        load_path = self.init_model_path

                policy, lr_scheduler, _, _, = load_all_weights(
                    load_path=load_path, train_env=train_env, instantiate_model=True,
                    model_info=self.model_info, device=self.device)
                if self.optimizer_reload:
                    policy.optimizer.__setstate__({'state': defaultdict(dict)})
            self.log_start_time = time.time()
            self.start_time = time.time()
            policy.train()
            policy.set_training_mode(True)
            
            # Set the dataset (Does not change in not prob modes):
            program_list = self.construct_training_data(train_state)
            dataset.dataset.update_program_list(program_list)
            
            for iter_ind, (obs_tensor, target) in enumerate(dataset):
                
                stats_dict_it = train_state.get_state_stats()

                # obs_tensor, target = format_data(batch)
                if self.rl_mode:
                    reward_weights =  obs_tensor.pop("reward") ** self.rl_reward_pow - self.bpds.mean_reward ** self.rl_reward_pow
                else:
                    reward_weights = None
                # print("target min", target.min())
                # with th.cuda.amp.autocast():
                loss, stats_dict_loss, stats_dict_acc = self._calculate_loss(
                    policy, obs_tensor, target, reward_weights)

                if self.collect_gradients:
                    loss = loss / self.gradient_step_count
                    cur_ind = iter_ind % self.gradient_step_count
                    if cur_ind == (self.gradient_step_count-1) or (iter_ind+1) % self.n_iters_per_epoch == 0:
                        loss.backward()
                        # Clip grad norm
                        # th.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                        policy.optimizer.step()
                        policy.optimizer.zero_grad(set_to_none=True)
                        if self.per_iter_scheduler:
                            lr_scheduler.step()
                        train_state.n_updates += 1
                    else:
                        loss.backward()
                        # th.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                else:
                    policy.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    policy.optimizer.step()
                    train_state.n_updates += 1
                
                if self.per_iter_scheduler:
                    lr_scheduler.step()

                train_state.n_forwards += 1
                train_state.tensorboard_step += 1
                self.log_training_details(policy,
                    stats_dict_it, stats_dict_loss, stats_dict_acc, log_interval, train_state)

                if on_batch_end is not None:
                    x = on_batch_end.on_step()
                epoch_loss_list.append(loss.item())

            
            if (train_state.cur_epoch % self.eval_epoch == 0):
                cur_score, best_reached = self.evaluate(train_eval, val_eval, policy, lr_scheduler, train_state.tensorboard_step)
                train_state.cur_score = cur_score
                if best_reached:
                    save_all_weights(policy, lr_scheduler, train_state, best_weights_path)

            if (train_state.cur_epoch % self.save_epoch == 1):
                save_path = os.path.join(
                    self.save_dir, "weights_%d.ptpkl" % train_state.cur_epoch)
                save_all_weights(policy, lr_scheduler, train_state, save_path)

            train_state.epoch_avg_loss = np.nanmean(epoch_loss_list)
            # train_state.best_search_loss = min(train_state.best_search_loss, train_state.epoch_avg_loss)

            if (cur_score - self.score_threshold) > train_state.best_score:
                # We can train more:
                train_state.poor_epochs = 0
                train_state.best_score = cur_score
                train_state.best_epoch = train_state.cur_epoch
            else:
                train_state.poor_epochs += 1


            if train_state.poor_epochs >= self.search_patience:
                do_search = True
            elif train_state.post_search_epoch >= self.max_search_wait:
                do_search = True
            else:
                do_search = False
            if do_search:
                train_state.poor_epochs = 0

                if not self.load_best_before_search:
                    # Always save the latest before search.
                    save_all_weights(policy, lr_scheduler, train_state, best_weights_path)
                    
                self.perform_search = True
                train_state.post_search_epoch = 0
                del policy, lr_scheduler
                policy = None
                lr_scheduler = None
            else:
                train_state.post_search_epoch += 1
            th.cuda.empty_cache()

    def length_alpha_curriculum_update(self, train_state):
        
        new_length_alpha = self.init_length_alpha + train_state.cur_epoch / \
            float(self.n_epochs) * (self.final_length_alpha - self.init_length_alpha)
        self.beam_search_generator.length_alpha = new_length_alpha
        self.wake_sleep_generator.length_alpha = new_length_alpha
        self.diff_opt_rewriter.length_alpha = new_length_alpha
        self.graph_sweep_rewriter.length_alpha = new_length_alpha
        self.code_splice_rewriter.length_alpha = new_length_alpha
        # Update the previous rewards in the BPDS?
        for key, progs in self.bpds.bpd.items():
            for prog in progs:
                if prog['origin'] != "WS":
                    prog['reward'] += (new_length_alpha - self.length_alpha) * len(prog['expression'])
        self.length_alpha = new_length_alpha
                
                
    def evaluate(self, train_eval, val_eval, policy, lr_scheduler, tensorboard_step):
        # Epoch End Part:
        th.cuda.empty_cache()
        policy.optimizer.zero_grad(set_to_none=True)
        policy.eval()
        policy.set_training_mode(False)
        policy.action_dist.distribution = None

        best_reached = []
        for cur_eval in [train_eval, val_eval]:
            cur_eval.n_calls += 1
            cur_eval.num_timesteps = tensorboard_step
            best = cur_eval._on_step(policy, lr_scheduler)
            best_reached.append(best)
        cur_score = val_eval.main_metric
        if self.load_best_training_weights:
            best_reached = best_reached[0]
        else:
            best_reached = best_reached[1]
        return cur_score, best_reached
