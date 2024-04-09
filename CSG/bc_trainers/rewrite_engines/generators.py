from fileinput import filename
import time
import os
import math
import torch as th
import copy
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.multiprocessing import Pool, Process, set_start_method
import CSG.env as csg_env
import _pickle as cPickle
from CSG.utils.eval_utils import parallel_CSG_beam_evaluate, batch_CSG_beam_evaluate, CSG_evaluate
from CSG.utils.train_utils import load_all_weights, save_all_weights
from .rewriters import DifferentiableOptimizer
from .dataset import MultipleProgramBCEnvDataset
from .utils import format_data, ntcsg_to_pcsg, get_program_list, MERGABLE_SANS_BS_TYPES, LOG_MULTIPLIER, MERGABLE_ORIGIN_TYPES
from .train_state import BCTrainState, WSTrainState
from CSG.utils.profile_utils import profileit
import traceback
try:
     set_start_method('spawn')
except RuntimeError:
    pass


class BeamSearcher(DifferentiableOptimizer):

    def __init__(self, config, max_length, save_dir, logger, model_info, device, init_model_path, length_alpha, *args, **kwargs):

        # BEAM SEARCH
        self.enable = config.ENABLE
        self.bs_n_proc = config.N_PROC
        self.bs_deterministic = config.DETERMINISTIC
        self.bs_beam_size = config.BEAM_SIZE
        self.bs_beam_selector = config.BEAM_SELECTOR
        self.bs_batch_size = config.BATCH_SIZE
        self.bs_return_multiple = config.RETURN_MULTIPLE
        self.language_name = config.LANGUAGE_NAME
        self.stochastic_bs = config.STOCHASTIC_BS

        self.device = device
        self.model_info = model_info
        self.init_model_path = init_model_path
        self.length_alpha = length_alpha

        if "SA" in self.language_name:
            self.reward_evaluation_limit = 15
        else:
            self.reward_evaluation_limit = 10000
        self.common_setting(max_length, save_dir, logger)

    def generate_programs(self, save_path, temp_env, tensorboard_step):
        

        lang_class = temp_env.language_name 
        if "CSG3D" in lang_class:
            extractor_class = "CSG3DMetricExtractor"
        elif "CSG2D" in lang_class:
            extractor_class = "CSG2DMetricExtractor"
        elif "SA" in lang_class:
            extractor_class = "HSA3DMetricExtractor"

        start_time = time.time()
        print("LOADING %s for Beam Search" % save_path)
        if not os.path.exists(save_path):
            save_path = self.init_model_path
        policy, _, _, _ = load_all_weights(save_path, temp_env, instantiate_model=True,
                                        model_info=self.model_info, device=self.device)
        # Important
        policy.training = False
        policy.eval()
        policy.set_training_mode(False)
        keep_trying = True
        cur_n_proc = self.bs_n_proc
        cur_beam_n_batch=self.bs_batch_size
        if self.stochastic_bs:
            exhaustive = False
            n_eval_episodes = 4096
        else:
            # exhaustive = False
            # n_eval_episodes = 1500
            exhaustive = True
            n_eval_episodes = 0

        while(keep_trying):
            try:
                th.cuda.empty_cache()
                metric_obj, all_program_metric_obj, new_conf = parallel_CSG_beam_evaluate(policy, temp_env, gt_program=temp_env.gt_program, deterministic=self.bs_deterministic, 
                                                                                         callback=None, beam_k=self.bs_beam_size, beam_state_size=self.bs_beam_size,
                                                                                         beam_selector=self.bs_beam_selector, save_predictions=True, logger=None,
                                                                                         save_loc=os.path.join(self.logger.get_dir(), 'beam_%d' % self.bs_beam_size),
                                                                                         beam_n_proc=cur_n_proc, beam_n_batch=cur_beam_n_batch, return_all=self.bs_return_multiple,
                                                                                         n_eval_episodes=n_eval_episodes, # Since we will do an exhaustive enumeration
                                                                                         stochastic_bs=self.stochastic_bs,
                                                                                         exhaustive=exhaustive, extractor_class=extractor_class, reward_evaluation_limit=self.reward_evaluation_limit,
                                                                                         length_alpha=self.length_alpha)
                print("BS successful!")
                keep_trying = False
            except Exception as ex:
                print(ex)
                print(traceback.format_exc())
                print("failed with %d procs" % cur_n_proc)
                cur_n_proc = cur_n_proc - 1
                cur_beam_n_batch = int(cur_beam_n_batch / 6) # make the batch size smaller.
                print("Trying with %d processes" % cur_n_proc)
                if cur_n_proc == 0:
                    print("Programs are too complex!! Find a different way.")
                    raise ValueError("Cannot do this")
        # Keep the settings for next run
        self.bs_n_proc = new_conf['beam_n_proc']
        self.bs_batch_size = new_conf['beam_n_batch']
        print("Running settings %d processes and %d batch size" % (self.bs_n_proc, self.bs_batch_size))
                
        temp_env.program_generator.set_execution_mode(th.device("cuda"), th.float32)

        policy.cpu()
        del policy, _
        # # Save metric and use that.
        # file_name = "file_name"
        # with open(file_name, 'rb') as f:
        #     metric_obj, all_program_metric_obj  = cPickle.load(f)
        mean_metrics, all_metrics, predictions = metric_obj.return_metrics()
        if self.bs_return_multiple:
            _, all_metrics, predictions = all_program_metric_obj.return_metrics()
        
        # Merge the slot and predicts into one list:
        prog_objs = []
        for ind, prediction in enumerate(predictions):
            # if prediction[3] > 0: # Do Not use overly complex objects. 
            prog_obj = dict(expression=prediction[0],
                        slot_id=prediction[1],
                        target_id=prediction[2],
                        reward=prediction[3],
                        origin="BS",
                        do_fail=False,
                        cs_fail=False,
                        log_prob=prediction[4])
            prog_objs.append(prog_obj)
        # Have all the rewriters here
        end_time = time.time()
        for key, value in mean_metrics.items():
            self.logger.record("search/BS_ %s" % (key), value)
        self.logger.record('Training data/BS time', end_time - start_time)
        self.logger.record("Training data/New BS programs", len(prog_objs))
        self.logger.dump(tensorboard_step)

        return prog_objs
        


class WakeSleeper(BeamSearcher):
    
    def __init__(self, ws_config, config, bc_config, seed, model_info, save_dir, logger, device, reset_train_state, *args, **kwargs):

        self.enable  = ws_config.ENABLE
        self.save_dir = save_dir
        self.logger = logger
        self.device = device
        self.model_info = {
            "policy_model": model_info['policy_model'],
            "policy_class": model_info['policy_class'],
            "policy_kwargs": copy.deepcopy(model_info['policy_kwargs']),
            "config": model_info['config'].clone()
        }
        self.model_info['config'].MODEL = ws_config.MODEL.clone()
        self.model_info['policy_kwargs']['features_extractor_kwargs']['config'] = ws_config.MODEL.CONFIG.clone()
        self.model_info['train_state'] = WSTrainState
        

        self.config = config
        self.bc_config = bc_config
        self.seed = seed
        self.n_iters_per_epoch = bc_config.N_ITERS
        self.latent_execution = False
        self.init_ler = 0
        self.le_add_noise = False
        self.reset_train_state = reset_train_state
        self.num_workers = bc_config.NUM_WORKERS
        self.batch_size = bc_config.BATCH_SIZE

        # For multiple gradients
        self.collect_gradients = bc_config.COLLECT_GRADIENTS
        self.gradient_step_count = bc_config.GRADIENT_STEP_COUNT
        self.max_grad_norm = bc_config.MAX_GRAD_NORM

        self.score_threshold = ws_config.SCORE_THRESHOLD
        self.training_patience = ws_config.TRAINING_PATIENCE
        self.n_epochs = ws_config.N_EPOCHS
        self.kld_weight = ws_config.KLD_WEIGHT
        self.ent_weight = ws_config.ENT_WEIGHT
        self.sample_n_proc = ws_config.SAMPLE_N_PROC
        self.sample_count = ws_config.SAMPLE_COUNT
        self.sample_batch_size = ws_config.SAMPLE_BATCH_SIZE
        self.le_only_origins = bc_config.PLAD.LE_ONLY_ORIGINS
        self.best_prog_count = bc_config.PLAD.BPDS.BEST_PROG_COUNT
        self.language_name = ws_config.LANGUAGE_NAME
        self.sample_deviance = 10
        self.max_sampling_count = 3

        if ws_config.INPUT_GATING:
            self.valid_input_origins = ["BS"]
        else:
            self.valid_input_origins = ["BS", "DO", "GS", "CS"]

        self.init_weights = model_info['config'].MODEL.LOAD_WEIGHTS

        # For Scheduler
        if config.TRAIN.LR_SCHEDULER.TYPE in["WARM_UP", "ONE_CYCLE_LR"]:
            self.per_iter_scheduler = True
        else:
            self.per_iter_scheduler = False


    def generate_programs(self, bpds, master_ts, log_interval):

        # Create dataset:
        st = time.time()
        dataset = self.load_dataset()
        train_env = dataset.dataset.env.envs[0]

        if self.reset_train_state:
            self.reset_train_state = False
            load_path = self.init_weights
            reset_epoch = True
            strict = False
            load_optim = False
        else:
            new_load_path = os.path.join(self.save_dir, "WS_model.ptpkl")
            if os.path.exists(new_load_path):
                strict = True
                reset_epoch = False
                load_optim = False
                load_path = new_load_path
            else:
                load_path = self.init_weights
                reset_epoch = True
                strict = False
                load_optim = True

        policy, lr_scheduler, train_state, _ = load_all_weights(load_path=load_path, train_env=train_env, instantiate_model=True,
                                                   model_info=self.model_info, device=self.device, strict=strict, load_optim=load_optim)
        if reset_epoch:
            train_state = WSTrainState(0, self.n_epochs, self.n_iters_per_epoch)
        # Train the generative model:
        load_path = os.path.join(self.save_dir, "WS_model.ptpkl")
        st_1 = time.time()
        policy, lr_scheduler, train_state = self.train_model(policy, lr_scheduler, train_state, 
                                                             dataset, log_interval, bpds, load_path)
        train_state.ws_iteration += 1
        self.logger.record("search/WS training tme", time.time() - st_1)

        # Do this only if loss is better than previous best.
        del policy, lr_scheduler
        policy, _, _, _ = load_all_weights(load_path=load_path, train_env=train_env, instantiate_model=True,
                                                   model_info=self.model_info, device=self.device, strict=strict, load_optim=False)
        # sample programs:
        count = 0
        sample_count = self.sample_count
        total_prog_objs = []
        trials = 0
        while(count < self.sample_count):
            prog_objs = self.sample_from_model(policy, train_env, master_ts, train_state, sample_count)
            count += len(prog_objs)
            total_prog_objs.extend(prog_objs)
            sample_count = self.sample_count - count
            if sample_count < self.sample_deviance:
                break
            trials += 1
            if trials > self.max_sampling_count:
                break
        policy.cpu()
        del policy
        
        et = time.time()
        self.logger.record("Training data/WS time", et - st)
        self.logger.record("Training data/New WS programs", len(total_prog_objs))
        self.logger.dump(master_ts)

        return total_prog_objs

    def sample_from_model(self, policy, train_env, tensorboard_step, train_state, sample_count):
        # THe way to do it will be with the CSG forward and save, except using the mode to sample random noise instead of the CNN features. 
        policy.features_extractor.extractor.cnn_extractor.mode = "sample"

        lang_class = train_env.language_name 
        if "CSG3D" in lang_class:
            extractor_class = "CSG3DMetricExtractor"
        elif "CSG2D" in lang_class:
            extractor_class = "CSG2DMetricExtractor"
        elif "SA" in lang_class:
            extractor_class = "HSA3DMetricExtractor"
        start_time = time.time()
        policy.optimizer.zero_grad(set_to_none=True)
        policy.eval()
        policy.set_training_mode(False)
        policy.action_dist.distribution = None
        policy.features_extractor.extractor.cnn_extractor.kld_loss = None
        th.cuda.empty_cache()
        # train_env.program_generator.reload_data()
        keep_trying = True
        cur_n_proc = self.sample_n_proc
        cur_beam_n_batch=self.sample_batch_size
        while(keep_trying):
            try:
                metric_obj, _, new_conf = parallel_CSG_beam_evaluate(policy, train_env, gt_program=train_env.gt_program, deterministic=True, 
                                                        callback=None, beam_k=1, beam_state_size=1, beam_selector="rewards", save_predictions=True, logger=None,
                                                        save_loc=os.path.join(self.save_dir, 'samples'),
                                                        beam_n_proc=cur_n_proc, beam_n_batch=cur_beam_n_batch, return_all=False,
                                                        n_eval_episodes=sample_count, # Since we will do an exhaustive enumeration
                                                        stochastic_bs=True,
                                                        exhaustive=False, extractor_class=extractor_class)
                print("WS sampling successful!")
                keep_trying = False
            except Exception as ex:
                print(ex)
                print(traceback.format_exc())
                print("failed with %d procs" % cur_n_proc)
                cur_n_proc = cur_n_proc - 1
                print("Trying with %d processes" % cur_n_proc)
                if cur_n_proc == 0:
                    raise ValueError("Cannot do this!")
        
        self.sample_n_proc = new_conf['beam_n_proc']
        self.sample_batch_size = new_conf['beam_n_batch']
        print("Running settings %d processes and %d batch size" % (self.sample_n_proc, self.sample_batch_size))


        mean_metrics, all_metrics, predictions = metric_obj.return_metrics()
        # I think we should have avg. lengths and diversity.
        end_time = time.time()
        for key, value in mean_metrics.items():
            self.logger.record("search/WS_sample %s" % (key), value)
        self.logger.record('search/WS sampling time', end_time - start_time)
        self.logger.dump(tensorboard_step)
        
        # Merge the slot and predicts into one list:
        prog_objs = []
        for ind, prediction in enumerate(predictions):
            if prediction[3] >= 0:
                cur_log = prediction[4]
                action_len = train_env.action_space.expression_to_action(prediction[0]).shape[0]
                avg_nll = cur_log/action_len
                # if avg_nll < train_state.best_epoch_loss * LOG_MULTIPLIER:
                prog_obj = dict(expression=prediction[0],
                            slot_id="WS",
                            target_id=ind,
                            reward=0.0,
                            origin="WS",
                            do_fail=False,
                            cs_fail=False,
                            log_prob=prediction[4])
                prog_objs.append(prog_obj)
        policy.features_extractor.extractor.cnn_extractor.mode = "encode"

        return prog_objs


    def load_dataset(self):

        bc_env_class = getattr(csg_env, self.bc_config.ENV.TYPE)
        # bc_env = bc_env_class(config, config.BC, seed=seed)
        bc_env = DummyVecEnv([lambda ind=i: bc_env_class(config=self.config, phase_config=self.bc_config,
                                                         seed=self.seed + ind, n_proc=self.bc_config.N_ENVS, proc_id=ind) for i in range(self.bc_config.N_ENVS)])
        for env in bc_env.envs:
            env.program_generator.set_execution_mode(th.device("cuda"), th.float16)
        bc_env.reset()
        batch_iters = math.ceil(self.batch_size/self.bc_config.N_ENVS)
        bc_dataset = MultipleProgramBCEnvDataset(bc_env, self.n_iters_per_epoch * batch_iters,
                                                 self.init_ler, le_add_noise=self.le_add_noise, 
                                                 le_only_origins=self.le_only_origins)
        # Do not set latent execution only origins - we will not get them at sampling time.

        dataset = th.utils.data.DataLoader(bc_dataset, batch_size=batch_iters, pin_memory=False, collate_fn=format_data,
                                           num_workers=self.num_workers, shuffle=False)
        # TODO: Remove Hack.
        self.max_length = bc_env.envs[0].perm_max_len
        # program_list = get_program_list(best_program_dict, valid_origins=self.valid_input_origins, merge_origins=self.valid_input_origins)
        return dataset

    def construct_training_data(self, train_state, bpds):
        bc_env_class = getattr(csg_env, self.bc_config.ENV.TYPE)
        temp_env = bc_env_class(
            config=self.config, phase_config=self.bc_config, seed=self.seed, n_proc=1, proc_id=0)
        temp_env.mode = "EVAL"
        # Construct without generative:
        program_list = bpds.construct_training_data(None, temp_env, train_state, tag="WS", remove_generative=True)
        
        return program_list
    
    def train_model(self, policy, lr_scheduler, train_state, 
                    dataset, log_interval, bpds, load_path):


        dataset.dataset.update_ler(0.0)
        start_epoch = train_state.cur_epoch + 1
        for epoch in range(start_epoch, start_epoch + self.n_epochs):
        
            program_list = self.construct_training_data(train_state, bpds)
            dataset.dataset.update_program_list(program_list)
            
            self.log_start_time = time.time()
            self.start_time = time.time()
            policy.train()
            policy.set_training_mode(True)
            epoch_loss_list = []
            train_state.cur_epoch = epoch
            for iter_ind, (obs_tensor, target) in enumerate(dataset):
                
                stats_dict_it = train_state.get_state_stats()    
                # print("target min", target.min())
                # with th.cuda.amp.autocast():
                loss, stats_dict_loss, stats_dict_acc = self._calculate_loss(
                    policy, obs_tensor, target)
                # Optimization step
                if self.collect_gradients:
                    loss = loss / self.gradient_step_count
                    cur_ind = iter_ind % self.gradient_step_count
                    if cur_ind == (self.gradient_step_count-1) or (iter_ind+1) % self.n_iters_per_epoch == 0:
                        loss.backward()
                        policy.optimizer.step()
                        policy.optimizer.zero_grad(set_to_none=True)
                        if self.per_iter_scheduler:
                            lr_scheduler.step()
                        train_state.n_updates += 1
                    else:
                        loss.backward()
                else:
                    policy.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    policy.optimizer.step()
                    train_state.n_updates += 1
                    
                if self.per_iter_scheduler:
                    lr_scheduler.step()

                epoch_loss_list.append(loss.item())
                self.log_training_details(policy, stats_dict_it, stats_dict_loss, stats_dict_acc, log_interval, train_state)
                train_state.n_forwards += 1
                
            train_state.epoch_loss = np.nanmean(epoch_loss_list)

            train_state.cur_score = - train_state.epoch_loss

            if (train_state.epoch_loss + self.score_threshold) < train_state.best_epoch_loss:
                # We can train more:
                train_state.poor_epochs = 0
                train_state.best_epoch_loss = train_state.epoch_loss
                train_state.best_score = - train_state.epoch_loss
                train_state.best_epoch = epoch
                save_all_weights(policy, lr_scheduler, train_state, load_path)
            else:
                train_state.poor_epochs += 1

            if train_state.poor_epochs >= self.training_patience:
                # finish_search
                train_state.poor_epochs = 0
                break
        return policy, lr_scheduler, train_state

    def _calculate_loss(self, policy, obs, acts):

        policy.disable_mask()
        _, all_log_prob, entropy, max_action = policy.tformer_evaluate_actions_and_acc(obs, acts)
        policy.enable_mask()
        prob_true_act = th.exp(all_log_prob).mean()
        all_log_prob = all_log_prob.mean()
        
        neglogp = -all_log_prob
        entropy = entropy.mean()
        ent_loss = -self.ent_weight * entropy
        
        kld_loss = policy.features_extractor.extractor.cnn_extractor.kld_loss
        loss = neglogp + self.kld_weight * kld_loss + ent_loss

        # Calculate accuracy:
        acc_dict = policy.action_space.get_action_accuracy(acts, max_action)
        
        stats_dict = dict(
            neglogp=neglogp.item(),
            loss=loss.item(),
            ent_loss=ent_loss.item(),
            kld_loss=kld_loss.item(),
            prob_true_act=prob_true_act.item(),
        )
        acc_dict = {x:y.item() for x, y in acc_dict.items()}

        return loss, stats_dict, acc_dict


    def log_training_details(self, policy, stats_dict_it, stats_dict_loss, stats_dict_acc, log_interval, train_state):
        # Logging:
        stats_dict_it['LR'] = policy.optimizer.param_groups[0]['lr']
        stats_dict_it['Iters'] = train_state.n_forwards % self.n_iters_per_epoch
        train_state.tensorboard_step += 1
        if train_state.n_forwards % log_interval == 0:
            for k, v in stats_dict_it.items():
                self.logger.record(f"WS train Iter/{k}", v)
            for k, v in stats_dict_loss.items():
                self.logger.record(f"WS train Loss/{k}", v)
            for k, v in stats_dict_acc.items():
                self.logger.record(f"WS train Acc/{k}", v)
            fps = (log_interval) / float(time.time() - self.log_start_time)
            self.logger.record("time/WS fps", fps)
            self.log_start_time = time.time()
            self.logger.dump(train_state.tensorboard_step)
