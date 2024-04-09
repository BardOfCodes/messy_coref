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
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv
import CSG.env as csg_env
from .rewrite_engines.dataset import BCEnvDataset
from .rewrite_engines.utils import format_data, LabelSmoothing
from CSG.utils.train_utils import load_all_weights, save_all_weights, resume_checkpoint_filename
import _pickle as cPickle
from .rewrite_engines.train_state import BCTrainState
from stable_baselines3.common.vec_env import SubprocVecEnv

        
class AdaptedBC():
    """Behavioral cloning (BC).

    Recovers a policy via supervised learning from observation-action pairs.
    """
    def __init__(self, bc_config, save_dir, seed, config, action_space, observation_space,
                 model_info, train_env, custom_logger,
                 device: Union[str, th.device] = "auto",
                 *args, **kwargs):
        
        # super(BC, self).__init__(*args, **kwargs)

        self.logger = custom_logger
        
        self.action_space = action_space
        self.observation_space = observation_space
        
        self.device = utils.get_device(device)

        # Create policy and lr

        # self._policy = None
        self.mode = "TFORMER"

        self.label_smoothing = bc_config.LABEL_SMOOTHING
        self.ls_size = bc_config.LS_SIZE
        self.ls_padding_idx = bc_config.LS_PADDING_IDX
        self.batch_size = bc_config.BATCH_SIZE
        self.ent_weight = bc_config.ENT_WEIGHT
        self.l2_weight = bc_config.L2_WEIGHT
        self.n_epochs = bc_config.EPOCHS
        self.n_iters_per_epoch = bc_config.N_ITERS
        self.num_workers = bc_config.NUM_WORKERS

        # For multiple gradients
        self.collect_gradients = bc_config.COLLECT_GRADIENTS
        self.gradient_step_count = bc_config.GRADIENT_STEP_COUNT
        self.max_grad_norm = bc_config.MAX_GRAD_NORM
        
        if self.label_smoothing:
            self.nllloss = LabelSmoothing(self.ls_size, self.ls_padding_idx)
        else:
            self.nllloss = th.nn.NLLLoss()
            
        self.save_dir = save_dir
        self.seed = seed
        self.config = config
        self.bc_config = bc_config
        self.reward_weighting = False # bc_config.REWARD_WEIGHTING
        self.reset_epoch = True
        if config.TRAIN.RESUME_CHECKPOINT:
            self.init_model_path = resume_checkpoint_filename(config.SAVE_DIR)
            self.reset_epoch = False
            if self.init_model_path is None:
                self.init_model_path = config.MODEL.LOAD_WEIGHTS
                self.reset_epoch = True
        else:
            self.init_model_path = config.MODEL.LOAD_WEIGHTS
            self.reset_epoch = True
        self.model_info = model_info
        self.model_info['train_state'] = BCTrainState
        
        # For Scheduler
        if config.TRAIN.LR_SCHEDULER.TYPE in["WARM_UP", "ONE_CYCLE_LR"]:
            self.per_iter_scheduler = True
        else:
            self.per_iter_scheduler = False
            
        # self.init_data_loader()
        self.save_epoch = bc_config.SAVE_EPOCHS

    def init_data_loader(self):
        
        
        bc_env_class = getattr(csg_env, self.bc_config.ENV.TYPE)
        # bc_env = bc_env_class(config, config.BC, seed=seed)
        bc_env = DummyVecEnv([lambda ind=i: bc_env_class(config=self.config, phase_config=self.bc_config, 
                                                         seed=self.seed + ind, n_proc=self.bc_config.N_ENVS, proc_id=ind) for i in range(self.bc_config.N_ENVS)])
        for env in bc_env.envs:
            env.program_generator.set_execution_mode(th.device("cpu"), th.float32)
        # bc_env.reset()
        batch_iters = math.ceil(self.batch_size/self.bc_config.N_ENVS)
        bc_dataset = BCEnvDataset(bc_env, self.n_iters_per_epoch * self.n_epochs * batch_iters)
        dataset = th.utils.data.DataLoader(bc_dataset, batch_size=batch_iters, pin_memory=False,
                                            num_workers=self.num_workers, shuffle=False, collate_fn=format_data)
        return dataset, bc_env
        

        
    def _calculate_loss(
        self,
        policy,
        obs: Union[th.Tensor, np.ndarray],
        acts: Union[th.Tensor, np.ndarray],
        reward_weights=None
    ) -> Tuple[th.Tensor, Mapping[str, float]]:

        # obs = th.as_tensor(obs, device=self.device).detach()
        # acts = th.as_tensor(acts, device=self.device).detach()
        # y_lens = obs['cur_step'].clone().detach()
        policy.disable_mask()
        _, all_log_prob, entropy, max_action = policy.tformer_evaluate_actions_and_acc(obs, acts)
        policy.enable_mask()
        prob_true_act = th.exp(all_log_prob).mean()
        if self.reward_weighting:
            # Multiply by probability, with no backprop:
            # all_log_prob = all_log_prob * (reward_weights) * th.exp(all_log_prob.detach())
            seq_avg_logprob = []
            seq_rewards = []
            cumalative_index = 0
            batch_size = obs['obs'].shape[0]
            for i in range(batch_size):
                cur_seq_log_prob = th.sum(all_log_prob[cumalative_index:cumalative_index + y_lens[i]])
                # cur_seq_log_prob = th.mean(all_log_prob[cumalative_index:cumalative_index + y_lens[i]])
                seq_avg_logprob.append(cur_seq_log_prob)
                seq_rewards.append(reward_weights[cumalative_index])
                cumalative_index += y_lens[i]
            seq_rewards = th.stack(seq_rewards, 0)
            seq_avg_logprob = th.stack(seq_avg_logprob, 0)
            seq_rewards = th.where(seq_rewards > 0, seq_rewards, 0)
            all_log_prob = seq_avg_logprob * seq_rewards
            
        all_log_prob = all_log_prob.mean()
        entropy = entropy.mean()
        l2_norms = [th.sum(th.square(w)) for w in policy.parameters()]
        l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square

        ent_loss = -self.ent_weight * entropy
        if self.label_smoothing:
            neglogp = self.nllloss(policy.action_dist.distribution.logits, acts)
        else:
            neglogp = - all_log_prob
        l2_loss = self.l2_weight * l2_norm
        loss = neglogp + ent_loss + l2_loss

        # Calculate accuracy:
        acc_dict = policy.action_space.get_action_accuracy(acts, max_action)
        
        stats_dict = dict(
            neglogp=neglogp.item(),
            loss=loss.item(),
            entropy=entropy.item(),
            ent_loss=ent_loss.item(),
            prob_true_act=prob_true_act.item(),
            l2_norm=l2_norm.item(),
            l2_loss=l2_loss.item(),
        )
        
        acc_dict = {x:y.item() for x, y in acc_dict.items()}

        return loss, stats_dict, acc_dict
    
    
    def train(
        self,
        train_eval,
        val_eval,
        log_interval,
        on_epoch_end=None,
        on_batch_end=None,
    ):

        dataset, train_env = self.init_data_loader()
        policy, lr_scheduler, train_state, _, = load_all_weights(load_path=self.init_model_path, train_env=train_env, model_info=self.model_info,
                                                    device=self.device, instantiate_model=True)
        # print("Reinitializing the entire net")
        if self.reset_epoch:
            train_state = BCTrainState(0, self.n_epochs, self.n_iters_per_epoch)

        self.start_time = time.time()
        self.log_start_time = time.time()
        for iter_ind, (obs_tensor, target) in enumerate(dataset):
            # for key, value in obs_tensor.items():
            #     obs_tensor[key] = value.to("cuda", non_blocking=True)
            # target = target.to("cuda", non_blocking=True)
            stats_dict_it = train_state.get_state_stats()
            # with th.cuda.amp.autocast():
            loss, stats_dict_loss, stats_dict_acc = self._calculate_loss(policy, obs_tensor, target)
            # Optimization step
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
            self.log_training_details(policy, stats_dict_it, stats_dict_loss, stats_dict_acc, log_interval, train_state)
            
            if on_batch_end is not None:
                x = on_batch_end.on_step()
                
            if (train_state.n_updates + 1) % self.n_iters_per_epoch == 0:
                # Epoch End Part:

                th.cuda.empty_cache()
                policy.optimizer.zero_grad(set_to_none=True)
                policy.eval()
                policy.set_training_mode(False)
                policy.action_dist.distribution = None

                for cur_eval in [train_eval, val_eval]:
                    cur_eval.n_calls += 1
                    cur_eval.num_timesteps = train_state.tensorboard_step
                    cur_eval._on_step(policy, lr_scheduler)


                train_state.cur_score = val_eval.main_metric
                if train_state.cur_score > train_state.best_score:
                    train_state.best_score = train_state.cur_score
                    train_state.best_epoch = train_state.cur_epoch


                # To save or not:
                if (train_state.cur_epoch % self.save_epoch == 0):
                    save_path = os.path.join(self.save_dir, "weights_%d.ptpkl" % train_state.cur_epoch)
                    save_all_weights(policy, lr_scheduler, train_state, save_path)

                train_state.cur_epoch += 1
                th.cuda.empty_cache()
                policy.train()
                policy.set_training_mode(True)
                self.start_time = time.time()

    def log_training_details(self, policy, stats_dict_it, stats_dict_loss, stats_dict_acc, log_interval, train_state):
        # Logging:
        stats_dict_it['LR'] = policy.optimizer.param_groups[0]['lr']
        stats_dict_it['Iters'] = train_state.n_forwards % self.n_iters_per_epoch
        if train_state.n_forwards % log_interval == 0:
            for k, v in stats_dict_it.items():
                self.logger.record(f"BC train Iter/{k}", v)
            for k, v in stats_dict_loss.items():
                self.logger.record(f"BC train Loss/{k}", v)
            for k, v in stats_dict_acc.items():
                self.logger.record(f"BC train Acc/{k}", v)
            fps = (log_interval) / float(time.time() - self.log_start_time)
            self.logger.record("time/fps", fps)
            
            self.logger.dump(train_state.tensorboard_step)
            self.log_start_time = time.time()
