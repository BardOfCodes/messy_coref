from termios import N_PPP
import torch as th
from torch.nn import functional as F
from gym import spaces
import numpy as np
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from collections import defaultdict
class ModExpTrainer:
    """
    Class for training on Modified Experience. 
    Goal is to keep it separate from the normal training.
    Basic Trainer only has BC loss and Entropy loss.
    
    Basline is simply Behavior Cloning on A2C + BC.  
    """
    def __init__(self, config, logger, rollout_buffer,
                 policy, mod_exp_collector, max_grad_norm=0.5) -> None:
        self._n_updates = 0
        self.enable = config.ENABLE
        self.batch_size = config.BATCH_SIZE
        self.max_epochs = config.MAX_EPOCH
        self.train_ratio_thres = config.TRAIN_RATIO_THRES
        self.recompute_advantages = config.RECOMPUTE_ADVANTAGES
        
        # logger:
        self.logger = logger
        self.rollout_buffer = rollout_buffer
        self.policy = policy
        self.max_grad_norm = max_grad_norm
        self.mod_exp_collector = mod_exp_collector
        self.action_space = rollout_buffer.action_space
        
        # Batch Update requirement
        self.update_batch_size_threshold =  0.25 * self.batch_size
        self.specific_init(config)
        
        
    def specific_init(self, config):
        self.bc_coef = config.LOSS.BC_COEF
        self.entropy_coef = config.LOSS.ENTROPY_COEF
    
    def train(self, normal_updates=None, progress_remaining=None):
        
        if not self.enable:
            # Log message
            print("Trained Disabled. No Training")
            return None
        
        buffer_size = self.rollout_buffer.buffer_size
        loss_avg_dict = defaultdict(lambda : list() )

        self.policy.set_training_mode(True)
        
        # train for n_epochs epochs
        self.rollout_buffer.enable_training_fetch()
        
        self.epoch_init(progress_remaining)
        total_updates  = 0
        mismatch_ratio = 1.0
        # self.policy.disable_mask()
        for epoch in range(self.max_epochs):
            n_updates = 0
            print("Epoch", epoch, "Mismatch ratio", mismatch_ratio)
            if normal_updates:
                condition_1 = total_updates/float(normal_updates) > self.train_ratio_thres
                # condition_2 = mismatch_ratio < 0.2
                if condition_1:
                    print("REACHED TRAIN THRESHOLD")
                    break
            total_batches = self.rollout_buffer.get_total_batches_per_epoch(self.batch_size)
            self.rollout_buffer.init_buffer_get(self.batch_size)
            while(self.rollout_buffer.iter_allowed(self.batch_size)):
                rollout_data = self.rollout_buffer.get_next(self.batch_size)
                actions = rollout_data.actions
                cur_batch_size = actions.shape[0]
                if cur_batch_size < self.update_batch_size_threshold:
                    # Too small batch
                    # Will result in high variance in RL loss
                    print("Found batch of size %d. Avoiding update." % cur_batch_size)
                    continue
                n_updates += 1
                total_updates += 1
                
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                # self.policy.disable_mask()
                loss, stats = self.get_loss(rollout_data, actions)
                # self.policy.enable_mask()
                for key, value in stats.items():
                    loss_avg_dict[key].append(value)
                
                self.policy.mod_exp_optimizer.zero_grad()
                # self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.mod_exp_optimizer.step()
                # self.policy.optimizer.step()
                mismatch_ratio = np.nanmean(loss_avg_dict['max_mismatch_ratio'])
            # post Epoch Update?
            if self.recompute_advantages and (epoch < self.max_epochs -1):
                self.policy.set_training_mode(False)
                self.mod_exp_collector.post_collection_update(self.policy)
                self.policy.set_training_mode(True)
                    
            # Logging:
            
            self._n_updates += n_updates
            # Logs
            for key, value in loss_avg_dict.items():
                self.logger.record("me_train/%s" % key, np.nanmean(value))
            self.logger.record("me_train/n_updates", self._n_updates)
            self.logger.dump(self._n_updates)
        update_ratio = total_updates/float(normal_updates)
        self.logger.record("me_train/update_ratio", update_ratio)
        self.logger.dump(self._n_updates)
        # self.policy.enable_mask()
        self.rollout_buffer.reset()
    
    def get_loss(self, rollout_data, actions):
        
        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

        # BC_LOSS 
        bc_loss = - th.mean(log_prob)
        
        entropy_loss = -th.mean(entropy)
        
        loss = self.entropy_coef * entropy_loss + self.bc_coef * bc_loss
        
        stats = dict(
            bc_loss=bc_loss.item(),
            entropy_loss=entropy_loss.item(),
        )
        
        return loss, stats

    def epoch_init(self, *args, **kwargs):
        pass
class A2CTrainer(ModExpTrainer):
    """
    Train with A2C loss.

    Returns:
        [type]: [description]
    """
    def specific_init(self, config):
        self.normalize_advantage = config.NORMALIZE_ADVANTAGE
        self.bc_coef = config.LOSS.BC_COEF
        self.entropy_coef = config.LOSS.ENTROPY_COEF
        self.policy_coef = config.LOSS.POLICY_COEF
        self.value_coef = config.LOSS.VALUE_COEF
        
    def get_loss(self, rollout_data, actions):
        
        values, log_prob, entropy, max_action = self.policy.evaluate_actions_and_acc(rollout_data.observations, actions)
        max_action_mismatch = (actions != max_action).float()
        # selective_bc = log_prob * max_action_mismatch
        # bc_loss = - th.mean(selective_bc)
        # BC_LOSS 
        # values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        bc_loss = - th.mean(log_prob)
        
        values = values.flatten()
        # Normalize advantage
        advantages = rollout_data.advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        policy_loss = -(advantages * log_prob).mean()
        
        # Value loss using the TD(gae_lambda) target
        value_loss = F.mse_loss(rollout_data.returns, values)
        entropy_loss = -th.mean(entropy)
        reinforce_loss = self.policy_coef * policy_loss + self.entropy_coef * entropy_loss + self.value_coef * value_loss
            
        loss = reinforce_loss + self.bc_coef * bc_loss
        
        stats = dict(
            bc_loss=bc_loss.item(),
            entropy_loss=entropy_loss.item(),
            policy_loss=policy_loss.item(),
            loss=loss.item(),
            value_loss=value_loss.item(),
            reinforce_loss=reinforce_loss.item(),
            max_mismatch_ratio = max_action_mismatch.mean().item()
        )
        
        return loss, stats
        
class SILTrainer(A2CTrainer):
    """Train with only positive advantage transitions.

    Returns:
        [type]: [description]
    """
    
    def get_loss(self, rollout_data, actions):
        
        # values, log_prob, entropy, max_action = self.policy.evaluate_actions_and_acc(rollout_data.observations, actions)
        # selective_bc = log_prob * (actions != max_action).float()
        # BC_LOSS 
        # bc_loss = - th.mean(selective_bc)
        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        bc_loss = - th.mean(log_prob)
        
        values = values.flatten()
        # Normalize advantage
        advantages = rollout_data.advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantage_indicator = (advantages>0).float()
        denom = ( 1e-10 + advantage_indicator.sum())
        policy_loss =(-(advantages * log_prob) * advantage_indicator).sum() / denom
        value_loss = F.mse_loss(rollout_data.returns * advantage_indicator,
                                values * advantage_indicator, reduction="sum") / denom
        entropy_loss = -th.mean(entropy)
        
        reinforce_loss = self.policy_coef * policy_loss + self.entropy_coef * entropy_loss + self.value_coef * value_loss
            
        loss = reinforce_loss + self.bc_coef * bc_loss
        
        stats = dict(
            bc_loss=bc_loss.item(),
            entropy_loss=entropy_loss.item(),
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            reinforce_loss=reinforce_loss.item()
        )
        
        return loss, stats
    


class PPOTrainer(A2CTrainer):
    """
    Use PPO training loss with the modified experience.
    """
    def specific_init(self, config):
        super(PPOTrainer, self).specific_init(config)
        self.clip_range_val = config.CLIP_RANGE
        self.clip_range_fn = get_schedule_fn(self.clip_range_val)
        self.clip_range_vf = config.CLIP_RANGE_VF
    
    def epoch_init(self, progress_remaining):
        
        self.clip_range = self.clip_range_fn(progress_remaining)
        
        if self.clip_range_vf:
            self.clip_range_vf = self.clip_range_vf(progress_remaining)
        
    def get_loss(self, rollout_data, actions):
        stats = {}
            
        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        values = values.flatten()
        # Normalize advantage
        advantages = rollout_data.advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # advantages = (advantages) / (advantages.std() + 1e-8)

        # Negative advantage with extremely high ratio - that is the problem.
        # ratio between old and new policy, should be one at the first iteration
        old_log_prob = th.clamp(rollout_data.old_log_prob,-10, 10)
        ratio = th.exp(log_prob - old_log_prob)
        
        stats['max_ratio'] = ratio.max().item()
        stats['min_ratio'] = ratio.min().item()
        ratio = th.clamp(ratio, 0, 2)

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio # th.clamp(ratio, 0.1, 100)
        policy_loss_2 = advantages * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

        # Logging
        stats['policy_loss'] = policy_loss.item()
        clip_fraction = th.mean((th.abs(ratio - 1) > self.clip_range).float()).item()
        stats['clip_fraction'] = clip_fraction

        if self.clip_range_vf is None:
            # No clipping
            values_pred = values
        else:
            # Clip the different between old and new value
            # NOTE: this depends on the reward scaling
            values_pred = rollout_data.old_values + th.clamp(
                values - rollout_data.old_values, -self.clip_range_vf, self.clip_range_vf
            )
        # Value loss using the TD(gae_lambda) target
        value_loss = F.mse_loss(rollout_data.returns, values_pred)
        stats['value_loss'] = value_loss.item()

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)

        stats['entropy_loss'] = entropy_loss.item()

        reinforce_loss = self.policy_coef * policy_loss + self.entropy_coef * entropy_loss + self.value_coef * value_loss
        
        if self.bc_coef:
            bc_loss = - th.mean(log_prob)
            stats["bc_loss"] = bc_loss.item()
        else: 
            bc_loss = 0.0
            stats["bc_loss"] = 0.0
        loss = reinforce_loss + self.bc_coef * bc_loss
        
        stats["reinforce_loss"] = reinforce_loss.item()
        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        # with th.no_grad():
        #     log_ratio = log_prob - rollout_data.old_log_prob
        #     approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
        #     stats['approx_kl_div'] = approx_kl_div.item()
        
        return loss, stats
    
    
class OffLineTrainer(PPOTrainer):
    """Train with only positive advantage transitions.

    Returns:
        [type]: [description]
    """
    
    def train(self, normal_updates=None, progress_remaining=None):
        
        if not self.enable:
            # Log message
            print("Trained Disabled. No Training")
            return None
        
        buffer_size = self.rollout_buffer.buffer_size
        if self.batch_size > buffer_size:
            # Don't train
            print("ME Buffer is not sufficiently filled. No Training")
            return None
        loss_avg_dict = defaultdict(lambda : list() )

        self.policy.set_training_mode(True)
        
        n_updates = 0
        # train for n_epochs epochs
        self.rollout_buffer.enable_training_fetch()
        
        self.epoch_init(progress_remaining)
        
        for epoch in range(self.max_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                cur_batch_size = actions.shape[0]
                if cur_batch_size < self.update_batch_size_threshold:
                    # Too small batch
                    # Will result in high variance in RL loss
                    print("Found batch of size %d. Avoiding update." % cur_batch_size)
                    break
                
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                
                loss, stats = self.get_q_function_loss(rollout_data, actions)
                
                
                for key, value in stats.items():
                    loss_avg_dict[key].append(value)
                


                self.policy.optimizer.zero_grad()
                # self.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                # self.optimizer.step()
                
                    
            # post Epoch Update?
            if self.recompute_advantages and (epoch < self.max_epochs -1):
                self.policy.set_training_mode(False)
                self.mod_exp_collector.post_collection_update(self.policy)
                self.policy.set_training_mode(True)
        
        for epoch in range(self.max_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                n_updates += 1
                actions = rollout_data.actions
                cur_batch_size = actions.shape[0]
                if cur_batch_size < self.update_batch_size_threshold:
                    # Too small batch
                    # Will result in high variance in RL loss
                    print("Found batch of size %d. Avoiding update." % cur_batch_size)
                    break
                
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                
                loss, stats = self.get_policy_function_loss(rollout_data, actions)
                
                
                for key, value in stats.items():
                    loss_avg_dict[key].append(value)
                


                self.policy.optimizer.zero_grad()
                # self.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                # self.optimizer.step()
                
                if normal_updates:
                    if n_updates/float(normal_updates) > self.train_ratio_thres:
                        print("REACHED TRAIN THRESHOLD")
                        break
                    
            # post Epoch Update?
            # if self.recompute_advantages and (epoch < self.max_epochs -1):
            #     self.policy.set_training_mode(False)
            #     self.mod_exp_collector.post_collection_update(self.policy)
            #     self.policy.set_training_mode(True)
            
                
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        # Logging:
        
        self._n_updates += n_updates
        update_ratio = n_updates/float(normal_updates)
        # Logs
        for key, value in loss_avg_dict.items():
            self.logger.record("me_train/%s" % key, np.nanmean(value))
        self.logger.record("me_train/explained_variance", explained_var)
        self.logger.record("me_train/update_ratio", update_ratio)
        self.logger.record("me_train/n_updates", self._n_updates)
        
        self.rollout_buffer.reset()
        
        
    def get_q_function_loss(self, rollout_data, actions):
        
        # Q-function maximizing loss.
        values = self.policy.predict_action_values(rollout_data.observations, actions)
        values = values.flatten()
        # if self.clip_range_vf is None:
        #     # No clipping
        #     values_pred = values
        # else:
        #     # Clip the different between old and new value
        #     # NOTE: this depends on the reward scaling
        #     values_pred = rollout_data.old_values + th.clamp(
        #         values - rollout_data.old_values, -self.clip_range_vf, self.clip_range_vf
        #     )
        values_pred = values
        value_loss = self.value_coef * F.mse_loss(rollout_data.returns, values_pred)

        
        stats = dict(
            value_loss=value_loss.item(),
        )
        
        return value_loss, stats

    def get_policy_function_loss(self, rollout_data, actions):
        
        # Q-function maximizing loss.
        action_distr, value_distr  = self.policy.get_action_and_value_distr(rollout_data.observations)

        # BC_LOSS 
        q_val = th.sum(action_distr.distribution.probs * value_distr.detach(), -1)
        q_val_loss = - th.mean(q_val)
        
        entropy = self.action_space.get_entropy(action_distr,  rollout_data.observations)
        entropy_loss = -th.mean(entropy)
        loss = self.policy_coef * q_val_loss + self.entropy_coef * entropy_loss
        
        stats = dict(
            q_val_loss=q_val_loss.item(),
            entropy_loss=entropy_loss.item()
        )
        
        return loss, stats
