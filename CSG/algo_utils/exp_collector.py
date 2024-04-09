

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from .exp_buffer import ModExpBuffer, SymbolicStateBuffer
from .ep_selector import EP_SELECTOR
from .ep_modifier import EP_MODIFIER
class ModExpCollector():

    def __init__(self, config, rollout_buffer, logger):
        
        # Initialize Variables
        self.total_time_steps = 0
        
        # Episode count:
        self.episode_count = 0
        self.accepted_episode_count = 0
        self.rejected_episode_count = 0
        
        self.gamma = rollout_buffer.gamma
        self.gae_lambda = config.GAE_LAMBDA # Completely MC Estimate
        
        self.mod_exp_buffer = ModExpBuffer(
            config.ME_BUFFER,
            rollout_buffer.observation_space,
            rollout_buffer.action_space,
            device=rollout_buffer.device,
        )
        
        ep_selector_cls = EP_SELECTOR[config.EP_SELECTOR.TYPE]
        self.ep_selector = ep_selector_cls(config.EP_SELECTOR)
        
        ep_modifier_cls = EP_MODIFIER[config.EP_MODIFIER.TYPE]
        self.ep_modifier = ep_modifier_cls(config.EP_MODIFIER)
        
        # logger:
        self.logger = logger
        
        
    
    def reset_stats(self):
        
        # Episode count:
        self.episode_count = 0
        self.accepted_episode_count = 0
        self.rejected_episode_count = 0
    
    def fetch_post_rollout_stats(self):
        acceptance_rate = self.accepted_episode_count / float(self.episode_count)
        rejection_rate = self.rejected_episode_count / float(self.episode_count)
        stats = {
            'total_ep_acr': acceptance_rate,
            'total_ep_rcr': rejection_rate,
            'total_rollout_episodes': self.episode_count,
            'total_time_steps': self.total_time_steps,
            'total_accepted_episodes': self.mod_exp_buffer.episode_count,
        }
        
        selector_stats = self.ep_selector.get_stats()
        stats.update(selector_stats)
        return stats
    
    def update_selector(self, *args, **kwargs):
        self.ep_selector.update_conditionals(*args, **kwargs)
        
    def flush_rollout(self):
        self.ep_selector.flush_data()
        self.mod_exp_buffer.flush_data()
    
    def collect_episode(self, env_id, reward, info, rollout_buffer, collection_step=None):
        
        # First extract the episode. 
        self.episode_count += 1
        
        if self.ep_selector.select(info=info, reward=reward, 
                                   env_id=env_id, rollout_buffer=rollout_buffer):
            # find the episode start pointer.
            # store the episode counter. 
            
            cur_pos = rollout_buffer.pos
            episode_start = cur_pos - 1
            while(True):
                start_signal = rollout_buffer.episode_starts[episode_start, env_id]
                if start_signal:
                    break
                else:
                    # When episode is from a certain point onwards.
                    if episode_start == 0:
                        break
                    episode_start -= 1
            ## Now replace the observation, reward
            if not episode_start == 0:
                episode_end = cur_pos
                # Perform relabelling - this will change based on setting.
                updated_episode_list = self.ep_modifier.modify(rollout_buffer=rollout_buffer, 
                                                            info=info, 
                                                            env_id=env_id, 
                                                            episode_start=episode_start, 
                                                            episode_end=episode_end)
                
                # Add the length: 
                for updated_episode in updated_episode_list:
                    episode_length = updated_episode['length']
                    updated_episode['collection_step'] = collection_step
                    self.total_time_steps += episode_length
                    # Add to buffer.
                    self.mod_exp_buffer.add(updated_episode)
                
                    self.accepted_episode_count += 1
            else:
                self.rejected_episode_count += 1
        else: 
            self.rejected_episode_count += 1
    
    def buffer_array_init(self):
        if self.mod_exp_buffer.buffer_size == 0:
            self.mod_exp_buffer.buffer_to_array()
            # IF even after this it is not 
            if self.mod_exp_buffer.buffer_size == 0:
                return None
            
    def post_collection_episodic_update(self, policy):
        old_ep_dict = self.mod_exp_buffer.buffer_dict
        new_episodes, stats= self.ep_modifier.bulk_modify(old_ep_dict, policy)
        if new_episodes:
            self.flush_rollout()
        
            for updated_episode in new_episodes:
                episode_length = updated_episode['length']
                updated_episode['collection_step'] = None
                self.total_time_steps += episode_length
                # Add to buffer.
                self.mod_exp_buffer.add(updated_episode)
        return stats
        
    def post_collection_update(self, model):
        """
        This is where we collect the value and log values 
        for each episode based on the latest policy.
        """
        
        # Update the buffer array - Observations, actions, rewards, and episode starts. 
        self.buffer_array_init()
        
        # Get large batches to compute log_probs and values: 
        log_prob_list = []
        value_list = []
        episode_start_list = []
        reward_list = []
        # store value and log probs
        for partial_rollout in self.mod_exp_buffer.partial_rollout():
            observations, actions, rewards, episode_starts = partial_rollout
            
            with th.no_grad():
                values, log_probs, _ = model.evaluate_actions_offline(observations, actions)
            if len(values.shape) == 2:
                values = values[:,0]
            values = values.cpu().numpy()
            log_probs  = log_probs.cpu().numpy()
            
            value_list.append(values)
            log_prob_list.append(log_probs)
            reward_list.append(rewards)
            episode_start_list.append(episode_starts)
        
        value_list = np.concatenate(value_list, 0)
        log_prob_list = np.concatenate(log_prob_list, 0)
        reward_list = np.concatenate(reward_list, 0)
        episode_start_list = np.concatenate(episode_start_list, 0)
        
        
        current_buffer_size = value_list.shape[0]
        approx_parallel_count = int(np.sqrt(current_buffer_size))
        approx_batch_size = max(28, current_buffer_size//approx_parallel_count)
        
        batch_indices = self.mod_exp_buffer.get_splitting_indices(current_buffer_size, 
                                                                  approx_batch_size, 
                                                                  episode_start_list)
        
        updated_args, lengths = self.split_pad_stack_array(batch_indices, value_list, 
                                                           reward_list, episode_start_list)
        values, rewards, episode_starts = updated_args
        # Make it a sequential array before splitting it up.
        
        returns, advantages = self.compute_returns_and_advantages(values, rewards,
                                                                  episode_starts)
        
        # Unstack the returns, advantages, values
        buffer_size, parallel_threads = returns.shape
        
        return_list = [returns[:lengths[i], i] for i in range(parallel_threads)]
        return_list = np.concatenate(return_list, 0)
        advantage_list = [advantages[:lengths[i], i] for i in range(parallel_threads)]
        advantage_list = np.concatenate(advantage_list, 0)
        
        # Return single sequence of log_probs, values, returns, advantages. 
        self.mod_exp_buffer.update_array(value_list, log_prob_list, return_list, advantage_list)
        
    
    # TODO: Should I move this to ModExpBuffer for consistency?
    def compute_returns_and_advantages(self, values,  rewards, episode_starts):
        """
        Based on code for buffer.
        """
        buffer_size, parallel_threads = values.shape
        # updated_returns = np.zeros((buffer_size, parallel_threads))
        advantages = np.zeros((buffer_size, parallel_threads))
        
        
        last_values = np.zeros((parallel_threads,))
        dones = np.ones((parallel_threads,))

        last_gae_lam = 0
        for step in reversed(range(buffer_size)):
            if step == buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - episode_starts[step + 1]
                next_values = values[step + 1]
            delta = rewards[step] + self.gamma * next_values * next_non_terminal - values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        returns = advantages + values
        return returns, advantages
    
    def buffer_stats(self):
        self.logger.record('me_train/new_transitions', self.mod_exp_buffer.b2a_new_count)
        self.logger.record('me_train/removed_transitions', self.mod_exp_buffer.b2a_removal_count)
        self.logger.record('me_train/transition_count', self.mod_exp_buffer.buffer_size)
        
        # Average age of the buffer:
        avg_age = self.mod_exp_buffer.get_avg_ep_age()
        self.logger.record('me_train/buffer_avg_age', avg_age)
        
        
    def split_pad_stack_array(self, batch_indices, *args):
        updated_args = []
        for arg in args:
            arg = np.split(arg, batch_indices)
            lens = [x.shape[0] for x in arg]
            max_length = max(lens)
            arg = [np.pad(x, pad_width=(0, max_length - x.shape[0])) for x in arg]
            arg = np.stack(arg, 1)
            updated_args.append(arg)
        return updated_args, lens
    

class BaselineModExpCollector(ModExpCollector):
    
    def __init__(self, *args, **kwargs):
        super(BaselineModExpCollector, self).__init__(*args, **kwargs)
        self.baseline_alpha = 0.7
        self.baseline_reward = 0
        self.cur_reward_baseline  = 0
        
    def update_selector(self, *args, **kwargs):
        self.ep_selector.update_conditionals(*args, **kwargs)
        self.cur_reward_baseline = args[0]

        
        
    # Calculate the Advantage as gamma * (Rt - Baseline)
    def compute_returns_and_advantages(self, values,  rewards, episode_starts):
        """
        Based on code for buffer.
        """
        buffer_size, parallel_threads = values.shape
        # updated_returns = np.zeros((buffer_size, parallel_threads))
        advantages = np.zeros((buffer_size, parallel_threads))
        
        self.baseline_reward = self.baseline_alpha * self.baseline_reward + (1-self.baseline_alpha) * self.cur_reward_baseline
        last_values = np.zeros((parallel_threads,))
        dones = np.ones((parallel_threads,))
        

        last_gae_lam = 0
        for step in reversed(range(buffer_size)):
            if step == buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - episode_starts[step + 1]
                next_values = advantages[step + 1]
            delta = rewards[step] - (1-next_non_terminal) * self.baseline_reward + self.gamma * next_values * next_non_terminal 
            advantages[step] = delta
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        
        # Since returns are not used:
        returns = advantages * 0
        return returns, advantages

        
class ReComputingModExpCollector(ModExpCollector):

    def buffer_array_init(self):
        
        self.mod_exp_buffer.buffer_to_array()
        if self.mod_exp_buffer.buffer_size == 0:
            return
        self.buffer_stats()
        

class SymbolicBufferExpCollector(ReComputingModExpCollector):
    
    def __init__(self, config, rollout_buffer, logger):
        
        # Initialize Variables
        self.total_time_steps = 0
        
        # Episode count:
        self.episode_count = 0
        self.accepted_episode_count = 0
        self.rejected_episode_count = 0
        
        self.gamma = rollout_buffer.gamma
        self.gae_lambda = config.GAE_LAMBDA # Completely MC Estimate
        
        self.mod_exp_buffer = SymbolicStateBuffer(
            config.ME_BUFFER,
            rollout_buffer.observation_space,
            rollout_buffer.action_space,
            device=rollout_buffer.device,
        )
        
        ep_selector_cls = EP_SELECTOR[config.EP_SELECTOR.TYPE]
        self.ep_selector = ep_selector_cls(config.EP_SELECTOR)
        
        
        ep_modifier_cls = EP_MODIFIER[config.EP_MODIFIER.TYPE]
        self.ep_modifier = ep_modifier_cls(config.EP_MODIFIER)
        
        self.mod_exp_buffer.mod_env = self.ep_modifier.mod_env
        # logger:
        self.logger = logger
        
    
    def post_collection_update(self, model):
        """
        This is where we collect the value and log values 
        for each episode based on the latest policy.
        """
        # Can also just skip this.
        return 0
    
    def update_selector(self, *args, **kwargs):
        super(SymbolicBufferExpCollector, self).update_selector()
        self.mod_exp_buffer.update_baseline_reward(*args, **kwargs)
