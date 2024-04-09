
from attr import asdict
from gym import spaces
import torch as th
from typing import Union
from collections import defaultdict
import numpy as np
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None
    
class ModExpBuffer(DictRolloutBuffer):
    
    def __init__(self,
                 config,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 device: Union[th.device, str] = "cpu"):
    
        self.approx_batch_size = config.APPROX_BATCH_SIZE
        self.episode_budget = config.EPISODE_BUDGET
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        

        self.action_dim = get_action_dim(action_space)
        self.full = False
        self.device = device
        
        self.buffer_dict = defaultdict(lambda: defaultdict(lambda : None))
        
        self.buffer_idx = 0
        self.episode_count = 0
        self.n_envs = 1
        self.updated_episodes = []
        self.episode_lengths = np.zeros((self.episode_budget))
        
        # buffer related:
        self.buffer_idx_in_array = []
        self.buffer_idx_length = np.zeros((self.episode_budget))
        
        
        self.init_rollout_buffer()
    def flush_data(self):
        """Remove All the data
        """
        
        self.buffer_dict = defaultdict(lambda: defaultdict(lambda : None))
        self.buffer_idx = 0
        
        self.updated_episodes = []
        self.episode_lengths = np.zeros((self.episode_budget))
        self.buffer_idx_in_array = []
        self.buffer_idx_length = np.zeros((self.episode_budget))
        self.init_rollout_buffer()
        
    def get_avg_ep_age(self):
        avg_age = [ep['collection_step'] for id, ep in self.buffer_dict.items()]
        if avg_age:
            avg_age = np.nanmean(avg_age)
        else:
            avg_age = 0
        return avg_age
            
    def add(self, episode):
        self.buffer_dict[self.buffer_idx] = episode
        
        self.episode_lengths[self.buffer_idx] = episode['length']
        if self.buffer_idx in self.updated_episodes:
            self.updated_episodes.remove(self.buffer_idx)
        self.updated_episodes.append(self.buffer_idx)
        
        self.episode_count += 1
        self.buffer_idx = (self.buffer_idx + 1) % self.episode_budget
        
    def remove_obsolete_episodes(self):
        """
        The goal is to remove episodes which are too old.
        """
        # Log the removal of obselete episodes. 
        # 
        raise NotImplemented
    
    def create_empty_buffer(self, count):
        
        observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            observations[key] = np.zeros((count,) + obs_input_shape, dtype=np.float32)
        actions = np.zeros((count, self.action_dim), dtype=np.int32)
        rewards = np.zeros((count), dtype=np.float32)
        episode_starts = np.zeros((count), dtype=np.float32)
        
        # Do we even need these arrays in init?
        values = np.zeros((count, 1), dtype=np.float32)
        log_probs = np.zeros((count, 1), dtype=np.float32)
        advantages = np.zeros((count, 1), dtype=np.float32)
        returns = np.zeros((count, 1), dtype=np.float32)
        return observations, actions, rewards, returns, episode_starts, values, log_probs, advantages
        
    def init_rollout_buffer(self):
        observations, actions, rewards, returns, episode_starts, values, log_probs, advantages = self.create_empty_buffer(0)
        
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.returns = returns
        self.episode_starts = episode_starts
        self.values = values
        self.log_probs = log_probs
        self.advantages = advantages
        self.buffer_size = 0
        # super(RolloutBuffer, self).reset()
        
        
    def buffer_to_array(self):
        """
        Convert the collected experiences into DictRolloutBufferSamples
        """
        transition_count = np.sum([self.episode_lengths[i] for i in self.updated_episodes])
        transition_count = int(transition_count)
        observations, actions, rewards, returns, episode_starts, values, log_probs, advantages = self.create_empty_buffer(transition_count)
        
        # temp log:
        print("Updating array for %d episodes with %d transitions" % (len(self.updated_episodes),
                                                                         transition_count))
        # now fill in the new observations:
        idx = 0
        removal_length = 0
        for new_episode_id in self.updated_episodes:
            episode_length = int(self.episode_lengths[new_episode_id])
            
            start_idx = idx
            end_idx = idx + episode_length
            cur_episode = self.buffer_dict[new_episode_id]
            
            cur_obs = cur_episode['observations']
            for key, value in cur_obs.items():
                observations[key][start_idx: end_idx] = value
            
            actions[start_idx: end_idx] = cur_episode['actions']
            
            rewards[start_idx: end_idx] = cur_episode['rewards']
            episode_starts[start_idx] = 1
            # Values and Log probs calculated before backward pass. 
            # Advantage and returns calculated after value and log-probs.
            
            # Now, if this episode is in the buffer idx, then remove its last instance. 
            # Since episodes will be called sequentially, we can remove the initial k-transitions.
            if new_episode_id in self.buffer_idx_in_array:
                # We have to remove this item
                removal_length += self.buffer_idx_length[new_episode_id]
            else:
                self.buffer_idx_in_array.append(new_episode_id)
            self.buffer_idx_length[new_episode_id] = episode_length
            
            idx += episode_length
        
        # Now, update all:
        removal_length = int(removal_length)
        
        # TODO: Clean this up.
        for key, value in self.observations.items():
            self.observations[key] = np.concatenate([self.observations[key][removal_length:], observations[key]], 0)
        
        self.actions = np.concatenate([self.actions[removal_length:], actions], 0) 
        self.rewards = np.concatenate([self.rewards[removal_length:], rewards], 0) 
        self.episode_starts = np.concatenate([self.episode_starts[removal_length:], episode_starts], 0) 
        
        self.values = np.concatenate([self.values[removal_length:], values], 0) 
        self.log_probs = np.concatenate([self.log_probs[removal_length:], log_probs], 0) 
        self.returns = np.concatenate([self.returns[removal_length:], returns], 0) 
        self.advantages = np.concatenate([self.advantages[removal_length:], advantages], 0)
        
        # stats:
        self.b2a_new_count = transition_count    
        self.b2a_removal_count = removal_length
        
        self.buffer_size = self.actions.shape[0]
        
        self.updated_episodes = [] 
        
    @staticmethod
    def get_splitting_indices(current_buffer_size, batch_size, episode_starts):
        """[summary]
        Given current buffer size, return splitting indexest to create a buffer
        based on the episode starts.
        Args:
            current_buffer_size ([type]): [description]
            batch_size ([type]): [description]
            episode_starts ([type]): [description]

        Returns:
            [type]: [description]
        """
        batch_indices = []
        cur_batch_idx = 0
        append_idx = True
        while((cur_batch_idx + batch_size) < current_buffer_size):
            new_idx = cur_batch_idx + batch_size
            ep_start =  episode_starts[new_idx]
            while(not ep_start):
                new_idx += 1
                if new_idx == current_buffer_size:
                    # Reached the end of the buffer
                    append_idx = False
                    break
                ep_start =  episode_starts[new_idx]
            if append_idx:
               batch_indices.append(new_idx)
            cur_batch_idx = new_idx
        return batch_indices
    
    def partial_rollout(self):
        """
        Since training requires the latest values and log-probs for correct estimate,
        we update the log probs and values of all episodes. 
        """
        
        current_buffer_size = self.actions.shape[0]
        batch_indices = self.get_splitting_indices(current_buffer_size, 
                                                self.approx_batch_size, 
                                                self.episode_starts)
        
        batch_indices.insert(0, 0)
        batch_indices.append(current_buffer_size)
        
        # now create a iterator which the trainer can take. 
        # also perform conversion to torch.
        batch_count = len(batch_indices) -1
        
        
        for ind in range(batch_count):
            cur_batch_start = batch_indices[ind]
            cur_batch_end = batch_indices[ind+1]
            
            # Need to send only observations, actions, rewards, episode starts
            observations = {}
            for key, value in self.observations.items():
                observations[key] = self.observations[key][cur_batch_start:cur_batch_end]
            actions = self.actions[cur_batch_start:cur_batch_end]
            rewards = self.rewards[cur_batch_start:cur_batch_end]
            episode_starts = self.episode_starts[cur_batch_start:cur_batch_end]
            
            for key, value in observations.items():
                observations[key] = self.to_torch(value)
            actions = self.to_torch(actions)
            
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()
            yield (observations, actions, rewards, episode_starts)
                
    def update_array(self, values, log_probs, returns, advantages):
        
        self.values[:,0] = values
        self.log_probs[:, 0] = log_probs
        self.returns[:, 0] = returns
        self.advantages[:, 0] = advantages
        
    def reset(self):
        self.generator_ready = False
        self.full = False
        
    def enable_training_fetch(self):
        self.generator_ready = True
        self.full = True
 

class SymbolicStateBuffer(ModExpBuffer):
    
    def __init__(self,
                 config,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 device: Union[th.device, str] = "cpu"):
        
        self.episode_batch_size = config.EPISODE_BATCH_SIZE
        self.cur_episode_batch_idx = 0
        super(SymbolicStateBuffer, self).__init__(config, observation_space, action_space, device)
        self.mod_env = None
        self.total_batches = 0
        # HACK - set if from exp modifier. 
        self.gamma = config.GAMMA
        self.gae_lambda = config.GAE_LAMBDA
        self.baseline_alpha = config.BASELINE_ALPHA
        self.baseline_reward = 0
        self.cur_reward_baseline  = 0
        
    def add(self, episode):
        # DONOT save the obs
        episode['observations'] = {}
        assert 'target_id' in episode.keys()
        assert 'slot_id' in episode.keys()
        super(SymbolicStateBuffer, self).add(episode)
        
    def flush_data(self):
        super(SymbolicStateBuffer, self).flush_data()
        self.cur_episode_batch_idx = 0
    
    def get_total_batches(self):
        self.total_batches = np.ceil(len(self.updated_episodes)/self.episode_batch_size).astype(int)
        return self.total_batches
    
    def get_total_batches_per_epoch(self, batch_size):
        
        transition_count = np.sum([self.episode_lengths[i] for i in self.updated_episodes]).astype(int)
        
        total_batches = np.ceil(transition_count/batch_size).astype(int)
        return total_batches
        
    def buffer_to_array(self):
        """
        Chief difference - create env from scratch for selected episodes. 
        """
        
        self.total_batches = np.ceil(len(self.updated_episodes)/self.episode_batch_size).astype(int)
        cur_idx = self.cur_episode_batch_idx % self.total_batches
        
        ##  Also allow random selection
        selected_episodes = self.updated_episodes[cur_idx * self.episode_batch_size: (cur_idx +1) * self.episode_batch_size]
        print('self.cur_episode_batch_idx', self.cur_episode_batch_idx)
        
        transition_count = np.sum([self.episode_lengths[i] for i in selected_episodes]).astype(int)
        
        observations, actions, rewards, returns, episode_starts, values, log_probs, advantages = self.create_empty_buffer(transition_count)
    
        # temp log:
        print("Updating array for %d episodes with %d transitions" % (len(selected_episodes),
                                                                        transition_count))
        # now fill in the new observations:
        idx = 0
        removal_length = 0
        for new_episode_id in selected_episodes:
            episode_length = int(self.episode_lengths[new_episode_id])
            
            start_idx = idx
            end_idx = idx + episode_length
            cur_episode = self.buffer_dict[new_episode_id]
            
            slot_id, target_id = cur_episode['slot_id'], cur_episode['target_id']
            
            cur_actions = cur_episode['actions']
            cur_obs = self.mod_env.generate_observations(slot_id, target_id, cur_actions)
            
            for key, value in cur_obs.items():
                observations[key][start_idx: end_idx] = value
                
            actions[start_idx: end_idx] = cur_actions[:,0,:]
            
            rewards[start_idx: end_idx] = cur_episode['rewards']
            episode_starts[start_idx] = 1
            # Values and Log probs calculated before backward pass. 
            # Advantage and returns calculated after value and log-probs.
            
            # Now, if this episode is in the buffer idx, then remove its last instance. 
            # Since episodes will be called sequentially, we can remove the initial k-transitions.
            if new_episode_id in self.buffer_idx_in_array:
                # We have to remove this item
                removal_length += self.buffer_idx_length[new_episode_id]
            else:
                self.buffer_idx_in_array.append(new_episode_id)
            self.buffer_idx_length[new_episode_id] = episode_length
            
            idx += episode_length
        
        # Now, update all:
        # removal_length = int(removal_length)
        
        # TODO: Clean this up.
        for key, value in self.observations.items():
            self.observations[key] = observations[key]
        
        self.actions = actions # np.concatenate([self.actions[removal_length:], actions], 0) 
        self.rewards = rewards # np.concatenate([self.rewards[removal_length:], rewards], 0) 
        self.episode_starts = episode_starts# np.concatenate([self.episode_starts[removal_length:], episode_starts], 0) 
        
        self.values = values# np.concatenate([self.values[removal_length:], values], 0) 
        self.log_probs = log_probs# np.concatenate([self.log_probs[removal_length:], log_probs], 0) 
        self.returns = returns # np.concatenate([self.returns[removal_length:], returns], 0) 
        self.advantages = advantages# np.concatenate([self.advantages[removal_length:], advantages], 0)
        
        # stats:
        self.b2a_new_count = transition_count    
        self.b2a_removal_count = removal_length
        
        self.buffer_size = self.actions.shape[0]
        
        # self.updated_episodes = [] 
        self.cur_episode_batch_idx += 1
            
    def get(self, batch_size):
        # Shuffle updated episodes:
        np.random.shuffle(self.updated_episodes)
        _ = self.get_total_batches()
        j = 0
        start_idx = np.inf
        while j < self.total_batches:
            if start_idx > self.buffer_size * self.n_envs:
                start_idx = 0
                self.buffer_to_array()
                ## Function to compute the values etc if required.
                print('cur j from total batches', j)
                self.indices = np.random.permutation(self.buffer_size * self.n_envs)
                # Prepare the data
                j += 1
            yield self._get_samples(self.indices[start_idx : start_idx + batch_size])
            print(start_idx, len(self.indices)) 
            start_idx += batch_size
            
    def init_buffer_get(self, batch_size):
        
        np.random.shuffle(self.updated_episodes)
        _ = self.get_total_batches()
        self.j = -1
        self.start_idx = np.inf
    
    def get_next(self, batch_size):
            # Prepare the data
        output =  self._get_samples(self.indices[self.start_idx : self.start_idx + batch_size])
        self.start_idx += batch_size
        return output
    
    def iter_allowed(self, batch_size):
        if self.j < self.total_batches:
            if self.start_idx > self.buffer_size * self.n_envs:
                self.j += 1
                self.start_idx = 0
                if self.j < self.total_batches:
                    self.buffer_to_array()
                    ## If enabled do it:
                    self.compute_returns_and_advantages()
                    self.indices = np.random.permutation(self.buffer_size * self.n_envs)
                    return True
                else:
                    return False
            else:
                return True
        else:
            return False

    def compute_returns_and_advantages(self):
        """
        Based on code for buffer.
        """
        
        buffer_size, parallel_threads = self.values.shape
        # updated_returns = np.zeros((buffer_size, parallel_threads))
        # advantages = np.zeros((buffer_size, parallel_threads))
        
        last_values = np.zeros((parallel_threads,))
        dones = np.ones((parallel_threads,))
        

        last_gae_lam = 0
        for step in reversed(range(buffer_size)):
            if step == buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.advantages[step + 1]
            delta = self.rewards[step] - (1-next_non_terminal) * self.baseline_reward + self.gamma * next_values * next_non_terminal 
            self.advantages[step] = delta
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        
        # Since returns are not used:

    def update_baseline_reward(self, *args, **kwargs):
        self.cur_reward_baseline = args[0]
        self.baseline_reward = self.baseline_alpha * self.baseline_reward + (1-self.baseline_alpha) * self.cur_reward_baseline
        print('baseline reward in mod exp buffer', self.baseline_reward)

class DiverseExpPPO(ModExpBuffer):
    """Additional Content which delianates different episodes.

    Args:
        ModExpBuffer ([type]): [description]
    """
    pass