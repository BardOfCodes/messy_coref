from typing import Any, DefaultDict, Dict, Optional, Type, Union

import torch as th
from gym import spaces
import gym
import numpy as np
from torch.nn import functional as F
from collections import defaultdict
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

from .ppo_mod import ModPPO


class BaselineDictRolloutBuffer(DictRolloutBuffer):
    
    def __init__(self, *args, **kwargs):
        super(BaselineDictRolloutBuffer, self).__init__(*args, **kwargs)
        self.seq_length = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.alpha = 0.7
        self.baseline_reward = 0
    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ):
        super(BaselineDictRolloutBuffer, self).add(obs, action, reward, episode_start, value, log_prob)
        
        self.pos = self.pos - 1
        if self.pos == 0:
            ind = -1
        else:
            ind = self.pos - 1 
        self.seq_length[self.pos] = 1 + (1 - self.episode_starts[self.pos]) * self.seq_length[ind] 
        self.pos = self.pos + 1
        
        
    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()
        cur_reward_baseline = np.sum(self.rewards)/np.sum(self.episode_starts)
        self.baseline_reward = self.alpha * self.baseline_reward + (1-self.alpha) * cur_reward_baseline
        
        mean_length = np.round(self.rewards.shape[0] * self.rewards.shape[1] / np.sum(self.episode_starts))
        
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = self.gamma ** (mean_length - self.seq_length[step]) * self.baseline_reward
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.advantages[step + 1]
            delta = self.rewards[step] - (1-next_non_terminal) * self.baseline_reward + self.gamma * next_values * next_non_terminal 
            # last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = delta
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        # self.returns = self.advantages + self.values

    
    
class ReinforceWithBaseline(ModPPO):
    
    
    def _setup_model(self) -> None:
        super(ReinforceWithBaseline, self)._setup_model()
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = BaselineDictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        
    def get_loss(self, rollout_data, actions):
        
        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

        # BC_LOSS 
        
        values = values.flatten()
        # Normalize advantage
        advantages = rollout_data.advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        policy_gradient_loss = -(advantages * log_prob).mean()

        # Value loss using the TD(gae_lambda) target
        # value_loss = F.mse_loss(rollout_data.returns, values)
        entropy_loss = -th.mean(entropy)
        
        loss = policy_gradient_loss + self.ent_coef * entropy_loss # + self.value_coef * value_loss
            
        
        stats = dict(
            entropy_loss=entropy_loss.item(),
            policy_gradient_loss=policy_gradient_loss.item(),
            loss=loss.item(),
            approx_kl=0,
            value_loss=0,
            clip_fractions=0,
            
        )
        
        return loss, stats
    

