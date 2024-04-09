import gym
from typing import Any, Dict, List, Optional, Type
import torch.nn as nn
import torch as th
from stable_baselines3.dqn.policies import QNetwork
from stable_baselines3.common.torch_layers import create_mlp
"""
Restrict action based on the observation space dict.
Not this is specific to the action space (400 tokens)
"""
class RestrictedQ(QNetwork):
    """
    Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """
    

    def forward(self, obs: th.Tensor, mask_input=True) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        # real_obs = obs['observation']
        features = self.extract_features(obs)
        # now these 400 dim feature must have some strong reduction based on the obs
        
        q_vals = self.q_net(features)
        if mask_input:
            q_vals = self.action_space.restrict_pred_action(q_vals, obs)
        
        return q_vals
    