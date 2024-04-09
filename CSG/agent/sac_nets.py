
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from .policy import OldRestrictedActorCritic
from stable_baselines3.common.type_aliases import Schedule
import torch.nn as nn
import numpy as np
from functools import partial
import torch as th
from stable_baselines3.common.distributions import CategoricalDistribution, Distribution, MultiCategoricalDistribution
import gym
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.torch_layers import MlpExtractor


from stable_baselines3.dqn.policies import MultiInputPolicy
from stable_baselines3.common.policies import MultiInputActorCriticPolicy, ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution, Distribution, MultiCategoricalDistribution
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule

from torch.nn import functional as F
import warnings

class RestrictedActorActionCritic(OldRestrictedActorCritic):
    
    ## Define the Value function to be a Action space sized output.
    
    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        ## VALUE NET CREATES ACTION VALUES
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, self.action_dist.action_dim)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            } 
            # How much does this matter?
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.lr_schedule = lr_schedule(1)
        self.sac_alpha = nn.Parameter(th.Tensor(1), requires_grad=True)
        nn.init.constant_(self.sac_alpha, 0.05)
        
        # self.lr_schedule_func = lr_schedule
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
    
    def get_distribution_values(self, action_values, distribution, obs):
        interm = th.sum(distribution.distribution.probs * action_values, 1)
        values = interm # - self.sac_alpha * self.action_space.get_entropy(distribution, obs)
        return values
    
    def get_argmax_values(self, action_values, actions, obs):
        action_one_hot = th.nn.functional.one_hot(actions, self.action_space.n)
        values = th.sum(action_values * action_one_hot, 1)
        return values
        pass
        
    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi, obs)
        action_values = self.value_net(latent_vf)
        values = self.get_argmax_values(action_values, actions, obs)
        # For evaluating actions no need to mask
        if self.mask_output:
            # print("masking is on!")
            entropy = self.action_space.get_entropy(distribution,  obs)
            log_prob = self.action_space.get_log_prob(distribution, actions, obs)
        else:
            entropy = distribution.entropy()
            log_prob = distribution.log_prob(actions)
        return values, log_prob, entropy
    
    def evaluate_actions_and_acc(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the max prediction as well.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        distribution = self._get_action_dist_from_latent(latent_pi, obs)
        _, max_actions = th.max(distribution.distribution.logits, 1)
        
        action_values = self.value_net(latent_vf)
        values = self.get_argmax_values(action_values, distribution, obs)
        if self.mask_output:
            entropy = self.action_space.get_entropy(distribution,  obs)
            log_prob = self.action_space.get_log_prob(distribution, actions, obs)
        else:
            entropy = distribution.entropy()
            log_prob = distribution.log_prob(actions)
        return values, log_prob, entropy, max_actions
    
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        action_values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, obs)
        values = self.get_distribution_values(action_values, distribution, obs)
        actions = distribution.get_actions(deterministic=deterministic)
        
        if self.mask_output:
            # print("masking is on!")
            log_prob = self.action_space.get_log_prob(distribution, actions, obs)
        else:
            log_prob = distribution.log_prob(actions)
        return actions, values, log_prob
    
    def predict_values(self, obs):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        action_values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, obs)
        values = self.get_distribution_values(action_values, distribution, obs)
        return values
    
    # For forward which requires action values as well.
    def special_forward(self):
        pass
    

class DualRestrictedActorActionCritic(RestrictedActorActionCritic):
    """Target: Learn a Q function for the agent, and a Q function for the target agent.
    The agent is trained with q-distillation from the modified experiences.  

    Args:
        RestrictedActorActionCritic (_type_): _description_
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU, #nn.ReLU, # Use RELU
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super(ActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        # This has to be our version:
        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        
        self.offline_features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.extractor.output_dim

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution definition.
        # Since Action Space is RestrictedAction, the existing initialization will throw error. 
        self._init_action_dist(action_space)
        self.mask_output = True

        self._build(lr_schedule)
        
        
    def _build_offline_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.offline_mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )


    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()
        self._build_offline_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        ## VALUE NET CREATES ACTION VALUES
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, self.action_dist.action_dim)
        self.offline_value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, self.action_dist.action_dim)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.offline_features_extractor: np.sqrt(2),
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.offline_mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
                self.offline_value_net: 1,
            } 
            # How much does this matter?
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.lr_schedule = lr_schedule(1)
        self.sac_alpha = nn.Parameter(th.Tensor(1), requires_grad=True)
        nn.init.constant_(self.sac_alpha, 0.05)
        
        # self.lr_schedule_func = lr_schedule
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
    
    
    def evaluate_actions_offline(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        offline_features = self.offline_extract_features(obs)
        _, latent_vf = self.offline_mlp_extractor(offline_features)
        action_values = self.offline_value_net(latent_vf)
        values = self.get_argmax_values(action_values, actions, obs)
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi, obs)
        ## CHANGE
        # For evaluating actions no need to mask
        if self.mask_output:
            # print("masking is on!")
            entropy = self.action_space.get_entropy(distribution,  obs)
            log_prob = self.action_space.get_log_prob(distribution, actions, obs)
        else:
            entropy = distribution.entropy()
            log_prob = distribution.log_prob(actions)
        return values, log_prob, entropy
    
    
    def predict_action_values(self, obs, actions):
        offline_features = self.offline_extract_features(obs)
        # latent_pi, _ = self.mlp_extractor(features)
        _, latent_vf = self.offline_mlp_extractor(offline_features)
        action_values = self.offline_value_net(latent_vf)
        # distribution = self._get_action_dist_from_latent(latent_pi, obs)
        values = self.get_argmax_values(action_values, actions, obs)
        return values
    
    def get_action_and_value_distr(self, obs):
        offline_features = self.offline_extract_features(obs)
        _, latent_vf = self.offline_mlp_extractor(offline_features)
        action_values = self.offline_value_net(latent_vf)
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi, obs)
        return distribution, action_values
        
    def offline_extract_features(self, obs: th.Tensor) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.

        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No features extractor was set"
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.offline_features_extractor(preprocessed_obs)
