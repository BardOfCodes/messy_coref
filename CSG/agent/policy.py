import copy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
from gym.spaces.discrete import Discrete
import numpy as np
import torch as th
import torch.nn as nn
from functools import partial
from torch.nn import functional as F
import warnings
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.dqn.policies import MultiInputPolicy
from stable_baselines3.common.policies import MultiInputActorCriticPolicy, ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution, Distribution, MultiCategoricalDistribution
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule

from .q_net import RestrictedQ
from CSG.env.action_spaces import RestrictedAction, RefactoredActionSpace

class RestrictedActionPolicy(MultiInputPolicy):

    def make_q_net(self) -> RestrictedQ:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(
            self.net_args, features_extractor=None)
        # Get the features_dim from a forward pass:
        # Compute shape by doing one forward pass
        features_dim = net_args['features_extractor'].extractor.output_dim
        net_args['features_dim'] = features_dim
        self.mask_output = True
        return RestrictedQ(**net_args).to(self.device)

    def get_logits(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        out = self.q_net(obs, self.mask_output)
        return out


class RestrictedActorCritic(MultiInputActorCriticPolicy):
    
    # Can't skip init since it uses make_proba_distribution directly to define self.action_dist.
    
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
        initial_temperature: float=0,
        use_temperature:bool=False
    ):

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-6

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
        self.lr_schedule_func = lr_schedule
        
        # self.initial_temperature = th.FloatTensor([initial_temperature]).cuda()
        # self.temperature = self.initial_temperature.clone()
        self.use_temperature = use_temperature
        self._build(lr_schedule)
        
        epoch = th.LongTensor([0])
        self.register_buffer("epoch", epoch)
        
        self.beam_mode = False
        self.beam_partial_init = False
        
    def get_minimal_policy_clone(self):
        """Return an object will all the functionality, but only the essential NN parts. 
        """
        return MinimalOldRestrictedPolicyClone(self)
        
        
    def set_temperature(self, progress_remaining):
        
        self.temperature = 1 + self.initial_temperature.clone() * progress_remaining

    def enable_mask(self):
        self.mask_output = True
    def disable_mask(self):
        self.mask_output = False
        
    def _init_action_dist(self, action_space):
        if isinstance(action_space, RestrictedAction):
            self.action_dist = CategoricalDistribution(action_space.n)
            self.masked_action_dist = CategoricalDistribution(action_space.n)
        elif isinstance(action_space, RefactoredActionSpace):
            self.action_dist = MultiCategoricalDistribution(action_space.nvec)
        elif isinstance(action_space, Discrete):
            self.action_dist = CategoricalDistribution(action_space.n)
        else:
            raise Exception("Action space issue!")
    
    def reinit_value_network(self):
        for name, param in self.mlp_extractor.named_children():
            if name == 'value_net':
                print("reinitializing %s" % name)
                param.apply(self.features_extractor.extractor.initialize_weights)
                
        self.value_net.apply(self.features_extractor.extractor.initialize_weights)
        
    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
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
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.lr_schedule = lr_schedule(1)
        # self.lr_schedule_func = lr_schedule
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, obs: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        
        if self.use_temperature:
            mean_actions = mean_actions / self.temperature

        # if self.mask_output:
        masked_mean_actions = self.action_space.restrict_pred_action(mean_actions.clone(), obs)
        out = [self.action_dist.proba_distribution(action_logits=mean_actions),
               self.masked_action_dist.proba_distribution(action_logits=masked_mean_actions)]
        
        return out
    def evaluate_actions_offline(self, observations, actions):
        return self.evaluate_actions(observations, actions)
    
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
        distribution, masked_distribution = self._get_action_dist_from_latent(latent_pi, obs)
        values = self.value_net(latent_vf)
        # For evaluating actions no need to mask
        entropy = self.action_space.get_conditional_entropy(distribution)
        log_prob = self.action_space.get_all_log_prob(distribution, actions)
        return values, log_prob, entropy
    
    def evaluate_actions_and_acc(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the max prediction as well.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        distribution, masked_distribution = self._get_action_dist_from_latent(latent_pi, obs)
        _, max_actions = th.max(masked_distribution.distribution.logits, 1)
        # _, max_actions = th.max(distribution.distribution.logits, 1)
        values = self.value_net(latent_vf)
        entropy = self.action_space.get_conditional_entropy(distribution)
        log_prob = self.action_space.get_all_log_prob(distribution, actions)
        
        return values, log_prob, entropy, max_actions
    
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        action_distr, masked_action_distr =self.get_distribution(observation)
        return masked_action_distr.get_actions(deterministic=deterministic)
    
    def get_logits(self, obs: th.Tensor):
        # Used only in rl_csg Beam Search
        action_distr, masked_action_distr =self.get_distribution(obs)
        out = masked_action_distr.distribution.logits
        return out
    
    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi, obs)

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
        values = self.value_net(latent_vf)
        distribution, masked_distribution = self._get_action_dist_from_latent(latent_pi, obs)
        actions = masked_distribution.get_actions(deterministic=deterministic)
        log_prob = self.action_space.get_all_log_prob(distribution, actions)
        
        return actions, values, log_prob
    
    def predict_with_logits(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # TODO (GH/1): add support for RNN policies
        # if state is None:
        #     state = self.initial_state
        # if episode_start is None:
        #     episode_start = [False for _ in range(self.n_envs)]
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions, logits = self._predict_with_logits(observation, deterministic=deterministic)
        # Convert to numpy
        actions = actions.cpu().numpy()
        logits = logits.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions[0]
            logits = logits[0]

        return actions, logits, state

    def tformer_evaluate_actions_and_acc(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the max prediction as well.
        """
        # Preprocess the observation if needed
        
        self.features_extractor.extractor.return_all = True
        features = self.extract_features(obs)
        self.features_extractor.extractor.return_all = False
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        distribution = self._get_action_dist_from_latent(latent_pi, obs)
        
        max_actions = self.action_space.get_max_action(distribution)
        # _, max_actions = th.max(distribution.distribution.logits, 1)
        
        values = self.value_net(latent_vf)
        if self.mask_output:
            entropy = self.action_space.get_entropy(distribution,  obs)
            log_prob = self.action_space.get_log_prob(distribution, actions, obs)
        else:
            entropy = self.action_space.get_conditional_entropy(distribution)
            log_prob = self.action_space.get_all_log_prob(distribution, actions)
        return values, log_prob, entropy, max_actions
    
    def enable_beam_mode(self):
        self.features_extractor.extractor.enable_beam_mode()
        self.beam_mode = True
        self.beam_partial_init = True
        
    def disable_beam_mode(self):
        self.features_extractor.extractor.disable_beam_mode()
        self.beam_mode = False
        self.beam_partial_init = False

class OldRestrictedActorCritic(RestrictedActorCritic):

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, obs: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        
        if self.use_temperature:
            mean_actions = mean_actions / self.temperature        
        if self.mask_output:
            # mean_actions = mean_actions * 0 + th.randn(mean_actions.shape).to(mean_actions.device)
            mean_actions = self.action_space.restrict_pred_action(mean_actions, obs)
            
        out = self.action_dist.proba_distribution(action_logits=mean_actions)
        return out
    
    
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
        values = self.value_net(latent_vf)
        # For evaluating actions no need to mask
        if self.mask_output:
            # print("masking is on!")
            entropy = self.action_space.get_entropy(distribution,  obs)
            log_prob = self.action_space.get_log_prob(distribution, actions, obs)
        else:
            entropy = self.action_space.get_conditional_entropy(distribution)
            log_prob = self.action_space.get_all_log_prob(distribution, actions)
        return values, log_prob, entropy
    
    def evaluate_actions_and_acc(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the max prediction as well.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        distribution = self._get_action_dist_from_latent(latent_pi, obs)
        
        max_actions = self.action_space.get_max_action(distribution)
        # _, max_actions = th.max(distribution.distribution.logits, 1)
        
        values = self.value_net(latent_vf)
        if self.mask_output:
            entropy = self.action_space.get_entropy(distribution,  obs)
            log_prob = self.action_space.get_log_prob(distribution, actions, obs)
        else:
            entropy = self.action_space.get_conditional_entropy(distribution)
            log_prob = self.action_space.get_all_log_prob(distribution, actions)
        return values, log_prob, entropy, max_actions
    
    
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        self.temp_action_distr = self.get_distribution(observation)
        # max_actions = self.action_space.get_max_action(self.temp_action_distr)
        max_actions = self.temp_action_distr.get_actions(deterministic=deterministic)
        return max_actions
    
    def _predict_with_logits(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        ## TODO: Untested with Batch size > 1.
        self.temp_action_distr = self.get_distribution(observation)
        # max_actions = self.action_space.get_max_action(self.temp_action_distr)
        max_actions = self.temp_action_distr.get_actions(deterministic=deterministic)
        action_logit = self.temp_action_distr.distribution.logits[0, max_actions]
        return max_actions, action_logit
        
    def get_logits(self, obs: th.Tensor):
        # Used only in rl_csg Beam Search
        action_distr = self.get_distribution(obs)
        out = action_distr.logits()
        return out
    

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
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, obs)
        actions = distribution.get_actions(deterministic=deterministic)
        
        if self.mask_output:
            # print("masking is on!")
            log_prob = self.action_space.get_log_prob(distribution, actions, obs)
        else:
            log_prob = self.action_space.get_all_log_prob(distribution, actions)
        return actions, values, log_prob
    
    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.

        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No features extractor was set"
        key_name = "previous_steps"
        retain_value = obs.pop(key_name).long()
        key_name_2 = "cur_step"
        retain_value_2 = obs.pop(key_name_2).long()
        obs_key = "obs"
        if self.beam_mode:
            if not self.beam_partial_init:
                _ = obs.pop(obs_key)
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        preprocessed_obs[key_name] = retain_value
        preprocessed_obs[key_name_2] = retain_value_2
        if self.beam_mode:
            if self.beam_partial_init:
                self.beam_partial_init = False 
            else:
                preprocessed_obs[obs_key] = None
        return self.features_extractor(preprocessed_obs)
    
    
class MinimalOldRestrictedPolicyClone(OldRestrictedActorCritic):
    
    def __init__(self, policy):
        
        nn.Module.__init__(self)
        self.observation_space = copy.deepcopy(policy.observation_space)
        self.action_space = copy.deepcopy(policy.action_space)
        self.features_extractor = copy.deepcopy(policy.features_extractor)
        self.mlp_extractor = copy.deepcopy(policy.mlp_extractor)
        self.action_net = copy.deepcopy(policy.action_net)
        self.value_net = copy.deepcopy(policy.value_net)
        self.action_dist = copy.deepcopy(policy.action_dist)
        # Random things
        self.normalize_images = copy.deepcopy(policy.normalize_images)
        self.use_temperature = copy.deepcopy(policy.use_temperature)
        # self.temperature = copy.deepcopy(policy.temperature)
        self.mask_output = copy.deepcopy(policy.mask_output)