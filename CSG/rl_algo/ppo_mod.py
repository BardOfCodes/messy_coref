
import gym
import time
import torch as th
from gym import spaces
import numpy as np
import io
import pathlib
from torch.nn import functional as F
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.ppo import PPO
from CSG.env.modified_env import ModifierCSG
import CSG.algo_utils.exp_collector as exp_collector
import CSG.algo_utils.exp_trainer as exp_trainer
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space, is_image_space_channels_first
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
    get_device,
    get_schedule_fn,
    get_system_info,
    set_random_seed,
    update_learning_rate,
)

from stable_baselines3.common.utils import get_schedule_fn

class ModPPO(PPO):
    """
    Main update:
    1) Logging
    2) Learning Rate Scheduler.
    3) Link Initialization to Config YACS object.
    4) Making Advantage normalization optional.
    """
    def __init__(self, policy, env, policy_kwargs, config, *args, **kwargs):
        
        policy_config = config.POLICY.PPO
        scheduler_config = config.TRAIN.LR_SCHEDULER
        
        super(ModPPO, self).__init__(policy, 
                                     env, 
                                     batch_size=policy_config.BATCH_SIZE, 
                                     learning_rate=config.TRAIN.LR_INITIAL,
                                     n_steps=policy_config.N_STEPS, 
                                     n_epochs=policy_config.N_EPOCHS, 
                                     vf_coef=policy_config.VF_COEF, 
                                     ent_coef=policy_config.ENT_COEF, 
                                     gamma=policy_config.GAMMA, 
                                     gae_lambda=policy_config.GAE_LAMBDA,
                                     sde_sample_freq=policy_config.SDE_FREQ, 
                                     policy_kwargs=policy_kwargs, 
                                     verbose=1, 
                                     tensorboard_log=config.LOG_DIR + '_tf')

        # initiate lr_scheduler
        self.per_train_updates = self.n_epochs * (self.rollout_buffer.buffer_size * self.n_envs // self.batch_size)
        
        if scheduler_config.TYPE == "REDUCE_PLATEAU":
            self.lr_scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(self.policy.optimizer, 
                                                                            factor=scheduler_config.FACTOR, 
                                                                            patience=scheduler_config.PATIENCE)
        elif scheduler_config.TYPE == "EXPONENTIAL":
            self.lr_scheduler = th.optim.lr_scheduler.ExponentialLR(self.policy.optimizer, gamma=scheduler_config.GAMMA)
        elif scheduler_config.TYPE == "ONE_CYCLE_LR":
            self.lr_scheduler = th.optim.lr_scheduler.OneCycleLR(self.policy.optimizer, max_lr=scheduler_config.MAX_LR,
                                                                total_steps=scheduler_config.TOTAL_STEPS, pct_start=scheduler_config.PCT_START,
                                                                verbose=scheduler_config.VERBOSE)
        elif scheduler_config.TYPE == "WARM_UP":
            def warm_up_func(step, model_size, factor, warmup):
                if step == 0:
                    step = 1
                return factor * (
                    model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
                )
            self.lr_scheduler = th.optim.lr_scheduler.LambdaLR(self.policy.optimizer, lr_lambda=lambda step: warm_up_func(step, scheduler_config.MODEL_SIZE, 
                                                                                                                  scheduler_config.FACTOR, scheduler_config.WARMUP)
        )
        else:
            raise Exception("Scheduler Not Defined.")
        
        self.train_log_keys = ['entropy_loss', 'policy_gradient_loss', 'value_loss', 'clip_fraction', 
                               'approx_kl', 'loss', 'explained_variance', 'clip_range', 'n_updates', 'learning_rate']
        
        # Optional Advantage Normalization:
        self.normalize_advantage = config.POLICY.NORMALIZE_ADVANTAGE
        self.debug_eval_list = None
        self.debug_eval_env = None
        
        ## Multi step train:
        self.collect_gradients = config.POLICY.COLLECT_GRADIENTS
        self.gradient_step_count = config.POLICY.GRADIENT_STEP_COUNT
        
    def _get_torch_save_params(self):
        state_dicts, _ = super(ModPPO, self)._get_torch_save_params()
        state_dicts.append("lr_scheduler")
        return state_dicts, _
    
    def set_parameters(
        self,
        load_path_or_dict: Union[str, Dict[str, Dict]],
        exact_match: bool = True,
        device: Union[th.device, str] = "auto",
    ) -> None:
        """
        Load parameters from a given zip-file or a nested dictionary containing parameters for
        different modules (see ``get_parameters``).

        :param load_path_or_iter: Location of the saved data (path or file-like, see ``save``), or a nested
            dictionary containing nn.Module parameters used by the policy. The dictionary maps
            object names to a state-dictionary returned by ``torch.nn.Module.state_dict()``.
        :param exact_match: If True, the given parameters should include parameters for each
            module and each of their parameters, otherwise raises an Exception. If set to False, this
            can be used to update only specific parameters.
        :param device: Device on which the code should run.
        """
        params = None
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)

        # Keep track which objects were updated.
        # `_get_torch_save_params` returns [params, other_pytorch_variables].
        # We are only interested in former here.
        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception:
                raise ValueError(f"Key {name} is an invalid object name.")

            if isinstance(attr, th.optim.Optimizer):
                attr.load_state_dict(params[name])
            elif isinstance(attr, th.optim.lr_scheduler.ReduceLROnPlateau):
                attr.load_state_dict(params[name])
            elif isinstance(attr, th.optim.lr_scheduler.LambdaLR):
                attr.load_state_dict(params[name])
            else:
                # Assume attr is th.nn.Module
                attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )
            
    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        config,
        policy_kwargs,
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ):
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path, device=device, custom_objects=custom_objects, print_system_info=print_system_info
        )

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        # noinspection PyArgumentList
        model = cls(  # pytype: disable=not-instantiable,wrong-keyword-args
            policy=data["policy_class"],
            env=env,
            policy_kwargs=policy_kwargs,
            config=config,
            device=device,
            _init_setup_model=False,  # pytype: disable=not-instantiable,wrong-keyword-args
        ) 
        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        # put state_dicts back in place
        model.set_parameters(params, exact_match=True, device=device)

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, name + ".data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # pytype: disable=attribute-error
        return model
    
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        Modifications:
        1) Optimizer scheduler for learning rate control.
        2) Optimizer LR Logging.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        stat_dict = {'entropy_loss': [],
                     'policy_gradient_loss': [],
                     'value_loss': [],
                     'clip_fractions': [],
                     'approx_kl': [],
                     'bc_loss': [],
                     'max_mismatch_ratio': [],
                     "loss" : []}

        continue_training = True
        # train for n_epochs epochs
        last_iter = self.rollout_buffer.buffer_size * self.n_envs/self.batch_size - 1
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for iter_ind, rollout_data in enumerate(self.rollout_buffer.get(self.batch_size)):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                    
                    
                loss, stats = self.get_loss(rollout_data, actions)
                
                # Optimization step
                if self.collect_gradients:
                    loss = loss / self.gradient_step_count
                    cur_ind = iter_ind % self.gradient_step_count
                    if cur_ind == (self.gradient_step_count-1) or iter_ind == last_iter:
                        loss.backward()
                        # Clip grad norm
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        self.policy.optimizer.step()
                        self.policy.optimizer.zero_grad()
                    else:
                        loss.backward()
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                else:
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

                for key, stat in stats.items():
                    stat_dict[key].append(stat)
        
        # Modified to reflect the true number of NN updates. 
        self._n_updates += self.per_train_updates
        
        for key, value_list in stat_dict.items():
            self.logger.record("train/%s" % key, np.nanmean(value_list))
        # Additional:
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        # Logs
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("train/learning_rate", self.policy.optimizer.param_groups[0]['lr'])
        self.logger.record("train/n_updates", self._n_updates)
        self.logger.dump(step=self._n_updates)
        
        

    def get_loss(self, rollout_data, actions):
        stats = {}
            
        # values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        values, log_prob, entropy, max_action = self.policy.evaluate_actions_and_acc(rollout_data.observations, actions)
        max_action_mismatch = (actions != max_action).float()
        
        stats['max_mismatch_ratio'] = max_action_mismatch.mean().item()
        
        values = values.flatten()
        # Normalize advantage
        advantages = rollout_data.advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # advantages = (advantages) / (advantages.std() + 1e-8)

        # Negative advantage with extremely high ratio - that is the problem.
        # ratio between old and new policy, should be one at the first iteration
        ratio = th.exp(log_prob - rollout_data.old_log_prob)
        
        clip_range = self.clip_range(self._current_progress_remaining)
        # clipped surrogate loss
        policy_loss_1 = advantages * ratio # th.clamp(ratio, 0.1, 100)
        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

        # Logging
        stats['policy_gradient_loss'] = policy_loss.item()
        clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
        stats['clip_fractions'] = clip_fraction

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
        bc_loss = -th.mean(log_prob)
        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
        stats['loss'] = loss.item()
        stats['bc_loss'] = bc_loss.item()
        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        with th.no_grad():
            log_ratio = log_prob - rollout_data.old_log_prob
            approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
            stats['approx_kl'] = approx_kl_div.item()
        
        return loss, stats

class ModExpPPO(ModPPO):
    """
    PPO along with a modified Experience Buffer (MEB). 
    Pseudo-Algorithm:
    1) During rollout collection:
        1) On each episode completion send it to MEB
    2) Post rollout update:
        1) Perform an update using NN for items like log probability etc.
    3) During training:
        1) Mix the batches with losses from both Normal Buffer and MEB.
    4) MEB reset.
    
    """
    def __init__(self, policy, env, policy_kwargs, config, *args, **kwargs):
        super(ModExpPPO, self).__init__(policy, env, 
                                        policy_kwargs, config, *args, **kwargs)
        
        self.default_train = True
        self.collect_mod_exp = True
        self.mod_exp_train = True
        self.value_train_only = False
        ## TODO: Fix
        self._n_value_updates = 0
        if config.POLICY.VALUE_PRETRAIN >0:
            self.num_rollout_collections = - config.POLICY.VALUE_PRETRAIN
        else:
            self.num_rollout_collections = 0 
        self.allow_mod_exp_train = config.POLICY.ALLOW_MOD_EXP_TRAIN
        self.allow_collect_mod_exp = config.POLICY.ALLOW_COLLECT_MOD_EXP
        
        self.mod_exp_timesteps = 0
        self.value_train_timesteps = 0
        self.n_default_train = 1
        
        # Create an additional her buffer
        mec_config = config.POLICY.ME_C
        met_config = config.POLICY.ME_T
        
        collector_cls = getattr(exp_collector, mec_config.TYPE)
        self.mod_exp_collector = collector_cls(mec_config, self.rollout_buffer, self.logger)
        trainer_cls = getattr(exp_trainer, met_config.TYPE)
        self.mod_exp_trainer = trainer_cls(met_config, self.logger, 
                                             self.mod_exp_collector.mod_exp_buffer,
                                             self.policy, self.mod_exp_collector)
        ### HACK For CR Reward:
        ## TODO: Remove Hack
        ep_config = mec_config.EP_MODIFIER
        # mod_env = ModifierCSG(config=ep_config.CONFIG, phase_config=ep_config.PHASE_CONFIG, 
        #                            seed=0, n_proc=1, proc_id=0)
        # mod_env = self.mod_exp_collector.ep_modifier.mod_env
        # for env in self.env.envs:
        #     env.reward.mod_env = mod_env
            
        ## For Value Train: 
        # self.policy.mod_exp_optimizer = self.policy.optimizer_class(self.policy.parameters(), 
        #                                                          lr=get_schedule_fn(met_config.LR_INITIAL)(1), **self.policy.optimizer_kwargs)
        # Value train optimizer:
        self.policy.value_optimizer = self.policy.optimizer_class([x for x in self.policy.value_net.parameters()] + 
                                                                 [x for x in self.policy.mlp_extractor.value_net.parameters()], 
                                                                 lr=self.policy.lr_schedule_func(1), **self.policy.optimizer_kwargs)
        
    def _setup_learn(self, *args, **kwargs):
        returns = super(ModExpPPO, self)._setup_learn(*args, **kwargs)
        self.mod_exp_collector.logger = self.logger
        self.mod_exp_trainer.logger = self.logger
        
        return returns
        
        
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()
            
            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            
            if self.default_train:
                self.num_timesteps += env.num_envs

                # Give access to local variables
                # callback.update_locals(locals())
                if callback.on_step() is False:
                    return False

            if self.collect_mod_exp:
                self.mod_exp_timesteps += env.num_envs
                
            if self.value_train_only:
                self.value_train_timesteps += env.num_envs

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value
                    
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            
            # MEC call
            for idx, done in enumerate(dones):
                # Simply episode is over, refit episode and add to the other buffer:
                if done and self.collect_mod_exp:
                    self.mod_exp_collector.collect_episode(idx, 
                                                           rewards[idx], 
                                                           infos[idx], 
                                                           rollout_buffer,
                                                           collection_step=self.num_timesteps)
                    
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        
        self.num_rollout_collections += 1
        
        
        # measure avg. train rewards:
        
        rewards = rollout_buffer.rewards.sum()
        n_episodes = rollout_buffer.episode_starts.sum()
        avg_rewards = rewards/float(n_episodes)

        self.mod_exp_collector.update_selector(avg_rewards)
        
        if self.default_train:
            total_steps = rollout_buffer.buffer_size * self.n_envs 
            self.logger.record('train/avg_rewards', avg_rewards)
            self.logger.record('train/avg_episode_len', total_steps/float(n_episodes))
            self.logger.record('train/rollout_episodes', n_episodes)
            self.logger.record('train/rollout_index', self.num_rollout_collections)
            self.logger.dump(step=self.num_timesteps)
        
        
        if self.collect_mod_exp:
            # number of episodes in buffer from this cycle
            mod_exp_stats = self.mod_exp_collector.fetch_post_rollout_stats()
            for key, value in mod_exp_stats.items():    
                self.logger.record('me_train/%s' % key, value)
                self.logger.record('me_train/rollout_index', self.num_rollout_collections)
                
            self.logger.dump(step=self.mod_exp_timesteps)
        
        if self.value_train_only:
            total_steps = rollout_buffer.buffer_size * self.n_envs 
            self.logger.record('value_train/avg_rewards', avg_rewards)
            self.logger.record('value_train/avg_episode_len', total_steps/float(n_episodes))
            self.logger.record('value_train/rollout_episodes', n_episodes)
            self.logger.record('value_train/rollout_index', self.num_rollout_collections)
            self.logger.dump(step=self.value_train_timesteps)
        

        callback.on_rollout_end()

        return True

    
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())
        
        ## Place at the right location
        if self.num_rollout_collections < 0:
            print("Reinitializing Value Net!")
            self.policy.reinit_value_network()

        while self.num_timesteps < total_timesteps:
                
            self.set_train_modes()
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
            # Using the collected Rollouts, create HER rollouts.

            if continue_training is False:
                break

            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            
            self.policy_update(self._current_progress_remaining)
            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                
            cur_rollout = self.num_rollout_collections
            if self.default_train:
                # self.test_performance_on_debug()
                self.train()
                self.logger.dump(step=self._n_updates)
            if self.value_train_only:
                self.value_train()
            if self.mod_exp_train:
                if len(self.mod_exp_collector.mod_exp_buffer.updated_episodes) > 0:
                    post_collection_stats = self.mod_exp_collector.post_collection_episodic_update(self.policy)
                    
                    for key, value in post_collection_stats.items():    
                        self.logger.record('me_train/%s' % key, value)
                        
                    self.logger.dump(step=self.mod_exp_timesteps)
                    self.mod_exp_collector.post_collection_update(self.policy)
                    # Check train performance
                    self.test_performance_on_debug()
                    self.mod_exp_trainer.train(self.per_train_updates * self.n_default_train, self._current_progress_remaining)
                    self.test_performance_on_debug()
                    self.mod_exp_collector.flush_rollout()
            # Update reward threshold.
            
                
        callback.on_training_end()
        

        return self
    
    def test_performance_on_debug(self):
        
        for eval in self.debug_eval_list:
            print("eval for", eval)
            # Assigns the model and logger
            eval.init_callback(self)
            eval.n_calls = 1
            # starts eval
            eval.on_step()
        
    def policy_update(self, progress_remaining):
        # Progress remaining goes from 1 to 0
        self.policy.set_temperature(progress_remaining)
        
        
    def set_train_modes(self):
        
        if self.allow_mod_exp_train:
            self.mod_exp_train = True
        else:
            self.mod_exp_train = False
        if self.allow_collect_mod_exp:
            self.collect_mod_exp = True
        else:
            self.collect_mod_exp = False
        if self.num_rollout_collections < 0:   
            self.default_train = False
            self.value_train_only = True
        else:
            self.default_train = True
            self.value_train_only = False
        
    def value_train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        Modifications:
        1) Optimizer scheduler for learning rate control.
        2) Optimizer LR Logging.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        stat_dict = {'value_loss': [],
                     "loss" : []}

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            last_iter = self.rollout_buffer.buffer_size * self.n_envs/self.batch_size - 1
            # Do a complete pass on the rollout buffer
            for iter_ind, rollout_data in enumerate(self.rollout_buffer.get(self.batch_size)):
                actions = rollout_data.actions
                self._n_value_updates += 1
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                    
                    
                loss, stats = self.get_value_loss(rollout_data, actions)
                
                
                # Optimization step
                if self.collect_gradients:
                    loss = loss / self.gradient_step_count
                    cur_ind = iter_ind % self.gradient_step_count
                    if cur_ind == (self.gradient_step_count-1) or iter_ind == last_iter:
                        loss.backward()
                        # Clip grad norm
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        self.policy.value_optimizer.step()
                        self.policy.value_optimizer.zero_grad()
                    else:
                        loss.backward()
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                else:
                    self.policy.value_optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

                for key, stat in stats.items():
                    stat_dict[key].append(stat)
        
            # Modified to reflect the true number of NN updates. 
            for key, value_list in stat_dict.items():
                self.logger.record("value_train/%s" % key, np.nanmean(value_list))
            self.logger.record("value_train/n_updates", self._n_value_updates)
            self.logger.dump(step=self._n_value_updates)
        
        

    def get_value_loss(self, rollout_data, actions):
        stats = {}
            
        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        values = values.flatten()
        # Normalize advantage


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

        loss = self.vf_coef * value_loss
        stats['loss'] = loss.item()
        
        return loss, stats
        

class PhasicModExpPPO(ModExpPPO):
    
    def __init__(self, policy, env, policy_kwargs, config, *args, **kwargs):
        super(PhasicModExpPPO, self).__init__(policy, env, 
                                        policy_kwargs, config, *args, **kwargs)
        
        ## Also add to the scheduler
        
        
        
        self.n_default_train = config.POLICY.DEFAULT_TRAIN_ROLLOUTS
        self.n_collect_mod_exp = 0 # config.POLICY.COLLECT_MOD_EXP_ROLLOUTS
        self.n_mod_exp_train = config.POLICY.MOD_EXP_TRAIN_ROLLOUTS
        self.n_value_train_only = config.POLICY.VALUE_TRAIN_ROLLOUTS
        
        self.cyclic_rollout_size = self.n_default_train + self.n_collect_mod_exp + self.n_value_train_only
        
    
    def set_train_modes(self):
        
        cur_rollout = self.num_rollout_collections % self.cyclic_rollout_size
        if cur_rollout < self.n_default_train:
            self.default_train = False
        else:
            self.default_train = False
        if cur_rollout < self.n_default_train: 
            self.collect_mod_exp = True
        else:
            self.collect_mod_exp = False
        
        if cur_rollout == self.n_default_train:
            self.mod_exp_train = True
        else:
            self.mod_exp_train = False
        
        if cur_rollout > (self.cyclic_rollout_size - self.n_value_train_only):
            self.value_train_only = True
        else:        
            self.value_train_only = False
        
    def real_set_train_modes(self):
        
        cur_rollout = self.num_rollout_collections % self.cyclic_rollout_size
        if cur_rollout < self.n_default_train:
            self.default_train = True
        else:
            self.default_train = False
        if self.n_default_train < cur_rollout <= (self.n_default_train + self.n_collect_mod_exp): 
            self.collect_mod_exp = True
        else:
            self.collect_mod_exp = False
        
        if cur_rollout == self.n_default_train + self.n_collect_mod_exp:
            self.mod_exp_train = True
        else:
            self.mod_exp_train = False
        
        if cur_rollout > (self.cyclic_rollout_size - self.n_value_train_only):
            self.value_train_only = True
        else:        
            self.value_train_only = False
        
    