from collections import defaultdict
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
from imitation.algorithms.bc import BC, EpochOrBatchIteratorWithProgress
from .bc_trainer import BCEnvDataset, AdaptedBC

    
class ReinforceWithBaseline(AdaptedBC):
    """Behavioral cloning (BC).

    Recovers a policy via supervised learning from observation-action pairs.
    """
    def __init__(self, rwb_config, save_dir, *args, **kwargs):
        self.n_iters = rwb_config.N_ITERS
        self.nllloss = th.nn.NLLLoss()
        self.logsoftmax = th.nn.LogSoftmax()
        super(AdaptedBC, self).__init__(*args, **kwargs)
        # print("Reinitializing the entire net")
        # self.policy.apply(self.policy.features_extractor.extractor.initialize_weights)
        
        self.optimizer.param_groups[0]['lr'] = rwb_config.INIT_LR
        if rwb_config.SCHEDULER == "EXPONENTIAL":
            # TODO: Remove Hardcode:
            self.scheduler = th.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        elif rwb_config.SCHEDULER == "REDUCE_PLATEAU":
            self.scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=12)
            
        self.save_dir = save_dir
        # self.scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=4, cooldown=2)

    def set_demonstrations(self, demonstrations) -> None:
        # demonstration is a env
        # create a torch dataloader with env
        
        batch_iters = math.ceil(self.batch_size/len(demonstrations.envs))
        bc_dataset = BCEnvDataset(demonstrations, self.n_iters * batch_iters)
        self._demo_data_loader = th.utils.data.DataLoader(bc_dataset, batch_size=batch_iters,
                                            num_workers=8, shuffle=True)

        
    def _calculate_loss(
        self,
        obs: Union[th.Tensor, np.ndarray],
        acts: Union[th.Tensor, np.ndarray],
    ) -> Tuple[th.Tensor, Mapping[str, float]]:
        """Calculate the supervised learning loss used to train the behavioral clone.

        Args:
            obs: The observations seen by the expert. If this is a Tensor, then
                gradients are detached first before loss is calculated.
            acts: The actions taken by the expert. If this is a Tensor, then its
                gradients are detached first before loss is calculated.

        Returns:
            loss: The supervised learning loss for the behavioral clone to optimize.
            stats_dict: Statistics about the learning process to be logged.

        """
        # obs = th.as_tensor(obs, device=self.device).detach()
        # acts = th.as_tensor(acts, device=self.device).detach()
        _, log_prob, entropy, max_action = self.policy.evaluate_actions_and_acc(obs, acts)
        prob_true_act = th.exp(log_prob).mean()
        print("max_action and prob", max_action.item(), prob_true_act.item())
        log_prob = log_prob.mean()
        entropy = entropy.mean()

        l2_norms = [th.sum(th.square(w)) for w in self.policy.parameters()]
        l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square

        ent_loss = -self.ent_weight * entropy
        neglogp = -log_prob
        l2_loss = self.l2_weight * l2_norm
        loss = neglogp + ent_loss + l2_loss

        # Calculate accuracy:
        acc = th.mean((acts == max_action).float())
        
        stats_dict = dict(
            neglogp=neglogp.item(),
            loss=loss.item(),
            entropy=entropy.item(),
            ent_loss=ent_loss.item(),
            prob_true_act=prob_true_act.item(),
            l2_norm=l2_norm.item(),
            l2_loss=l2_loss.item(),
            accuracy=acc.item()
        )

        return loss, stats_dict
    
    def get_data(self):
        
        output_dict = self._demo_data_loader.reset()
        for j in range(self.n_iters - 1):
            cur_dict = self._demo_data_loader.reset()
            for key, value in cur_dict.items():
                output_dict[key] = np.concatenate([output_dict[key], value], 0)
                
        obs_tensor = obs_as_tensor(output_dict, self.device)
        target = obs_tensor.pop("target").long()
        return obs_tensor, target
    
    def format_data(self, batch):
        for key, item in batch.items():
            item = [x.squeeze(0) for x in item.split(1, 0)]
            item = th.cat(item, 0)
            batch[key] = item.cuda().detach()
        target = batch.pop("target")
        return batch, target
    
    def train(
        self,
        *,
        n_epochs: Optional[int] = None,
        on_epoch_end: Callable[[], None] = None,
        on_batch_end: Callable[[], None] = None,
        log_interval: int = 500,
        log_rollouts_venv: Optional[vec_env.VecEnv] = None,
        log_rollouts_n_episodes: int = 5,
        progress_bar: bool = True,
        reset_tensorboard: bool = False,
    ):
        """Train with supervised learning for some number of epochs.

        Here an 'epoch' is just a complete pass through the expert data loader,
        as set by `self.set_expert_data_loader()`.

        Args:
            n_epochs: Number of complete passes made through expert data before ending
                training. Provide exactly one of `n_epochs` and `n_batches`.
            n_batches: Number of batches loaded from dataset before ending training.
                Provide exactly one of `n_epochs` and `n_batches`.
            on_epoch_end: Optional callback with no parameters to run at the end of each
                epoch.
            on_batch_end: Optional callback with no parameters to run at the end of each
                batch.
            log_interval: Log stats after every log_interval batches.
            log_rollouts_venv: If not None, then this VecEnv (whose observation and
                actions spaces must match `self.observation_space` and
                `self.action_space`) is used to generate rollout stats, including
                average return and average episode length. If None, then no rollouts
                are generated.
            log_rollouts_n_episodes: Number of rollouts to generate when calculating
                rollout stats. Non-positive number disables rollouts.
            progress_bar: If True, then show a progress bar during training.
            reset_tensorboard: If True, then start plotting to Tensorboard from x=0
                even if `.train()` logged to Tensorboard previously. Has no practical
                effect if `.train()` is being called for the first time.
        """

        batch_num = 0
        min_epoch_loss = np.inf
        if on_batch_end:
            for cb in on_batch_end.callbacks:
                cb.logger = self.logger
        if on_epoch_end:
            for cb in on_epoch_end.callbacks:
                cb.logger = self.logger
        
        stats_dict_it = {}
        stats_dict_it['Total Epochs'] = n_epochs
        for epoch in range(n_epochs):
            avg_dict = defaultdict(list)
            stats_dict_it['Epoch'] = epoch
            stats_dict_it['Iters/per Epoch'] = self.n_iters
            # for iter_ind in range(self.n_iters):
            for iter_ind, batch in enumerate(self._demo_data_loader):
                # self.policy.enable_mask()
                obs_tensor, target = self.format_data(batch)
                loss, stats_dict_loss = self._calculate_loss(obs_tensor, target)
                # loss, stats_dict_loss = self._new_loss(obs_tensor, target)
                print("Target, loss ", target.item(), loss.item(), 'state', obs_tensor['draw_allowed'].item(),
                      obs_tensor['op_allowed'].item(), obs_tensor['stop_allowed'].item())
                self.optimizer.zero_grad()
                loss.backward()
                # See the gradient quality by means of mean, std
                z = [x for x in self.policy.parameters()]
                avg_dict['GRAD/first_layer'].append(z[0].grad.mean().item())
                avg_dict['GRAD/last_layer'].append(z[-4].grad.mean().item())
                print('first layer', z[0].grad.mean().item(), z[0].grad.std().item())
                print('last layer', z[-4].grad.mean().item(), z[-4].grad.std().item())
                print("----------next start---------")
                self.optimizer.step()
                
                # Logging:
                stats_dict_it['LR'] = self.optimizer.param_groups[0]['lr']
                stats_dict_it['Iters'] = iter_ind
                for k, v in stats_dict_loss.items():
                    # self.logger.record(f"cur_losses/{k}", v)
                    avg_dict[k].append(v)
                if batch_num % log_interval == 0:
                    for k, v in stats_dict_it.items():
                        self.logger.record(f"BC train/{k}", v)
                    for k, v in avg_dict.items():
                        self.logger.record(f"BC moving_avg/{k}_mv", np.mean(v[-log_interval:]))
                    self.logger.dump(self.tensorboard_step)
                batch_num += 1
                self.tensorboard_step += 1
                
                if on_batch_end is not None:
                    self.policy.enable_mask()
                    x = on_batch_end.on_step()
                    self.policy.disable_mask()
            # self.scheduler.step(np.mean(avg_dict['loss']))
            
            if on_epoch_end is not None:
                self.policy.enable_mask()
                self.policy.eval()
                on_epoch_end.on_step()
                # TODO: FIX THIS HARD CODE
                loss = on_epoch_end.callbacks[0].mean_cdist
                self.policy.disable_mask()
                self.policy.train()
            self.scheduler.step(loss)
                