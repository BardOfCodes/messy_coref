import traceback
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import gym
import numpy as np
import os
import torch
import time
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.vec_env import is_vecenv_wrapped
from .env.reward_function import chamfer
from .utils.eval_utils import parallel_CSG_beam_evaluate, batch_CSG_beam_evaluate, CSG_evaluate
from pathlib import Path
import json
import _pickle as cPickle
from stable_baselines3.common.logger import Figure
# Metric tracker:

class Evaluator(EvalCallback):

    def __init__(self, beam_search=False, beam_selector="log_probability", beam_k=1,
                 beam_state_size=1, gt_program=True, save_predictions=False, 
                 beam_n_proc=1, beam_n_batch=1, exhaustive=False, log_prefix=None, perform_lr_scheduling=False, 
                 return_all=False, *args, **kwargs):
        super(Evaluator, self).__init__(*args, **kwargs)
        # list for storing evaluation dicts.
        self.beam_search = beam_search
        self.beam_selector = beam_selector
        self.gt_program = gt_program
        self.save_predictions = save_predictions
        self.log_prefix = log_prefix
        self.exhaustive = exhaustive
        self.perform_lr_scheduling = perform_lr_scheduling
        self.eval_func = parallel_CSG_beam_evaluate
        self.beam_k = beam_k
        self.beam_state_size = beam_state_size
        self.beam_n_proc = beam_n_proc
        self.beam_n_batch = beam_n_batch
        self.best_call = 0
        self.best_main_metric = -np.inf
        self.main_metric = -np.inf
        self.return_all = return_all
        lang_name = self.eval_env.envs[0].language_name 
        if "CSG3D" in lang_name:
            self.extractor_class = "CSG3DMetricExtractor"
            self.reward_evaluation_limit = 10000
            self.main_metric_name = "iou"
        elif "CSG2D" in lang_name:
            self.extractor_class = "CSG2DMetricExtractor"
            self.reward_evaluation_limit = 10000
            self.main_metric_name = "chamfer"
        elif "SA" in lang_name:
            self.extractor_class = "HSA3DMetricExtractor"
            self.reward_evaluation_limit = 15
            # self.reward_evaluation_limit = 100
            self.main_metric_name = "iou"

    @property    
    def lr_metric(self):
        # Add modification based on other strategies.
        return self.main_metric
    
    def __repr__(self):
        
        str_def = 'Version: Beam ' + str(self.beam_search) + ' Beam-size= ' + str( self.beam_k)
        return str_def
        
    def _on_step(self, policy, lr_scheduler) -> bool:
        
        if (self.n_calls-1) % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            self.logger.name_to_value.clear()
            self.logger.name_to_count.clear()
            self.logger.name_to_excluded.clear()
            # Reset success rate buffer
            self._is_success_buffer = []
            
            # Reset Eval set:
            for env in self.eval_env.envs:
                env.program_generator.reset_data_distr()
            
            start_time = time.time()
            
            keep_trying = True
            cur_n_proc = self.beam_n_proc
            cur_beam_n_batch=self.beam_n_batch
            while(keep_trying):
                try:
                    metric_obj, all_program_metric_obj, new_conf = self.eval_func(
                        policy,
                        self.eval_env,
                        gt_program=self.gt_program,
                        n_eval_episodes=self.n_eval_episodes,
                        deterministic=self.deterministic,
                        callback=self._log_success_callback,
                        beam_k=self.beam_k,
                        beam_state_size=self.beam_state_size,
                        beam_selector=self.beam_selector,
                        save_predictions=self.save_predictions,
                        logger=self.logger,
                        n_call=self.n_calls,
                        save_loc=self.log_path + '_beam_%d' % self.beam_k,
                        beam_n_proc=cur_n_proc,
                        beam_n_batch=cur_beam_n_batch,
                        exhaustive=self.exhaustive,
                        extractor_class=self.extractor_class,
                        reward_evaluation_limit=self.reward_evaluation_limit,
                    )
                    print("BS sampling successful!")
                    keep_trying = False
                except Exception as ex:
                    print(ex)
                    print(traceback.format_exc())
                    print("failed with %d procs" % cur_n_proc)
                    cur_n_proc = cur_n_proc - 1
                    print("Trying with %d processes" % cur_n_proc)
                    if cur_n_proc == 0:
                        raise ValueError("Cannot do this!")

            for key, value in new_conf.items():
                setattr(self, key, value)

            mean_metrics, all_metrics, predictions = metric_obj.return_metrics()
            if self.return_all:
                _,_, predictions = all_program_metric_obj.return_metrics()
            end_time = time.time()
            print('Version: Beam ',self.beam_search, ' Beam-size = ', self.beam_k, ' Beam State Size = ', self.beam_state_size)
            print('Time for eval', end_time - start_time)

            if self.log_path is not None:
                with open(self.log_path + '_metrics.pkl', 'wb') as f:
                    cPickle.dump(dict(mean_metrics=mean_metrics, all_metrics=all_metrics), f)
                if self.save_predictions:
                    print("Saving data at %s" % self.log_path)
                    with open(self.log_path + '_data.pkl', 'wb') as f:
                        cPickle.dump(dict(predictions=predictions), f)

            self.main_metric = mean_metrics[self.main_metric_name]
            # Select based on self.lr_metric
            best_reached = self.main_metric > self.best_main_metric
            if best_reached:
                if self.verbose > 0:
                    print("New best Main Metric of value %f!" % self.main_metric)
                if self.best_model_save_path is not None:
                    Path(self.best_model_save_path).mkdir(parents=True, exist_ok=True)
                    save_path = os.path.join(self.best_model_save_path, "best_model.pt")
                    torch.save(policy.state_dict(), save_path)
                    best_info = os.path.join(self.best_model_save_path, "best_model.txt")
                    with open(best_info, 'w') as f:
                        f.write("Best stats: %f %f \n" % (mean_metrics['chamfer'], self.main_metric))
                        f.write("Eval epoch: %d" % self.n_calls)
                    self.best_call = self.n_calls
                    
                # self.best_mean_reward = mean_reward
                self.best_main_metric = self.main_metric
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()
                
            # Scheduler
            loss = self.lr_metric
            if self.perform_lr_scheduling:
                lr_scheduler.step(loss)
            if self.log_prefix:
                log_prefix = self.log_prefix
            else:
                log_prefix = ""
            for key, value in mean_metrics.items():
                self.logger.record("eval%s/%s" % (log_prefix, key), value)
            self.logger.record("eval%s/best call" % log_prefix, self.best_call)
            self.logger.record("eval%s/best value" % log_prefix, self.best_main_metric)
            self.logger.record("time%s/total timesteps" % log_prefix, self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

        return best_reached
