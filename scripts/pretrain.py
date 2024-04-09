"""
A Basic Run of DQN to check the environment is functioning.
"""

import sys
import numpy as np
import os
from typing import Callable
# env imports
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
# SB Imports
from stable_baselines3.common.env_checker import check_env
# argparser
from CSG.utils.train_utils import arg_parser, load_config, prepare_model_config_and_env
from CSG.utils.notification_utils import slack_sender, SlackNotifier
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import DummyVecEnv
from CSG.evaluator import Evaluator
import CSG.bc_trainers as bc_trainers
from imitation.algorithms.bc import BC
from pathlib import Path
import torch as th
from yacs.config import CfgNode as CN
from stable_baselines3.common import utils


if __name__ == "__main__":

    # th.autograd.set_detect_anomaly(True)
    th.backends.cudnn.benchmark = True
    try:
        th.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    args = arg_parser.parse_args()
    config = load_config(args)
    print(config)
    train_env, eval_env, model_info = prepare_model_config_and_env(config)
    

    bc_config = config.BC
    # Get BC:
    # venv = common.make_venv()
    bc_class = getattr(bc_trainers, bc_config.TYPE)
    logger = utils.configure_logger(1, config.MACHINE_SPEC.LOG_DIR , "BC_%s" % config.EXP_NAME, False)
    bc_trainer = bc_class(
        bc_config=bc_config,
        save_dir=config.SAVE_DIR,
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        seed=config.SEED,
        config=config,
        model_info=model_info,
        demonstrations=None,
        train_env=train_env,
        custom_logger=logger
    )
    perform_lr_scheduling_in_eval = config.TRAIN.LR_SCHEDULER.TYPE in ['REDUCE_PLATEAU']
    eval_beam_size = config.EVAL.BEAM_SIZE
    train_beam_size = config.TRAIN.BEAM_SIZE

    train_eval = Evaluator(eval_env=train_env, n_eval_episodes=config.TRAIN.EVAL_EPISODES, eval_freq=1,
                                log_path=config.MACHINE_SPEC.LOG_DIR + '_train_env', best_model_save_path=config.MACHINE_SPEC.SAVE_DIR + '_train_env', 
                                beam_search=(train_beam_size>1), beam_k=train_beam_size, beam_state_size=train_beam_size, 
                                beam_n_proc=config.EVAL.BEAM_N_PROC, beam_n_batch=config.EVAL.BEAM_BATCH_SIZE,
                                perform_lr_scheduling=False,
                                gt_program=config.TRAIN.ENV.GT_PROGRAM, beam_selector=config.EVAL.BEAM_SELECTOR, log_prefix="_train_env", 
                                exhaustive=False)

    val_eval = Evaluator(eval_env=eval_env, n_eval_episodes=config.TRAIN.EVAL_EPISODES, eval_freq=1,
                                  log_path=config.MACHINE_SPEC.LOG_DIR + '_eval_env', best_model_save_path=config.MACHINE_SPEC.SAVE_DIR + '_eval_env', 
                                  beam_search=(eval_beam_size>1), beam_k=eval_beam_size, beam_state_size=eval_beam_size, 
                                  beam_n_proc=config.EVAL.BEAM_N_PROC, beam_n_batch=config.EVAL.BEAM_BATCH_SIZE,
                                  perform_lr_scheduling=perform_lr_scheduling_in_eval,
                                  gt_program=config.EVAL.ENV.GT_PROGRAM, beam_selector=config.EVAL.BEAM_SELECTOR, log_prefix="_eval_env", exhaustive=True)
    
    train_eval.logger = logger
    val_eval.logger = logger

    # train_env.envs[0].test_state_machine()

    notif = SlackNotifier(config.EXP_NAME, config.NOTIFICATION)
    try:
        notif.start_exp()
        bc_trainer.train(train_eval=train_eval, val_eval=val_eval, log_interval=bc_config.LOG_INTERVAL)
    except Exception as ex:
        notif.exp_failed(ex)
        raise ex