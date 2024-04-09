from gc import get_stats
from typing import Callable
from configs.subconfig.machine.cluster import MACHINE_SPEC
import numpy as np
import os
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse
from logging import exception
import torch as th 
import os
import CSG.rl_algo as ALGOS
import configs.subconfig.machine as machine_config
from pathlib import Path
from yacs.config import CfgNode as CN
from stable_baselines3.common.vec_env import DummyVecEnv
import CSG.env as csg_env
import CSG.agent as csg_agent
import _pickle as cPickle

# Evaluation
# Saving weights
# logging
# LR Schedules

arg_parser = argparse.ArgumentParser(description="singular parser")
arg_parser.add_argument('--config-file', type=str, default="configs/config.yml")
arg_parser.add_argument('--machine', type=str, default="MIKOSHI")
arg_parser.add_argument('--model-weights', type=str, default="")
arg_parser.add_argument('--job-desc', type=str, default="")
arg_parser.add_argument("--bs-only", help="", action="store_true")
arg_parser.add_argument("--test", help="print the config",
                    action="store_true")


def load_config(args):
    config = CN._load_cfg_py_source(args.config_file)
    # Add exp name to log_dir, save_dir
    DEFAULT_SPEC = getattr(machine_config, "MIKOSHI")
    MACHINE_SPEC = getattr(machine_config, args.machine)
    config.MACHINE_SPEC.DATA_PATH = config.MACHINE_SPEC.DATA_PATH.replace(DEFAULT_SPEC.DATA_ROOT, MACHINE_SPEC.DATA_ROOT)
    config.MACHINE_SPEC.TERMINAL_FILE = config.MACHINE_SPEC.TERMINAL_FILE.replace(DEFAULT_SPEC.DATA_ROOT, MACHINE_SPEC.DATA_ROOT)
    config.MACHINE_SPEC.SAVE_DIR = config.MACHINE_SPEC.SAVE_DIR.replace(DEFAULT_SPEC.PROJECT_ROOT, MACHINE_SPEC.PROJECT_ROOT)
    config.MACHINE_SPEC.LOG_DIR = config.MACHINE_SPEC.LOG_DIR.replace(DEFAULT_SPEC.PROJECT_ROOT, MACHINE_SPEC.PROJECT_ROOT)
    
    config.MACHINE_SPEC.LOG_DIR = os.path.join(config.MACHINE_SPEC.LOG_DIR, config.EXP_NAME)
    config.MACHINE_SPEC.SAVE_DIR = os.path.join(config.MACHINE_SPEC.SAVE_DIR, config.EXP_NAME)
    config.LOG_DIR = config.MACHINE_SPEC.LOG_DIR
    config.SAVE_DIR = config.MACHINE_SPEC.SAVE_DIR
    config.MACHINE_SPEC.DATA_ROOT = MACHINE_SPEC.DATA_ROOT
    config.MACHINE_SPEC.PROJECT_ROOT = MACHINE_SPEC.PROJECT_ROOT
    Path(config.LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.SAVE_DIR).mkdir(parents=True, exist_ok=True)
    
    # config.freeze()
    return config

def prepare_model_config_and_env(config):
    # For DQN:
    seed = config.SEED

    # Init env
    train_env_class = getattr(csg_env, config.TRAIN.ENV.TYPE)
    eval_env_class = getattr(csg_env, config.EVAL.ENV.TYPE)
    policy_config = config.POLICY.PPO
    train_env = DummyVecEnv([lambda ind=i: train_env_class(config=config, phase_config=config.TRAIN, 
                                                            seed=seed + ind, n_proc=policy_config.N_ENVS, proc_id=ind) for i in range(policy_config.N_ENVS)])

    eval_env = DummyVecEnv([lambda ind=i: eval_env_class(config=config, phase_config=config.EVAL, 
                                                         seed=seed + ind, n_proc=policy_config.N_ENVS, proc_id=ind) for i in range(policy_config.N_ENVS)])
    
    
    feature_extractor_class = getattr(csg_agent, config.MODEL.EXTRACTOR)
    # Init Model:
        
    policy_model = getattr(csg_agent, config.POLICY.MODEL)
    
    if config.TRAIN.OPTIM.TYPE =="ADAM_SPECIFIC":
        optimizer_kwargs =dict(betas=(config.TRAIN.OPTIM.BETA_1, config.TRAIN.OPTIM.BETA_2), 
                                eps=config.TRAIN.OPTIM.EPSILON)
    else:
        optimizer_kwargs = {}
    policy_kwargs = dict(
        features_extractor_class=feature_extractor_class,
        features_extractor_kwargs=dict(features_dim=config.MODEL.FEATURE_DIM,
                                        config=config.MODEL.CONFIG,
                                        dropout=config.MODEL.DROPOUT),
        optimizer_kwargs=optimizer_kwargs,
        net_arch=[dict(pi=policy_config.PI_CONF, vf=policy_config.VF_CONF)],
        initial_temperature = policy_config.INITIAL_TEMP,
        use_temperature=policy_config.ENABLE_TEMP,
    )
    policy_class = getattr(ALGOS, config.POLICY.TYPE)

    info = {
        'policy_model': policy_model,
        'policy_class': policy_class,
        'policy_kwargs': policy_kwargs,
        'config': config,
    }
    return train_env, eval_env, info
        


def save_all_weights(policy, lr_scheduler, train_state, save_path, best_program_dict=None, save_programs=False):
    save_dir = os.path.dirname(save_path)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    dictionary_items = {
        "model_weights": policy.state_dict(),
        "optimizer_weights": policy.optimizer.state_dict(),
        "lr_scheduler_weights": lr_scheduler.state_dict(),
        "epoch": train_state.cur_epoch,
        "score": train_state.cur_score,
        "train_state": train_state.get_state(),
    }
    if save_programs:
        dictionary_items['programs'] = best_program_dict        
    # save all the three?
    with open(save_path, 'wb') as f:
        cPickle.dump(dictionary_items, f)


def initialize_model(train_env, model_info, device):

    policy_model = model_info['policy_model']
    policy_class = model_info['policy_class']
    policy_kwargs = model_info['policy_kwargs']
    config = model_info['config']
    model = policy_class(policy_model, train_env, policy_kwargs, config)

    policy = model.policy.to(device)
    lr_scheduler = model.lr_scheduler
    train_state = model_info['train_state']()

    return policy, lr_scheduler, train_state
    
def load_all_weights(load_path, train_env, model_info=None, instantiate_model=False, 
                     device="cpu", strict=True, load_programs=False,
                     policy=None, lr_scheduler=None, train_state=None, load_optim=True):
    """
    Can instantiate model.
    Can load programs.
    if given polich and lr_scheduler, will load weights to that.
    """
    
    if instantiate_model:
        policy, lr_scheduler, train_state = initialize_model(train_env, model_info, device)
    else:
        lr_scheduler = None
        train_state = None
    best_program_dict = None
    if load_path:
        print("loading weights %s" % load_path)
        extension = load_path.split('/')[-1].split('.')[-1]
        if extension == "ptpkl":
            with open(load_path, "rb") as f:
                dictionary_items = cPickle.load(f)
            model_weights = dictionary_items["model_weights"]
            model_weights['features_extractor.extractor.start_token'] = model_weights['features_extractor.extractor.start_token'][:1, :]
            policy.load_state_dict(model_weights, strict=strict)
            if load_optim:
                try:
                    optimizer_weights = dictionary_items['optimizer_weights']
                    policy.optimizer.load_state_dict(optimizer_weights)
                    lr_scheduler_weights = dictionary_items['lr_scheduler_weights']
                    lr_scheduler.load_state_dict(lr_scheduler_weights)
                except:
                    print("There is some bug here")
            if "train_state" in dictionary_items.keys():
                cur_train_state = dictionary_items['train_state']
                train_state.set_state(cur_train_state)
            if load_programs:
                best_program_dict = dictionary_items['programs']
            del dictionary_items
        elif extension == 'pt':
            load_obj = th.load(load_path)
            load_obj['features_extractor.extractor.start_token'] = load_obj['features_extractor.extractor.start_token'][:1, :]
            policy.load_state_dict(load_obj, strict=strict)


    return policy, lr_scheduler, train_state, best_program_dict


def resume_checkpoint_filename(save_dir):
    """ Return latest save file, and epoch.
    """
    ckpt_dir = save_dir
    
    ckpts = [os.path.join(ckpt_dir, x) for x in os.listdir(ckpt_dir) if x.split('.')[-1] == 'ptpkl' and x.split('_')[0] == 'weights']
    # select latest checkpoint:
    if ckpts:
        ckpts.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_checkpoint = ckpts[-1]
    else:
        latest_checkpoint = None
        print("No Checkpoint files found yet!")
    return latest_checkpoint
