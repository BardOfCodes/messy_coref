
import os
from yacs.config import CfgNode as CN
from configs.subconfig.base_policy.ppo import POLICY
from configs.subconfig.envs.train_cad import ENV as train_cad
from configs.subconfig.envs.eval_cad import ENV as eval_cad
from configs.subconfig.envs.train_random import ENV as train_random
from configs.subconfig.envs.eval_random import ENV as eval_random

from configs.basic import cfg
import numpy as np


# cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('csgnet', 'csgnet_updated') 
# MAX_LENGTH = 19
# MIN_LENGTH = 3
# COMPLEXITY_LENGTHS = [x for x in range(MIN_LENGTH, MAX_LENGTH+1,2)]
# power = 0.5
# PROPORTIONS = [np.power(x, power)/np.sum(np.power(COMPLEXITY_LENGTHS, power)) for x in COMPLEXITY_LENGTHS]
# # from configs.cluster import cfg
# train_random.PROGRAM_LENGTHS = COMPLEXITY_LENGTHS
# train_random.PROGRAM_PROPORTIONS = PROPORTIONS
# eval_random.PROGRAM_LENGTHS = COMPLEXITY_LENGTHS
# eval_random.PROGRAM_PROPORTIONS = PROPORTIONS

cfg = cfg.clone()



cfg.EXP_NAME = "debug"

cfg.MACHINE_SPEC.LOG_DIR = os.path.join(cfg.MACHINE_SPEC.LOG_DIR, "stage_14")
cfg.MACHINE_SPEC.SAVE_DIR = os.path.join(cfg.MACHINE_SPEC.SAVE_DIR, "stage_14")

# MODEL:
# cfg.OBSERVABLE_STACK = 8
cfg.OBSERVABLE_STACK = 11

# Change
# cfg.MODEL.LOAD_WEIGHTS = "../weights/stage_13/pretrain_refactored/weights_9.pt"
# cfg.MODEL.LOAD_WEIGHTS = "../weights/stage_13/pretrain/weights_9.pt"

# cfg.MODEL.LOAD_WEIGHTS = "../weights/stage_13/value_pretrain/value_pretrain_value_ft_16000000_steps.pt"

POLICY.PPO.N_ENVS = 2
POLICY.PPO.BATCH_SIZE = 512
cfg.POLICY = POLICY.clone()
# Environment:
cfg.TRAIN.ENV = train_cad.clone()
# Change the path
cfg.TRAIN.EVAL_EPISODES = 3000 # 0 * 9
# cfg.TRAIN.ENV.DATA_PATH = "/data/drlab/aditya/data/ranked_cad/"
cfg.EVAL.ENV  = eval_cad.clone()
# cfg.TRAIN.ENV = train_random.clone()
# cfg.EVAL.ENV  = eval_random.clone()

# Change the eval and save rate:
cfg.TRAIN.EVAL_FREQ = int(1e5/POLICY.PPO.N_ENVS * 2) # How many steps between EVALs
cfg.TRAIN.SAVE_FREQ = int(2e6/POLICY.PPO.N_ENVS * 2) # How many steps

# For reduce plateau:
cfg.TRAIN.LR_INITIAL = 0.0003
