
import os
from yacs.config import CfgNode as CN
from configs.subconfig.base_policy.ppo import POLICY
from configs.subconfig.envs.train_random import ENV as train_random
from configs.subconfig.envs.eval_random import ENV as eval_random

from configs.basic import cfg
import numpy as np


cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('csgnet', 'csgnet_updated') 
MAX_LENGTH = 7
MIN_LENGTH =3
COMPLEXITY_LENGTHS = [x for x in range(MIN_LENGTH, MAX_LENGTH+1,2)]
power = 0.5
PROPORTIONS = [np.power(x, power)/np.sum(np.power(COMPLEXITY_LENGTHS, power)) for x in COMPLEXITY_LENGTHS]
# from configs.cluster import cfg
train_random.PROGRAM_LENGTHS = COMPLEXITY_LENGTHS
train_random.PROGRAM_PROPORTIONS = PROPORTIONS
eval_random.PROGRAM_LENGTHS = COMPLEXITY_LENGTHS
eval_random.PROGRAM_PROPORTIONS = PROPORTIONS

cfg = cfg.clone()



cfg.EXP_NAME = "debug"

cfg.MACHINE_SPEC.LOG_DIR = os.path.join(cfg.MACHINE_SPEC.LOG_DIR, "stage_15")
cfg.MACHINE_SPEC.SAVE_DIR = os.path.join(cfg.MACHINE_SPEC.SAVE_DIR, "stage_15")

# MODEL:
cfg.OBSERVABLE_STACK = 3


POLICY.PPO.N_ENVS = 8
POLICY.PPO.BATCH_SIZE = 512
POLICY.PPO.N_STEPS = 512
POLICY.PPO.ENT_COEF = 5e-1
cfg.POLICY = POLICY.clone()
# Environment:
# Change the path
# cfg.TRAIN.ENV.DATA_PATH = "/data/drlab/aditya/data/ranked_cad/"
cfg.TRAIN.ENV = train_random.clone()
cfg.EVAL.ENV  = eval_random.clone()

# Change the eval and save rate:
cfg.TRAIN.EVAL_FREQ = int(5e5/POLICY.PPO.N_ENVS) # How many steps between EVALs
cfg.TRAIN.SAVE_FREQ = int(2e6) # How many steps

cfg.TRAIN.NUM_STEPS = int(5e6)
# For reduce plateau:
cfg.TRAIN.LR_INITIAL = 0.0003
