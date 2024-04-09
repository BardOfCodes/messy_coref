
import os
from yacs.config import CfgNode as CN
from configs.subconfig.base_policy.ppo import POLICY
from configs.subconfig.envs.train_random import ENV as train_random
from configs.subconfig.envs.eval_random import ENV as eval_random
from configs.subconfig.behavior_cloning.baseline import BC

# from configs.cluster import cfg
from configs.basic import cfg as base_cfg
import numpy as np
cfg = base_cfg.clone()

MAX_LENGTH = 13
MIN_LENGTH = 3
COMPLEXITY_LENGTHS = [x for x in range(MIN_LENGTH, MAX_LENGTH+1,2)]
power = 0.5
PROPORTIONS = [np.power(x, power)/np.sum(np.power(COMPLEXITY_LENGTHS, power)) for x in COMPLEXITY_LENGTHS]
# PROPORTIONS = [1/len(COMPLEXITY_LENGTHS) for x in COMPLEXITY_LENGTHS]

cfg.EXP_NAME = "debug"

cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('csgnet', 'csgnet_refactored') 
cfg.MACHINE_SPEC.LOG_DIR = os.path.join(cfg.MACHINE_SPEC.LOG_DIR, "stage_16")
cfg.MACHINE_SPEC.SAVE_DIR = os.path.join(cfg.MACHINE_SPEC.SAVE_DIR, "stage_16")

# MODEL:
cfg.OBSERVABLE_STACK = 7

train_random.PROGRAM_LENGTHS = COMPLEXITY_LENGTHS
train_random.PROGRAM_PROPORTIONS = PROPORTIONS
eval_random.PROGRAM_LENGTHS = COMPLEXITY_LENGTHS
eval_random.PROGRAM_PROPORTIONS = PROPORTIONS

POLICY.PPO.N_ENVS = 32
POLICY.MODEL = "OldRestrictedActorCritic"
POLICY.PPO.PI_CONF = [1024, 512, 256]
cfg.POLICY = POLICY.clone()
# Environment:
cfg.TRAIN.ENV = train_random.clone()
cfg.EVAL.ENV  = eval_random.clone()

# Change the eval and save rate:
cfg.TRAIN.EVAL_FREQ = int(1e5/POLICY.PPO.N_ENVS) # How many steps between EVALs
cfg.TRAIN.SAVE_FREQ = int(1) # How many steps


cfg.TRAIN.EVAL_EPISODES = 3000 # How many episodes in EVAL
cfg.MODEL.FEATURE_DIM = 64 * 4 * 4 * 2
cfg.MODEL.EXTRACTOR = "WrapperReplCNNExtractor"# "WrapperRes18Extractor" # Extractor architecture. 
# For reduce plateau:
cfg.TRAIN.LR_INITIAL = 0.0003

cfg.TRAIN.LR_SCHEDULER.PATIENCE = 8

cfg.BC = BC.clone()
cfg.BC.ENV = train_random.clone()
cfg.BC.ENV.DYNAMIC_MAX_LEN = False
cfg.BC.ENV.TYPE = 'BCRestrictedCSG'
cfg.BC.ENV.DYNAMIC_MAX_LEN = True