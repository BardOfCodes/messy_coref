
import os
from yacs.config import CfgNode as CN
from configs.subconfig.base_policy.ppo import POLICY
# from configs.subconfig.envs.train_cad import ENV as train_cad
# from configs.subconfig.envs.eval_cad import ENV as eval_cad
from configs.subconfig.envs.train_cad_rnn import ENV as train_cad
from configs.subconfig.envs.eval_cad_rnn import ENV as eval_cad
from configs.subconfig.envs.train_random import ENV as train_random
from configs.subconfig.envs.eval_random import ENV as eval_random

from configs.basic import cfg
import numpy as np



cfg = cfg.clone()


cfg.MODEL.EXTRACTOR = "WrapperTransformerExtractor" # Extractor architecture. 


cfg.EXP_NAME = "debug"

cfg.MACHINE_SPEC.LOG_DIR = os.path.join(cfg.MACHINE_SPEC.LOG_DIR, "stage_17")
cfg.MACHINE_SPEC.SAVE_DIR = os.path.join(cfg.MACHINE_SPEC.SAVE_DIR, "stage_17")

cfg.ACTION_SPACE_TYPE = "MultiRefactoredActionSpace" 

# MODEL:
cfg.OBSERVABLE_STACK = 7
# cfg.OBSERVABLE_STACK = 11

# Change
# cfg.MODEL.LOAD_WEIGHTS = "../weights/stage_13/pretrain_refactored/weights_9.pt"
# cfg.MODEL.LOAD_WEIGHTS = "../weights/stage_13/pretrain/weights_9.pt"
# cfg.MODEL.LOAD_WEIGHTS = "../weights/stage_13/value_pretrain/value_pretrain_value_ft_16000000_steps.pt"

POLICY.PPO.N_ENVS = 16
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
cfg.TRAIN.EVAL_FREQ = int(1e2/POLICY.PPO.N_ENVS * 2) # How many steps between EVALs
cfg.TRAIN.SAVE_FREQ = int(2e6/POLICY.PPO.N_ENVS * 2) # How many steps

# For reduce plateau:
cfg.TRAIN.LR_INITIAL = 0.0003
