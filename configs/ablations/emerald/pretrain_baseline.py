
import os
from yacs.config import CfgNode as CN
import numpy as np

from configs.basic_3d import cfg as base_cfg
from configs.subconfig.envs.csg3d_train_random_rnn import ENV as train_random
from configs.subconfig.envs.csg3d_eval_shapenet_rnn import ENV as eval_shapenet
from configs.subconfig.lr_schedulers.cyclic_lr import LR_SCHEDULER as cyclic_lr
from configs.subconfig.behavior_cloning.baseline import BC
from configs.ablations.upgrade.helpers import set_program_lens, set_action_specs, set_half_resolution, set_lang_mode

# Config FLAGS
DEBUG = False
LANGUAGE_ID = 0
ENABLE_NOTIFICATION = True

LANGUAGE_MAP = {
    0: "PCSG3D",
    1: "FCSG3D",
    2: "HCSG3D",
    3: "MCSG3D"
}

action_space_size = {
    "PCSG3D": 6,
    "FCSG3D": 7,
    "HCSG3D": 10,
    "MCSG3D": 14
}

LANGUAGE_MODE = LANGUAGE_MAP[LANGUAGE_ID]

DRAW_DIRECT = True



cfg = base_cfg.clone()
cfg.TRAIN.RESUME_CHECKPOINT = False
cfg.NOTIFICATION.ENABLE = ENABLE_NOTIFICATION

# Replace the stage: 

cfg.EXP_NAME = "%s_pretrain_baseline" % LANGUAGE_MODE

cfg.MACHINE_SPEC.LOG_DIR = os.path.join(cfg.MACHINE_SPEC.LOG_DIR, "stage_28")
cfg.MACHINE_SPEC.SAVE_DIR = os.path.join(cfg.MACHINE_SPEC.SAVE_DIR, "stage_28")

# EVAL ENV
cfg.EVAL.ENV = eval_shapenet.clone()
cfg.EVAL.EXHAUSTIVE = True
cfg.EVAL.BEAM_N_PROC = 2
cfg.EVAL.BEAM_BATCH_SIZE = 48
cfg.EVAL.BEAM_SIZE = 10
cfg.TRAIN.EVAL_EPISODES = 500 # How many episodes in EVAL
cfg.TRAIN.BEAM_SEARCH = True
# Cyclic optimizer
# cfg.TRAIN.LR_SCHEDULER = cyclic_lr.clone()
cfg.TRAIN.LR_SCHEDULER.FACTOR = 0.5
cfg.TRAIN.LR_SCHEDULER.PATIENCE = 600
cfg.TRAIN.LR_INITIAL = 0.0001

# BC
cfg.BC = BC.clone()
cfg.BC.ENV = train_random.clone()
cfg.BC.ENV.TYPE = 'CSG3DBaseBC'
cfg.BC.ENV.DYNAMIC_MAX_LEN = True
cfg.BC.BATCH_SIZE = 96
cfg.BC.EPOCHS = 600
cfg.BC.N_ITERS = int(5000)
# Logging
cfg.BC.SAVE_EPOCHS = 5
cfg.BC.LOG_INTERVAL = 100
# ENV
# Compute
cfg.BC.ENV.NUM_WORKERS = 1
cfg.BC.N_ENVS = 1
cfg.POLICY.PPO.N_ENVS = 1
# Loss
cfg.BC.ENT_WEIGHT = 0.05
cfg.BC.L2_WEIGHT = 0.00005


# Label Smoothing
cfg.BC.LABEL_SMOOTHING = False
cfg.BC.LS_SIZE = 75
cfg.BC.LS_PADDING_IDX = 75

# Set the action space:
cfg.ACTION_SPACE_TYPE = "%sAction" %  LANGUAGE_MODE
cfg.ACTION_RESOLUTION = 33
cfg.MODEL.CONFIG.OUTPUT_TOKEN_COUNT = cfg.ACTION_RESOLUTION + action_space_size[LANGUAGE_MODE]
# Set length
MAX_LENGTH = 11
MIN_LENGTH = 1
cfg = set_program_lens(cfg, MIN_LENGTH, MAX_LENGTH)
if LANGUAGE_MODE  == "PCSG3D":
    max_len = 11 + 12 * 7 + 1
    batch_size = 400
    bs_batch_size = 48
    bs_n_proc = 2
    collect_gradients = False
    gradient_step_count = 1
    data_loader_num_workers = 3
    cfg.BC.N_ENVS = 1

elif LANGUAGE_MODE  == "FCSG3D":
    max_len = 11 + 12 * 10 + 1
    batch_size = 350
    bs_batch_size = 32
    bs_n_proc = 2
    collect_gradients = False
    gradient_step_count = 1
    data_loader_num_workers = 1
    cfg.BC.N_ENVS = 1

elif LANGUAGE_MODE  == "HCSG3D":
    max_len = 192
    batch_size = 200
    bs_batch_size = 32
    bs_n_proc = 2
    collect_gradients = True
    gradient_step_count = 2
    data_loader_num_workers = 1

elif LANGUAGE_MODE  == "MCSG3D":
    max_len = 192
    batch_size = 200
    bs_batch_size = 32
    bs_n_proc = 2
    collect_gradients = True
    gradient_step_count = 2
    data_loader_num_workers = 1

cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('csgnet', 'synthetic_data/%s_data' % LANGUAGE_MODE)
cfg = set_action_specs(cfg, max_len, 11, LANGUAGE_MODE)
cfg = set_lang_mode(cfg, LANGUAGE_MODE, retain_action_type=True)
cfg = set_half_resolution(cfg)
cfg.BC.BATCH_SIZE = batch_size
cfg.EVAL.BEAM_N_PROC = bs_n_proc
cfg.EVAL.BEAM_BATCH_SIZE = bs_batch_size
cfg.BC.NUM_WORKERS = data_loader_num_workers
cfg.BC.COLLECT_GRADIENTS = collect_gradients
cfg.BC.GRADIENT_STEP_COUNT = gradient_step_count

        
if DRAW_DIRECT:
    cfg.TRAIN.ENV.CSG_CONF.DRAW_MODE = "direct"
    cfg.EVAL.ENV.CSG_CONF.DRAW_MODE = "direct"
    cfg.BC.ENV.CSG_CONF.DRAW_MODE = "direct"


if DEBUG:
    cfg.BC.N_ITERS = int(150)
    cfg.BC.BATCH_SIZE = int(150)
    cfg.NOTIFICATION.ENABLE = False
    cfg.EXP_NAME += "_debug"
    # cfg.MODEL.LOAD_WEIGHTS = "/home/aditya/projects/rl/weights/stage_28/MCSG3D_pretrain_baseline_debug/weights_0.ptpkl"