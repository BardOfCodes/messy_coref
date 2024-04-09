
import os
from tkinter import FALSE
from yacs.config import CfgNode as CN
import numpy as np

from configs.basic_3d import cfg as base_3d_cfg
from configs.basic_2d import cfg as base_2d_cfg

from configs.subconfig.envs.shape_assembly_env import RANDOM_TRAIN_ENV as sa_train_random, SHAPENET_EVAL_ENV as sa_eval_shapenet
from configs.subconfig.envs.csg2d_env import RANDOM_TRAIN_ENV as csg2d_train_random, SHAPENET_EVAL_ENV as csg2d_eval_shapenet
from configs.subconfig.envs.csg3d_train_random_rnn import ENV as csg3d_train_random
from configs.subconfig.envs.csg3d_eval_shapenet_rnn import ENV as csg3d_eval_shapenet

from configs.subconfig.lr_schedulers.cyclic_lr import LR_SCHEDULER as cyclic_lr
from configs.subconfig.behavior_cloning.baseline import BC
from configs.ablations.upgrade.helpers import set_program_lens, set_action_specs, set_half_resolution, set_lang_mode, set_full_resolution

# Config FLAGS
DEBUG = False
ENABLE_NOTIFICATION = True

# 0 1 = PSA and HSA
# 2 5 = PCSG & MCSG
# 6 7 = for 2D CSG
LANGUAGE_MODE = 2

LANGUAGE_NAME_DICT = {
    0: "PSA3D",
    1: "HSA3D",
    2: "PCSG3D",
    3: "FCSG3D",
    4: "HCSG3D",
    5: "MCSG3D",
    6: "PCSG2D",
    7: "FCSG2D",
    8: "MCSG2D"
}

ACTION_SPACE_SIZE_DICT = {
    0: 26 - 5 + 11 + 1,
    1: 26 + 1,
    2: 6,
    3: 7,
    4: 10,
    5: 14,
    6: 5,
    7: 6,
    8: 12,
}

LANGUAGE_NAME = LANGUAGE_NAME_DICT[LANGUAGE_MODE]
ACTION_SPACE_SIZE = ACTION_SPACE_SIZE_DICT[LANGUAGE_MODE]

# Replace the stage: 
if LANGUAGE_MODE < 6:
    cfg = base_3d_cfg.clone()
else:
    cfg = base_2d_cfg.clone()

cfg.TRAIN.RESUME_CHECKPOINT = False
cfg.NOTIFICATION.ENABLE = ENABLE_NOTIFICATION

cfg.EXP_NAME = "%s_length_tax" % LANGUAGE_NAME

cfg.LANGUAGE_NAME = LANGUAGE_NAME

cfg.MACHINE_SPEC.LOG_DIR = os.path.join(cfg.MACHINE_SPEC.LOG_DIR, "stage_49")
cfg.MACHINE_SPEC.SAVE_DIR = os.path.join(cfg.MACHINE_SPEC.SAVE_DIR, "stage_49")

cfg.BC = BC.clone()


if "CSG3D" in LANGUAGE_NAME:
    cfg.TRAIN.ENV = csg3d_train_random.clone()
    cfg.EVAL.ENV = csg3d_eval_shapenet.clone()
    cfg.BC.ENV = csg3d_train_random.clone()
elif "CSG2D" in LANGUAGE_NAME:
    cfg.TRAIN.ENV = csg2d_train_random.clone()
    cfg.EVAL.ENV = csg2d_eval_shapenet.clone()
    cfg.BC.ENV = csg2d_train_random.clone()

else:
    cfg.TRAIN.ENV = sa_train_random.clone()
    cfg.EVAL.ENV = sa_eval_shapenet.clone()
    cfg.BC.ENV = sa_train_random.clone()

cfg.EVAL.EXHAUSTIVE = True
if "3D" in LANGUAGE_NAME:
    cfg.EVAL.BEAM_N_PROC = 4
    cfg.EVAL.BEAM_BATCH_SIZE = 64
    cfg.EVAL.BEAM_SIZE = 3 # For pretrain only.
    cfg.TRAIN.BEAM_SIZE = 1 # For pretrain only.
    cfg.TRAIN.EVAL_EPISODES = 1000 # How many episodes in EVAL
    cfg.TRAIN.BEAM_SEARCH = False
    cfg.TRAIN.LR_INITIAL = 0.001
else:
    cfg.EVAL.BEAM_N_PROC = 2
    cfg.EVAL.BEAM_BATCH_SIZE = 48
    cfg.EVAL.BEAM_SIZE = 3 # For pretrain only.
    cfg.TRAIN.BEAM_SIZE = 1 # For pretrain only.
    cfg.TRAIN.EVAL_EPISODES = 1000 # How many episodes in EVAL
    cfg.TRAIN.BEAM_SEARCH = False
    cfg.TRAIN.LR_INITIAL = 0.0001

# Cyclic optimizer
# cfg.TRAIN.LR_SCHEDULER = cyclic_lr.clone()
cfg.TRAIN.LR_SCHEDULER.FACTOR = 0.5
cfg.TRAIN.LR_SCHEDULER.PATIENCE = 600

# BC
cfg.BC.ENV.TYPE = cfg.BC.ENV.TYPE + "BC"
cfg.BC.ENV.DYNAMIC_MAX_LEN = True
cfg.BC.EPOCHS = 600
cfg.BC.N_ITERS = int(5000)
# Logging
cfg.BC.SAVE_EPOCHS = 5
cfg.BC.LOG_INTERVAL = 100
# ENV
# Compute
cfg.BC.N_ENVS = 1
cfg.POLICY.PPO.N_ENVS = 1
cfg.BC.ENV.NUM_WORKERS = 1 # NOT USED
cfg.BC.ENV.N_ENVS = 1 # NOT USED
# Loss
cfg.BC.ENT_WEIGHT = 0.01
cfg.BC.L2_WEIGHT = 0.00001
cfg.BC.BATCH_SIZE = 96

cfg.ACTION_RESOLUTION = 33# 129
cfg.ACTION_SPACE_TYPE = LANGUAGE_NAME + "Action"
cfg.MODEL.CONFIG.OUTPUT_TOKEN_COUNT = cfg.ACTION_RESOLUTION + ACTION_SPACE_SIZE
# Set language mode.

# Set length
if "3D" in LANGUAGE_NAME:
    MAX_LENGTH = 7
else:
    MAX_LENGTH = 7
MIN_LENGTH = 5
cfg = set_program_lens(cfg, MIN_LENGTH, MAX_LENGTH)

if LANGUAGE_NAME  == "PCSG3D":
    max_len = 11 + 12 * 7 + 1
    batch_size = 400
    bs_batch_size = 48
    bs_n_proc = 3
    collect_gradients = False
    gradient_step_count = 1
    data_loader_num_workers = 4
    n_envs = 4

elif LANGUAGE_NAME  == "FCSG3D":
    max_len = 11 + 12 * 10 + 1
    batch_size = 400
    bs_batch_size = 32
    bs_n_proc = 3
    collect_gradients = False
    gradient_step_count = 1
    data_loader_num_workers = 4
    cfg.BC.N_ENVS = 1
    n_envs = 4

elif LANGUAGE_NAME  == "HCSG3D":
    max_len = 192
    batch_size = 200
    bs_batch_size = 32
    bs_n_proc = 2
    collect_gradients = True
    gradient_step_count = 2
    data_loader_num_workers = 2
    n_envs = 2

elif LANGUAGE_NAME  == "MCSG3D":
    max_len = 192
    batch_size = 250
    bs_batch_size = 32
    bs_n_proc = 2
    collect_gradients = True
    gradient_step_count = 2
    data_loader_num_workers = 2
    n_envs = 2

elif LANGUAGE_NAME  == "PSA3D":
    max_len = 192
    batch_size = 300
    bs_batch_size = 32
    bs_n_proc = 2
    collect_gradients = False
    gradient_step_count = 1
    data_loader_num_workers = 2
    n_envs = 2

elif LANGUAGE_NAME  == "HSA3D":
    max_len = 192
    batch_size = 300
    bs_batch_size = 32
    bs_n_proc = 2
    collect_gradients = False
    gradient_step_count = 1
    data_loader_num_workers = 2
    n_envs = 2

elif LANGUAGE_NAME  == "PCSG2D":
    max_len = 84
    batch_size = 512
    bs_batch_size = 48
    bs_n_proc = 5
    collect_gradients = False
    gradient_step_count = 1
    data_loader_num_workers = 2
    n_envs = 2
elif LANGUAGE_NAME  == "FCSG2D":
    max_len = 128
    batch_size = 512
    bs_batch_size = 48
    bs_n_proc = 5
    collect_gradients = False
    gradient_step_count = 1
    data_loader_num_workers = 5
    n_envs = 5

elif LANGUAGE_NAME  == "MCSG2D":
    max_len = 128
    batch_size = 512
    bs_batch_size = 48
    bs_n_proc = 5
    collect_gradients = False
    gradient_step_count = 1
    data_loader_num_workers = 4
    n_envs = 4

if "3D" in LANGUAGE_NAME:
    cfg = set_half_resolution(cfg)
else:
    cfg = set_full_resolution(cfg)
    cfg.CANVAS_SHAPE = [64, 64]
    cfg.MODEL.CONFIG.INPUT_SEQ_LENGTH = 16

cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('csgnet', 'synthetic_data/%s_data' % LANGUAGE_NAME)
cfg.BC.BATCH_SIZE = batch_size
cfg.EVAL.BEAM_N_PROC = bs_n_proc
cfg.EVAL.BEAM_BATCH_SIZE = bs_batch_size
cfg.BC.NUM_WORKERS = data_loader_num_workers
cfg.BC.COLLECT_GRADIENTS = collect_gradients
cfg.BC.GRADIENT_STEP_COUNT = gradient_step_count
cfg.BC.N_ENVS = n_envs

if "3D" in LANGUAGE_NAME:
    cfg = set_action_specs(cfg, max_len, 11, LANGUAGE_NAME)
else:
    cfg = set_action_specs(cfg, max_len, 16, LANGUAGE_NAME)

if "CSG" in LANGUAGE_NAME:
    cfg = set_lang_mode(cfg, LANGUAGE_NAME, retain_action_type=True)
elif "SA" in LANGUAGE_NAME:
    cfg.TRAIN.ENV.SA_CONF.LANGUAGE_NAME = LANGUAGE_NAME
    cfg.EVAL.ENV.SA_CONF.LANGUAGE_NAME = LANGUAGE_NAME
    cfg.BC.ENV.SA_CONF.LANGUAGE_NAME = LANGUAGE_NAME

    cfg.MODEL.CONFIG.OUTPUT_SEQ_LENGTH = max_len
    cfg.BC.ENV.SA_CONF.PERM_MAX_LEN = max_len
    cfg.TRAIN.ENV.SA_CONF.PERM_MAX_LEN = max_len
    cfg.EVAL.ENV.SA_CONF.PERM_MAX_LEN = max_len

if DEBUG:
    cfg.BC.N_ITERS = int(100)
    # cfg.BC.BATCH_SIZE = int(20)
    cfg.BC.LOG_INTERVAL = 10
    cfg.NOTIFICATION.ENABLE = False
    cfg.EXP_NAME += "_debug"
    # cfg.MODEL.LOAD_WEIGHTS = "/home/aditya/projects/rl/weights/stage_28/MCSG3D_pretrain_baseline_debug/weights_0.ptpkl"
