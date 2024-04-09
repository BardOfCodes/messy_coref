
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
NT_CSG32_MODE = True 
NR_CSG64_MODE = False
CNR_CSG64_MODE = False
MNR_CSG64_MODE = False

NRCSG_ORDERED = False
NRCSG_NO_EXTRA = False
CNRCSG_ROUNDED = False

DRAW_DIRECT = False

cfg = base_cfg.clone()


# Replace the stage: 

cfg.EXP_NAME = "CSG_pretrain_baseline"
if NT_CSG32_MODE:
    cfg.EXP_NAME = "NTCSG_" + cfg.EXP_NAME
elif NR_CSG64_MODE:
    cfg.EXP_NAME = "NRCSG_" + cfg.EXP_NAME
    if NRCSG_ORDERED:
        cfg.EXP_NAME += "_ordered"
    if NRCSG_NO_EXTRA:
        cfg.EXP_NAME += "_no_extra"
elif CNR_CSG64_MODE:
    cfg.EXP_NAME = "CNRCSG_" + cfg.EXP_NAME
    cfg.EXP_NAME += "_ordered"
    if CNRCSG_ROUNDED:
        cfg.EXP_NAME += "_rounded"
elif MNR_CSG64_MODE:
    cfg.EXP_NAME = "MCNRCSG_" + cfg.EXP_NAME

    
    


cfg.MACHINE_SPEC.LOG_DIR = os.path.join(cfg.MACHINE_SPEC.LOG_DIR, "stage_27")
cfg.MACHINE_SPEC.SAVE_DIR = os.path.join(cfg.MACHINE_SPEC.SAVE_DIR, "stage_27")

# EVAL ENV
cfg.EVAL.ENV = eval_shapenet.clone()
cfg.EVAL.EXHAUSTIVE = True
cfg.EVAL.BEAM_N_PROC = 2
cfg.EVAL.BEAM_BATCH_SIZE = 48
cfg.EVAL.BEAM_SIZE = 10
cfg.TRAIN.EVAL_EPISODES = 1000 # How many episodes in EVAL
cfg.TRAIN.BEAM_SEARCH = False
# Cyclic optimizer
# cfg.TRAIN.LR_SCHEDULER = cyclic_lr.clone()
cfg.TRAIN.LR_SCHEDULER.FACTOR = 0.5
cfg.TRAIN.LR_SCHEDULER.PATIENCE = 600

cfg.TRAIN.LR_INITIAL = 0.00005

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



# Set length
MAX_LENGTH = 11
MIN_LENGTH = 1
cfg = set_program_lens(cfg, MIN_LENGTH, MAX_LENGTH)
if NT_CSG32_MODE:
    cfg = set_action_specs(cfg, 96, 11)
    cfg = set_lang_mode(cfg, "NTCSG3D")
    cfg = set_half_resolution(cfg)
    cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('csgnet', 'revised_nt_csgnet_large')
    cfg.BC.BATCH_SIZE = 400
    cfg.EVAL.BEAM_BATCH_SIZE = 64
    cfg.BC.ENV.NUM_WORKERS = 3
    cfg.BC.N_ENVS = 3
    cfg.POLICY.PPO.N_ENVS = 1
else:
    cfg = set_action_specs(cfg, 192, 11)
    cfg = set_half_resolution(cfg)
    # cfg.BC.BATCH_SIZE = 200
    # cfg.BC.ENV.NUM_WORKERS = 1
    # cfg.BC.N_ENVS = 1
    # cfg.POLICY.PPO.N_ENVS = 1
    # cfg.BC.COLLECT_GRADIENTS = True
    # cfg.BC.GRADIENT_STEP_COUNT = 2
    if NR_CSG64_MODE:
        cfg = set_lang_mode(cfg, "NRCSG3D")
        if NRCSG_ORDERED:
            if NRCSG_NO_EXTRA:
                cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('csgnet', 'len_corrected/no_extra_ordered_nr_csgnet_large')
            else:
                cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('csgnet', 'len_corrected/ordered_nr_csgnet_large')
        else:
            cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('csgnet', 'len_corrected/revised_nr_csgnet_large')
    elif CNR_CSG64_MODE:
        cfg = set_lang_mode(cfg, "CNRCSG3D")
        if CNRCSG_ROUNDED:
            cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('csgnet', 'len_corrected/rounded_cnr_csgnet_large')
        else:
            cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('csgnet', 'len_corrected/ordered_cnr_csgnet_large')
    elif MNR_CSG64_MODE:
        cfg = set_lang_mode(cfg, "MNRCSG3D")
        cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('csgnet', 'mirror_nr_csgnet_large')
    else:
        cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('csgnet', 'revised_csgnet_large')
        
if DRAW_DIRECT:
    cfg.TRAIN.ENV.CSG_CONF.DRAW_MODE = "direct"
    cfg.EVAL.ENV.CSG_CONF.DRAW_MODE = "direct"
    cfg.BC.ENV.CSG_CONF.DRAW_MODE = "direct"