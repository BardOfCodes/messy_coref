# from configs.cluster import cfg
# from configs.ablations.upgrade.pretrain_baseline import cfg as base_cfg
from configs.ablations.upgrade.tf_ablation import cfg as base_cfg
import os
import numpy as np

cfg = base_cfg.clone()

MATCH_PLAD_OPT = True
MATCH_PLAD_ARCH = False
MATCH_PLAD_DATA = False

old_exp_name = "CSG_pretrain_baseline_till_l10"
new_exp_name = "NT_CSG32_pretrain_plad"

if MATCH_PLAD_OPT:
    new_exp_name = "_".join([new_exp_name, "opt_match"])
if MATCH_PLAD_ARCH:
    new_exp_name = "_".join([new_exp_name, "arch_match"])
if MATCH_PLAD_DATA:
    new_exp_name = "_".join([new_exp_name, "data_match"])

cfg.EXP_NAME = base_cfg.EXP_NAME.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)

# Change env and action space:
cfg.TRAIN.ENV.CSG_CONF.LANG_TYPE = "NTCSG3D"
cfg.EVAL.ENV.CSG_CONF.LANG_TYPE = "NTCSG3D"
cfg.BC.ENV.CSG_CONF.LANG_TYPE = "NTCSG3D"

# cfg.ACTION_SPACE_TYPE = "NTCSG3DAction64" 
cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('revised_csgnet_large', 'revised_nt_csgnet_large') 

cfg.ACTION_SPACE_TYPE = "NTCSG3DAction32" 
cfg.TRAIN.ENV.CSG_CONF.RESOLUTION = 32
cfg.EVAL.ENV.CSG_CONF.RESOLUTION = 32
cfg.BC.ENV.CSG_CONF.RESOLUTION = 32
cfg.TRAIN.ENV.CSG_CONF.SCALE = 32
cfg.EVAL.ENV.CSG_CONF.SCALE = 32
cfg.BC.ENV.CSG_CONF.SCALE = 32
cfg.CANVAS_SHAPE = [32, 32, 32]
cfg.MODEL.CONFIG.INPUT_SEQ_LENGTH = 8
# cfg.BC.N_ITERS = int(500)
# cfg.TRAIN.EVAL_EPISODES = 500 
cfg.EVAL.BEAM_SIZE = 1
cfg.EVAL.BEAM_N_PROC = 1
cfg.EVAL.BEAM_BATCH_SIZE = 50

# Length Optimization
max_len = 96
cfg.MODEL.CONFIG.OUTPUT_SEQ_LENGTH = max_len
cfg.BC.ENV.CSG_CONF.PERM_MAX_LEN = max_len
cfg.TRAIN.ENV.CSG_CONF.PERM_MAX_LEN = max_len
cfg.EVAL.ENV.CSG_CONF.PERM_MAX_LEN = max_len

if MATCH_PLAD_OPT:
    cfg.TRAIN.LR_SCHEDULER.PATIENCE = 600
    cfg.BC.BATCH_SIZE = 400

if MATCH_PLAD_ARCH:
    cfg.MODEL.CONFIG.OLD_ARCH = False
    cfg.MODEL.CONFIG.OUTPUT_DIM = 256
    cfg.POLICY.PPO.PI_CONF = []

if MATCH_PLAD_DATA:
    ## New Data
    MAX_LENGTH = 12
    MIN_LENGTH = 1
    COMPLEXITY_LENGTHS = [x for x in range(MIN_LENGTH, MAX_LENGTH)]
    PROPORTIONS = [202034, 197226, 191934, 188293, 184874, 180798, 177654, 173719, 171261, 167963, 164244]
    PROPORTIONS = [x/2000000.0 for x in PROPORTIONS]
    cfg.TRAIN.ENV.PROGRAM_LENGTHS = COMPLEXITY_LENGTHS
    cfg.TRAIN.ENV.PROGRAM_PROPORTIONS = PROPORTIONS
    cfg.BC.ENV.PROGRAM_LENGTHS = COMPLEXITY_LENGTHS
    cfg.BC.ENV.PROGRAM_PROPORTIONS = PROPORTIONS
    cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('revised_csgnet_large', 'plad_nt_csg_large') 

