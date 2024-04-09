# from configs.cluster import cfg
# from configs.ablations.upgrade.pretrain_baseline import cfg as base_cfg
from configs.ablations.upgrade.tf_ablation import cfg as base_cfg
import os
import numpy as np

cfg = base_cfg.clone()


old_exp_name = "CSG_pretrain_baseline_till_l10"
new_exp_name = "NR_CSG32_pretrain_plad"

cfg.EXP_NAME = base_cfg.EXP_NAME.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)

old_exp_name = "stage_20"
new_exp_name = "stage_21"
cfg.EXP_NAME = cfg.EXP_NAME.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)

COMPLEXITY_LENGTHS = [1,10, 2, 9, 3]
COMPLEXITY_LENGTHS = [8, 4, 7, 5, 6]
PROPORTIONS = [1/len(COMPLEXITY_LENGTHS) for x in COMPLEXITY_LENGTHS]
cfg.TRAIN.ENV.PROGRAM_LENGTHS = COMPLEXITY_LENGTHS
cfg.TRAIN.ENV.PROGRAM_PROPORTIONS = PROPORTIONS
cfg.BC.ENV.PROGRAM_LENGTHS = COMPLEXITY_LENGTHS
cfg.BC.ENV.PROGRAM_PROPORTIONS = PROPORTIONS

# Change env and action space:
cfg.TRAIN.ENV.CSG_CONF.LANG_TYPE = "NRCSG3D"
cfg.EVAL.ENV.CSG_CONF.LANG_TYPE = "NRCSG3D"
cfg.BC.ENV.CSG_CONF.LANG_TYPE = "NRCSG3D"
cfg.TRAIN.ENV.CSG_CONF.GENERATOR_N_PROCS = 5
cfg.EVAL.ENV.CSG_CONF.GENERATOR_N_PROCS = 5
cfg.BC.ENV.CSG_CONF.GENERATOR_N_PROCS = 5
cfg.TRAIN.ENV.CSG_CONF.VALID_DRAWS = ["sphere", "cylinder", "cuboid"]
cfg.EVAL.ENV.CSG_CONF.VALID_DRAWS = ["sphere", "cylinder", "cuboid"]
cfg.BC.ENV.CSG_CONF.VALID_DRAWS = ["sphere", "cylinder", "cuboid"]

cfg.ACTION_SPACE_TYPE = "NRCSG3DAction64" 
cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('revised_csgnet_large', 'ordered_nr_csgnet_large') 

# cfg.BC.N_ITERS = int(500)
# cfg.TRAIN.EVAL_EPISODES = 500 
cfg.EVAL.BEAM_SIZE = 1
cfg.EVAL.BEAM_N_PROC = 1
cfg.EVAL.BEAM_BATCH_SIZE = 50

