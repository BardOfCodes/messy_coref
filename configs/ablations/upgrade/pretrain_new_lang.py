from configs.ablations.upgrade.pretrain_baseline import cfg as base_cfg
from configs.subconfig.envs.csg3d_train_random_rnn import ENV as train_random
from configs.subconfig.envs.csg3d_eval_shapenet_rnn import ENV as eval_shapenet

import os
import numpy as np
import scipy.stats as spyst

MODE = "2_prim_translate"
# MODE = "4_prim_translate"
# MODE = "2_prim_no_rotate"
# MODE = "2_prim_no_scale"
# MODE = "4_prim_all"
COLLECT_GRADIENTS = True

cfg = base_cfg.clone()

old_exp_name = "CSG_pretrain_baseline"
cfg = base_cfg.clone()

## New Sampling Distribution
max_bool = 10
train_random.CSG_CONF.BOOLEAN_COUNT = max_bool
eval_shapenet.CSG_CONF.BOOLEAN_COUNT = max_bool

max_len = 128 + 64
train_random.CSG_CONF.PERM_MAX_LEN = max_len
eval_shapenet.CSG_CONF.PERM_MAX_LEN = max_len
cfg.MODEL.CONFIG.OUTPUT_SEQ_LENGTH = max_len

MAX_LENGTH = 11
MIN_LENGTH = 1
COMPLEXITY_LENGTHS = [x for x in range(MIN_LENGTH, MAX_LENGTH)]
beta_distr = spyst.beta(2,2)
complex_values = [beta_distr.pdf(x/MAX_LENGTH) for x in COMPLEXITY_LENGTHS]
PROPORTIONS = [x/np.sum(complex_values) for x in complex_values]
train_random.PROGRAM_LENGTHS = COMPLEXITY_LENGTHS
train_random.PROGRAM_PROPORTIONS = PROPORTIONS
train_random.CSG_CONF.GENERATOR_N_PROCS = MAX_LENGTH - MIN_LENGTH

cfg.ACTION_SPACE_TYPE = "CSG3DPartialAction64"

# New Language Specifications
if MODE == "2_prim_translate":
    tag = "2_prim_translate"
    train_random.CSG_CONF.VALID_DRAWS = ["cuboid", "ellipsoid"]
    train_random.CSG_CONF.VALID_TRANFORMS = ["translate"]
elif MODE == "4_prim_translate":
    tag = "4_prim_translate"
    train_random.CSG_CONF.VALID_DRAWS = ["sphere", "cylinder", "cuboid", "ellipsoid"]
    train_random.CSG_CONF.VALID_TRANFORMS = ["translate"]
elif MODE == "2_prim_no_rotate":
    tag = "2_prim_no_rotate"
    train_random.CSG_CONF.VALID_DRAWS = ["cuboid", "ellipsoid"]
    train_random.CSG_CONF.VALID_TRANFORMS = ["translate", "scale"]
elif MODE == "2_prim_no_scale":
    tag = "2_prim_no_scale"
    train_random.CSG_CONF.VALID_DRAWS = ["cuboid", "ellipsoid"]
    train_random.CSG_CONF.VALID_TRANFORMS = ["translate", "rotate"]
elif MODE == "4_prim_all":
    tag = "4_prim_all"
    train_random.CSG_CONF.VALID_DRAWS = ["sphere", "cylinder", "cuboid", "ellipsoid"]
    train_random.CSG_CONF.VALID_TRANFORMS = ["translate", "rotate", "scale"]

eval_shapenet.CSG_CONF = train_random.CSG_CONF
cfg.TRAIN.ENV = train_random.clone()
cfg.BC.ENV = train_random.clone()
cfg.EVAL.ENV = eval_shapenet.clone()

cfg.POLICY.PPO.N_ENVS = 1
cfg.BC.ENV.TYPE = 'CSG3DBaseBC'
cfg.BC.ENV.NUM_WORKERS = 1
cfg.BC.N_ENVS = 1
cfg.EVAL.BEAM_SIZE = 1
cfg.EVAL.BEAM_N_PROC = 1
cfg.EVAL.BEAM_BATCH_SIZE = 50
cfg.TRAIN.EVAL_EPISODES = 500 
# cfg.BC.N_ITERS = 300

# Optimization:
cfg.BC.BATCH_SIZE = 96
cfg.MODEL.CONFIG.OUTPUT_DIM = 256
cfg.POLICY.PPO.VF_CONF = []
cfg.POLICY.PPO.PI_CONF = [256]


if COLLECT_GRADIENTS:
    name_tag = tag + "_gradient_collect"
    cfg.BC.COLLECT_GRADIENTS = True
    cfg.BC.GRADIENT_STEP_COUNT = 4
    cfg.BC.N_ITERS = cfg.BC.N_ITERS * 4
else:
    name_tag = tag


new_exp_name = "CSG_%s_pretrain" % name_tag
cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('csgnet_large', 'len_corrected/revised_csg_%s' % tag) 
cfg.EXP_NAME = base_cfg.EXP_NAME.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)
