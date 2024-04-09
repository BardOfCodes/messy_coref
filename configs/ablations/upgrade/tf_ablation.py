
# from configs.cluster import cfg
from configs.ablations.upgrade.pretrain_baseline import cfg as base_cfg
from configs.subconfig.lr_schedulers.warmup import LR_SCHEDULER as warmup
import os
import numpy as np
import scipy.stats as spyst


old_exp_name = "CSG_pretrain_baseline"
new_exp_name = "debug"
cfg = base_cfg.clone()

# EncDec
# new_exp_name = "tf_pretrain_baseline_EncDec"
# cfg.BC.BATCH_SIZE = 48
# cfg.MODEL.CONFIG.TYPE = "DefaultTransformerExtractor" 

# PosFixed
# new_exp_name = "tf_pretrain_baseline_PosFixed"
# cfg.MODEL.CONFIG.POS_ENCODING_TYPE = "FIXED" # "" 

# Gating
# new_exp_name = "tf_pretrain_baseline_gating"
# cfg.MODEL.CONFIG.ATTENTION_TYPE = "GATED"# "GATED"

# Small Inp
# new_exp_name = "tf_pretrain_baseline_small_3d"
# cfg.MODEL.CONFIG.CNN_FIRST_STRIDE = 2
# cfg.MODEL.CONFIG.INPUT_SEQ_LENGTH = 8

# Fast Former
# new_exp_name = "tf_pretrain_baseline_fastformer"
# cfg.MODEL.CONFIG.TYPE = "FastTransformerExtractor" # DefaultTransformerExtractor
# cfg.MODEL.CONFIG.ATTENTION_TYPE = "FAST"# "GATED"
# cfg.BC.BATCH_SIZE = 8
# cfg.POLICY.PPO.N_ENVS = 4
# cfg.BC.ENV.NUM_WORKERS = 4
# cfg.BC.N_ENVS = 4
# cfg.BC.N_ITERS = int(500)
# cfg.TRAIN.EVAL_EPISODES = 500 # How many episodes in EVAL
# cfg.MODEL.CONFIG.NUM_HEADS = 8

# Warm Up
# new_exp_name = "tf_pretrain_baseline_warm_up"
# cfg.TRAIN.OPTIM.TYPE = "ADAM_SPECIFIC" # ""
# cfg.TRAIN.OPTIM.BETA_1 = 0.9
# cfg.TRAIN.OPTIM.BETA_2 = 0.98
# cfg.TRAIN.OPTIM.EPSILON = 1e-9
# cfg.TRAIN.LR_SCHEDULER = warmup.clone()

# label Smooth
# new_exp_name = "tf_pretrain_baseline_label_smooth"
# cfg.BC.LABEL_SMOOTHING = True




## New Sampling Distribution
if True:
    ## Larger:
    if True: 
        new_exp_name = "CSG_pretrain_baseline_till_l10" 
        MAX_LENGTH = 11
        cfg.BC.ENV.NUM_WORKERS = 1
        cfg.BC.N_ENVS = 1
        cfg.POLICY.PPO.N_ENVS = 1
        max_bool = 10
        cfg.BC.ENV.CSG_CONF.BOOLEAN_COUNT = max_bool
        cfg.TRAIN.ENV.CSG_CONF.BOOLEAN_COUNT = max_bool
        cfg.EVAL.ENV.CSG_CONF.BOOLEAN_COUNT = max_bool
        # MIN_LENGTH = 10
        if True:
            ## Also have to change max length specifications everywhere
            max_len = 128 + 64
            cfg.MODEL.CONFIG.OUTPUT_SEQ_LENGTH = max_len
            cfg.BC.ENV.CSG_CONF.PERM_MAX_LEN = max_len
            cfg.TRAIN.ENV.CSG_CONF.PERM_MAX_LEN = max_len
            cfg.EVAL.ENV.CSG_CONF.PERM_MAX_LEN = max_len

    else:
        new_exp_name = "CSG_pretrain_baseline_new_sampling"
        MAX_LENGTH = 6
    MIN_LENGTH = 1

    cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('csgnet_large', 'revised_csgnet_large') 
    COMPLEXITY_LENGTHS = [x for x in range(MIN_LENGTH, MAX_LENGTH)]
    beta_distr = spyst.beta(2,2)
    complex_values = [beta_distr.pdf(x/MAX_LENGTH) for x in COMPLEXITY_LENGTHS]
    PROPORTIONS = [x/np.sum(complex_values) for x in complex_values]
    # PROPORTIONS = [1/len(COMPLEXITY_LENGTHS) for x in COMPLEXITY_LENGTHS]
    cfg.TRAIN.ENV.PROGRAM_LENGTHS = COMPLEXITY_LENGTHS
    cfg.TRAIN.ENV.PROGRAM_PROPORTIONS = PROPORTIONS
    cfg.BC.ENV.PROGRAM_LENGTHS = COMPLEXITY_LENGTHS
    cfg.BC.ENV.PROGRAM_PROPORTIONS = PROPORTIONS
    # cfg.BC.N_ITERS = int(500)
    # cfg.TRAIN.EVAL_EPISODES = 500 

## New dataset:
if False:
    new_exp_name = "CSG_pretrain_baseline_new_data"
    cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('csgnet_large', 'revised_csgnet_large') 




cfg.EXP_NAME = base_cfg.EXP_NAME.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)