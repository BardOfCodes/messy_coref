
# from configs.cluster import cfg
from configs.ablations.upgrade.tf_plad import cfg as base_cfg
from configs.subconfig.lr_schedulers.warmup import LR_SCHEDULER as warmup
from configs.subconfig.envs.csg3d_train_random_rnn import ENV as train_random
from configs.subconfig.envs.csg3d_eval_random_rnn import ENV as eval_random
import os
import numpy as np

cfg = base_cfg.clone()

old_exp_name = "tf_plad_new_baseline"


# Try on Random CSG.
new_exp_name = "tf_plad_rand_csg_pretrained"
cfg.TRAIN.ENV = train_random.clone()
cfg.EVAL.ENV = eval_random.clone()
MAX_LENGTH = 2
MIN_LENGTH = 1
COMPLEXITY_LENGTHS = [x for x in range(MIN_LENGTH, MAX_LENGTH)]
power = 0.5
PROPORTIONS = [np.power(x, power)/np.sum(np.power(COMPLEXITY_LENGTHS, power)) for x in COMPLEXITY_LENGTHS]
# PROPORTIONS = [1/len(COMPLEXITY_LENGTHS) for x in COMPLEXITY_LENGTHS]
cfg.TRAIN.ENV.PROGRAM_LENGTHS = COMPLEXITY_LENGTHS
cfg.TRAIN.ENV.PROGRAM_PROPORTIONS = PROPORTIONS
cfg.EVAL.ENV.PROGRAM_LENGTHS = COMPLEXITY_LENGTHS
cfg.EVAL.ENV.PROGRAM_PROPORTIONS = PROPORTIONS
train_random.PROGRAM_LENGTHS = COMPLEXITY_LENGTHS
train_random.PROGRAM_PROPORTIONS = PROPORTIONS
cfg.Train.ENV.SET_LOADER_LIMIT = True
cfg.Train.ENV.LOADER_LIMIT = 2500
cfg.Train.ENV.SET_LOADER_LIMIT = True
cfg.Train.ENV.LOADER_LIMIT = 400
train_random.SET_LOADER_LIMIT = True
train_random.LOADER_LIMIT = 2500
cfg.BC.ENV = train_random.clone()
# Since there are more samples - train for longer as well:
cfg.BC.PLAD.BEAM_SEARCH_FREQUENCY = 15 # In terms of number of epochs
cfg.BC.ENV.TYPE = 'CSG3DBaseBC'
cfg.EVAL.EXHAUSTIVE = False

# Remove Pretraining:
# new_exp_name = "tf_plad_no_pretrain"
# cfg.MODEL.LOAD_WEIGHTS = ""

# Higher Beam Size:
# new_exp_name = "tf_plad_beam_20"
# cfg.BC.PLAD.BEAM_SEARCH_PARAMS.BEAM_SIZE = 20
# cfg.BC.PLAD.BEAM_SEARCH_PARAMS.BATCH_SIZE = 1

# Without Latent Execution:
# new_exp_name = "tf_plad_no_LE"
# cfg.BC.PLAD.LATENT_EXECUTION = False

# no_LS:
# new_exp_name = "tf_plad_no_LS"
# cfg.BC.LABEL_SMOOTHING = False

####### TBD

# With Multiple Best Programs:
# new_exp_name = "tf_plad_multiple_progs"
# cfg.BC.PLAD.BEST_PROG_COUNT = 5

# Reward Weighted regression:


cfg.EXP_NAME = base_cfg.EXP_NAME.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)