
import scipy.stats as spyst
import numpy as np


def set_lang_mode(config, mode, retain_action_type=False):
    config.TRAIN.ENV.CSG_CONF.LANG_TYPE = mode
    config.EVAL.ENV.CSG_CONF.LANG_TYPE = mode
    config.BC.ENV.CSG_CONF.LANG_TYPE = mode
    if not retain_action_type:
        config.ACTION_SPACE_TYPE = "%sAction%s" % (mode, config.ACTION_SPACE_TYPE[-2:])
    if "NRCSG" in mode:
        config.TRAIN.ENV.CSG_CONF.VALID_DRAWS = ["sphere", "cylinder", "cuboid"]
        config.EVAL.ENV.CSG_CONF.VALID_DRAWS = ["sphere", "cylinder", "cuboid"]
        config.BC.ENV.CSG_CONF.VALID_DRAWS = ["sphere", "cylinder", "cuboid"]
    elif mode == "PCSG3D":
        config.TRAIN.ENV.CSG_CONF.VALID_DRAWS = ["sphere", "cuboid"]
        config.EVAL.ENV.CSG_CONF.VALID_DRAWS = ["sphere", "cuboid"]
        config.BC.ENV.CSG_CONF.VALID_DRAWS = ["sphere", "cuboid"]
    elif "3D" in mode:
        config.TRAIN.ENV.CSG_CONF.VALID_DRAWS = ["sphere", "cylinder", "cuboid"]
        config.EVAL.ENV.CSG_CONF.VALID_DRAWS = ["sphere", "cylinder", "cuboid"]
        config.BC.ENV.CSG_CONF.VALID_DRAWS = ["sphere", "cylinder", "cuboid"]
    elif "2D" in mode:
        config.TRAIN.ENV.CSG_CONF.VALID_DRAWS = ["sphere", "cuboid"]
        config.EVAL.ENV.CSG_CONF.VALID_DRAWS = ["sphere", "cuboid"]
        config.BC.ENV.CSG_CONF.VALID_DRAWS = ["sphere", "cuboid"]

    return config


def set_half_resolution(config):

    config.ACTION_SPACE_TYPE = config.ACTION_SPACE_TYPE.replace("64", "32")
    if "CSG_CONF" in config.TRAIN.ENV:
        config.TRAIN.ENV.CSG_CONF.RESOLUTION = 32
        config.EVAL.ENV.CSG_CONF.RESOLUTION = 32
        config.BC.ENV.CSG_CONF.RESOLUTION = 32
        config.TRAIN.ENV.CSG_CONF.SCALE = 32
        config.EVAL.ENV.CSG_CONF.SCALE = 32
        config.BC.ENV.CSG_CONF.SCALE = 32
    else:
        config.TRAIN.ENV.SA_CONF.RESOLUTION = 32
        config.EVAL.ENV.SA_CONF.RESOLUTION = 32
        config.BC.ENV.SA_CONF.RESOLUTION = 32
        config.TRAIN.ENV.SA_CONF.SCALE = 32
        config.EVAL.ENV.SA_CONF.SCALE = 32
        config.BC.ENV.SA_CONF.SCALE = 32

    config.CANVAS_SHAPE = [32,] * len(config.CANVAS_SHAPE)
    config.MODEL.CONFIG.INPUT_SEQ_LENGTH = 2 ** (len(config.CANVAS_SHAPE))
    return config


def set_full_resolution(config):

    config.ACTION_SPACE_TYPE = config.ACTION_SPACE_TYPE.replace("32", "64")
    if "CSG_CONF" in config.TRAIN.ENV:
        config.TRAIN.ENV.CSG_CONF.RESOLUTION = 64
        config.EVAL.ENV.CSG_CONF.RESOLUTION = 64
        config.BC.ENV.CSG_CONF.RESOLUTION = 64
        config.TRAIN.ENV.CSG_CONF.SCALE = 64
        config.EVAL.ENV.CSG_CONF.SCALE = 64
        config.BC.ENV.CSG_CONF.SCALE = 64
    else:
        config.TRAIN.ENV.SA_CONF.RESOLUTION = 64
        config.EVAL.ENV.SA_CONF.RESOLUTION = 64
        config.BC.ENV.SA_CONF.RESOLUTION = 64
        config.TRAIN.ENV.SA_CONF.SCALE = 64
        config.EVAL.ENV.SA_CONF.SCALE = 64
        config.BC.ENV.SA_CONF.SCALE = 64

    config.CANVAS_SHAPE = [64, ] * len(config.CANVAS_SHAPE)
    config.MODEL.CONFIG.INPUT_SEQ_LENGTH = 4 ** (len(config.CANVAS_SHAPE))
    return config

def set_action_specs(config, length, n_bools, lang_name):

    if "CSG" in lang_name:
        config.MODEL.CONFIG.OUTPUT_SEQ_LENGTH = length
        
        config.BC.ENV.CSG_CONF.PERM_MAX_LEN = length
        config.TRAIN.ENV.CSG_CONF.PERM_MAX_LEN = length
        config.EVAL.ENV.CSG_CONF.PERM_MAX_LEN = length
        
        config.BC.ENV.CSG_CONF.BOOLEAN_COUNT = n_bools
        config.TRAIN.ENV.CSG_CONF.BOOLEAN_COUNT = n_bools
        config.EVAL.ENV.CSG_CONF.BOOLEAN_COUNT = n_bools
    else:
        if "PSA" in lang_name:
            config.TRAIN.ENV.SA_CONF.N_CUBOID_IND_STATES = 11
            config.EVAL.ENV.SA_CONF.N_CUBOID_IND_STATES = 11
            config.BC.ENV.SA_CONF.N_CUBOID_IND_STATES = 11
            config.TRAIN.ENV.SA_CONF.MASTER_MAX_PRIM = 9
            config.EVAL.ENV.SA_CONF.MASTER_MAX_PRIM = 9
            config.BC.ENV.SA_CONF.MASTER_MAX_PRIM = 9
            config.TRAIN.ENV.SA_CONF.MAX_SUB_PROGS = 0
            config.EVAL.ENV.SA_CONF.MAX_SUB_PROGS = 0
            config.BC.ENV.SA_CONF.MAX_SUB_PROG = 0
        elif "HSA" in lang_name:
            config.TRAIN.ENV.SA_CONF.N_CUBOID_IND_STATES = 5
            config.EVAL.ENV.SA_CONF.N_CUBOID_IND_STATES = 5
            config.BC.ENV.SA_CONF.N_CUBOID_IND_STATES = 5
    return config


def set_program_lens(config, min_len, max_len):

    COMPLEXITY_LENGTHS = [x for x in range(min_len, max_len)]
    beta_distr = spyst.beta(2, 4)
    complex_values = [beta_distr.pdf(x/max_len) for x in COMPLEXITY_LENGTHS]
    PROPORTIONS = [x/np.sum(complex_values) for x in complex_values]
    # PROPORTIONS = [1/len(COMPLEXITY_LENGTHS) for x in COMPLEXITY_LENGTHS]
    config.TRAIN.ENV.PROGRAM_LENGTHS = COMPLEXITY_LENGTHS
    config.TRAIN.ENV.PROGRAM_PROPORTIONS = PROPORTIONS
    config.BC.ENV.PROGRAM_LENGTHS = COMPLEXITY_LENGTHS
    config.BC.ENV.PROGRAM_PROPORTIONS = PROPORTIONS
    return config