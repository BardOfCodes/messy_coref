
from yacs.config import CfgNode as CN
from .base_reward import REWARD

RANDOM_TRAIN_ENV = CN()
RANDOM_TRAIN_ENV.REWARD = REWARD.clone()

RANDOM_TRAIN_ENV.TYPE = "CSG2DBase"
RANDOM_TRAIN_ENV.MODE = "TRAIN"
RANDOM_TRAIN_ENV.SAMPLING = "RANDOM"
RANDOM_TRAIN_ENV.PROGRAM_LENGTHS = [3]
RANDOM_TRAIN_ENV.PROGRAM_PROPORTIONS = [1.0]
RANDOM_TRAIN_ENV.DYNAMIC_MAX_LEN = False
# Reward Specs
# for BC
RANDOM_TRAIN_ENV.N_ENVS = 12
RANDOM_TRAIN_ENV.GT_PROGRAM = True
RANDOM_TRAIN_ENV.SET_LOADER_LIMIT = False
RANDOM_TRAIN_ENV.LOADER_LIMIT = 0

RANDOM_TRAIN_ENV.CSG_CONF = CN()
RANDOM_TRAIN_ENV.CSG_CONF.LANG_TYPE = "MCSG2D"
RANDOM_TRAIN_ENV.CSG_CONF.RESOLUTION = 64
RANDOM_TRAIN_ENV.CSG_CONF.SCALE = 64
RANDOM_TRAIN_ENV.CSG_CONF.SCAD_RESOLUTION = 30
RANDOM_TRAIN_ENV.CSG_CONF.DRAW_TYPE = "basic"
RANDOM_TRAIN_ENV.CSG_CONF.DRAW_MODE = "direct"
RANDOM_TRAIN_ENV.CSG_CONF.BOOLEAN_COUNT = 10
RANDOM_TRAIN_ENV.CSG_CONF.PERM_MAX_LEN = 128
RANDOM_TRAIN_ENV.CSG_CONF.GENERATOR_N_PROCS = 2
RANDOM_TRAIN_ENV.CSG_CONF.VALID_DRAWS = ["sphere", "cuboid"]
RANDOM_TRAIN_ENV.CSG_CONF.VALID_TRANFORMS = ["scale", "rotate", "translate"]
RANDOM_TRAIN_ENV.CSG_CONF.VALID_BOOL = ['union', 'intersection', 'difference']
RANDOM_TRAIN_ENV.CSG_CONF.MAX_EXPRESSION_COMPLEXITY = 35
RANDOM_TRAIN_ENV.CSG_CONF.DATAMODE = "PLAD"
RANDOM_TRAIN_ENV.CSG_CONF.RESTRICT_DATASIZE = False
RANDOM_TRAIN_ENV.CSG_CONF.DATASIZE = 0



RANDOM_EVAL_ENV = RANDOM_TRAIN_ENV.clone()
RANDOM_EVAL_ENV.MODE = "EVAL"
RANDOM_EVAL_ENV.REWARD = REWARD.clone()

SHAPENET_TRAIN_ENV = RANDOM_TRAIN_ENV.clone()
SHAPENET_TRAIN_ENV.TYPE = "CSG2DShapeNet"
SHAPENET_TRAIN_ENV.REWARD = REWARD.clone()
SHAPENET_TRAIN_ENV.PROGRAM_LENGTHS = ['CAD']
SHAPENET_TRAIN_ENV.PROGRAM_PROPORTIONS = [1]

SHAPENET_EVAL_ENV = SHAPENET_TRAIN_ENV.clone()
SHAPENET_EVAL_ENV.MODE = "EVAL"
SHAPENET_EVAL_ENV.REWARD = REWARD.clone()