
from yacs.config import CfgNode as CN
from .train_random import ENV
from .base_reward_3d import REWARD
RANDOM_TRAIN_ENV = ENV.clone()
RANDOM_TRAIN_ENV.TYPE = "SA3DBase"
RANDOM_TRAIN_ENV.SET_LOADER_LIMIT = False
RANDOM_TRAIN_ENV.LOADER_LIMIT = 0
RANDOM_TRAIN_ENV.SA_CONF = CN()
RANDOM_TRAIN_ENV.SA_CONF.RESOLUTION = 32
RANDOM_TRAIN_ENV.SA_CONF.SCALE = 32
RANDOM_TRAIN_ENV.SA_CONF.SCAD_RESOLUTION = 30
RANDOM_TRAIN_ENV.PROGRAM_LENGTHS = list(range(4, 12))
n = 12 - 4
RANDOM_TRAIN_ENV.PROGRAM_PROPORTIONS = [1/n for i in range(4, 12)]
# Add more limits here.
RANDOM_TRAIN_ENV.SA_CONF.MASTER_MIN_PRIM = 2
RANDOM_TRAIN_ENV.SA_CONF.MASTER_MAX_PRIM = 5
RANDOM_TRAIN_ENV.SA_CONF.SUB_MIN_PRIM = 1
RANDOM_TRAIN_ENV.SA_CONF.SUB_MAX_PRIM = 4
RANDOM_TRAIN_ENV.SA_CONF.MAX_SUB_PROGS = 4
RANDOM_TRAIN_ENV.SA_CONF.PERM_MAX_COUNT = 15# 192
RANDOM_TRAIN_ENV.SA_CONF.VARIED_SUBPROG_BBX = False
RANDOM_TRAIN_ENV.SA_CONF.N_CUBOID_IND_STATES = 5
RANDOM_TRAIN_ENV.SA_CONF.MAX_EXPRESSION_COMPLEXITY = 50


RANDOM_TRAIN_ENV.SA_CONF.BOOLEAN_COUNT = 5
RANDOM_TRAIN_ENV.SA_CONF.PERM_MAX_LEN = 256 # 192
RANDOM_TRAIN_ENV.SA_CONF.GENERATOR_N_PROCS = 5
RANDOM_TRAIN_ENV.SA_CONF.LANGUAGE_NAME = "PSA3D"
RANDOM_TRAIN_ENV.REWARD = REWARD.clone()
### For the PLAD data:
# PROPORTIONS = [202034, 197226, 191934, 188293, 184874, 180798, 177654, 173719, 171261, 167963, 164244]
# ENV.PROGRAM_PROPORTIONS = [1/2000000.0 for x in PROPORTIONS]

RANDOM_EVAL_ENV = RANDOM_TRAIN_ENV.clone()
RANDOM_EVAL_ENV.MODE = "EVAL"
RANDOM_EVAL_ENV.REWARD = REWARD.clone()

SHAPENET_TRAIN_ENV = RANDOM_TRAIN_ENV.clone()
SHAPENET_TRAIN_ENV.TYPE = "SA3DShapeNet"
SHAPENET_TRAIN_ENV.REWARD = REWARD.clone()
# ENV.PROGRAM_LENGTHS = ["04099429_rocket"]
# ENV.PROGRAM_PROPORTIONS = [0.5, 0.5]
# ENV.PROGRAM_LENGTHS = ["02828884_bench"]
# ENV.PROGRAM_PROPORTIONS = [1.0]
# ENV.PROGRAM_LENGTHS = ['03790512_motorbike', '03001627_chair', '03261776_earphone', '03624134_knife', '04379243_table', 
#              '03642806_laptop', '03797390_mug', '04225987_skateboard', '02773838_bag', '02954340_cap', 
#              '03467517_guitar', '03636649_lamp', '02691156_airplane', '04099429_rocket', '03948459_pistol']
# sample proportional to content:
# ENV.PROGRAM_PROPORTIONS = [1.0/len(ENV.PROGRAM_LENGTHS) for x in ENV.PROGRAM_LENGTHS]
SHAPENET_TRAIN_ENV.PROGRAM_LENGTHS = ['03001627_chair', '04379243_table', '02828884_bench', '04256520_couch']
SHAPENET_TRAIN_ENV.PROGRAM_PROPORTIONS = [3746/13998., 1816/13998., 3173/13998., 5263/13998.]
SHAPENET_TRAIN_ENV.SA_CONF.RESTRICT_DATASIZE = False
SHAPENET_TRAIN_ENV.SA_CONF.DATASIZE = 0
SHAPENET_TRAIN_ENV.SA_CONF.DATAMODE = "PLAD"
# ENV.MODE = "EVAL"
# ENV.PROGRAM_LENGTHS = ['00000000_temp']
# ENV.PROGRAM_PROPORTIONS = [1.0]


SHAPENET_EVAL_ENV = SHAPENET_TRAIN_ENV.clone()
SHAPENET_EVAL_ENV.MODE = "EVAL"
SHAPENET_EVAL_ENV.REWARD = REWARD.clone()