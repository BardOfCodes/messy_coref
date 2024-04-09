
from yacs.config import CfgNode as CN
from .csg3d_train_random_rnn import ENV
from .base_reward_3d import REWARD
UCSG_ENV = ENV.clone()
UCSG_ENV.TYPE = "CSG3DShapeNet"
UCSG_ENV.REWARD = REWARD.clone()
# ENV.PROGRAM_LENGTHS = ["04099429_rocket"]
# ENV.PROGRAM_PROPORTIONS = [0.5, 0.5]
# ENV.PROGRAM_LENGTHS = ["02828884_bench"]
# ENV.PROGRAM_PROPORTIONS = [1.0]
# ENV.PROGRAM_LENGTHS = ['03790512_motorbike', '03001627_chair', '03261776_earphone', '03624134_knife', '04379243_table', 
#              '03642806_laptop', '03797390_mug', '04225987_skateboard', '02773838_bag', '02954340_cap', 
#              '03467517_guitar', '03636649_lamp', '02691156_airplane', '04099429_rocket', '03948459_pistol']
# sample proportional to content:
# ENV.PROGRAM_PROPORTIONS = [1.0/len(ENV.PROGRAM_LENGTHS) for x in ENV.PROGRAM_LENGTHS]
UCSG_ENV.PROGRAM_LENGTHS = ['ucsgnet_data']
UCSG_ENV.PROGRAM_PROPORTIONS = [1.0]
# ENV.MODE = "EVAL"
# ENV.PROGRAM_LENGTHS = ['00000000_temp']
# ENV.PROGRAM_PROPORTIONS = [1.0]


UCSG_ENV_EVAL = UCSG_ENV.clone()
UCSG_ENV_EVAL.MODE = "EVAL"
UCSG_ENV_EVAL.REWARD = REWARD.clone()