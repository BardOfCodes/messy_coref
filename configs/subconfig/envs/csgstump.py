
from yacs.config import CfgNode as CN
from .csg3d_train_random_rnn import ENV
from .base_reward_3d import REWARD
ENV = ENV.clone()
ENV.TYPE = "CSG3DShapeNet"
ENV.REWARD = REWARD.clone()
ENV.CSG_CONF.DATAMODE = "CSGSTUMP"
# ENV.PROGRAM_LENGTHS = ['03001627_chair']
ENV.PROGRAM_LENGTHS = ['03001627_chair', '04379243_table', '02828884_bench', '04256520_couch']
x = 14198.
ENV.PROGRAM_PROPORTIONS = [4746/x, 5958/x, 1272/x, 2222/x]

EVAL_ENV = ENV.clone()
EVAL_ENV.MODE = "EVAL"
EVAL_ENV.REWARD = REWARD.clone()