

from yacs.config import CfgNode as CN
from .base_reward import REWARD
from .train_curriculum import ENV

MAX_LENGTH = 19
MIN_LENGTH = 3
COMPLEXITY_LENGTHS = [x for x in range(MIN_LENGTH, MAX_LENGTH+1,2)]
PROPORTIONS = [1.0/len(COMPLEXITY_LENGTHS),] * len(COMPLEXITY_LENGTHS)
# ENV = CN()
# ENV.TYPE = "CADCSG"
# ENV.MODE = "EVAL"
# # CAD ENV PARAMETERS
# ENV.CAD_MAX_LENGTH = 13
# ENV.GT_PROGRAM = True
# ENV.SAMPLING = "RANDOM"
# ENV.PROGRAM_LENGTHS = [13]
# ENV.PROGRAM_PROPORTIONS = [1.0]
# ENV.DYNAMIC_MAX_LEN = False
# # Reward Specs
# ENV.REWARD = REWARD.clone()
# # for BC
# ENV.N_ENVS = 12

### Create CSG ENV
ENV = CN()
ENV.TYPE = "RestrictedCSG"
ENV.MODE = "EVAL"
ENV.SAMPLING = "RANDOM"
ENV.PROGRAM_LENGTHS = COMPLEXITY_LENGTHS
ENV.PROGRAM_PROPORTIONS = PROPORTIONS
ENV.DYNAMIC_MAX_LEN = False
# Reward Specs
ENV.REWARD = REWARD.clone()
# for BC
ENV.N_ENVS = 12
# CAD ENV PARAMETERS
ENV.CAD_MAX_LENGTH = 3
ENV.GT_PROGRAM = True