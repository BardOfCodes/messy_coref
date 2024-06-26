

from yacs.config import CfgNode as CN
from .base_reward import REWARD

ENV = CN()
ENV.TYPE = "CADCSG"
ENV.MODE = "TRAIN"
# CAD ENV PARAMETERS
ENV.CAD_MAX_LENGTH = 13
ENV.GT_PROGRAM = True
ENV.SAMPLING = "RANDOM"
ENV.PROGRAM_LENGTHS = [13]
ENV.PROGRAM_PROPORTIONS = [1.0]
ENV.DYNAMIC_MAX_LEN = False
# Reward Specs
ENV.REWARD = REWARD.clone()
# for BC
# ENV.N_ENVS = 12