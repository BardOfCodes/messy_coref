
from yacs.config import CfgNode as CN
from .csg3d_train_random_rnn import ENV
from .base_reward_3d import REWARD
ENV = ENV.clone()
ENV.MODE = "EVAL"
ENV.REWARD = REWARD.clone()