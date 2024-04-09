

from yacs.config import CfgNode as CN
from .base_reward import REWARD
from .train_complexity import ENV

ENV = ENV.clone()
ENV.MODE = "EVAL"