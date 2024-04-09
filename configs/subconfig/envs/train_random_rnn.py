

from yacs.config import CfgNode as CN
from .train_random import ENV
ENV = ENV.clone()
ENV.TYPE = "RNNRestrictedCSG"