

from yacs.config import CfgNode as CN
from .eval_random import ENV

ENV = ENV.clone()
ENV.TYPE = "RNNRestrictedCSG"