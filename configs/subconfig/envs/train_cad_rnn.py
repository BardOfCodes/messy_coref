

from yacs.config import CfgNode as CN
from .train_cad import ENV
ENV = ENV.clone()
ENV.TYPE = "RNNCADCSG"