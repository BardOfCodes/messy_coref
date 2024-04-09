

from yacs.config import CfgNode as CN
from .eval_cad import ENV

ENV = ENV.clone()
ENV.TYPE = "RNNCADCSG"