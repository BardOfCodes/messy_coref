
from yacs.config import CfgNode as CN
from .base_reward import REWARD
REWARD = REWARD.clone()
REWARD.TYPE = "3DIOU"
REWARD.POWER = 1