from yacs.config import CfgNode as CN
from .ppo import POLICY

POLICY = POLICY.clone()

POLICY.TYPE = "SACStylePPO"
POLICY.MODEL = "RestrictedActorActionCritic"