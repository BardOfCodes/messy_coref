
from configs.subconfig.machine.mikoshi import MACHINE_SPEC as mikoshi
from configs.subconfig.base_policy.me_ppo import POLICY as ME_PPO
from configs.subconfig.base_policy.ppo import POLICY as PPO
from configs.subconfig.envs.csg2d_env import RANDOM_TRAIN_ENV as train_env, RANDOM_EVAL_ENV as eval_env
from configs.subconfig.behavior_cloning.baseline import BC as bc_baseline
from configs.subconfig.lr_schedulers.warmup import LR_SCHEDULER as warmup
from configs.subconfig.lr_schedulers.reduce_plateau import LR_SCHEDULER as reduce_plateau
from yacs.config import CfgNode as CN

from .basic_3d import cfg as base_cfg
SIZE = 32
cfg = base_cfg.clone()
# Defaults
cfg.CANVAS_SHAPE = [SIZE, SIZE]
cfg.TRAIN_PROPORTION = 0.8
# OPTIONS: 
cfg.ACTION_SPACE_TYPE = "CSG2DAction" 
cfg.ACTION_RESOLUTION = 32


# EVAL CONFIG
cfg.EVAL.ENV = eval_env.clone()
cfg.EVAL.BEAM_SELECTOR = "rewards"
cfg.EVAL.BEAM_STATE_SIZE = 10
cfg.EVAL.EXHAUSTIVE = False

# TRAIN CONFIG
cfg.TRAIN.ENV = train_env.clone()

cfg.TRAIN.LR_INITIAL = 0.003

cfg.MODEL.CONFIG.INPUT_DOMAIN = "2D"

##################################
############ FOR BC ##############

### Defaults: Not necessary to change: 
# cfg.BC = bc_baseline.clone()
# cfg.BC.ENV = train_env.clone() 

####################################
########### For Value Train ########

# Specifically for value pretraining
# cfg.TRAIN.VALUE_NUM_STEPS = int(2e5)