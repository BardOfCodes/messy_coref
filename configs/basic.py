
from configs.subconfig.machine.mikoshi import MACHINE_SPEC as mikoshi
from configs.subconfig.base_policy.dqn import POLICY as dqn
from configs.subconfig.envs.train_random import ENV as train_env
from configs.subconfig.envs.eval_random import ENV as eval_env
from configs.subconfig.behavior_cloning.baseline import BC as bc_baseline
from configs.subconfig.lr_schedulers.reduce_plateau import LR_SCHEDULER as reduce_plateau
from yacs.config import CfgNode as CN

SIZE = 64

cfg = CN()
cfg.EXP_NAME = "{0}_{1}".format(mikoshi.EXP_NAME, "debugging")
cfg.SEED = 1234
cfg.MACHINE_SPEC = mikoshi.clone()

# Defaults
cfg.CANVAS_SHAPE = [SIZE, SIZE]
cfg.CANVAS_SLATE = True
# Set this to 1 to make weaker agents.
cfg.OBSERVABLE_STACK = 2
cfg.TRAIN_PROPORTION = 0.7
# OPTIONS: 
cfg.ACTION_SPACE_TYPE = "RestrictedAction" 


# TRAIN CONFIG
cfg.TRAIN = CN()
cfg.TRAIN.ENV = train_env.clone()
cfg.TRAIN.NUM_STEPS = int(5e7)
cfg.TRAIN.EVAL_EPISODES = 3000 # How many episodes in EVAL
cfg.TRAIN.EVAL_FREQ = int(1e5) # How many steps between EVALs
cfg.TRAIN.SAVE_FREQ = int(2e6) # How many steps

cfg.TRAIN.LR_INITIAL = 0.003
cfg.TRAIN.LR_SCHEDULER = reduce_plateau.clone()
cfg.TRAIN.RESUME_CHECKPOINT = True


# EVAL CONFIG
cfg.EVAL = CN()
cfg.EVAL.ENV = eval_env.clone()
cfg.EVAL.BEAM_SELECTOR = "rewards"
cfg.EVAL.BEAM_STATE_SIZE = 1

# Model:
cfg.MODEL = CN()
cfg.MODEL.EXTRACTOR = "WrapperConvCoordExtractor" # Extractor architecture. 
cfg.MODEL.FEATURE_DIM = SIZE * 4 * 4
cfg.MODEL.POLICY_ARCH = [512, 256, 128] # ONly used in DQN
cfg.MODEL.LOAD_WEIGHTS = ""
cfg.MODEL.CONFIG = CN()
cfg.MODEL.DROPOUT = 0.2

cfg.POLICY = dqn.clone()

cfg.USE_MEMORY_STACK = False
##################################
############ FOR BC ##############

### Defaults: Not necessary to change: 
# cfg.BC = bc_baseline.clone()
# cfg.BC.ENV = train_env.clone() 

####################################
########### For Value Train ########

# Specifically for value pretraining
# cfg.TRAIN.VALUE_NUM_STEPS = int(2e5)