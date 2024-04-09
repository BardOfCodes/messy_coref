
from configs.subconfig.machine.mikoshi import MACHINE_SPEC as mikoshi
from configs.subconfig.base_policy.me_ppo import POLICY as ME_PPO
from configs.subconfig.base_policy.ppo import POLICY as PPO
from configs.subconfig.envs.csg3d_train_random_rnn import ENV as train_env
from configs.subconfig.envs.csg3d_eval_random_rnn import ENV as eval_env
from configs.subconfig.behavior_cloning.baseline import BC as bc_baseline
from configs.subconfig.lr_schedulers.warmup import LR_SCHEDULER as warmup
from configs.subconfig.lr_schedulers.reduce_plateau import LR_SCHEDULER as reduce_plateau
from yacs.config import CfgNode as CN

from .basic import cfg as base_cfg
SIZE = 32
cfg = base_cfg.clone()
# Defaults
cfg.CANVAS_SHAPE = [SIZE, SIZE, SIZE]
cfg.TRAIN_PROPORTION = 0.8
# OPTIONS: 
cfg.ACTION_SPACE_TYPE = "CSG3DAction64" 
cfg.ACTION_RESOLUTION = 32


# EVAL CONFIG
cfg.EVAL = CN()
cfg.EVAL.ENV = eval_env.clone()
cfg.EVAL.BEAM_SELECTOR = "rewards"
cfg.EVAL.BEAM_STATE_SIZE = 5
cfg.EVAL.EXHAUSTIVE = False

# TRAIN CONFIG
cfg.TRAIN = CN()
cfg.TRAIN.ENV = train_env.clone()
cfg.TRAIN.NUM_STEPS = int(5e7)
cfg.TRAIN.EVAL_EPISODES = 2000 # How many episodes in EVAL
cfg.TRAIN.EVAL_FREQ = int(1e5) # How many steps between EVALs
cfg.TRAIN.SAVE_FREQ = int(2e6) # How many steps
cfg.TRAIN.BEAM_SEARCH = False
cfg.TRAIN.BEAM_SIZE = 1

cfg.TRAIN.LR_INITIAL = 0.0003
cfg.TRAIN.OPTIM = CN()
cfg.TRAIN.OPTIM.TYPE = "DEFAULT" # "ADAM_SPECIFIC"
cfg.TRAIN.OPTIM.BETA_1 = 0.9
cfg.TRAIN.OPTIM.BETA_2 = 0.98
cfg.TRAIN.OPTIM.EPSILON = 1e-9
cfg.TRAIN.LR_SCHEDULER = reduce_plateau.clone()
cfg.TRAIN.RESUME_CHECKPOINT = True

# Model:
cfg.MODEL = CN()
cfg.MODEL.EXTRACTOR = "WrapperTransformerExtractor" # Extractor architecture. 
cfg.MODEL.FEATURE_DIM = SIZE * 4 * 4
cfg.MODEL.LOAD_WEIGHTS = ""

cfg.MODEL.CONFIG = CN()
cfg.MODEL.CONFIG.TYPE = "PLADTransformerExtractor" # DefaultTransformerExtractor
cfg.MODEL.CONFIG.CNN_FIRST_STRIDE = 1
cfg.MODEL.CONFIG.POS_ENCODING_TYPE = "LEARNABLE" # "FIXED" 
cfg.MODEL.CONFIG.ATTENTION_TYPE = "DEFAULT"# "GATED" "FAST"
cfg.MODEL.CONFIG.OUTPUT_DIM = 512
cfg.MODEL.CONFIG.DROPOUT = 0.1
cfg.MODEL.CONFIG.INPUT_SEQ_LENGTH = 64
cfg.MODEL.CONFIG.OUTPUT_SEQ_LENGTH = 128
cfg.MODEL.CONFIG.NUM_ENC_LAYERS = 8
cfg.MODEL.CONFIG.NUM_DEC_LAYERS = 8
cfg.MODEL.CONFIG.NUM_HEADS = 16
cfg.MODEL.CONFIG.OUTPUT_TOKEN_COUNT = 75
cfg.MODEL.CONFIG.HIDDEN_DIM = 256
cfg.MODEL.CONFIG.INIT_DEVICE = "cuda"
cfg.MODEL.CONFIG.RETURN_ALL = False
cfg.MODEL.CONFIG.OLD_ARCH = True
cfg.MODEL.CONFIG.INPUT_DOMAIN = "3D"

cfg.MODEL.DROPOUT = 0.2

cfg.POLICY = PPO.clone()
cfg.POLICY.PPO.PI_CONF = [256, 128]
cfg.POLICY.PPO.VF_CONF = [256, 128]
cfg.POLICY.PPO.N_ENVS = 1


cfg.NOTIFICATION = CN()
cfg.NOTIFICATION.ENABLE = False
cfg.NOTIFICATION.CHANNEL = "aditya"
cfg.NOTIFICATION.WEBHOOK = ""

##################################
############ FOR BC ##############

### Defaults: Not necessary to change: 
# cfg.BC = bc_baseline.clone()
# cfg.BC.ENV = train_env.clone() 

####################################
########### For Value Train ########

# Specifically for value pretraining
# cfg.TRAIN.VALUE_NUM_STEPS = int(2e5)
