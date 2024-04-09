
from configs.ablations.mod_exp.baseline import cfg
from configs.subconfig.base_policy.me_ppo import POLICY as ME_PPO
import os



cfg = cfg.clone()

old_exp_name = "debug"
new_exp_name = "PrePhasic_DiffOpt_Correct"

cfg.EXP_NAME = cfg.EXP_NAME.replace(old_exp_name, new_exp_name)

cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)

ME_PPO.TYPE = "PhasicModExpPPO"
## TRY SAC Like
# ME_PPO.MODEL = "RestrictedActorActionCritic"

cfg.TRAIN.LR_INITIAL = 0.0003


## Important changes
ME_PPO.PPO.N_ENVS = 32
ME_PPO.PPO.N_STEPS = 1024 * 2
ME_PPO.PPO.GAMMA = 0.999
ME_PPO.PPO.BATCH_SIZE = 512
# ME_PPO.PPO.ENT_COEF = 0.15
# cfg.TRAIN.ENV.REWARD.USE_EXPLORATION_REWARD = True
# cfg.TRAIN.ENV.REWARD.EXPLORATION_BETA = 0.5
cfg.TRAIN.ENV.CAD_MAX_LENGTH= 13
cfg.TRAIN.ENV.PROGRAM_LENGTHS = [13]
cfg.EVAL.ENV.CAD_MAX_LENGTH= 13
cfg.EVAL.ENV.PROGRAM_LENGTHS = [13]

# TEMP
ME_PPO.PPO.ENABLE_TEMP = False
ME_PPO.PPO.INITIAL_TEMP = 0
## Remove noramlization? 
# ME_PPO.MODEL = "RestrictedActorCritic"
# Basically, if we normalize, we have to include normal episodes in mod buffer. 
# Otherwise we need to uncomment the following line. 
ME_PPO.ME_C.EP_MODIFIER.COLLECT_BASE = False
ME_PPO.NORMALIZE_ADVANTAGE = True
ME_PPO.PPO.GAE_LAMBDA = 0.5

# Collector Modification
# ME_PPO.ME_C.TYPE = "BaselineModExpCollector"
ME_PPO.ME_C.TYPE = "SymbolicBufferExpCollector"
# ME_PPO.ME_C.EP_SELECTOR.TYPE = "AdvantageBasedSelector"
# ME_PPO.ME_C.EP_SELECTOR.TYPE = "RewardBasedSelector"
# ME_PPO.ME_C.EP_SELECTOR.TYPE = "AllowNoEpisodes"
# ME_PPO.ME_C.EP_SELECTOR.TYPE = "AllowAllEpisodes"
# ME_PPO.ME_C.EP_SELECTOR.TYPE = "AllowUniqueIDs"
# ME_PPO.ME_C.EP_SELECTOR.TYPE = "AllowUniqueEpisodes"
ME_PPO.ME_C.EP_SELECTOR.TYPE = "AllowUniqueEpisodeAndID"

# ME_PPO.ME_C.EP_MODIFIER.TYPE = "NoModification"
# ME_PPO.ME_C.EP_MODIFIER.TYPE = "BeamSearchModifier"
ME_PPO.ME_C.EP_MODIFIER.TYPE = "DiffOptModifier"
# ME_PPO.ME_C.EP_MODIFIER.TYPE = "RefactorModifier"
# ME_PPO.ME_C.EP_MODIFIER.TYPE = "HindsightModifier"
## HACK
ME_PPO.ME_C.EP_MODIFIER.CONFIG = cfg.clone()
ME_PPO.ME_C.EP_MODIFIER.PHASE_CONFIG = cfg.TRAIN.clone()
#### HACK EXTRA
# old_loc = "/home/aditya/"
# new_loc = "/users/aganesh8/data/aganesh8/"
# temp = ME_PPO.ME_C.EP_MODIFIER.CONFIG.MACHINE_SPEC.TERMINAL_FILE
# ME_PPO.ME_C.EP_MODIFIER.CONFIG.MACHINE_SPEC.TERMINAL_FILE = temp.replace(old_loc, new_loc)
# temp = ME_PPO.ME_C.EP_MODIFIER.CONFIG.MACHINE_SPEC.DATA_PATH
# ME_PPO.ME_C.EP_MODIFIER.CONFIG.MACHINE_SPEC.DATA_PATH = temp.replace(old_loc, new_loc)
# ME_PPO.ME_C.EP_MODIFIER.SAVE_LOCATION = "/users/aganesh8/data/aganesh8/projects/temp_dir"
ME_PPO.ME_C.EP_MODIFIER.SAVE_LOCATION = "/home/aditya/projects/rl/temp_dirr"

ME_PPO.ME_C.GAE_LAMBDA = 0.5
ME_PPO.ME_C.ME_BUFFER.EPISODE_BUDGET = int(30000)
ME_PPO.ME_C.EP_SELECTOR.EXPRESSION_LIMIT = ME_PPO.ME_C.ME_BUFFER.EPISODE_BUDGET

# Trainer modification
ME_PPO.ME_T.ENABLE = True
# ME_PPO.ME_T.TYPE = "PPOTrainer"
# ME_PPO.ME_T.TYPE = "SILTrainer"
ME_PPO.ME_T.TYPE = "A2CTrainer"
ME_PPO.ME_T.NORMALIZE_ADVANTAGE = False
ME_PPO.ME_T.TRAIN_RATIO_THRES = 1.0
ME_PPO.ME_T.LOSS.POLICY_COEF = 2.0# 50
ME_PPO.ME_T.LOSS.VALUE_COEF = 0.00# 25
ME_PPO.ME_T.LOSS.ENTROPY_COEF = 0.2# 50
ME_PPO.ME_T.LOSS.BC_COEF = 0.0
ME_PPO.ME_T.LR_INITIAL = 0.001
ME_PPO.ME_T.MAX_EPOCH = 10

### RELATED To PHASIC:
ME_PPO.DEFAULT_TRAIN_ROLLOUTS = 10
ME_PPO.COLLECT_MOD_EXP_ROLLOUTS = 10
ME_PPO.MOD_EXP_TRAIN_ROLLOUTS = 10
ME_PPO.VALUE_TRAIN_ROLLOUTS = 10

cfg.TRAIN.NUM_STEPS = int(5e7)

cfg.POLICY = ME_PPO.clone()
# Change the eval and save rate:
cfg.TRAIN.EVAL_FREQ = int(1e5/ME_PPO.PPO.N_ENVS * 2) # How many steps between EVALs
cfg.TRAIN.SAVE_FREQ = int(2e10/ME_PPO.PPO.N_ENVS * 2) # How many steps
