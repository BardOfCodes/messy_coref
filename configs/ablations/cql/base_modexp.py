
from configs.ablations.cql.basic_ppo import cfg
from configs.subconfig.base_policy.me_ppo import POLICY as ME_PPO
import os



cfg = cfg.clone()

old_exp_name = "debug"
new_exp_name = "debug"

cfg.EXP_NAME = cfg.EXP_NAME.replace(old_exp_name, new_exp_name)

cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)

ME_PPO.TYPE = "PhasicModExpPPO"
## TRY SAC Like
ME_PPO.MODEL = "OldRestrictedActorCritic"
## Important changes
ME_PPO.PPO.N_ENVS = 4
ME_PPO.PPO.BATCH_SIZE = 512
ME_PPO.PPO.N_STEPS = 1024
ME_PPO.PPO.ENT_COEF = 5e-1

# TEMP
## Remove noramlization? 
# ME_PPO.MODEL = "RestrictedActorCritic"
# Basically, if we normalize, we have to include normal episodes in mod buffer. 
# Otherwise we need to uncomment the following line. 
ME_PPO.ME_C.EP_MODIFIER.COLLECT_BASE = False
ME_PPO.NORMALIZE_ADVANTAGE = True
ME_PPO.GAE_LAMBDA = 0.5

# Collector Modification
# cfg.POLICY.ME_C.TYPE = "BaselineModExpCollector"
ME_PPO.ME_C.TYPE = "BaselineModExpCollector"
# ME_PPO.ME_C.EP_SELECTOR.TYPE = "AdvantageBasedSelector"
# ME_PPO.ME_C.EP_SELECTOR.TYPE = "RewardBasedSelector"
# ME_PPO.ME_C.EP_SELECTOR.TYPE = "AllowUniqueEpisodes"
ME_PPO.ME_C.EP_SELECTOR.TYPE = "AllowUniqueIDs"
# ME_PPO.ME_C.EP_MODIFIER.TYPE = "NoModification"
# ME_PPO.ME_C.EP_MODIFIER.TYPE = "DiffOptModifier"
ME_PPO.ME_C.EP_MODIFIER.TYPE = "BeamSearchModifier"
# ME_PPO.ME_C.EP_MODIFIER.TYPE = "OracleModifier"
# ME_PPO.ME_C.EP_MODIFIER.TYPE = "RefactorModifier"
# ME_PPO.ME_C.EP_MODIFIER.TYPE = "HindsightModifier"
## HACK
ME_PPO.ME_C.EP_MODIFIER.CONFIG = cfg.clone()
ME_PPO.ME_C.EP_MODIFIER.PHASE_CONFIG = cfg.TRAIN.clone()

ME_PPO.ME_C.GAE_LAMBDA = 0.5
ME_PPO.ME_C.ME_BUFFER.EPISODE_BUDGET = int(10000)
ME_PPO.ME_C.EP_SELECTOR.EXPRESSION_LIMIT = ME_PPO.ME_C.ME_BUFFER.EPISODE_BUDGET

# Trainer modification
ME_PPO.ME_T.ENABLE = True
# ME_PPO.ME_T.TYPE = "OffLineTrainer"
# ME_PPO.ME_T.TYPE = "SILTrainer"
ME_PPO.ME_T.TYPE = "A2CTrainer"
ME_PPO.ME_T.NORMALIZE_ADVANTAGE = False
ME_PPO.ME_T.TRAIN_RATIO_THRES = 1.5
ME_PPO.ME_T.LOSS.POLICY_COEF = 0.0# 50
ME_PPO.ME_T.LOSS.VALUE_COEF = 0.0# 25
ME_PPO.ME_T.LOSS.ENTROPY_COEF = 0.5# 50
ME_PPO.ME_T.LOSS.BC_COEF = 1.0
ME_PPO.ME_T.MAX_EPOCH = 5

### RELATED To PHASIC:
ME_PPO.DEFAULT_TRAIN_ROLLOUTS = 100
ME_PPO.COLLECT_MOD_EXP_ROLLOUTS = 100
ME_PPO.MOD_EXP_TRAIN_ROLLOUTS = 1
ME_PPO.VALUE_TRAIN_ROLLOUTS = 100



cfg.POLICY = ME_PPO.clone()
cfg.TRAIN.EVAL_FREQ = int(1e5/ME_PPO.PPO.N_ENVS) # How many steps between EVALs
