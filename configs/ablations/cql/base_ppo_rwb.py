
from configs.ablations.cql.base_modexp import cfg
import os



cfg = cfg.clone()

old_exp_name = "Mod_Exp_Offline"
new_exp_name = "Offline_RWB"

cfg.EXP_NAME = cfg.EXP_NAME.replace(old_exp_name, new_exp_name)


cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)

# cfg.POLICY.ME_T.TYPE = "OffLineTrainer"
# ME_PPO.ME_T.TYPE = "SILTrainer"
cfg.POLICY.ME_C.TYPE = "BaselineModExpCollector"

cfg.POLICY.ME_T.ENABLE = True
cfg.POLICY.ME_C.EP_MODIFIER.COLLECT_BASE = False
cfg.POLICY.ME_T.TYPE = "A2CTrainer"
cfg.POLICY.ME_T.NORMALIZE_ADVANTAGE = False
cfg.POLICY.ME_T.LOSS.POLICY_COEF = 1.0# 50
cfg.POLICY.ME_T.LOSS.VALUE_COEF = 0.0# 25
cfg.POLICY.ME_T.LOSS.ENTROPY_COEF = 0.5# 50
cfg.POLICY.ME_T.LOSS.BC_COEF = 0.0