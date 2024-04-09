
from configs.ablations.cql.basic_ppo import cfg
from configs.subconfig.base_policy.sac import POLICY as SAC_POLICY 
import os



cfg = cfg.clone()

old_exp_name = "debug"
new_exp_name = "sac_baseline"

cfg.EXP_NAME = cfg.EXP_NAME.replace(old_exp_name, new_exp_name)


cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)


cfg.POLICY = SAC_POLICY.clone()
# Change the eval and save rate:
cfg.TRAIN.EVAL_FREQ = int(2e5/SAC_POLICY.PPO.N_ENVS) # How many steps between EVALs
cfg.TRAIN.SAVE_FREQ = int(2e6/SAC_POLICY.PPO.N_ENVS) # How many steps