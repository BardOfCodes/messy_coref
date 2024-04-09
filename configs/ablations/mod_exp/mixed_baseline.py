
from configs.ablations.mod_exp.baseline import cfg
from configs.subconfig.base_policy.mixed_ppo import POLICY as MIX_PPO
import os



cfg = cfg.clone()

old_exp_name = "baseline"
new_exp_name = "mixed_exp_baseline"

cfg.EXP_NAME = cfg.EXP_NAME.replace(old_exp_name, new_exp_name)

MIX_PPO.N_ENVS = 32

cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)

cfg.POLICY = MIX_PPO.clone()
# Change the eval and save rate:
cfg.TRAIN.EVAL_FREQ = int(2e5/MIX_PPO.PPO.N_ENVS) # How many steps between EVALs
cfg.TRAIN.SAVE_FREQ = int(4e10/MIX_PPO.PPO.N_ENVS) # How many steps