
from configs.ablations.mod_exp.baseline import cfg
from configs.subconfig.base_policy.me_ppo import POLICY as ME_PPO
import os



cfg = cfg.clone()

old_exp_name = "baseline"
new_exp_name = "mod_exp_baseline"

cfg.EXP_NAME = cfg.EXP_NAME.replace(old_exp_name, new_exp_name)

ME_PPO.N_ENVS = 32

cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)

cfg.POLICY = ME_PPO.clone()
# Change the eval and save rate:
cfg.TRAIN.EVAL_FREQ = int(1e5/ME_PPO.PPO.N_ENVS * 2) # How many steps between EVALs
cfg.TRAIN.SAVE_FREQ = int(2e10/ME_PPO.PPO.N_ENVS * 2) # How many steps