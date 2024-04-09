
from configs.ablations.cql.basic_ppo import cfg
import os



cfg = cfg.clone()

old_exp_name = "debug"
new_exp_name = "RWB_baseline_normalized"

cfg.EXP_NAME = cfg.EXP_NAME.replace(old_exp_name, new_exp_name)


cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)


cfg.POLICY.TYPE = "ReinforceWithBaseline"
cfg.POLICY.PPO.VF_COEF = 0.0
cfg.POLICY.PPO.N_ENVS = 16
cfg.POLICY.PPO.BATCH_SIZE = 512
cfg.POLICY.PPO.ENT_COEF = 1e-1
cfg.POLICY.PPO.N_STEPS = 1024
cfg.POLICY.NORMALIZE_ADVANTAGE = True
cfg.TRAIN.EVAL_FREQ = int(2e5/cfg.POLICY.PPO.N_ENVS) # How many steps between EVALs