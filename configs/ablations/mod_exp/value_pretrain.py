
from configs.ablations.mod_exp.baseline import cfg
import os



cfg = cfg.clone()

old_exp_name = "debug"
new_exp_name = "value_pretrain"

cfg.EXP_NAME = cfg.EXP_NAME.replace(old_exp_name, new_exp_name)

cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)


# Specifically for value pretraining
cfg.TRAIN.VALUE_NUM_STEPS = int(2e7)