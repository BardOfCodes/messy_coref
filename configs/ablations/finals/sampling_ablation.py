
from configs.ablations.finals.icr import cfg as base_cfg
import os


old_exp_name = "_icr"
new_exp_name = "_sampling_ablation"

SAMPLE_COUNT = 1000

new_exp_name = new_exp_name + "_%d" % SAMPLE_COUNT

cfg = base_cfg.clone()

cfg.EXP_NAME = base_cfg.EXP_NAME.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)

cfg.BC.DO.SAMPLE_COUNT = SAMPLE_COUNT
cfg.BC.GS.SAMPLE_COUNT = SAMPLE_COUNT
cfg.BC.CS.SAMPLE_COUNT = SAMPLE_COUNT
