
# from configs.ablations.mod_exp.sil_baseline import cfg
from configs.ablations.mod_exp.sil_new_action import cfg
from configs.cluster import cluster
import os



old_exp_name = "R30_Refactor_pre"
new_exp_name = "eval"
cfg.EXP_NAME = cfg.EXP_NAME.replace(old_exp_name, new_exp_name)

# HER_PPO.PPO.ENT_COEF = 1e-1
cfg.POLICY.PPO.N_ENVS = 1
cfg.EVAL.BEAM_N_PROC = 3
cfg.EVAL.BEAM_BATCH_SIZE = 100

cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)

cfg.MACHINE_SPEC.PREDICTION_PATH = os.path.join(cfg.MACHINE_SPEC.SAVE_DIR, "stage_14", cfg.EXP_NAME)

# NENV Reset:
# cfg.MODEL.LOAD_WEIGHTS = "/home/aditya/projects/rl/weights/stage_13/R30 Comparison to PLAD/best_model.pt"