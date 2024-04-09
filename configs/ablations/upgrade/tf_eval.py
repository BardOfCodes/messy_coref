
# from configs.cluster import cfg
# from configs.ablations.upgrade.pretrain_baseline import cfg as base_cfg
# from configs.ablations.upgrade.tf_ablation import cfg as base_cfg
# from configs.ablations.upgrade.tf_nt_csg import cfg as base_cfg
from configs.ablations.upgrade.tf_plad import cfg as base_cfg
from configs.subconfig.envs.csg3d_eval_shapenet_rnn import ENV as eval_shapenet
from configs.subconfig.envs.csg3d_train_shapenet_rnn import ENV as train_shapenet
import os


# old_exp_name = "tf_pretrain_baseline"
# old_exp_name = "CSG_pretrain_baseline_till_l10"
old_exp_name = "NT_CSG32_pretrain_plad"
new_exp_name = "eval"
cfg = base_cfg.clone()

cfg.EXP_NAME = base_cfg.EXP_NAME.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)
# HER_PPO.PPO.ENT_COEF = 1e-1
cfg.POLICY.PPO.N_ENVS = 1
cfg.EVAL.BEAM_N_PROC = 4
cfg.EVAL.BEAM_BATCH_SIZE = 48

# CS Setting
cfg.BC.CS.CACHE_CONFIG.SEARCH_NPROBE = 7
cfg.BC.CS.CACHE_CONFIG.MERGE_NPROBE = 7
cfg.BC.CS.CACHE_CONFIG.SEARCH_NLIST = 25
cfg.BC.CS.CACHE_CONFIG.MERGE_NLIST = 30

cfg.EVAL.EXHAUSTIVE = True
# cfg.TRAIN.EVAL_EPISODES = 1000 # How many episodes in EVAL


cfg.MACHINE_SPEC.PREDICTION_PATH = os.path.join(cfg.MACHINE_SPEC.SAVE_DIR, "stage_20", cfg.EXP_NAME)

cfg.TRAIN.RESUME_CHECKPOINT = False
cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/NTCSG32_NTCSG32_plad_baseline_diffOpt_DCR_eval_env/best_model.pt"
# cfg.MODEL.LOAD_WEIGHTS = "/home/aditya/projects/rl/weights/oscar/stage_20/NT_CSG_pretrain_baseline_eval_env/best_model.pt"
# cfg.MODEL.LOAD_WEIGHTS = "/home/aditya/projects/rl/weights/stage_18/tf_pretrain_baseline_fastformer/weights_13.pt"
