
# from configs.cluster import cfg
# from configs.ablations.upgrade.pretrain_baseline import cfg as base_cfg
# from configs.ablations.upgrade.tf_ablation import cfg as base_cfg
# from configs.ablations.upgrade.tf_nt_csg import cfg as base_cfg
# from configs.ablations.emerald.search_opt import cfg as base_cfg
from configs.ablations.emerald.pcsg_cs_ablation import cfg as base_cfg
from configs.subconfig.envs.csg3d_eval_shapenet_rnn import ENV as eval_shapenet
from configs.subconfig.envs.csg3d_train_shapenet_rnn import ENV as train_shapenet
import os


# old_exp_name = "tf_pretrain_baseline"
# old_exp_name = "CSG_pretrain_baseline_till_l10"
old_exp_name = "_icr"
new_exp_name = "_eval"
cfg = base_cfg.clone()

cfg.EXP_NAME = base_cfg.EXP_NAME.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)
# HER_PPO.PPO.ENT_COEF = 1e-1
cfg.POLICY.PPO.N_ENVS = 1
cfg.EVAL.BEAM_N_PROC = 4
cfg.EVAL.BEAM_BATCH_SIZE = 48

cfg.EVAL.ITERATIVE_DO_CS = True
cfg.EVAL.N_ITERATIVE = 3


# CS Setting
# cfg.BC.CS.CACHE_CONFIG.SEARCH_NPROBE = 7
# cfg.BC.CS.CACHE_CONFIG.MERGE_NPROBE = 7
# cfg.BC.CS.CACHE_CONFIG.SEARCH_NLIST = 25
# cfg.BC.CS.CACHE_CONFIG.MERGE_NLIST = 30
cfg.BC.CS.N_PROC = 1
cfg.BC.CS.EXHAUSTIVE = True
cfg.BC.CS.REWRITE_LIMIT = 50
cfg.BC.CS.CACHE_CONFIG.CACHE_SIZE = 50000
cfg.BC.CS.TOP_K = 50
cfg.BC.CS.CACHE_CONFIG.SEARCH_NPROBE = 7
cfg.BC.CS.CACHE_CONFIG.SEARCH_NLIST = 150
cfg.BC.CS.DUMMY_NODE = True
cfg.BC.CS.NODE_MASKING_REQ = 1.0
cfg.BC.CS.RUN_GS = False
cfg.BC.CS.USE_CANONICAL = True
cfg.BC.CS.USE_PROBS = False
cfg.BC.CS.LOGPROB_THRESHOLD = 1.0
cfg.BC.CS.REWARD_BASED_THRESH = False

cfg.EVAL.EXHAUSTIVE = True
# cfg.TRAIN.EVAL_EPISODES = 1000 # How many episodes in EVAL


cfg.MACHINE_SPEC.PREDICTION_PATH = os.path.join(cfg.MACHINE_SPEC.SAVE_DIR, cfg.EXP_NAME)

cfg.TRAIN.RESUME_CHECKPOINT = False
# cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_28/PCSG3D_icr_low_ent_add_dummy_masking_req_0.90_high_lr_eval_env/best_model.ptpkl"
# cfg.BC.CS.CACHE_CONFIG.SUBEXPR_LOAD_PATH = "../weights/stage_28/PCSG3D_eval_ablation_new_DO_GS_CS_WS_csrewrites_10_cssamples_2500_le_all_progs_1.000000_prob_thres_traindata_bs_plus_best/prev_subexpr.pkl"
# cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_31/PCSG3D_icr_DO_GS_CS_WS_prob_thres_traindata_bs_plus_best_eval_env/best_model.ptpkl"
# cfg.BC.CS.CACHE_CONFIG.SUBEXPR_LOAD_PATH = "../weights/oscar/stage_31/PCSG3D_icr_DO_GS_CS_WS_prob_thres_traindata_bs_plus_best/oscar_subexpr.pkl"
# cfg.MODEL.LOAD_WEIGHTS = "/home/aditya/projects/rl/weights/oscar/stage_20/NT_CSG_pretrain_baseline_eval_env/best_model.pt"
# cfg.MODEL.LOAD_WEIGHTS = "/home/aditya/projects/rl/weights/stage_18/tf_pretrain_baseline_fastformer/weights_13.pt"

cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_31/PCSG3D_icr_more_DO_GS_CS_WS_traindata_bs_plus_best_eval_env/best_model.ptpkl"
cfg.BC.CS.CACHE_CONFIG.SUBEXPR_LOAD_PATH = "../weights/oscar/stage_31/PCSG3D_icr_more_DO_GS_CS_WS_traindata_bs_plus_best/all_subexpr.pkl"