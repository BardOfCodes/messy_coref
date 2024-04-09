
# from configs.cluster import cfg
# from configs.ablations.upgrade.pretrain_baseline import cfg as base_cfg
# from configs.ablations.upgrade.tf_ablation import cfg as base_cfg
# from configs.ablations.upgrade.tf_nt_csg import cfg as base_cfg
# from configs.ablations.emerald.search_opt import cfg as base_cfg
from configs.ablations.finals.icr import cfg as base_cfg
import os


# old_exp_name = "tf_pretrain_baseline"
# old_exp_name = "CSG_pretrain_baseline_till_l10"
old_exp_name = "_icr"
new_exp_name = "_plad_ttr_eval"
cfg = base_cfg.clone()

cfg.EXP_NAME = base_cfg.EXP_NAME.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)
# HER_PPO.PPO.ENT_COEF = 1e-1
cfg.POLICY.PPO.N_ENVS = 1
cfg.EVAL.BEAM_N_PROC = 6
cfg.EVAL.BEAM_BATCH_SIZE = 48

cfg.EVAL.ITERATIVE_DO_CS = True
cfg.EVAL.N_ITERATIVE = 3
# cfg.BC.BS.BATCH_SIZE = 1
# if BEST_PLUS_MODE:
cfg.BC.PLAD.BPDS.TRAINING_DATA_SELECTION = "BEST"
cfg.BC.PLAD.BPDS.BEST_PROG_COUNT = 1
cfg.BC.CS.MERGE_SPLICE.ENABLE = False

# CS Setting
# cfg.BC.CS.CACHE_CONFIG.SEARCH_NPROBE = 7
# cfg.BC.CS.CACHE_CONFIG.MERGE_NPROBE = 7
# cfg.BC.CS.CACHE_CONFIG.SEARCH_NLIST = 25
# cfg.BC.CS.CACHE_CONFIG.MERGE_NLIST = 30
# cfg.BC.CS.N_PROC = 1
# cfg.BC.CS.EXHAUSTIVE = True
# cfg.BC.CS.REWRITE_LIMIT = 10
cfg.BC.CS.CACHE_CONFIG.CACHE_SIZE = 40000
# cfg.BC.CS.TOP_K = 15
# cfg.BC.CS.CACHE_CONFIG.SEARCH_NPROBE = 10
# cfg.BC.CS.CACHE_CONFIG.SEARCH_NLIST = 150
# cfg.BC.CS.DUMMY_NODE = True
# cfg.BC.CS.NODE_MASKING_REQ = 0.95
# cfg.BC.CS.RUN_GS = False
# cfg.BC.CS.USE_CANONICAL = True
# cfg.BC.CS.USE_PROBS = True
# cfg.BC.CS.LOGPROB_THRESHOLD = 1.0
# cfg.BC.CS.REWARD_BASED_THRESH = False

cfg.EVAL.EXHAUSTIVE = True
# cfg.TRAIN.EVAL_EPISODES = 1000 # How many episodes in EVAL


cfg.BC.PLAD.LENGTH_ALPHA = -0.001


cfg.MACHINE_SPEC.PREDICTION_PATH = os.path.join(cfg.MACHINE_SPEC.SAVE_DIR, cfg.EXP_NAME)

cfg.TRAIN.RESUME_CHECKPOINT = False

## FCSG2D:
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/base_main/fcsg2d_plad/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/base_main/fcsg2d_plad_rewrite/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/base_main/fcsg2d_srt/best_model.ptpkl"
# cfg.BC.CS.CACHE_CONFIG.SUBEXPR_LOAD_PATH = "../weights/final_weights/FCSG2D_plad_third_mini_eval_env/all_subexpr.pkl"


## MCSG2D:
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/logs/iccv/mcsg2d_plad/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="../weights/final_weights/MCSG2D_icr_re_re_re_WS_DO_GS_CS_MS_le_rewrites_CS_NLL_best_plus_data_eval_env/best_model.ptpkl"
# cfg.BC.CS.CACHE_CONFIG.SUBEXPR_LOAD_PATH = "../weights/final_weights/MCSG2D_icr_re_re_re_WS_DO_GS_CS_MS_le_rewrites_CS_NLL_best_plus_data_eval_env/all_subexpr.pkl"

## PCSG3D:
cfg.EVAL.ENV.MODE = "TEST"
cfg.BC.ENV.MODE = "TEST"
# cfg.EVAL.ENV.MODE = "TRAIN"
# cfg.BC.ENV.MODE = "TRAIN" 
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/base_main/pcsg3d_plad/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/base_main/pcsg3d_plad_rewrite/best_model.ptpkl"
cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/base_main/pcsg3d_srt_2/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/new_exp/pcsg3d_srt_ttr_2/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/new_exp/pcsg_plad_ttr/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/rebuttal/length/siri/nil/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/rebuttal/length/siri/1/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/rebuttal/length/siri/2/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/rebuttal/length/siri/main/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/rebuttal/length/siri/3/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/rebuttal/length/siri/4/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/rebuttal/rate/lower25/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/rebuttal/rate/highr50/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/rebuttal/rate/highr75/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/rebuttal/rate/max/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/new_exp/pcsg_icr_csgstump/best_model.ptpkl"
# cfg.BC.CS.CACHE_CONFIG.SUBEXPR_LOAD_PATH  ="/home/aditya/projects/rl/weights/stage_47/PCSG3D_lengthttr_eval_WS_DO_GS_CS_MS_le_rewrites_best_plus_data_best_count_3_LA_-0.015000_NEW_data/all_subexpr.pkl"
# cfg.BC.CS.CACHE_CONFIG.SUBEXPR_LOAD_PATH  ="/home/aditya/projects/rl/weights/new_exp/pcsg3d_srt_ttr/all_subexpr.pkl"
# cfg.BC.CS.CACHE_CONFIG.SUBEXPR_LOAD_PATH  ="/home/aditya/projects/rl/weights/stage_47/PCSG3D_length_plad_ttr_eval_WS_DO_GS_CS_MS_le_rewrites_best_plus_data_best_count_3_LA_-0.015000_NEW_data/all_subexpr.pkl"
cfg.BC.CS.CACHE_CONFIG.SUBEXPR_LOAD_PATH  ="/home/aditya/projects/rl/weights/stage_47/PCSG3D_length_eval_train_WS_DO_GS_CS_MS_le_rewrites_best_plus_data_best_count_3_LA_-0.015000/all_subexpr.pkl"

## PCSG SIRI Ablation:

# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/framework_ablation/pcsg_mode_3/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/framework_ablation/pcsg_mode_4/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/framework_ablation/pcsg_mode_5/best_model.ptpkl"

cfg.EVAL.PROGRAM_LISTS = [
                        "/home/aditya/projects/rl/weights/rebuttal/rate/lower25/beam_10.pkl", 
                        "/home/aditya/projects/rl/weights/rebuttal/rate/highr50/beam_10.pkl", 
                        "/home/aditya/projects/rl/weights/rebuttal/rate/highr75/beam_10.pkl", 
                          ]

cfg.EVAL.LENGTH_ALPHAS = [-0.015,-0.015, -0.015]
# Rewriter ablation:
# cfg.EVAL.ENV.MODE = "EVAL"
# cfg.BC.ENV.MODE = "EVAL"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/rewriter_ablation/pcsg3d_srt/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/rewriter_ablation/pcsg3d_no_do/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/rewriter_ablation/pcsg3d_no_gs/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/rewriter_ablation/pcsg3d_no_cs/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/rewriter_ablation/pcsg3d_plad/best_model.ptpkl"
# cfg.EVAL.PROGRAM_LISTS = [
#                         "/home/aditya/projects/rl/weights/iccv/rewriter_ablation/pcsg3d_srt/beam_10.pkl",
#                         "/home/aditya/projects/rl/weights/iccv/rewriter_ablation/pcsg3d_no_do/beam_10.pkl",
#                         "/home/aditya/projects/rl/weights/iccv/rewriter_ablation/pcsg3d_no_gs/beam_10.pkl",
#                         "/home/aditya/projects/rl/weights/iccv/rewriter_ablation/pcsg3d_no_cs/beam_10.pkl",
#                         "/home/aditya/projects/rl/weights/iccv/rewriter_ablation/pcsg3d_plad/beam_10.pkl"
#                           ]

# data ablation:
# cfg.EVAL.ENV.MODE = "EVAL"
# cfg.BC.ENV.MODE = "EVAL"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/data_scarcity/pcsg3d_plad_0.1/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/data_scarcity/pcsg3d_srt_0.1/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/data_scarcity/pcsg3d_plad_0.5/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/data_ablation/pcsg3d_srt_50/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/data_ablation/pcsg3d_plad_100/best_model.ptpkl" # Use from the above table
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/data_ablation/pcsg3d_srt_100/best_model.ptpkl" # Use from the above table

# cfg.EVAL.PROGRAM_LISTS = [
                        # "/home/aditya/projects/rl/weights/iccv/data_scarcity/pcsg3d_plad_0.5/beam_10.pkl",
                        # "/home/aditya/projects/rl/weights/iccv/data_scarcity/pcsg3d_srt_0.1/beam_10.pkl",
                        # "/home/aditya/projects/rl/weights/iccv/data_scarcity/pcsg3d_plad_0.1/beam_10.pkl", 
#                         # "/home/aditya/projects/rl/weights/iccv/data_ablation/pcsg3d_srt_50/beam_10.pkl", 
                        #   ]



# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/oscar/eval_stage_41/PCSG3D_length_icr_WS_DO_GS_CS_MS_le_rewrites_best_plus_data_LA_-0.025000_eval_env/best_model.ptpkl"
# cfg.BC.CS.CACHE_CONFIG.SUBEXPR_LOAD_PATH = "/home/aditya/projects/rl/weights/stage_35/PCSG3D_eval_WS_DO_GS_CS_MS_le_rewrites_CS_NLL_best_plus_data/all_subexpr.pkl"

# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/csgstump/chair/pcsg3d_srt_chair/best_model.ptpkl"
# cfg.BC.CS.CACHE_CONFIG.SUBEXPR_LOAD_PATH = "/home/aditya/projects/rl/weights/iccv/csgstump/chair/pcsg3d_srt_chair/all_subexpr.pkl"

## CSGSTUMP
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/csgstump/fcsg3d_srt_2/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/csgstump/chair/pcsg3d_plad/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/csgstump/chair/pcsg3d_srt_latest/best_model.ptpkl"
# cfg.BC.CS.CACHE_CONFIG.SUBEXPR_LOAD_PATH = "/home/aditya/projects/rl/weights/iccv/csgstump/fcsg3d_srt_2/all_subexpr.pkl"
# cfg.EVAL.PROGRAM_LISTS = [
                        # "/home/aditya/projects/rl/weights/iccv/csgstump/fcsg3d_srt_2/beam_do_gs_cs_3.pkl",
                        # "/home/aditya/projects/rl/weights/iccv/csgstump/chair/pcsg3d_srt_latest/beam_10.pkl",
                        # "/home/aditya/projects/rl/weights/iccv/rewriter_ablation/pcsg3d_no_gs/beam_10.pkl",
                        # "/home/aditya/projects/rl/weights/iccv/rewriter_ablation/pcsg3d_no_cs/beam_10.pkl",
                        # "/home/aditya/projects/rl/weights/iccv/rewriter_ablation/pcsg3d_plad/beam_10.pkl"
                        #   ]

## MCSG3D:
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/higher_lang/mcsg3d_plad/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/higher_lang/mcsg3d_plad_rewrite/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/higher_lang/mcsg3d_srt_2/best_model.ptpkl"
# cfg.BC.CS.CACHE_CONFIG.SUBEXPR_LOAD_PATH = "/home/aditya/projects/rl/weights/iccv/higher_lang/mcsg3d_srt_2/all_subexpr.pkl"
# cfg.BC.CS.CACHE_CONFIG.SUBEXPR_LOAD_PATH = "/home/aditya/projects/rl/weights/stage_45/MCSG3D_length_icr_plad_eval_mode_2_WS_DO_GS_CS_MS_mode=BS+BEST_best_count_3_LA_-0.015000/all_subexpr.pkl"

# cfg.EVAL.PROGRAM_LISTS = [
                        # "/home/aditya/projects/rl/weights/iccv/higher_lang/mcsg3d_plad/beam_10.pkl",
                        # "/home/aditya/projects/rl/weights/iccv/higher_lang/mcsg3d_srt_2/beam_10.pkl",
                        # "/home/aditya/projects/rl/weights/iccv/higher_lang/mcsg3d_plad_rewrite/beam_10.pkl",
                        # "/home/aditya/projects/rl/weights/iccv/higher_lang/mcsg3d_srt_2/beam_do_gs_cs_no_tax.pkl",
                        # "/home/aditya/projects/rl/weights/iccv/higher_lang/mcsg3d_plad/beam_do_gs_cs_3_no_tax.pkl",
                        # "/home/aditya/projects/rl/weights/iccv/higher_lang/mcsg3d_srt_2/beam_do_gs_cs_3_no_tax.pkl",
                        #   ]


# CSGSTUMP:

# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/csgstump_converted/best_model.ptpkl"
# cfg.BC.CS.CACHE_CONFIG.SUBEXPR_LOAD_PATH  ="/home/aditya/projects/rl/weights/stage_45/MCSG3D_length_eval_WS_DO_GS_CS_MS_le_rewrites_best_plus_data_best_count_3_csgstump_eval_LA_-0.015000/all_subexpr.pkl"
# cfg.EVAL.PROGRAM_LISTS = [
                        # "/home/aditya/projects/rl/weights/iccv/csgstump_converted/csgstump_converted.pkl",
                        # "/home/aditya/projects/rl/weights/iccv/csgstump_converted/beam_do_gs_cs_3.pkl",
                        #   ]

## PSA3D:
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/base_main/psa3d_plad_longer/best_model.ptpkl"
# # cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/base_main/psa3d_plad_rewrite/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/base_main/psa3d_srt_longer/best_model.ptpkl"
# cfg.EVAL.PROGRAM_LISTS = [
#                         "/home/aditya/projects/rl/weights/iccv/base_main/psa3d_plad_longer/beam_do_gs_no_tax.pkl",
                        # "/home/aditya/projects/rl/weights/iccv/base_main/psa3d_plad_longer/beam_10.pkl",
                        # "/home/aditya/projects/rl/weights/iccv/base_main/psa3d_plad_rewrite/beam_10.pkl",
#                         "/home/aditya/projects/rl/weights/iccv/base_main/psa3d_srt_longer/beam_do_gs_cs_new.pkl",
                        #   ]

## HSA3D:
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/higher_lang/hsa_plad/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/higher_lang/hsa_plad_rewrite/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS ="/home/aditya/projects/rl/weights/iccv/higher_lang/hsa_srt_fin_again/best_model.ptpkl"
# cfg.EVAL.PROGRAM_LISTS = [
                        # "/home/aditya/projects/rl/weights/iccv/higher_lang/hsa_plad/beam_10_no_tax.pkl",
                        # "/home/aditya/projects/rl/weights/iccv/higher_lang/hsa_srt_fin/beam_10_no_tax.pkl",
                        # "/home/aditya/projects/rl/weights/iccv/higher_lang/hsa_srt_fin_again/beam_do_gs_cs_3.pkl",
                        #   ]
# cfg.MODEL.LOAD_WEIGHTS ="../weights/final_weights/HSA3D_icr_new_WS_DO_GS_CS_MS_le_rewrites_CS_NLL_best_plus_data_eval_env/best_model.ptpkl"
# cfg.BC.CS.CACHE_CONFIG.SUBEXPR_LOAD_PATH = "/home/aditya/projects/rl/weights/stage_45/HSA3D_length_eval_WS_DO_GS_CS_MS_le_rewrites_best_plus_data_best_count_3_csgstump_eval_LA_-0.015000/all_subexpr.pkl"
# cfg.BC.CS.CACHE_CONFIG.SUBEXPR_LOAD_PATH = "/home/aditya/projects/rl/weights/iccv/higher_lang/hsa_srt_fin_again/all_subexpr.pkl"

## Ablations:
# cfg.MODEL.LOAD_WEIGHTS = "../weights/final_weights/ablations/MCSG3D_icr_ablation_WS_DO_GS_MS_le_rewrites_CS_NLL_best_plus_data_eval_env/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS = "../weights/final_weights/ablations/MCSG3D_icr_ablation_no_gs_re_WS_DO_CS_MS_le_rewrites_CS_NLL_best_plus_data_eval_env/best_model.ptpkl"
# cfg.MODEL.LOAD_WEIGHTS = "../weights/final_weights/ablations/MCSG3D_icr_ablation_WS_GS_CS_MS_le_rewrites_CS_NLL_best_plus_data_eval_env/best_model.ptpkl"

# cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_28/PCSG3D_icr_low_ent_add_dummy_masking_req_0.90_high_lr_eval_env/best_model.ptpkl"
# cfg.BC.CS.CACHE_CONFIG.SUBEXPR_LOAD_PATH = "../weights/stage_28/PCSG3D_eval_ablation_new_DO_GS_CS_WS_csrewrites_10_cssamples_2500_le_all_progs_1.000000_prob_thres_traindata_bs_plus_best/prev_subexpr.pkl"

# cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_31/PCSG3D_icr_more_DO_GS_CS_WS_traindata_bs_plus_best_eval_env/best_model.ptpkl"
# cfg.BC.CS.CACHE_CONFIG.SUBEXPR_LOAD_PATH = "../weights/oscar/stage_31/PCSG3D_icr_more_DO_GS_CS_WS_traindata_bs_plus_best/all_subexpr.pkl"

# cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/eval_stage_33/mcsg_icr/best_model.ptpkl"
# cfg.BC.CS.CACHE_CONFIG.SUBEXPR_LOAD_PATH = "../weights/oscar/eval_stage_33/mcsg_icr/all_subexpr.pkl"

