
from configs.ablations.finals.pretrain_baseline import LANGUAGE_MODE, DEBUG, LANGUAGE_NAME
from configs.ablations.finals.plad_baseline import cfg as base_cfg


old_exp_name = "_plad"
new_exp_name = "_rl"

cfg = base_cfg.clone()

cfg.BC.PLAD.RESET_TRAIN_STATE = True

cfg.BC.BS.ENABLE = True
cfg.BC.WS.ENABLE = False
# cfg.BC.DO.ENABLE = False
# cfg.BC.GS.ENABLE = False
# cfg.BC.CS.ENABLE = False
cfg.BC.CS.MERGE_SPLICE.ENABLE = False

cfg.BC.DO.ENABLE = True
cfg.BC.GS.ENABLE = True
cfg.BC.CS.ENABLE = True
# cfg.BC.CS.MERGE_SPLICE.ENABLE = True
cfg.BC.DO.SAMPLE_COUNT = 100
cfg.BC.DO.N_PROC = 1
cfg.BC.GS.SAMPLE_COUNT = 4096
cfg.BC.CS.SAMPLE_COUNT = 100
cfg.BC.CS.CACHE_CONFIG.MERGE_NLIST = 20# 200
cfg.BC.CS.CACHE_CONFIG.SEARCH_NLIST = 20# 

cfg.EXP_NAME = cfg.EXP_NAME.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(
    old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(
    old_exp_name, new_exp_name)

cfg.BC.PLAD.INIT_LER = 0.0
cfg.BC.PLAD.FINAL_LER = 0.0
cfg.BC.BS.STOCHASTIC_BS = True
cfg.BC.BS.BEAM_SIZE = 1
cfg.BC.PLAD.RL_MODE = True
cfg.BC.PLAD.RL_REWARD_POW = 2
cfg.BC.PLAD.RL_MOVING_BASELINE_ALPHA = 0.8
cfg.BC.PLAD.REWARD_WEIGHTING = True
cfg.BC.PLAD.LOAD_BEST_BEFORE_SEARCH = False
cfg.BC.PLAD.EVAL_EPOCHS = 25
cfg.BC.SAVE_EPOCHS = 60
# cfg.BC.N_ITERS = 20
cfg.BC.N_ITERS = 40
cfg.BC.LOG_INTERVAL = 20
cfg.TRAIN.LR_INITIAL = 0.0005
cfg.BC.ENT_WEIGHT = 0.05
cfg.BC.L2_WEIGHT = 0.00005