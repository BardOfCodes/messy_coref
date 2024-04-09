
# from configs.cluster import cfg
import os

# from configs.ablations.upgrade.tf_ablation import cfg as base_cfg
from configs.subconfig.envs.csg3d_train_shapenet_rnn import ENV as train_shapenet
from configs.subconfig.envs.csg3d_train_random_rnn import ENV as train_random
from configs.subconfig.envs.csg3d_eval_shapenet_rnn import ENV as eval_shapenet
# from configs.subconfig.base_policy.me_ppo import POLICY
from configs.subconfig.behavior_cloning.plad import PLAD_BC
from configs.ablations.emerald.pretrain_baseline import cfg as base_cfg, DRAW_DIRECT, LANGUAGE_ID, DEBUG
from configs.ablations.upgrade.helpers import set_program_lens, set_action_specs, set_half_resolution, set_lang_mode


assert LANGUAGE_ID > 0, "Config not valid for PCSG"

ABLATION_MODE = 3

old_exp_name = "_pretrain_baseline"
new_exp_name = "_icr_higher_ablation"

if ABLATION_MODE == 0:
    DIFF_OPT, GS, CS = (False, False, False)
    WS = False
elif ABLATION_MODE == 1:
    DIFF_OPT, GS, CS = (False, False, False)
    WS = True
elif ABLATION_MODE == 2:
    DIFF_OPT, GS, CS = (True, True, False)
    WS = False
elif ABLATION_MODE == 3:
    DIFF_OPT, GS, CS = (True, True, True)
    WS = False
elif ABLATION_MODE == 4:
    DIFF_OPT, GS, CS = (True, True, True)
    WS = True

### Ablation options:
# Hyp based
SEARCH_PATIENCE = 5
EXHAUSTIVE = False
# Data based
OPT_RELOAD = False
BEST_PLUS_MODE = True
LE_ALL_REWRITES = False
LE_ALL_PROGS = False
CS_USE_PROBS = False


# Reward based
CS_LOGPROB_THRESHOLD = 1.0
CS_REWARD_BASED_THRESH = False
# Previous Settings:
DUMMY_NODE = True
NODE_MASK_REQ = 0.90
CS_RUN_GS = False
LOW_ENT = True
HIGH_LR = True

LE_ADD_NOISE = False

LOAD_BEST_TRAINING_WEIGHTS = False
LOSS_BASED_POOR_EPOCH_RESET = False
LOAD_BEST_BEFORE_SEARCH = True
CS_N_REWRITES = 10
CS_SAMPLES = 1000
N_BEST_PROG = 1
REWRITE_BS_ONLY = False

if DIFF_OPT:
    new_exp_name = new_exp_name + "_DO"
if GS:
    new_exp_name = new_exp_name + "_GS"
if CS:
    new_exp_name = new_exp_name + "_CS"
if WS:
    new_exp_name = new_exp_name + "_WS"


if LE_ALL_REWRITES:
    if LE_ALL_PROGS:
        new_exp_name = new_exp_name + "_le_all_progs"
    else:
        new_exp_name = new_exp_name + "_le_all_rewrites"
if REWRITE_BS_ONLY:
    new_exp_name += "_rewrite_bs_only"

if CS_USE_PROBS:
    if CS_REWARD_BASED_THRESH:
        new_exp_name = new_exp_name + "_reward_based_prob_thres"
    else:
        new_exp_name = new_exp_name + "_%f_prob_thres" % CS_LOGPROB_THRESHOLD

if EXHAUSTIVE:
    new_exp_name = new_exp_name + "_exhaustive"
if BEST_PLUS_MODE:
    new_exp_name = new_exp_name + "_bs_plus_data"

if N_BEST_PROG > 1:
    new_exp_name = new_exp_name + "_%s_best_progs" % N_BEST_PROG

cfg = base_cfg.clone()
cfg.TRAIN.RESUME_CHECKPOINT = True

cfg.EXP_NAME = cfg.EXP_NAME.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(
    old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(
    old_exp_name, new_exp_name)

old_lang_mode = cfg.TRAIN.ENV.CSG_CONF.LANG_TYPE
old_bool_count = cfg.BC.ENV.CSG_CONF.BOOLEAN_COUNT
cfg.TRAIN.ENV = train_shapenet.clone()
cfg.TRAIN.EVAL_EPISODES = 1000  # How many episodes in EVAL


# BC:
cfg.BC.TYPE = "PLADBC"
cfg.BC.PLAD = PLAD_BC.PLAD.clone()
cfg.BC.PLAD.SEARCH_PATIENCE = SEARCH_PATIENCE
cfg.BC.PLAD.BEST_PROG_COUNT = N_BEST_PROG


if LE_ALL_REWRITES:
    if LE_ALL_PROGS:
        cfg.BC.PLAD.LE_ONLY_ORIGINS = ["BS", "WS", "DO", "GS", "CS"]
    else:
        cfg.BC.PLAD.LE_ONLY_ORIGINS = ["WS", "DO", "GS", "CS"]

if N_BEST_PROG > 1:
    cfg.BC.BS.RETURN_MULTIPLE = True

cfg.BC.ENV = train_shapenet.clone()
cfg.BC.ENV.TYPE = "CSG3DShapeNetBC"
cfg.BC.ENV.DYNAMIC_MAX_LEN = True

cfg.BC.BS = PLAD_BC.BS.clone()
cfg.BC.DO = PLAD_BC.DO.clone()
cfg.BC.GS = PLAD_BC.GS.clone()
cfg.BC.CS = PLAD_BC.CS.clone()
cfg.BC.WS = PLAD_BC.WS.clone()

cfg.BC.N_ITERS = int(500)

if HIGH_LR:
    cfg.TRAIN.LR_INITIAL = 0.002
else:
    cfg.TRAIN.LR_INITIAL = 0.00005

if LOW_ENT:
    cfg.BC.ENT_WEIGHT = 0.0001
    cfg.BC.L2_WEIGHT = 0.000005
else:
    cfg.BC.ENT_WEIGHT = 0.1
    cfg.BC.L2_WEIGHT = 0.00005

cfg.TRAIN.BEAM_SEARCH = True

cfg = set_action_specs(cfg, cfg.MODEL.CONFIG.OUTPUT_SEQ_LENGTH, old_bool_count)
cfg = set_lang_mode(cfg, old_lang_mode, retain_action_type=True)
cfg = set_half_resolution(cfg)

cfg.BC.PLAD.LE_ADD_NOISE = LE_ADD_NOISE
if DRAW_DIRECT:
    cfg.TRAIN.ENV.CSG_CONF.DRAW_MODE = "direct"
    cfg.EVAL.ENV.CSG_CONF.DRAW_MODE = "direct"
    cfg.BC.ENV.CSG_CONF.DRAW_MODE = "direct"

cfg.BC.PLAD.LOAD_BEST_BEFORE_SEARCH = LOAD_BEST_BEFORE_SEARCH
cfg.BC.PLAD.LOAD_BEST_TRAINING_WEIGHTS = LOAD_BEST_TRAINING_WEIGHTS
cfg.BC.PLAD.LOSS_BASED_POOR_EPOCH_RESET = LOSS_BASED_POOR_EPOCH_RESET
if BEST_PLUS_MODE:
    cfg.BC.PLAD.TRAINING_DATA_SELECTION = "BS+BEST"
cfg.BC.PLAD.OPTIMIZER_RELOAD = OPT_RELOAD

cfg.BC.GS.ENABLE = GS
cfg.BC.GS.EXHAUSTIVE = True
cfg.BC.GS.SAMPLE_COUNT = 9998

cfg.BC.DO.ENABLE = DIFF_OPT
cfg.BC.DO.N_PROC = 6
cfg.BC.DO.SAMPLE_COUNT = 2000
cfg.BC.DO.FREQUENCY = 1
cfg.BC.DO.N_STEPS = 250
cfg.BC.DO.LR = 0.01
cfg.BC.DO.EXHAUSTIVE = EXHAUSTIVE

cfg.BC.CS.ENABLE = CS
cfg.BC.CS.N_PROC = 1
cfg.BC.CS.SAMPLE_COUNT = CS_SAMPLES
cfg.BC.CS.REWRITE_LIMIT = CS_N_REWRITES
cfg.BC.CS.CACHE_CONFIG.CACHE_SIZE = 25000
cfg.BC.CS.TOP_K = 15
cfg.BC.CS.CACHE_CONFIG.SEARCH_NPROBE = 7
cfg.BC.CS.CACHE_CONFIG.SEARCH_NLIST = 200
cfg.BC.CS.CACHE_CONFIG.MERGE_NLIST = 200
cfg.BC.CS.EXHAUSTIVE = EXHAUSTIVE
cfg.BC.CS.DUMMY_NODE = DUMMY_NODE
cfg.BC.CS.NODE_MASKING_REQ = NODE_MASK_REQ
cfg.BC.CS.RUN_GS = CS_RUN_GS
cfg.BC.CS.RETURN_TOP_K = N_BEST_PROG
cfg.BC.CS.USE_CANONICAL = True
cfg.BC.CS.USE_PROBS = CS_USE_PROBS
cfg.BC.CS.LOGPROB_THRESHOLD = CS_LOGPROB_THRESHOLD
cfg.BC.CS.REWARD_BASED_THRESH = CS_REWARD_BASED_THRESH

cfg.BC.WS.ENABLE = WS
cfg.BC.WS.MODEL = cfg.MODEL.clone()
cfg.BC.WS.MODEL.CONFIG.TYPE = "PLADTransVAE"
cfg.BC.WS.MODEL.CONFIG.LATENT_DIM = 128
cfg.BC.WS.N_EPOCHS = 10

# Input gating:
cfg.BC.DO.INPUT_GATING = REWRITE_BS_ONLY
cfg.BC.GS.INPUT_GATING = REWRITE_BS_ONLY
cfg.BC.CS.INPUT_GATING = REWRITE_BS_ONLY
cfg.BC.WS.INPUT_GATING = REWRITE_BS_ONLY


# Weights
if LANGUAGE_ID == 1:
    if cfg.ACTION_RESOLUTION == 33:
        cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_27/FCSG3D_pretrain_baseline_small/weights_26.pt"
    elif cfg.ACTION_RESOLUTION == 65:
        cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_27/FCSG3D_pretrain_baseline/weights_36.pt"
elif LANGUAGE_ID == 2:
    if cfg.ACTION_RESOLUTION == 33:
        cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_27/HCSG3D_pretrain_baseline_small/weights_56.pt"
    elif cfg.ACTION_RESOLUTION == 65:
        # cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_27/HCSG3D_pretrain_baseline/weights_41.pt"
        cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_27/HCSG3D_pretrain_baseline_old_data/weights_60.ptpkl"
elif LANGUAGE_ID == 3:
    if cfg.ACTION_RESOLUTION == 33:
        cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_27/MCSG3D_pretrain_baseline_small/weights_46.pt"
    elif cfg.ACTION_RESOLUTION == 65:
        cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_27/MCSG3D_pretrain_baseline/weights_71.pt"

## Language specific edits
if LANGUAGE_ID == 1:
    cfg.BC.BS.N_PROC = 3
    cfg.BC.BS.BATCH_SIZE = 48
    cfg.BC.DO.N_PROC = 4
    cfg.BC.CS.FCSG_MODE = True # To avoid rotation stacking
    cfg.BC.BATCH_SIZE = 350
elif LANGUAGE_ID in [2, 3]:
    cfg.BC.BS.N_PROC = 1
    cfg.BC.BS.BATCH_SIZE = 32
    cfg.BC.DO.N_PROC = 4
    cfg.BC.CS.HIGHER_LANGUAGE = True
    cfg.BC.BATCH_SIZE = 200


if DRAW_DIRECT:
    cfg.TRAIN.ENV.CSG_CONF.DRAW_MODE = "direct"
    cfg.EVAL.ENV.CSG_CONF.DRAW_MODE = "direct"
    cfg.BC.ENV.CSG_CONF.DRAW_MODE = "direct"


if DEBUG:
    cfg.BC.N_ITERS = int(50)
    cfg.BC.WS.N_EPOCHS = 2
    cfg.BC.CS.SAMPLE_COUNT = 100
    cfg.BC.DO.SAMPLE_COUNT = 100
    cfg.BC.GS.SAMPLE_COUNT = 100
    cfg.TRAIN.RESUME_CHECKPOINT = False
    cfg.BC.BATCH_SIZE = 20