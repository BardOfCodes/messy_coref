
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


old_exp_name = "_pretrain_baseline"
new_exp_name = "_icr"

DIFF_OPT = True
GS = True
CS = True
WS = False

SEARCH_PATIENCE = 5

EXHAUSTIVE = False
BEST_PLUS_MODE = True

DUMMY_NODE = True
NODE_MASK_REQ = 0.90
CS_RUN_GS = False
# Ablation OPTIONS:
LOW_ENT = True
HIGH_LR = True

N_BEST_PROG = 1
OPT_RELOAD = False
LE_ADD_NOISE = False

LOAD_BEST_BEFORE_SEARCH = True
LOAD_BEST_TRAINING_WEIGHTS = False
LOSS_BASED_POOR_EPOCH_RESET = False


if DIFF_OPT:
    new_exp_name = new_exp_name + "_DO"
if GS:
    new_exp_name = new_exp_name + "_GS"
if CS:
    new_exp_name = new_exp_name + "_CS"
if WS:
    new_exp_name = new_exp_name + "_WS"

if not LOAD_BEST_BEFORE_SEARCH:
    new_exp_name = new_exp_name + "_no_bwr"
if LOSS_BASED_POOR_EPOCH_RESET:
    new_exp_name = new_exp_name + "_loss_based_reset"
if LOAD_BEST_TRAINING_WEIGHTS:
    new_exp_name = new_exp_name + "_train_best_wts"
if EXHAUSTIVE:
    new_exp_name = new_exp_name + "_exhaustive"
if BEST_PLUS_MODE:
    new_exp_name = new_exp_name + "_traindata_bs_plus_best"


if LOW_ENT:
    new_exp_name = new_exp_name + "_low_ent"
if DUMMY_NODE:
    new_exp_name = new_exp_name + "_add_dummy"
if NODE_MASK_REQ < 1.0:
    new_exp_name = new_exp_name + "_masking_req_{:0.2f}".format(NODE_MASK_REQ)
if LE_ADD_NOISE:
    new_exp_name = new_exp_name + "_noisy_le"
if N_BEST_PROG > 1:
    new_exp_name = new_exp_name + "_%s_best_progs" % N_BEST_PROG

if CS_RUN_GS:
    new_exp_name = new_exp_name + "_cs_inner_loop_dcr"


if HIGH_LR:
    new_exp_name = new_exp_name + "_high_lr"
if OPT_RELOAD:
    new_exp_name = new_exp_name + "_opt_reload"

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
PLAD_BC.PLAD.BEST_PROG_COUNT = N_BEST_PROG

if N_BEST_PROG > 1:
    PLAD_BC.BS.RETURN_MULTIPLE = True

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
    cfg.BC.ENT_WEIGHT = 0.000
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
cfg.BC.GS.EXHAUSTIVE = False
cfg.BC.GS.SAMPLE_COUNT = 5000

cfg.BC.DO.ENABLE = DIFF_OPT
cfg.BC.DO.N_PROC = 6
cfg.BC.DO.SAMPLE_COUNT = 2500
cfg.BC.DO.FREQUENCY = 1
cfg.BC.DO.N_STEPS = 250
cfg.BC.DO.LR = 0.01
cfg.BC.DO.EXHAUSTIVE = EXHAUSTIVE

cfg.BC.CS.ENABLE = CS
cfg.BC.CS.N_PROC = 1
cfg.BC.CS.SAMPLE_COUNT = 1000
cfg.BC.CS.REWRITE_LIMIT = 50
cfg.BC.CS.CACHE_CONFIG.CACHE_SIZE = 40000
cfg.BC.CS.TOP_K = 50
cfg.BC.CS.CACHE_CONFIG.SEARCH_NPROBE = 15
cfg.BC.CS.CACHE_CONFIG.SEARCH_NLIST = 250
cfg.BC.CS.EXHAUSTIVE = EXHAUSTIVE
cfg.BC.CS.DUMMY_NODE = DUMMY_NODE
cfg.BC.CS.NODE_MASKING_REQ = NODE_MASK_REQ
cfg.BC.CS.RUN_GS = CS_RUN_GS
cfg.BC.CS.RETURN_TOP_K = N_BEST_PROG
cfg.BC.CS.USE_CANONICAL = True

cfg.BC.WS.ENABLE = WS
cfg.BC.WS.MODEL = cfg.MODEL.clone()
cfg.BC.WS.MODEL.CONFIG.TYPE = "PLADTransVAE"
cfg.BC.WS.MODEL.CONFIG.LATENT_DIM = 128
cfg.BC.WS.N_EPOCHS = 10

if LANGUAGE_ID == 0:
    if cfg.ACTION_RESOLUTION == 33:
        cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_27/PCSG3D_pretrain_baseline_small/weights_41.pt"
        cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_28/PCSG3D_icr_low_ent_add_dummy_masking_req_0.90_high_lr_eval_env/best_model.ptpkl"
        # cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_28/PCSG3D_icr_DO_GS_CS_low_ent_add_dummy_masking_req_0.90_high_lr_eval_env/best_model.ptpkl"
    elif cfg.ACTION_RESOLUTION == 65:
        cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_27/PCSG3D_pretrain_baseline_no_ent/weights_26.pt"
    cfg.BC.BS.N_PROC = 4
    cfg.BC.BS.BATCH_SIZE = 64
elif LANGUAGE_ID == 1:
    if cfg.ACTION_RESOLUTION == 33:
        # cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_27/FCSG3D_pretrain_baseline_small/weights_26.pt"
        cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_28/FCSG3D_icr_DO_GS_CS_low_ent_add_dummy_masking_req_0.90_high_lr_eval_env/best_model.ptpkl"
        # cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_28/FCSG3D_icr_WS_low_ent_add_dummy_masking_req_0.90_high_lr_eval_env/best_model.ptpkl"
        
    elif cfg.ACTION_RESOLUTION == 65:
        cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_27/FCSG3D_pretrain_baseline/weights_36.pt"
    cfg.BC.BS.N_PROC = 3
    cfg.BC.BS.BATCH_SIZE = 48
    cfg.BC.DO.N_PROC = 4
elif LANGUAGE_ID == 2:
    if cfg.ACTION_RESOLUTION == 33:
        cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_27/HCSG3D_pretrain_baseline_small/weights_56.pt"
    elif cfg.ACTION_RESOLUTION == 65:
        # cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_27/HCSG3D_pretrain_baseline/weights_41.pt"
        cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_27/HCSG3D_pretrain_baseline_old_data/weights_60.ptpkl"
    cfg.BC.BS.N_PROC = 4
    cfg.BC.BS.BATCH_SIZE = 32
    cfg.BC.BS.BEAM_SIZE = 10
elif LANGUAGE_ID == 3:
    if cfg.ACTION_RESOLUTION == 33:
        cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_27/MCSG3D_pretrain_baseline_small/weights_46.pt"
    elif cfg.ACTION_RESOLUTION == 65:
        cfg.MODEL.LOAD_WEIGHTS = "../weights/oscar/stage_27/MCSG3D_pretrain_baseline/weights_71.pt"
    cfg.BC.BS.N_PROC = 4
    cfg.BC.BS.BATCH_SIZE = 32
    cfg.BC.BS.BEAM_SIZE = 10

if DEBUG:
    cfg.BC.N_ITERS = int(50)
    cfg.BC.WS.N_EPOCHS = 2
    cfg.BC.CS.SAMPLE_COUNT = 100
    cfg.NOTIFICATION.ENABLE = False

if LANGUAGE_ID == 1:
    # cfg.BC.CS.VALID_NODES = ["B", "DUMMY"] # To avoid rotation stacking
    cfg.BC.CS.FCSG_MODE = True # To avoid rotation stacking