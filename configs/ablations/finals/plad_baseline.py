
# from configs.cluster import cfg
import os

# from configs.ablations.upgrade.tf_ablation import cfg as base_cfg
from configs.subconfig.envs.csg3d_train_shapenet_rnn import ENV as csg3d_train_shapenet
from configs.subconfig.envs.csg3d_eval_shapenet_rnn import ENV as csg3d_eval_shapenet
from configs.subconfig.envs.shape_assembly_env import SHAPENET_TRAIN_ENV as sa_train_shapenet, SHAPENET_EVAL_ENV as sa_eval_shapenet
from configs.subconfig.envs.csg2d_env import SHAPENET_TRAIN_ENV as csg2d_train_shapenet, SHAPENET_EVAL_ENV as csg2d_eval_shapenet
from configs.subconfig.envs.ucsgnet_train import UCSG_ENV as ucsgnet_train_shapenet, UCSG_ENV_EVAL as ucsgnet_eval_shapenet
from configs.subconfig.envs.csgstump import ENV as csgstump_train_shapenet, EVAL_ENV as csgstump_eval_shapenet
# from configs.subconfig.base_policy.me_ppo import POLICY
from configs.subconfig.behavior_cloning.plad import PLAD_BC
from configs.ablations.finals.pretrain_baseline import cfg as base_cfg, LANGUAGE_MODE, DEBUG, LANGUAGE_NAME
from configs.ablations.upgrade.helpers import set_program_lens, set_action_specs, set_half_resolution, set_lang_mode, set_full_resolution

old_exp_name = "_tax"
new_exp_name = "_plad"

WS_ENABLE = True
LENGTH_ALPHA = -0.015
LENGTH_CURRICULUM = False
INIT_ALPHA = -0.020
FINAL_ALPHA = -0.000

# DATA SET
USE_UCSG = False
USE_CSGSTUMP = False
USE_NEW_DATA = False
# USE_NEW_DATA = False

RESTRICT_DATASIZE = False
DATASIZE = 0.1

NOISY_AUG = False


if not LENGTH_ALPHA == 0:
    new_exp_name += "_LA_%f" % LENGTH_ALPHA

if not WS_ENABLE: 
    new_exp_name += "_no_WS"
if USE_UCSG:
    new_exp_name += "_UCSG_data"
    
if USE_CSGSTUMP:
    new_exp_name += "_CSGSTMP_data"
if USE_NEW_DATA:
    new_exp_name += "_NEW_data"
if NOISY_AUG:
    new_exp_name += "_noisy_aug_"
    
cfg = base_cfg.clone()

cfg.TRAIN.RESUME_CHECKPOINT = False

cfg.EXP_NAME = cfg.EXP_NAME.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(
    old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(
    old_exp_name, new_exp_name)

if "CSG3D" in LANGUAGE_NAME:
    old_lang_mode = cfg.TRAIN.ENV.CSG_CONF.LANG_TYPE
    old_bool_count = cfg.BC.ENV.CSG_CONF.BOOLEAN_COUNT
    if USE_UCSG:
        cfg.TRAIN.ENV = ucsgnet_train_shapenet.clone()
        cfg.BC.ENV = ucsgnet_train_shapenet.clone()
        cfg.EVAL.ENV = ucsgnet_eval_shapenet.clone()
    elif USE_CSGSTUMP:
        cfg.TRAIN.ENV = csgstump_train_shapenet.clone()
        cfg.BC.ENV = csgstump_train_shapenet.clone()
        cfg.EVAL.ENV = csgstump_eval_shapenet.clone()
    else:
        cfg.TRAIN.ENV = csg3d_train_shapenet.clone()
        cfg.BC.ENV = csg3d_train_shapenet.clone()
        cfg.EVAL.ENV = csg3d_eval_shapenet.clone()
        if USE_NEW_DATA:
            cfg.TRAIN.ENV.CSG_CONF.DATAMODE = "NEW"
            cfg.BC.ENV.CSG_CONF.DATAMODE = "NEW"
            cfg.EVAL.ENV.CSG_CONF.DATAMODE = "NEW"
            
elif "CSG2D" in LANGUAGE_NAME:
    old_lang_mode = cfg.TRAIN.ENV.CSG_CONF.LANG_TYPE
    old_bool_count = cfg.BC.ENV.CSG_CONF.BOOLEAN_COUNT
    cfg.TRAIN.ENV = csg2d_train_shapenet.clone()
    cfg.BC.ENV = csg2d_train_shapenet.clone()
else:
    old_lang_mode = cfg.TRAIN.ENV.SA_CONF.LANGUAGE_NAME
    old_bool_count = 100
    cfg.TRAIN.ENV = sa_train_shapenet.clone()
    cfg.BC.ENV = sa_train_shapenet.clone()

# SPEED
cfg.TRAIN.EVAL_EPISODES = 1000  # How many episodes in EVAL
cfg.TRAIN.BEAM_SEARCH = False
cfg.EVAL.BEAM_SIZE = 10
cfg.TRAIN.BEAM_SIZE = 1


# BC:
cfg.BC.TYPE = "PLADBC"
# To avoid changing settings from pretrain_baseline.py
cfg.BC.PLAD = PLAD_BC.PLAD.clone()
cfg.BC.BS = PLAD_BC.BS.clone()
cfg.BC.WS = PLAD_BC.WS.clone()
cfg.BC.DO = PLAD_BC.DO.clone()
cfg.BC.GS = PLAD_BC.GS.clone()
cfg.BC.CS = PLAD_BC.CS.clone()
cfg.BC.NR = PLAD_BC.NR.clone()



cfg.BC.BS.LANGUAGE_NAME = LANGUAGE_NAME
cfg.BC.WS.LANGUAGE_NAME = LANGUAGE_NAME
cfg.BC.NR.LANGUAGE_NAME = LANGUAGE_NAME
cfg.BC.DO.LANGUAGE_NAME = LANGUAGE_NAME
cfg.BC.GS.LANGUAGE_NAME = LANGUAGE_NAME
cfg.BC.CS.LANGUAGE_NAME = LANGUAGE_NAME

cfg.BC.PLAD.RESET_TRAIN_STATE = True

cfg.BC.PLAD.SEARCH_PATIENCE = 5
cfg.BC.PLAD.BPDS.BEST_PROG_COUNT = 1

cfg.BC.PLAD.LENGTH_ALPHA = LENGTH_ALPHA

if LENGTH_CURRICULUM:
    cfg.BC.PLAD.LENGTH_CURRICULUM = True
    cfg.BC.PLAD.INIT_LENGTH_ALPHA = INIT_ALPHA
    cfg.BC.PLAD.FINAL_LENGTH_ALPHA = FINAL_ALPHA
    
    
cfg.BC.ENV.TYPE = cfg.BC.ENV.TYPE + "BC"
cfg.BC.ENV.DYNAMIC_MAX_LEN = True

cfg.BC.NR.ENABLE = NOISY_AUG

cfg.BC.WS.ENABLE = WS_ENABLE
cfg.BC.WS.MODEL = cfg.MODEL.clone()
cfg.BC.WS.MODEL.CONFIG.TYPE = "PLADTransVAE"
cfg.BC.WS.MODEL.CONFIG.LATENT_DIM = 128
cfg.BC.WS.N_EPOCHS = 7

cfg.BC.N_ITERS = int(500)
cfg.BC.ENT_WEIGHT = 0.005
cfg.BC.L2_WEIGHT = 0.000005


# Because envs are reloaded:
if "3D" in LANGUAGE_NAME:
    cfg = set_half_resolution(cfg)
else:
    cfg = set_full_resolution(cfg)

if "CSG" in LANGUAGE_NAME:
    if "PCSG3D" in LANGUAGE_NAME:
        max_len = 96
    elif "FCSG3D" in LANGUAGE_NAME:
        max_len = 11 + 12 * 10 + 1
    elif "PCSG2D" in LANGUAGE_NAME:
        max_len = 84
    elif "FCSG2D" in LANGUAGE_NAME:
        max_len = 128
    elif "MCSG2D" in LANGUAGE_NAME:
        max_len = 128
    else:
        max_len = 192
    cfg = set_action_specs(cfg, cfg.MODEL.CONFIG.OUTPUT_SEQ_LENGTH, old_bool_count, LANGUAGE_NAME)
    cfg = set_lang_mode(cfg, old_lang_mode, retain_action_type=True)
else:
    max_len = 192
    cfg = set_action_specs(cfg, max_len, 11, LANGUAGE_NAME)

    cfg.TRAIN.ENV.SA_CONF.LANGUAGE_NAME = LANGUAGE_NAME
    cfg.EVAL.ENV.SA_CONF.LANGUAGE_NAME = LANGUAGE_NAME
    cfg.BC.ENV.SA_CONF.LANGUAGE_NAME = LANGUAGE_NAME

    cfg.MODEL.CONFIG.OUTPUT_SEQ_LENGTH = max_len
    cfg.BC.ENV.SA_CONF.PERM_MAX_LEN = max_len
    cfg.TRAIN.ENV.SA_CONF.PERM_MAX_LEN = max_len
    cfg.EVAL.ENV.SA_CONF.PERM_MAX_LEN = max_len

# High LR for PLAD
cfg.BC.N_ENVS = 1
cfg.TRAIN.LR_INITIAL = 0.001

# Weights to load
if "SA"  in LANGUAGE_NAME:
    if "PSA" in LANGUAGE_NAME:
        # cfg.MODEL.LOAD_WEIGHTS = "../weights/stage_33/PSA3D_cor_re_pretrain/weights_35.ptpkl"
        cfg.MODEL.LOAD_WEIGHTS = "../weights/stage_43/PSA3D_length_tax/weights_50.ptpkl"
        cfg.BC.BS.N_PROC = 4
    elif "HSA" in LANGUAGE_NAME:
        cfg.MODEL.LOAD_WEIGHTS = "../weights/stage_33/HSA3D_cor_re_pretrain/weights_40.ptpkl"
        cfg.BC.BS.N_PROC = 5
    cfg.BC.BS.BATCH_SIZE = 48
    cfg.BC.WS.SAMPLE_N_PROC = 2
    cfg.BC.WS.SAMPLE_BATCH_SIZE = 64
    cfg.TRAIN.LR_INITIAL = 0.0005
    cfg.EVAL.BEAM_N_PROC = 4
    cfg.EVAL.BEAM_BATCH_SIZE = 48
    # Set lower beam size
    cfg.EVAL.BEAM_SIZE = 3

elif "CSG" in LANGUAGE_NAME:
    # Non general settings
    if "PCSG3D" in LANGUAGE_NAME:
        cfg.MODEL.LOAD_WEIGHTS = "../weights/stage_27/PCSG3D_pretrain_baseline_small/weights_41.pt"
        cfg.BC.BS.N_PROC = 4
        cfg.BC.BS.BATCH_SIZE = 32
        cfg.BC.WS.SAMPLE_N_PROC = 2
        cfg.BC.WS.SAMPLE_BATCH_SIZE = 64
        cfg.TRAIN.LR_INITIAL = 0.001
    elif "FCSG3D" in LANGUAGE_NAME:
        cfg.MODEL.LOAD_WEIGHTS = "../weights/stage_27/FCSG3D_pretrain_baseline_small/weights_26.pt"
        cfg.BC.BS.N_PROC = 4
        cfg.BC.BS.BATCH_SIZE = 32
        cfg.BC.WS.SAMPLE_N_PROC = 2
        cfg.BC.WS.SAMPLE_BATCH_SIZE = 64
        cfg.TRAIN.LR_INITIAL = 0.0005
    elif "MCSG3D" in LANGUAGE_NAME:
        cfg.MODEL.LOAD_WEIGHTS = "../weights/stage_27/MCSG3D_pretrain_baseline_small/weights_46.pt"
        cfg.BC.BS.N_PROC = 4
        cfg.BC.BS.BATCH_SIZE = 48
        cfg.BC.WS.SAMPLE_N_PROC = 2
        cfg.BC.WS.SAMPLE_BATCH_SIZE = 64
        cfg.TRAIN.LR_INITIAL = 0.0005
        cfg.EVAL.BEAM_N_PROC = 4
        cfg.EVAL.BEAM_BATCH_SIZE = 32
    elif "CSG2D" in LANGUAGE_NAME:
        if "MCSG" in LANGUAGE_NAME:
            cfg.MODEL.LOAD_WEIGHTS = "../weights/stage_35/MCSG2D_pretrain_baseline/weights_85.ptpkl"
        if "FCSG" in LANGUAGE_NAME:
            cfg.MODEL.LOAD_WEIGHTS = "../weights/stage_35/FCSG2D_pretrain_baseline/weights_135.ptpkl"
        # cfg.MODEL.LOAD_WEIGHTS = "../weights/stage_27/PCSG3D_pretrain_baseline_small/weights_41.pt"
        cfg.BC.BS.N_PROC = 4
        cfg.BC.BS.BATCH_SIZE = 48
        cfg.BC.WS.SAMPLE_N_PROC = 2
        cfg.BC.WS.SAMPLE_BATCH_SIZE = 64
        cfg.TRAIN.LR_INITIAL = 0.001
        cfg.EVAL.BEAM_N_PROC = 4
        cfg.EVAL.BEAM_BATCH_SIZE = 32
        
if USE_CSGSTUMP:
    cfg.BC.WS.SAMPLE_COUNT = 14198
    cfg.BC.NR.SAMPLE_COUNT = 14198
    # cfg.BC.N_ITERS = 250
    
    
if RESTRICT_DATASIZE:
    if "CSG" in LANGUAGE_NAME:
        cfg.TRAIN.ENV.CSG_CONF.RESTRICT_DATASIZE = True
        cfg.TRAIN.ENV.CSG_CONF.DATASIZE = DATASIZE
        cfg.BC.ENV.CSG_CONF.RESTRICT_DATASIZE = True
        cfg.BC.ENV.CSG_CONF.DATASIZE = DATASIZE
    elif "SA" in LANGUAGE_NAME:
        cfg.TRAIN.ENV.SA_CONF.RESTRICT_DATASIZE = True
        cfg.TRAIN.ENV.SA_CONF.DATASIZE = DATASIZE
        cfg.BC.ENV.SA_CONF.RESTRICT_DATASIZE = True
        cfg.BC.ENV.SA_CONF.DATASIZE = DATASIZE
    # Should the iterations be shorter as well? 
    cfg.BC.WS.SAMPLE_COUNT = int(cfg.BC.WS.SAMPLE_COUNT * DATASIZE)
    # cfg.BC.NR.SAMPLE_COUNT = int(cfg.BC.NR.SAMPLE_COUNT * DATASIZE)
    cfg.BC.N_ITERS = 250# int((DATASIZE)/5)
    
    

if DEBUG:
    cfg.BC.BS.BATCH_SIZE = 32
    cfg.BC.BS.N_PROC = 4
    cfg.BC.WS.SAMPLE_N_PROC = 2
    cfg.BC.WS.SAMPLE_BATCH_SIZE = 32
    cfg.TRAIN.LR_INITIAL = 0.001
    cfg.EVAL.BEAM_N_PROC = 4
    cfg.EVAL.BEAM_BATCH_SIZE = 32

    cfg.BC.BATCH_SIZE = int(100)
    cfg.BC.N_ITERS = int(100)
    cfg.BC.WS.SAMPLE_BATCH_SIZE = 64
    cfg.BC.WS.SAMPLE_COUNT = 8000
    cfg.TRAIN.RESUME_CHECKPOINT = False
    cfg.BC.WS.N_EPOCHS = 2
    # cfg.BC.BATCH_SIZE = 20


# ../weights/stage_32/MCSG3D_icr_BS_LE_WS_bs_plus_data_eval_env/best_model.ptpkl
# ../weights/stage_32/MCSG3D_icr_BS_LE_WS_DO_GS_CS_bs_plus_data_eval_env/best_model.ptpkl
# ../weights/stage_32/PCSG3D_icr_BS_LE__WS_le_all_rewrites_best_plus_data_eval_env/best_model.ptpkl
