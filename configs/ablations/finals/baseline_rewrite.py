
from configs.ablations.finals.pretrain_baseline import LANGUAGE_MODE, DEBUG, LANGUAGE_NAME
from configs.ablations.finals.plad_baseline import cfg as base_cfg, USE_UCSG, NOISY_AUG


old_exp_name = "_plad"
new_exp_name = "_icr_rewrite_ablation"

cfg = base_cfg.clone()

baseline_mode = 6

cfg.BC.PLAD.RESET_TRAIN_STATE = True

cfg.BC.BS.ENABLE = True
if NOISY_AUG:
    cfg.BC.WS.ENABLE = False
else:
    cfg.BC.WS.ENABLE = True
    
cfg.BC.DO.ENABLE = True
cfg.BC.GS.ENABLE = True
cfg.BC.CS.ENABLE = True
cfg.BC.CS.MERGE_SPLICE.ENABLE = True
 
# Configurations
if "PCSG" in LANGUAGE_NAME:
    CS_SAMPLES = 1500
    DO_SAMPLES = 5000
    if USE_UCSG:
        CS_SAMPLES = int(1500 * 3.6)
        DO_SAMPLES = int(5000 * 3.6)
        
    DO_N_PROC = 4
    HIGHER_LANGUAGE = False
    CS_TOP_K = 15
    CS_RETURN_TOP_K = 1
    CS_REWRITE_LIMIT = 10
elif "FCSG" in LANGUAGE_NAME:
    CS_SAMPLES = 1500
    DO_SAMPLES = 5000
    DO_N_PROC = 4
    HIGHER_LANGUAGE = False
    CS_TOP_K = 15
    CS_RETURN_TOP_K = 1
    CS_REWRITE_LIMIT = 10
elif "MCSG2D" in LANGUAGE_NAME:
    CS_SAMPLES = 1500
    DO_SAMPLES = 5000
    DO_N_PROC = 4
    HIGHER_LANGUAGE = True
    CS_TOP_K = 15
    CS_RETURN_TOP_K = 1
    CS_REWRITE_LIMIT = 10

else:
    # For MCSG, PSA, HSA
    if "MCSG" in LANGUAGE_NAME:
        CS_SAMPLES = 1500
        DO_SAMPLES = 5000
        CS_TOP_K = 15
        CS_REWRITE_LIMIT = 10
    else:
        CS_SAMPLES = 1500
        DO_SAMPLES = 5000
        CS_TOP_K = 15
        CS_REWRITE_LIMIT = 10

    DO_N_PROC = 4
    CS_RETURN_TOP_K = 1
    HIGHER_LANGUAGE = True

GS_EXHAUSTIVE = True
# PURE Baseline: 
LE_ALL = False
STORE_SINGLE = True
if baseline_mode == 0:
    EXHAUSTIVE = True
    LE_ALL_REWRITES = False
    BEST_PLUS_MODE = False
    BEST_PROG_COUNT = 1
elif baseline_mode == 1:
    EXHAUSTIVE = False
    LE_ALL_REWRITES = True
    BEST_PLUS_MODE = True
    BEST_PROG_COUNT = 3
elif baseline_mode == 2:
    EXHAUSTIVE = False
    LE_ALL_REWRITES = False
    BEST_PLUS_MODE = True
    BEST_PROG_COUNT = 3
elif baseline_mode == 3:
    EXHAUSTIVE = False
    LE_ALL_REWRITES = True
    BEST_PLUS_MODE = False
    BEST_PROG_COUNT = 3
elif baseline_mode == 4:
    EXHAUSTIVE = False
    LE_ALL_REWRITES = True
    BEST_PLUS_MODE = True
    BEST_PROG_COUNT = 1
elif baseline_mode == 5:
    EXHAUSTIVE = False
    LE_ALL_REWRITES = True
    BEST_PLUS_MODE = True
    BEST_PROG_COUNT = 1
    LE_ALL = True
elif baseline_mode == 6:
    EXHAUSTIVE = False
    LE_ALL_REWRITES = True
    BEST_PLUS_MODE = True
    BEST_PROG_COUNT = 3
    LE_ALL = True
elif baseline_mode == 7:
    EXHAUSTIVE = False
    LE_ALL_REWRITES = True
    BEST_PLUS_MODE = True
    BEST_PROG_COUNT = 5
    LE_ALL = True
    STORE_SINGLE = False
elif baseline_mode == 8:
    EXHAUSTIVE = False
    LE_ALL_REWRITES = False
    BEST_PLUS_MODE = True
    BEST_PROG_COUNT = 5
    LE_ALL = False
    STORE_SINGLE = False
    

new_exp_name = new_exp_name + "_mode_%d" % baseline_mode
CS_USE_PROBS = False

if cfg.BC.WS.ENABLE:
    new_exp_name = new_exp_name + "_WS"
if cfg.BC.DO.ENABLE:
    new_exp_name = new_exp_name + "_DO"
if cfg.BC.GS.ENABLE:
    new_exp_name = new_exp_name + "_GS"
if cfg.BC.CS.ENABLE:
    new_exp_name = new_exp_name + "_CS"
if cfg.BC.CS.MERGE_SPLICE.ENABLE:
    new_exp_name = new_exp_name + "_MS"


if LE_ALL_REWRITES:
    if LE_ALL:
        new_exp_name = new_exp_name + "_le_all"
    else:
        new_exp_name = new_exp_name + "_le_rewrites"

if CS_USE_PROBS:
    new_exp_name = new_exp_name + "_CS_NLL"

if EXHAUSTIVE:
    new_exp_name = new_exp_name + "_exhaustive"

if BEST_PLUS_MODE:
    new_exp_name = new_exp_name + "_best_plus_data"
    
if not STORE_SINGLE:
    new_exp_name = new_exp_name + "_store_multiple"
    

if BEST_PROG_COUNT > 1:
    new_exp_name = new_exp_name + "_best_count_%d" % BEST_PROG_COUNT

cfg.EXP_NAME = cfg.EXP_NAME.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(
    old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(
    old_exp_name, new_exp_name)

if BEST_PLUS_MODE:
    cfg.BC.PLAD.BPDS.TRAINING_DATA_SELECTION = "BS+BEST"
if LE_ALL_REWRITES:
    cfg.BC.PLAD.LE_ONLY_ORIGINS = ["WS", "DO", "GS", "CS", "NR"]
    if LE_ALL:
       cfg.BC.PLAD.LE_ONLY_ORIGINS.append("BS") 
       
# 3 more than "10 k BS and 10k WS"
cfg.BC.PLAD.BPDS.BEST_PROG_COUNT = BEST_PROG_COUNT
# Since its best +  - Use only best from BS
cfg.BC.BS.RETURN_MULTIPLE = False
cfg.BC.PLAD.BPDS.STORE_SINGLE = STORE_SINGLE

# DO
cfg.BC.DO.N_PROC = DO_N_PROC
cfg.BC.DO.SAMPLE_COUNT = DO_SAMPLES
cfg.BC.DO.EXHAUSTIVE = EXHAUSTIVE

# GS 
cfg.BC.GS.EXHAUSTIVE = GS_EXHAUSTIVE

# CS
if "PSA" in LANGUAGE_NAME:
    cfg.BC.CS.ENABLE = False    

cfg.BC.CS.HIGHER_LANGUAGE = HIGHER_LANGUAGE
cfg.BC.CS.SAMPLE_COUNT = CS_SAMPLES
cfg.BC.CS.EXHAUSTIVE = EXHAUSTIVE
cfg.BC.CS.USE_PROBS = CS_USE_PROBS
cfg.BC.CS.TOP_K = CS_TOP_K
cfg.BC.CS.CS_RETURN_TOP_K = CS_RETURN_TOP_K
cfg.BC.CS.REWRITE_LIMIT = CS_REWRITE_LIMIT

if "2D" in LANGUAGE_NAME:
    cfg.BC.CS.CACHE_CONFIG.CACHE_SIZE = 40000
    cfg.BC.CS.CACHE_CONFIG.MERGE_BIT_DISTANCE = 1
    cfg.BC.CS.USE_CANONICAL = False
    cfg.BC.CS.N_PROC = 1
    cfg.BC.CS.MAX_BOOL_COUNT = 15 
    cfg.BC.CS.CACHE_CONFIG.N_PROGRAM_FOR_MERGE = 30000

if "FCSG" in LANGUAGE_NAME:
    cfg.BC.CS.FCSG_MODE = True

if "HSA" in LANGUAGE_NAME:
    cfg.BC.CS.CACHE_CONFIG.MERGE_NLIST = 10# 200
    cfg.BC.CS.CACHE_CONFIG.SEARCH_NLIST = 10# 
    cfg.BC.CS.CACHE_CONFIG.CACHE_SIZE = 25000

if DEBUG:
    # cfg.BC.DO.N_PROC = 1
    cfg.BC.DO.SAMPLE_COUNT = 200
    cfg.BC.CS.SAMPLE_COUNT = 200
    # cfg.BC.CS.CACHE_CONFIG.MERGE_BIT_DISTANCE = 100
    cfg.BC.CS.CACHE_CONFIG.MERGE_NLIST = 10# 150
    cfg.BC.CS.CACHE_CONFIG.SEARCH_NLIST = 10# 150