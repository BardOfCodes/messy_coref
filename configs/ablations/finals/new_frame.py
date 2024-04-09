
from configs.ablations.finals.pretrain_baseline import LANGUAGE_MODE, DEBUG, LANGUAGE_NAME
from configs.ablations.finals.plad_baseline import cfg as base_cfg, USE_UCSG, NOISY_AUG, RESTRICT_DATASIZE, DATASIZE


old_exp_name = "_plad"
new_exp_name = "_icr_rewrite_ablation"

cfg = base_cfg.clone()

baseline_mode = 1

REWRITE_INPUT_BS_ONLY = False
REWRITE_INPUT_RANDOM = False
MULTIPLY_INPUT = False
STORE_SINGLE = False

if baseline_mode == 0:
    # BS + one best program
    BEST_PROG_COUNT = 1
    DATA_SELECTION_POLICY = "BS+BEST"
elif baseline_mode == 1:
    # BS + 3 best programs
    BEST_PROG_COUNT = 3
    DATA_SELECTION_POLICY = "BS+BEST"
    STORE_SINGLE = True
    
elif baseline_mode == 2:
    # BS + 5 best programs
    BEST_PROG_COUNT = 5
    DATA_SELECTION_POLICY = "BS+BEST"
    
elif baseline_mode == 3:
    # BS + inf best programs
    BEST_PROG_COUNT = 100000
    DATA_SELECTION_POLICY = "BS+BEST"
    
elif baseline_mode == 4:
    # BS + 3 Best programs 
    # probability weighted PONE
    BEST_PROG_COUNT = 3
    DATA_SELECTION_POLICY = "BS+BEST+PROB-ONE"
    
elif baseline_mode == 5:
    # BS + 5 best programs
    # probability weighted PONE
    BEST_PROG_COUNT = 5
    DATA_SELECTION_POLICY = "BS+BEST+PROB-ONE"
    
elif baseline_mode == 6:
    # BS + inf best programs
    # probability weighted PONE
    BEST_PROG_COUNT = 100000
    DATA_SELECTION_POLICY = "BS+BEST+PROB-ONE"
    
elif baseline_mode == 7:
    # BS + 3 best programs
    # probability weighted PTWO
    BEST_PROG_COUNT = 3
    DATA_SELECTION_POLICY = "BS+BEST+PROB-TWO"
    
elif baseline_mode == 8:
    # BS + 5 best programs
    # probability weighted PTWO
    BEST_PROG_COUNT = 5
    DATA_SELECTION_POLICY = "BS+BEST+PROB-TWO"
    
elif baseline_mode == 9:
    # BS + inf best programs
    # probability weighted PTWO
    BEST_PROG_COUNT = 100000
    DATA_SELECTION_POLICY = "BS+BEST+PROB-TWO"
    
elif baseline_mode == 10:
    # BS + inf best programs
    # probability weighted PTWO
    # Rewrite Origin fixed to latest BS
    BEST_PROG_COUNT = 100000
    DATA_SELECTION_POLICY = "BS+BEST+PROB-TWO"
    REWRITE_INPUT_BS_ONLY = True
    
elif baseline_mode == 11:
    # BS + inf best programs
    # probability weighted PTWO
    # Rewrite Origin fixed to random
    BEST_PROG_COUNT = 100000
    DATA_SELECTION_POLICY = "BS+BEST+PROB-TWO"
    REWRITE_INPUT_RANDOM = True
elif baseline_mode == 12:
    # BS + inf best programs
    # probability weighted PTWO
    # Rewrite Origin fixed to random
    BEST_PROG_COUNT = 100000
    DATA_SELECTION_POLICY = "BS+BEST+PROB-TWO"
    REWRITE_INPUT_RANDOM = True
    MULTIPLY_INPUT = True
    
    
    
    
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
CS_SAMPLES = 1500
DO_SAMPLES = 5000
CS_TOP_K = 15
CS_RETURN_TOP_K = 1
CS_REWRITE_LIMIT = 10
DO_N_PROC = 4

if "PCSG" in LANGUAGE_NAME:
    if USE_UCSG:
        CS_SAMPLES = int(1500 * 3.6)
        DO_SAMPLES = int(5000 * 3.6)
        
    HIGHER_LANGUAGE = False
elif "FCSG2D" in LANGUAGE_NAME:
    HIGHER_LANGUAGE = False
    CS_TOP_K = 15
    CS_RETURN_TOP_K = 1
    CS_REWRITE_LIMIT = 10
elif "MCSG" in LANGUAGE_NAME:
    HIGHER_LANGUAGE = True
else:
    # For MCSG, PSA, HSA
    if "SA" in LANGUAGE_NAME:
        CS_SAMPLES = 1500
        DO_SAMPLES = 5000
    HIGHER_LANGUAGE = True

GS_EXHAUSTIVE = True
# PURE Baseline: 
LE_ALL = False
EXHAUSTIVE = False
LE_ALL_REWRITES = False
CS_USE_PROBS = False


    

new_exp_name = new_exp_name + "_mode_%d" % baseline_mode


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



new_exp_name = new_exp_name + "_mode=%s" % DATA_SELECTION_POLICY
    
if not STORE_SINGLE:
    new_exp_name = new_exp_name + "_store_multiple"

if MULTIPLY_INPUT:
    new_exp_name = new_exp_name + "_n_mult"
    
new_exp_name = new_exp_name + "_best_count_%d" % BEST_PROG_COUNT



cfg.EXP_NAME = cfg.EXP_NAME.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(
    old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(
    old_exp_name, new_exp_name)

cfg.BC.PLAD.BPDS.TRAINING_DATA_SELECTION = DATA_SELECTION_POLICY# "BS+BEST"
cfg.BC.PLAD.BPDS.MULTIPlY_INPUT = MULTIPLY_INPUT
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

# Input gating:
if REWRITE_INPUT_BS_ONLY:
    cfg.BC.DO.INPUT_GATING = True
    cfg.BC.GS.INPUT_GATING = True
    cfg.BC.CS.INPUT_GATING = True
    cfg.BC.CS.MERGE_SPLICE.INPUT_GATING = True
if REWRITE_INPUT_RANDOM:
    cfg.BC.DO.SELECT_RANDOM = True
    cfg.BC.GS.SELECT_RANDOM = True
    cfg.BC.CS.SELECT_RANDOM = True
    


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
    
    
if RESTRICT_DATASIZE:
    cfg.BC.DO.SAMPLE_COUNT = int(cfg.BC.DO.SAMPLE_COUNT * DATASIZE)
    cfg.BC.GS.SAMPLE_COUNT = int(cfg.BC.GS.SAMPLE_COUNT * DATASIZE)
    cfg.BC.CS.SAMPLE_COUNT = int(cfg.BC.CS.SAMPLE_COUNT * DATASIZE)
    cfg.BC.CS.MERGE_SPLICE.SAMPLE_COUNT = int(cfg.BC.CS.MERGE_SPLICE.SAMPLE_COUNT * DATASIZE)

if DEBUG:
    # cfg.BC.DO.N_PROC = 1
    cfg.BC.DO.SAMPLE_COUNT = 20
    cfg.BC.CS.SAMPLE_COUNT = 20
    # cfg.BC.CS.CACHE_CONFIG.MERGE_BIT_DISTANCE = 100
    cfg.BC.CS.CACHE_CONFIG.MERGE_NLIST = 10# 150
    cfg.BC.CS.CACHE_CONFIG.SEARCH_NLIST = 10# 150