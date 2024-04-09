
from configs.ablations.finals.pretrain_baseline import LANGUAGE_MODE, DEBUG, LANGUAGE_NAME
from configs.ablations.finals.plad_baseline import cfg as base_cfg, USE_UCSG, NOISY_AUG, RESTRICT_DATASIZE, DATASIZE, USE_CSGSTUMP


old_exp_name = "_plad"
new_exp_name = "_icr_plad"

cfg = base_cfg.clone()

baseline_mode = 2

REWRITE_INPUT_BS_ONLY = False
REWRITE_INPUT_RANDOM = False
STORE_SINGLE = False

GS_EXHAUSTIVE = False
# PURE Baseline: 
EXHAUSTIVE = False
changed_CS_params = False

if baseline_mode == 0:
    # PLAD + Rewrites
    EXHAUSTIVE = True
    BEST_PROG_COUNT = 1
    DATA_SELECTION_POLICY = "BEST"
    STORE_SINGLE = True
elif baseline_mode == 1:
    # PLAD + R not exhaustive
    EXHAUSTIVE = False
    DATA_SELECTION_POLICY = False
    BEST_PROG_COUNT = 1
    DATA_SELECTION_POLICY = "BEST"
    STORE_SINGLE = True
    
elif baseline_mode == 2:
    # Basic SRT = PLAD + 3 best programs
    EXHAUSTIVE = False
    BEST_PROG_COUNT = 3
    DATA_SELECTION_POLICY = "BS+BEST"
    STORE_SINGLE = True # MS will not get a chance...
    
elif baseline_mode == 3:
    # SRT with diff sampling:
    EXHAUSTIVE = False
    BEST_PROG_COUNT = 5
    DATA_SELECTION_POLICY = "BS+BEST+PROB-ONE"
    STORE_SINGLE = False
    REWRITE_INPUT_BS_ONLY = True
elif baseline_mode == 4:
    # SRT with real prob. sampling
    EXHAUSTIVE = False
    BEST_PROG_COUNT = 5
    DATA_SELECTION_POLICY = "BS+BEST+PROB-THREE"
    STORE_SINGLE = False
elif baseline_mode == 5:
    # Basic SRT = PLAD + 3 best programs
    EXHAUSTIVE = False
    BEST_PROG_COUNT = 3
    DATA_SELECTION_POLICY = "BEST"
    STORE_SINGLE = True # MS will not get a chance...

elif baseline_mode == 6:
    # PLAD + Rewrites no early stopping
    EXHAUSTIVE = True
    BEST_PROG_COUNT = 1
    DATA_SELECTION_POLICY = "BEST"
    STORE_SINGLE = True
    cfg.BC.PLAD.LOAD_BEST_BEFORE_SEARCH = False
    
    

    
    
    
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

if changed_CS_params:
    CS_SAMPLES = 5000
    CS_REWRITE_LIMIT = 2


if USE_UCSG:
    CS_SAMPLES = int(CS_SAMPLES * 3.6)
    DO_SAMPLES = int(CS_SAMPLES * 3.6)
        
if "PCSG" in LANGUAGE_NAME:
        
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



if changed_CS_params:
    new_exp_name = new_exp_name + "_diffCS"
    

new_exp_name = new_exp_name + "_mode=%s" % DATA_SELECTION_POLICY
    
if not STORE_SINGLE:
    new_exp_name = new_exp_name + "_store_multiple"

    
new_exp_name = new_exp_name + "_best_count_%d" % BEST_PROG_COUNT



cfg.EXP_NAME = cfg.EXP_NAME.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(
    old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(
    old_exp_name, new_exp_name)

cfg.BC.PLAD.BPDS.TRAINING_DATA_SELECTION = DATA_SELECTION_POLICY# "BS+BEST"
       
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
    cfg.BC.CS.CACHE_CONFIG.MERGE_NLIST = 200
    cfg.BC.CS.CACHE_CONFIG.SEARCH_NLIST = 100# 
    cfg.BC.CS.CACHE_CONFIG.CACHE_SIZE = 25000
    
if USE_CSGSTUMP:
    SIZE_RATIO = 1.4198
    cfg.BC.DO.SAMPLE_COUNT = int(cfg.BC.DO.SAMPLE_COUNT * SIZE_RATIO)
    cfg.BC.GS.SAMPLE_COUNT = int(cfg.BC.GS.SAMPLE_COUNT * SIZE_RATIO)
    cfg.BC.CS.SAMPLE_COUNT = 5000 # reducing 1500 might be too costly.
    cfg.BC.CS.SAMPLE_COUNT = int(cfg.BC.CS.SAMPLE_COUNT * SIZE_RATIO)
    cfg.BC.CS.MERGE_SPLICE.SAMPLE_COUNT = int(cfg.BC.CS.MERGE_SPLICE.SAMPLE_COUNT * SIZE_RATIO)
    
    
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