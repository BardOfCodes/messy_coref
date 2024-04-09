from configs.ablations.mod_exp.pretrain import cfg
from yacs.config import CfgNode as CN

cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('csgnet_refactored', 'csgnet') 
cfg.REFACTOR_CONFIG = CN()
## DO this change before changing the header.
cfg.REFACTOR_CONFIG.CONFIG = cfg.clone()
cfg.REFACTOR_CONFIG.PHASE_CONFIG = cfg.TRAIN.clone()

cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('csgnet', 'csgnet_refactored') 
cfg.TRAIN.ENV.TYPE = "RefactoredComplexityCSG"
cfg.EVAL.ENV.TYPE = "RefactoredComplexityCSG"