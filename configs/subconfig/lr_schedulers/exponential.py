from yacs.config import CfgNode as CN

LR_SCHEDULER = CN()
LR_SCHEDULER.TYPE = "EXPONENTIAL"
LR_SCHEDULER.GAMMA = 0.98
LR_SCHEDULER.LAST_EPOCH = -1 # Must be set manually