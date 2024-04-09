from yacs.config import CfgNode as CN

LR_SCHEDULER = CN()
LR_SCHEDULER.TYPE = "ONE_CYCLE_LR"
LR_SCHEDULER.MAX_LR = 0.0005
LR_SCHEDULER.TOTAL_STEPS = 50 * 500
LR_SCHEDULER.PCT_START = 0.10
LR_SCHEDULER.VERBOSE = False

