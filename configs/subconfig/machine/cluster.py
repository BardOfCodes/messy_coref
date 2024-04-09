
from yacs.config import CfgNode as CN
import os

MACHINE_SPEC = CN()
MACHINE_SPEC.EXP_NAME = "CLUSTER"
MACHINE_SPEC.DATA_ROOT = "/data/drlab/aditya/data/"
MACHINE_SPEC.PROJECT_ROOT = "/data/drlab/aditya/projects/rl/"
MACHINE_SPEC.DATA_PATH = os.path.join(MACHINE_SPEC.DATA_ROOT, "csgnet")
MACHINE_SPEC.TERMINAL_FILE = os.path.join(MACHINE_SPEC.DATA_ROOT, "csgnet/terminals.txt")
MACHINE_SPEC.SAVE_DIR = os.path.join(MACHINE_SPEC.PROJECT_ROOT, "weights/")
MACHINE_SPEC.LOG_DIR = os.path.join(MACHINE_SPEC.PROJECT_ROOT, "logs/")
