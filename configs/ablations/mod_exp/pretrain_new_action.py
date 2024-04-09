
import os
from yacs.config import CfgNode as CN
from configs.subconfig.base_policy.ppo import POLICY
from configs.subconfig.envs.train_random import ENV as train_random
from configs.subconfig.envs.eval_random import ENV as eval_random
from configs.subconfig.behavior_cloning.baseline import BC

from configs.ablations.mod_exp.pretrain import cfg as pretrain_cfg

old_exp_name = "debug"
new_exp_name = "multi_all_large_full_action_repl"

cfg= pretrain_cfg.clone()
cfg.EXP_NAME = cfg.EXP_NAME.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)

cfg.ACTION_SPACE_TYPE = "MultiRefactoredActionSpace" 

