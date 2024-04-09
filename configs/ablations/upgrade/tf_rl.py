
# from configs.cluster import cfg
from configs.ablations.upgrade.pretrain_baseline import cfg as base_cfg
# from configs.ablations.upgrade.tf_ablation import cfg as base_cfg
from configs.subconfig.envs.csg3d_train_shapenet_rnn import ENV as train_shapenet
from configs.subconfig.envs.csg3d_eval_shapenet_rnn import ENV as eval_shapenet
from configs.subconfig.base_policy.me_ppo import POLICY
import os


old_exp_name = "tf_pretrain_baseline"
new_exp_name = "tf_rl_baseline_lambda_0.5"
cfg = base_cfg.clone()
cfg.EXP_NAME = base_cfg.EXP_NAME.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)
# cfg.MACHINE_SPEC.PREDICTION_PATH = os.path.join(cfg.MACHINE_SPEC.SAVE_DIR, "stage_17", cfg.EXP_NAME)

cfg.MACHINE_SPEC.DATA_PATH = cfg.MACHINE_SPEC.DATA_PATH.replace('csgnet_large', '3d_csg/data') 
cfg.TRAIN.ENV = train_shapenet.clone()
cfg.EVAL.ENV = eval_shapenet.clone()
cfg.EVAL.EXHAUSTIVE = True
# cfg.TRAIN.EVAL_EPISODES = 500 # How many episodes in EVAL


cfg.MODEL.LOAD_WEIGHTS = "../weights/stage_18/tf_pretrain_baseline_no_sdf/weights_25.pt"
# cfg.MODEL.LOAD_WEIGHTS = "../weights/stage_18/tf_pretrain_baseline_no_sdf/value_pretrained.pt"
# cfg.MODEL.LOAD_WEIGHTS = "/users/aganesh8/data/aganesh8/projects/rl/weights/stage_18/tf_pretrain_baseline_no_sdf/weights_25.pt"

# PPO Settings
cfg.POLICY.PPO.N_ENVS = 12
cfg.POLICY.PPO.N_STEPS = 1024
cfg.POLICY.PPO.BATCH_SIZE = 128
POLICY.PPO = cfg.POLICY.PPO.clone()
cfg.POLICY = POLICY.clone()
cfg.POLICY.ME_C.EP_SELECTOR.TYPE = "AllowNoEpisodes"
cfg.POLICY.ME_C.EP_MODIFIER.TYPE = "NoModification"
cfg.POLICY.ME_T.TYPE = "A2CTrainer"
cfg.POLICY.ME_T.ENABLE = False
cfg.POLICY.ME_C.EP_SELECTOR.EXPRESSION_LIMIT = 0
cfg.POLICY.ME_C.ME_BUFFER.EPISODE_BUDGET = 0

cfg.POLICY.ALLOW_COLLECT_MOD_EXP = False
cfg.POLICY.ALLOW_MOD_EXP_TRAIN = False
cfg.POLICY.VALUE_PRETRAIN = 10
cfg.POLICY.COLLECT_GRADIENTS = False
cfg.POLICY.GRADIENT_STEP_COUNT = 0

## Maybe Mod PPO style update? 
cfg.TRAIN.LR_INITIAL = 0.00001
# Save settings
cfg.TRAIN.EVAL_FREQ = int(5e3) # How many steps between EVALs
cfg.TRAIN.SAVE_FREQ = int(5e20) # How many steps
