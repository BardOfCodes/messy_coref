from configs.ablations.mod_exp.sil_baseline import cfg as sil_cfg
import os
cfg = sil_cfg.clone()

old_exp_name = "PrePhasic_DiffOpt_Correct"
# new_exp_name = "PP_MRA_B_Occupancy"
new_exp_name = "debug"

cfg.EXP_NAME = cfg.EXP_NAME.replace(old_exp_name, new_exp_name)

cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)
cfg.POLICY.ME_C.EP_MODIFIER.SAVE_LOCATION = os.path.join(cfg.POLICY.ME_C.EP_MODIFIER.SAVE_LOCATION, cfg.EXP_NAME)
cfg.OBSERVABLE_STACK = 7
cfg.POLICY.ME_C.EP_MODIFIER.CONFIG.OBSERVABLE_STACK = 7


cfg.ACTION_SPACE_TYPE = "MultiRefactoredActionSpace" 
cfg.POLICY.ME_C.EP_MODIFIER.CONFIG.ACTION_SPACE_TYPE = "MultiRefactoredActionSpace" 

cfg.POLICY.PPO.PI_CONF = [1024, 512, 256]
cfg.MODEL.FEATURE_DIM = 64 * 4 * 4 * 2
cfg.MODEL.EXTRACTOR = "WrapperReplCNNExtractor"#
# cfg.TRAIN.LR_SCHEDULER.PATIENCE = 30
# cfg.MODEL.LOAD_WEIGHTS = "../weights/stage_16/multi_all_large_full_action_repl/weights_61.pt"
cfg.MODEL.LOAD_WEIGHTS = "../weights/stage_14/PrePhasic_MultiRefactoredAction_DiffOpt/mid_point.pt"
# cfg.MODEL.LOAD_WEIGHTS = "../weights/stage_14/PrePhasic_MultiRefactoredAction_DiffOpt/best_model.pt"

# FOr testing purpose:
cfg.POLICY.PPO.N_ENVS = 32
cfg.POLICY.DEFAULT_TRAIN_ROLLOUTS = 1
cfg.POLICY.COLLECT_MOD_EXP_ROLLOUTS = 1
cfg.POLICY.MOD_EXP_TRAIN_ROLLOUTS = 1
cfg.POLICY.VALUE_TRAIN_ROLLOUTS = 0
# cfg.POLICY.ME_C.EP_SELECTOR.TYPE = "AllowNoEpisodes"
cfg.POLICY.ME_C.EP_SELECTOR.TYPE = "AllowAllEpisodes"
# cfg.POLICY.ME_C.EP_SELECTOR.TYPE = "AllowUniqueEpisodeAndID"

# cfg.POLICY.ME_C.EP_MODIFIER.TYPE = "NoModification"
# ME_PPO.ME_C.EP_MODIFIER.TYPE = "BeamSearchModifier"
# cfg.POLICY.ME_C.EP_MODIFIER.TYPE = "DiffOptModifier"
cfg.POLICY.ME_C.EP_MODIFIER.TYPE = "PerturbAndDiffOptModifier"

# cfg.TRAIN.EVAL_FREQ = int(1e4) # How many steps between EVALs
cfg.POLICY.ME_T.ENABLE = True

## For advantage: 
cfg.POLICY.ME_C.ME_BUFFER.GAMMA = cfg.POLICY.PPO.GAMMA
cfg.POLICY.ME_C.ME_BUFFER.GAE_LAMBDA = cfg.POLICY.PPO.GAE_LAMBDA
cfg.POLICY.ME_C.ME_BUFFER.BASELINE_ALPHA = 0.8