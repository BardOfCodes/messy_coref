from configs.ablations.mod_exp.sil_new_action import cfg as sil_cfg

cfg = sil_cfg.clone()

old_exp_name = "PP_MRA_B_Occupancy"
# new_exp_name = "CRR No_R_gamma_999"
new_exp_name = "debug"

cfg.EXP_NAME = cfg.EXP_NAME.replace(old_exp_name, new_exp_name)

cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)

# No Mod Exp:
cfg.POLICY.ME_C.EP_SELECTOR.TYPE = "AllowNoEpisodes"
cfg.POLICY.ME_C.EP_MODIFIER.TYPE = "NoModification"
cfg.POLICY.ME_T.ENABLE = False

# No Gamma
cfg.POLICY.PPO.GAMMA = 1.0
# cfg.POLICY.PPO.GAE_LAMBDA = 1.0


cfg.POLICY.PPO.N_ENVS = 32
cfg.POLICY.DEFAULT_TRAIN_ROLLOUTS = 10
cfg.POLICY.COLLECT_MOD_EXP_ROLLOUTS = 10
cfg.POLICY.MOD_EXP_TRAIN_ROLLOUTS = 10
cfg.POLICY.VALUE_TRAIN_ROLLOUTS = 10

# Reward function update:
cfg.TRAIN.ENV.REWARD.USE_CR_REWARD = False
cfg.TRAIN.ENV.REWARD.CR_REWARD_COEF = 0.1
cfg.EVAL.ENV.REWARD.USE_CR_REWARD = False
cfg.EVAL.ENV.REWARD.CR_REWARD_COEF = 0.5