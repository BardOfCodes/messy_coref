
from configs.ablations.upgrade.tf_rl import cfg as base_cfg


old_exp_name = "tf_rl_baseline_lambda_0.5"
new_exp_name = "tf_rl_grad_collect_lambda_0.5"
cfg = base_cfg.clone()
cfg.EXP_NAME = base_cfg.EXP_NAME.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.LOG_DIR = cfg.MACHINE_SPEC.LOG_DIR.replace(old_exp_name, new_exp_name)
cfg.MACHINE_SPEC.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR.replace(old_exp_name, new_exp_name)


cfg.POLICY.COLLECT_GRADIENTS = True
cfg.POLICY.GRADIENT_STEP_COUNT = 4