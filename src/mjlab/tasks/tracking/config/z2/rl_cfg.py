"""RL configuration wrappers for Z2 tracking tasks."""

from mjlab.tasks.tracking.config.g1.rl_cfg import (
  _unitree_g1_tracking_ppo_runner_cfg,
  unitree_g1_rough_tracking_late_phase_dr_finetune_ppo_runner_cfg,
  unitree_g1_tracking_late_phase_dr_finetune_ppo_runner_cfg,
)


def nubot_z2_tracking_ppo_runner_cfg():
  return _unitree_g1_tracking_ppo_runner_cfg(experiment_name="z2_tracking")


def nubot_z2_rough_tracking_ppo_runner_cfg():
  return _unitree_g1_tracking_ppo_runner_cfg(experiment_name="z2_tracking_rough")


def nubot_z2_tracking_late_phase_dr_finetune_ppo_runner_cfg():
  cfg = unitree_g1_tracking_late_phase_dr_finetune_ppo_runner_cfg()
  cfg.experiment_name = "z2_tracking_late_phase_dr_finetune"
  return cfg


def nubot_z2_rough_tracking_late_phase_dr_finetune_ppo_runner_cfg():
  cfg = unitree_g1_rough_tracking_late_phase_dr_finetune_ppo_runner_cfg()
  cfg.experiment_name = "z2_tracking_rough_late_phase_dr_finetune"
  return cfg

