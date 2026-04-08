"""RL configuration wrappers for the isolated mode_15-aligned G1 tracking tasks."""

from mjlab.tasks.tracking.config.g1.rl_cfg import (
  _unitree_g1_tracking_ppo_runner_cfg,
)


def unitree_g1_new_tracking_ppo_runner_cfg():
  return _unitree_g1_tracking_ppo_runner_cfg(experiment_name="g1_new_tracking")


def unitree_g1_new_rough_tracking_ppo_runner_cfg():
  return _unitree_g1_tracking_ppo_runner_cfg(experiment_name="g1_new_tracking_rough")
