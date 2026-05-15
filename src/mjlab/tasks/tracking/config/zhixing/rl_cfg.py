"""RL configuration for Zhixing tracking tasks."""

from mjlab.tasks.tracking.config.g1.rl_cfg import (
  _unitree_g1_tracking_ppo_runner_cfg,
)


def zhixing_tracking_ppo_runner_cfg():
  return _unitree_g1_tracking_ppo_runner_cfg(experiment_name="zhixing_tracking")


def zhixing_rough_tracking_ppo_runner_cfg():
  return _unitree_g1_tracking_ppo_runner_cfg(
    experiment_name="zhixing_tracking_rough"
  )
