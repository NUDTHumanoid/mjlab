"""RL configuration wrappers for the isolated mode_15-aligned G1 velocity tasks."""

from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg


def unitree_g1_new_ppo_runner_cfg():
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_new_velocity"
  return cfg
