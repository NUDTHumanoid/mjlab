from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_new_flat_env_cfg,
  unitree_g1_new_rough_env_cfg,
)
from .rl_cfg import unitree_g1_new_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-G1-New",
  env_cfg=unitree_g1_new_rough_env_cfg(),
  play_env_cfg=unitree_g1_new_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_new_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-New",
  env_cfg=unitree_g1_new_flat_env_cfg(),
  play_env_cfg=unitree_g1_new_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_new_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
