from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_acrobatics_env_cfg,
  unitree_g1_flat_crouch_to_lie_down_env_cfg,
  unitree_g1_flat_tracking_env_cfg,
)
from .rl_cfg import (
  unitree_g1_crouch_to_lie_down_ppo_runner_cfg,
  unitree_g1_tracking_acrobatics_finetune_runner_cfg,
  unitree_g1_tracking_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_tracking_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-CrouchToLieDown",
  env_cfg=unitree_g1_flat_crouch_to_lie_down_env_cfg(),
  play_env_cfg=unitree_g1_flat_crouch_to_lie_down_env_cfg(play=True),
  rl_cfg=unitree_g1_crouch_to_lie_down_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-Acrobatics",
  env_cfg=unitree_g1_flat_acrobatics_env_cfg(),
  play_env_cfg=unitree_g1_flat_acrobatics_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_acrobatics_finetune_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
