from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  nubot_z2_flat_tracking_env_cfg,
  nubot_z2_rough_tracking_env_cfg,
)
from .rl_cfg import (
  nubot_z2_rough_tracking_ppo_runner_cfg,
  nubot_z2_tracking_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Z2",
  env_cfg=nubot_z2_flat_tracking_env_cfg(),
  play_env_cfg=nubot_z2_flat_tracking_env_cfg(play=True),
  rl_cfg=nubot_z2_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Rough-Z2",
  env_cfg=nubot_z2_rough_tracking_env_cfg(),
  play_env_cfg=nubot_z2_rough_tracking_env_cfg(play=True),
  rl_cfg=nubot_z2_rough_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

