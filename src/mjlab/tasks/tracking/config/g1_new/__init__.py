from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  unitree_g1_new_flat_late_phase_dr_finetune_env_cfg,
  unitree_g1_new_flat_tracking_env_cfg,
  unitree_g1_new_rough_late_phase_dr_finetune_env_cfg,
  unitree_g1_new_rough_tracking_env_cfg,
)
from .rl_cfg import (
  unitree_g1_new_rough_tracking_late_phase_dr_finetune_ppo_runner_cfg,
  unitree_g1_new_rough_tracking_ppo_runner_cfg,
  unitree_g1_new_tracking_late_phase_dr_finetune_ppo_runner_cfg,
  unitree_g1_new_tracking_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Rough-Unitree-G1-New",
  env_cfg=unitree_g1_new_rough_tracking_env_cfg(),
  play_env_cfg=unitree_g1_new_rough_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_new_rough_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-New",
  env_cfg=unitree_g1_new_flat_tracking_env_cfg(),
  play_env_cfg=unitree_g1_new_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_new_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-New-Late-Phase-DR-Finetune",
  env_cfg=unitree_g1_new_flat_late_phase_dr_finetune_env_cfg(),
  play_env_cfg=unitree_g1_new_flat_late_phase_dr_finetune_env_cfg(play=True),
  rl_cfg=unitree_g1_new_tracking_late_phase_dr_finetune_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Rough-Unitree-G1-New-Late-Phase-DR-Finetune",
  env_cfg=unitree_g1_new_rough_late_phase_dr_finetune_env_cfg(),
  play_env_cfg=unitree_g1_new_rough_late_phase_dr_finetune_env_cfg(play=True),
  rl_cfg=unitree_g1_new_rough_tracking_late_phase_dr_finetune_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-New-No-State-Estimation",
  env_cfg=unitree_g1_new_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_g1_new_flat_tracking_env_cfg(
    has_state_estimation=False,
    play=True,
  ),
  rl_cfg=unitree_g1_new_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
