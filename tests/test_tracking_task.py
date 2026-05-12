"""Tests specific to motion tracking tasks."""

import pytest

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.asset_zoo.robots.unitree_g1.g1_constants_new import G1_NEW_ACTION_SCALE
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.scripts.play import _enable_tracking_late_phase_play_dr
from mjlab.scripts.train import _enable_tracking_late_phase_train_dr
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.config.late_phase_dr import LATE_PHASE_DR_START_RATIO
from mjlab.tasks.tracking.mdp import MotionCommandCfg


@pytest.fixture(scope="module")
def tracking_task_ids() -> list[str]:
  """Get all tracking task IDs."""
  return [t for t in list_tasks() if "Tracking" in t]


@pytest.fixture(scope="module")
def g1_tracking_task_ids(tracking_task_ids: list[str]) -> list[str]:
  """Get all G1 tracking task IDs."""
  return [t for t in tracking_task_ids if "G1" in t]


def test_tracking_tasks_have_motion_command(tracking_task_ids: list[str]) -> None:
  """All tracking tasks should have a 'motion' command of type MotionCommandCfg."""
  for task_id in tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "motion" in cfg.commands, f"Task {task_id} missing 'motion' command"

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} motion command is not MotionCommandCfg"
    )


def test_tracking_tasks_have_self_collision_sensor(
  tracking_task_ids: list[str],
) -> None:
  """All tracking tasks should have a self_collision sensor."""
  for task_id in tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "self_collision" in sensor_names, (
      f"Task {task_id} missing self_collision sensor"
    )


def test_tracking_no_state_estimation_observations() -> None:
  """No-state-estimation tasks remove observations that depend on state estimation."""
  task_id = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"

  # Test both training and play modes
  for play_mode in [False, True]:
    cfg = load_env_cfg(task_id, play=play_mode)
    mode_str = "play mode" if play_mode else "training mode"

    assert "actor" in cfg.observations, (
      f"Task {task_id} ({mode_str}) missing policy observations"
    )
    actor_terms = cfg.observations["actor"].terms

    assert "motion_anchor_pos_b" not in actor_terms, (
      f"Task {task_id} ({mode_str}) has motion_anchor_pos_b in policy, "
      "expected it to be removed for no-state-estimation variant"
    )
    assert "base_lin_vel" not in actor_terms, (
      f"Task {task_id} ({mode_str}) has base_lin_vel in policy, "
      "expected it to be removed for no-state-estimation variant"
    )


def test_tracking_play_disables_rsi_randomization() -> None:
  """Tracking play tasks should disable RSI randomization."""
  tracking_tasks = [
    "Mjlab-Tracking-Flat-Unitree-G1",
    "Mjlab-Tracking-Flat-Unitree-G1-Late-Phase-DR-Finetune",
    "Mjlab-Tracking-Rough-Unitree-G1-Late-Phase-DR-Finetune",
    "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
    "Mjlab-Tracking-Flat-Unitree-G1-New-Late-Phase-DR-Finetune",
    "Mjlab-Tracking-Rough-Unitree-G1-New-Late-Phase-DR-Finetune",
  ]

  for task_id in tracking_tasks:
    cfg = load_env_cfg(task_id, play=True)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} (play mode) motion command is not MotionCommandCfg"
    )

    assert motion_cmd.pose_range == {}, (
      f"Task {task_id} (play mode) has non-empty pose_range={motion_cmd.pose_range}, "
      "expected empty dict for disabled RSI"
    )
    assert motion_cmd.velocity_range == {}, (
      f"Task {task_id} (play mode) has non-empty velocity_range={motion_cmd.velocity_range}, "
      "expected empty dict for disabled RSI"
    )


def test_tracking_play_uses_start_sampling_mode() -> None:
  """Tracking play tasks should use sampling_mode='start'."""
  tracking_tasks = [
    "Mjlab-Tracking-Flat-Unitree-G1",
    "Mjlab-Tracking-Flat-Unitree-G1-Late-Phase-DR-Finetune",
    "Mjlab-Tracking-Rough-Unitree-G1-Late-Phase-DR-Finetune",
    "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
    "Mjlab-Tracking-Flat-Unitree-G1-New-Late-Phase-DR-Finetune",
    "Mjlab-Tracking-Rough-Unitree-G1-New-Late-Phase-DR-Finetune",
  ]

  for task_id in tracking_tasks:
    cfg = load_env_cfg(task_id, play=True)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} (play mode) motion command is not MotionCommandCfg"
    )

    assert motion_cmd.sampling_mode == "start", (
      f"Task {task_id} (play mode) sampling_mode={motion_cmd.sampling_mode}, expected 'start'"
    )


def test_removed_experimental_finetune_tasks_are_not_registered() -> None:
  """Old landing/recovery experimental tasks should be removed from the registry."""
  registered_tasks = set(list_tasks())

  assert "Mjlab-Tracking-Flat-Unitree-G1-Landing-Finetune" not in registered_tasks
  assert "Mjlab-Tracking-Flat-Unitree-G1-Recovery-Mixed" not in registered_tasks
  assert "Mjlab-Tracking-Flat-Unitree-G1-New-Landing-Finetune" not in registered_tasks
  assert "Mjlab-Tracking-Flat-Unitree-G1-New-Recovery-Mixed" not in registered_tasks


def test_late_phase_dr_finetune_task_keeps_full_motion_sampling_and_adds_late_disturbance() -> None:
  """Late-phase-DR finetune should keep full-motion sampling and add late-only DR."""
  base_task_id = "Mjlab-Tracking-Flat-Unitree-G1-New"
  task_id = "Mjlab-Tracking-Flat-Unitree-G1-New-Late-Phase-DR-Finetune"
  base_cfg = load_env_cfg(base_task_id)
  cfg = load_env_cfg(task_id)

  base_motion_cmd = base_cfg.commands["motion"]
  motion_cmd = cfg.commands["motion"]
  assert isinstance(base_motion_cmd, MotionCommandCfg)
  assert isinstance(motion_cmd, MotionCommandCfg)
  assert motion_cmd.sampling_start_frame is None
  assert motion_cmd.sampling_end_frame is None
  assert motion_cmd.sampling_mode == "adaptive"
  assert motion_cmd.pose_range == base_motion_cmd.pose_range
  assert motion_cmd.velocity_range == base_motion_cmd.velocity_range
  assert motion_cmd.joint_position_range == base_motion_cmd.joint_position_range
  assert motion_cmd.joint_velocity_range == base_motion_cmd.joint_velocity_range

  assert "late_phase_dr_disturbance" in cfg.events
  assert cfg.events["late_phase_dr_disturbance"].mode == "step"
  assert cfg.events["late_phase_dr_disturbance"].params["late_phase_start_ratio"] == (
    LATE_PHASE_DR_START_RATIO
  )
  assert cfg.events["late_phase_dr_disturbance"].params["late_phase_onset_scale"] == 0.35
  assert cfg.events["late_phase_dr_disturbance"].params["stand_up_push_center_frame"] == 150
  assert cfg.events["late_phase_dr_disturbance"].params["stand_up_push_half_window"] == 4
  assert cfg.events["late_phase_dr_disturbance"].params["stand_up_push_post_trigger_frame"] == 165
  assert cfg.events["late_phase_dr_disturbance"].params["stand_up_recovery_probability"] == 0.4
  assert cfg.events["late_phase_dr_disturbance"].params["stand_up_push_forward_force_range"] == (
    0.0,
    0.0,
  )
  assert cfg.events["late_phase_dr_disturbance"].params["stand_up_push_lateral_force_range"] == (
    0.0,
    0.0,
  )
  assert cfg.events["late_phase_dr_disturbance"].params["stand_up_overshoot_trigger_frame"] == 165
  assert cfg.events["late_phase_dr_disturbance"].params["stand_up_overshoot_half_window"] == 2
  assert cfg.events["late_phase_dr_disturbance"].params["stand_up_overshoot_effort_scale_range"] == (
    1.15,
    1.27,
  )
  assert cfg.events["late_phase_dr_disturbance"].params["stand_up_overshoot_pd_scale_range"] == (
    1.06,
    1.15,
  )
  assert cfg.events["late_phase_dr_disturbance"].params["stand_up_overshoot_action_scale_range"] == (
    1.225,
    1.42,
  )
  assert cfg.events["late_phase_dr_disturbance"].params["stand_up_overshoot_pitch_ang_vel_range"] == (
    1.05,
    1.65,
  )
  assert cfg.events["late_phase_dr_disturbance"].params["stand_up_underpowered_probability"] == 0.5
  assert cfg.events["late_phase_dr_disturbance"].params["stand_up_underpowered_trigger_frame"] == 135
  assert cfg.events["late_phase_dr_disturbance"].params["stand_up_underpowered_half_window"] == 2
  assert cfg.events["late_phase_dr_disturbance"].params["stand_up_underpowered_effort_scale_range"] == (
    0.73,
    0.88,
  )
  assert cfg.events["late_phase_dr_disturbance"].params["stand_up_underpowered_pd_scale_range"] == (
    0.76,
    0.88,
  )
  assert cfg.events["late_phase_dr_disturbance"].params["stand_up_underpowered_action_scale_range"] == (
    0.58,
    0.82,
  )
  assert cfg.events["late_phase_dr_disturbance"].params["stand_up_underpowered_pitch_ang_vel_range"] == (
    -1.95,
    -1.35,
  )
  assert cfg.events["late_phase_dr_disturbance"].params["stand_up_pitch_ang_vel_kick_duration_steps"] == 3
  assert cfg.events["late_phase_dr_disturbance"].params["stand_up_velocity_kick_forward_range"] == (
    0.0,
    0.0,
  )
  assert cfg.rewards["motion_joint_pos"].weight == 0.25
  assert cfg.rewards["motion_joint_pos"].params["std"] == 0.5
  assert cfg.rewards["motion_joint_vel"].weight == 0.1
  assert cfg.rewards["motion_joint_vel"].params["std"] == 2.5
  assert cfg.events["base_com"].params["ranges"] == base_cfg.events["base_com"].params["ranges"]
  assert cfg.events["encoder_bias"].params["bias_range"] == (
    base_cfg.events["encoder_bias"].params["bias_range"]
  )
  assert cfg.events["foot_friction"].params["ranges"] == (
    base_cfg.events["foot_friction"].params["ranges"]
  )
  assert cfg.terminations["anchor_ori"].params["threshold"] == 1.2


def test_rough_late_phase_dr_finetune_preserves_rough_setup_and_adds_late_disturbance() -> None:
  """Rough late-phase finetune should keep rough terrain logic and add late-only DR."""
  base_task_id = "Mjlab-Tracking-Rough-Unitree-G1-New"
  task_id = "Mjlab-Tracking-Rough-Unitree-G1-New-Late-Phase-DR-Finetune"
  base_cfg = load_env_cfg(base_task_id)
  cfg = load_env_cfg(task_id)

  base_motion_cmd = base_cfg.commands["motion"]
  motion_cmd = cfg.commands["motion"]
  assert isinstance(base_motion_cmd, MotionCommandCfg)
  assert isinstance(motion_cmd, MotionCommandCfg)
  assert motion_cmd.sampling_start_frame is None
  assert motion_cmd.sampling_end_frame is None
  assert motion_cmd.sampling_mode == "adaptive"
  assert cfg.scene.terrain is not None
  assert cfg.scene.extent == 2.0
  assert "staged_terrain_sampling" in cfg.events
  assert "late_phase_dr_disturbance" in cfg.events
  assert cfg.events["base_com"].params["ranges"] == base_cfg.events["base_com"].params["ranges"]
  assert cfg.events["encoder_bias"].params["bias_range"] == (
    base_cfg.events["encoder_bias"].params["bias_range"]
  )
  assert cfg.events["foot_friction"].params["ranges"] == (
    base_cfg.events["foot_friction"].params["ranges"]
  )
  assert "anchor_pos" not in cfg.terminations
  assert "ee_body_pos" not in cfg.terminations
  assert cfg.terminations["anchor_ori"].params["threshold"] == 1.2
  assert cfg.rewards["motion_joint_pos"].weight == 0.25
  assert cfg.rewards["motion_joint_vel"].weight == 0.1


def test_play_can_inject_late_phase_aggressive_dr_into_full_motion_task() -> None:
  """Play helper should add late-phase disturbances without changing full-motion start."""
  task_id = "Mjlab-Tracking-Flat-Unitree-G1-New"
  cfg = load_env_cfg(task_id, play=True)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  assert motion_cmd.sampling_mode == "start"
  assert motion_cmd.sampling_start_frame is None
  assert motion_cmd.sampling_end_frame is None

  _enable_tracking_late_phase_play_dr(cfg)

  assert "late_phase_play_disturbance" in cfg.events
  assert cfg.events["late_phase_play_disturbance"].mode == "step"
  assert cfg.events["late_phase_play_disturbance"].params["late_phase_start_ratio"] == (
    LATE_PHASE_DR_START_RATIO
  )
  assert cfg.events["late_phase_play_disturbance"].params["late_phase_onset_scale"] == 0.35
  assert cfg.events["late_phase_play_disturbance"].params["stand_up_push_center_frame"] == 150
  assert cfg.events["late_phase_play_disturbance"].params["stand_up_push_post_trigger_frame"] == 165
  assert cfg.events["late_phase_play_disturbance"].params["stand_up_overshoot_trigger_frame"] == 165
  assert cfg.events["late_phase_play_disturbance"].params["stand_up_overshoot_half_window"] == 2
  assert cfg.events["late_phase_play_disturbance"].params["stand_up_recovery_probability"] == 1.0
  assert cfg.events["late_phase_play_disturbance"].params["stand_up_underpowered_probability"] == 0.5
  assert cfg.events["late_phase_play_disturbance"].params["stand_up_underpowered_trigger_frame"] == 135
  assert cfg.events["late_phase_play_disturbance"].params["stand_up_underpowered_half_window"] == 2
  assert cfg.events["late_phase_play_disturbance"].params["pose_range"]["yaw"] == (0.0, 0.0)
  assert cfg.events["late_phase_play_disturbance"].params["velocity_range"]["x"] == (0.0, 0.0)
  assert cfg.events["late_phase_play_disturbance"].params["joint_velocity_range"] == (0.0, 0.0)
  assert cfg.events["late_phase_play_disturbance"].params["force_range"] == (0.0, 0.0)
  assert cfg.events["late_phase_play_disturbance"].params["stand_up_push_forward_force_range"] == (
    0.0,
    0.0,
  )
  assert cfg.events["late_phase_play_disturbance"].params["stand_up_push_lateral_force_range"] == (
    0.0,
    0.0,
  )
  assert cfg.events["late_phase_play_disturbance"].params["stand_up_overshoot_effort_scale_range"] == (
    1.15,
    1.27,
  )
  assert cfg.events["late_phase_play_disturbance"].params["stand_up_overshoot_pd_scale_range"] == (
    1.06,
    1.15,
  )
  assert cfg.events["late_phase_play_disturbance"].params["stand_up_overshoot_action_scale_range"] == (
    1.225,
    1.42,
  )
  assert cfg.events["late_phase_play_disturbance"].params["stand_up_overshoot_pitch_ang_vel_range"] == (
    1.05,
    1.65,
  )
  assert cfg.events["late_phase_play_disturbance"].params["stand_up_underpowered_effort_scale_range"] == (
    0.73,
    0.88,
  )
  assert cfg.events["late_phase_play_disturbance"].params["stand_up_underpowered_pd_scale_range"] == (
    0.76,
    0.88,
  )
  assert cfg.events["late_phase_play_disturbance"].params["stand_up_underpowered_action_scale_range"] == (
    0.58,
    0.82,
  )
  assert cfg.events["late_phase_play_disturbance"].params["stand_up_underpowered_pitch_ang_vel_range"] == (
    -1.95,
    -1.35,
  )
  assert cfg.events["late_phase_play_disturbance"].params["log_stand_up_recovery_disturbance"] is True
  assert cfg.events["late_phase_play_disturbance"].params["stand_up_pitch_ang_vel_kick_duration_steps"] == 3
  assert cfg.events["late_phase_play_disturbance"].params["stand_up_velocity_kick_forward_range"] == (
    0.0,
    0.0,
  )
  assert cfg.events["late_phase_play_disturbance"].params["stand_up_push_duration_s"] == (0.0, 0.0)


def test_play_can_scale_overshoot_and_underpowered_independently() -> None:
  """Play helper should allow separate scale knobs for overshoot and underpowered."""
  task_id = "Mjlab-Tracking-Flat-Unitree-G1-New"
  cfg = load_env_cfg(task_id, play=True)

  _enable_tracking_late_phase_play_dr(
    cfg,
    overshoot_scale=3.0,
    underpowered_scale=0.75,
  )

  params = cfg.events["late_phase_play_disturbance"].params
  assert params["stand_up_overshoot_pitch_ang_vel_range"] == (2.1, 3.3)
  assert params["stand_up_overshoot_action_scale_range"] == (1.45, 1.84)
  assert params["stand_up_underpowered_pitch_ang_vel_range"] == (-0.975, -0.675)
  assert params["stand_up_underpowered_action_scale_range"] == (0.79, 0.91)


def test_train_and_play_late_phase_scales_match() -> None:
  """Train and play should map the same scale values to the same disturbance params."""
  train_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1-New-Late-Phase-DR-Finetune")
  play_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1-New", play=True)

  _enable_tracking_late_phase_train_dr(
    train_cfg,
    overshoot_scale=7.0,
    underpowered_scale=3.0,
  )
  _enable_tracking_late_phase_play_dr(
    play_cfg,
    overshoot_scale=7.0,
    underpowered_scale=3.0,
  )

  train_params = train_cfg.events["late_phase_dr_disturbance"].params
  play_params = play_cfg.events["late_phase_play_disturbance"].params
  for key in (
    "stand_up_push_forward_force_range",
    "stand_up_push_lateral_force_range",
    "stand_up_overshoot_effort_scale_range",
    "stand_up_overshoot_pd_scale_range",
    "stand_up_overshoot_action_scale_range",
    "stand_up_overshoot_pitch_ang_vel_range",
    "stand_up_underpowered_effort_scale_range",
    "stand_up_underpowered_pd_scale_range",
    "stand_up_underpowered_action_scale_range",
    "stand_up_underpowered_pitch_ang_vel_range",
  ):
    assert train_params[key] == play_params[key]


def test_late_phase_dr_finetune_runner_is_more_conservative() -> None:
  """Late-phase finetune PPO config should be tighter than the base tracking config."""
  for task_id in (
    "Mjlab-Tracking-Flat-Unitree-G1-New-Late-Phase-DR-Finetune",
    "Mjlab-Tracking-Rough-Unitree-G1-New-Late-Phase-DR-Finetune",
  ):
    cfg = load_rl_cfg(task_id)
    assert cfg.algorithm.learning_rate == 1.0e-4
    assert cfg.algorithm.entropy_coef == 0.001
    assert cfg.algorithm.desired_kl == 0.003


def test_late_phase_dr_finetune_play_keeps_full_motion_without_extra_disturbance() -> None:
  """Late-phase-DR play config should default to clean full motion."""
  base_task_id = "Mjlab-Tracking-Flat-Unitree-G1-New"
  task_id = "Mjlab-Tracking-Flat-Unitree-G1-New-Late-Phase-DR-Finetune"
  base_cfg = load_env_cfg(base_task_id, play=True)
  cfg = load_env_cfg(task_id, play=True)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  assert motion_cmd.sampling_mode == "start"
  assert motion_cmd.sampling_start_frame is None
  assert motion_cmd.sampling_end_frame is None
  assert "late_phase_dr_disturbance" not in cfg.events
  assert cfg.events["base_com"].params["ranges"] == base_cfg.events["base_com"].params["ranges"]
  assert cfg.events["encoder_bias"].params["bias_range"] == base_cfg.events["encoder_bias"].params["bias_range"]
  assert cfg.events["foot_friction"].params["ranges"] == base_cfg.events["foot_friction"].params["ranges"]


def test_rough_late_phase_dr_finetune_play_keeps_staged_terrain_without_extra_disturbance() -> None:
  """Rough late-phase play should stay clean while preserving rough-terrain staging."""
  base_task_id = "Mjlab-Tracking-Rough-Unitree-G1-New"
  task_id = "Mjlab-Tracking-Rough-Unitree-G1-New-Late-Phase-DR-Finetune"
  base_cfg = load_env_cfg(base_task_id, play=True)
  cfg = load_env_cfg(task_id, play=True)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  assert motion_cmd.sampling_mode == "start"
  assert motion_cmd.sampling_start_frame is None
  assert motion_cmd.sampling_end_frame is None
  assert "late_phase_dr_disturbance" not in cfg.events
  assert "staged_terrain_sampling" in cfg.events
  assert cfg.scene.terrain is not None
  assert cfg.events["base_com"].params["ranges"] == base_cfg.events["base_com"].params["ranges"]
  assert cfg.events["encoder_bias"].params["bias_range"] == base_cfg.events["encoder_bias"].params["bias_range"]
  assert cfg.events["foot_friction"].params["ranges"] == base_cfg.events["foot_friction"].params["ranges"]


def test_g1_tracking_has_correct_action_scale(g1_tracking_task_ids: list[str]) -> None:
  """G1 tracking tasks should use the action scale of their robot variant."""
  for task_id in g1_tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    expected_scale = G1_NEW_ACTION_SCALE if "-G1-New" in task_id else G1_ACTION_SCALE
    assert joint_pos_action.scale == expected_scale, (
      f"Task {task_id} action scale mismatch for its robot variant"
    )
