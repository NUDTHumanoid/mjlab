"""Tests specific to motion tracking tasks."""

import pytest

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
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
    "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
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
    "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
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


def test_g1_tracking_has_correct_action_scale(g1_tracking_task_ids: list[str]) -> None:
  """G1 tracking tasks should use G1_ACTION_SCALE."""
  for task_id in g1_tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == G1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected G1_ACTION_SCALE"
    )


def test_crouch_to_lie_down_task_is_registered() -> None:
  """The crouch-to-lie-down tracking task should be registered."""
  task_id = "Mjlab-Tracking-Flat-Unitree-G1-CrouchToLieDown"

  assert task_id in list_tasks(), f"Task {task_id} is not registered"


def test_crouch_to_lie_down_training_motion_and_episode_length() -> None:
  """Crouch-to-lie-down training config should be one-shot and RSI-free."""
  task_id = "Mjlab-Tracking-Flat-Unitree-G1-CrouchToLieDown"
  cfg = load_env_cfg(task_id)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg), (
    f"Task {task_id} training motion command is not MotionCommandCfg"
  )
  assert motion_cmd.sampling_mode == "start", (
    f"Task {task_id} training sampling_mode={motion_cmd.sampling_mode}, expected 'start'"
  )
  assert motion_cmd.pose_range == {}, (
    f"Task {task_id} training pose_range={motion_cmd.pose_range}, expected empty dict"
  )
  assert motion_cmd.velocity_range == {}, (
    f"Task {task_id} training velocity_range={motion_cmd.velocity_range}, expected empty dict"
  )
  assert motion_cmd.joint_position_range == (0.0, 0.0), (
    f"Task {task_id} training joint_position_range={motion_cmd.joint_position_range}, "
    "expected (0.0, 0.0)"
  )
  assert cfg.episode_length_s == 6.5, (
    f"Task {task_id} training episode_length_s={cfg.episode_length_s}, expected 6.5"
  )


def test_crouch_to_lie_down_play_motion_uses_start_sampling_and_no_rsi() -> None:
  """Crouch-to-lie-down play config should also start from the first frame."""
  task_id = "Mjlab-Tracking-Flat-Unitree-G1-CrouchToLieDown"
  cfg = load_env_cfg(task_id, play=True)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg), (
    f"Task {task_id} play motion command is not MotionCommandCfg"
  )
  assert motion_cmd.sampling_mode == "start", (
    f"Task {task_id} play sampling_mode={motion_cmd.sampling_mode}, expected 'start'"
  )
  assert motion_cmd.pose_range == {}, (
    f"Task {task_id} play pose_range={motion_cmd.pose_range}, expected empty dict"
  )
  assert motion_cmd.velocity_range == {}, (
    f"Task {task_id} play velocity_range={motion_cmd.velocity_range}, expected empty dict"
  )
  assert motion_cmd.joint_position_range == (0.0, 0.0), (
    f"Task {task_id} play joint_position_range={motion_cmd.joint_position_range}, "
    "expected (0.0, 0.0)"
  )
  assert cfg.episode_length_s == int(1e9), (
    f"Task {task_id} play episode_length_s={cfg.episode_length_s}, expected infinite horizon"
  )
  assert not cfg.observations["actor"].enable_corruption, (
    f"Task {task_id} play actor corruption is enabled, expected disabled"
  )
  assert "push_robot" not in cfg.events, (
    f"Task {task_id} play config still has push_robot event"
  )


def test_crouch_to_lie_down_loading_does_not_pollute_baseline_cfg() -> None:
  """Loading the crouch-to-lie-down variant should not mutate the baseline config."""
  baseline_task_id = "Mjlab-Tracking-Flat-Unitree-G1"
  crouch_task_id = "Mjlab-Tracking-Flat-Unitree-G1-CrouchToLieDown"

  baseline_before = load_env_cfg(baseline_task_id)
  baseline_before_motion = baseline_before.commands["motion"]
  assert isinstance(baseline_before_motion, MotionCommandCfg)

  load_env_cfg(crouch_task_id)

  baseline_after = load_env_cfg(baseline_task_id)
  baseline_after_motion = baseline_after.commands["motion"]
  assert isinstance(baseline_after_motion, MotionCommandCfg)

  assert baseline_after.episode_length_s == baseline_before.episode_length_s, (
    "Baseline training episode length changed after loading crouch-to-lie-down variant"
  )
  assert baseline_after_motion.sampling_mode == baseline_before_motion.sampling_mode, (
    "Baseline motion sampling mode changed after loading crouch-to-lie-down variant"
  )
  assert baseline_after_motion.pose_range == baseline_before_motion.pose_range, (
    "Baseline motion pose_range changed after loading crouch-to-lie-down variant"
  )
  assert baseline_after_motion.velocity_range == baseline_before_motion.velocity_range, (
    "Baseline motion velocity_range changed after loading crouch-to-lie-down variant"
  )
  assert (
    baseline_after_motion.joint_position_range
    == baseline_before_motion.joint_position_range
  ), "Baseline motion joint_position_range changed after loading crouch-to-lie-down variant"
  assert baseline_after.terminations["anchor_pos"].params["threshold"] == baseline_before.terminations["anchor_pos"].params["threshold"], (
    "Baseline anchor_pos threshold changed after loading crouch-to-lie-down variant"
  )
  assert baseline_after.terminations["ee_body_pos"].params["threshold"] == baseline_before.terminations["ee_body_pos"].params["threshold"], (
    "Baseline ee_body_pos threshold changed after loading crouch-to-lie-down variant"
  )
  assert baseline_after.terminations["anchor_ori"].params["threshold"] == baseline_before.terminations["anchor_ori"].params["threshold"], (
    "Baseline anchor_ori threshold changed after loading crouch-to-lie-down variant"
  )


def test_crouch_to_lie_down_increases_contact_capacity() -> None:
  """Crouch-to-lie-down should increase contact capacity relative to baseline."""
  baseline_task_id = "Mjlab-Tracking-Flat-Unitree-G1"
  crouch_task_id = "Mjlab-Tracking-Flat-Unitree-G1-CrouchToLieDown"

  baseline_cfg = load_env_cfg(baseline_task_id)
  crouch_cfg = load_env_cfg(crouch_task_id)

  assert crouch_cfg.sim.nconmax == 55, (
    f"Task {crouch_task_id} sim.nconmax={crouch_cfg.sim.nconmax}, expected 55"
  )
  assert crouch_cfg.sim.nconmax > baseline_cfg.sim.nconmax, (
    f"Task {crouch_task_id} sim.nconmax={crouch_cfg.sim.nconmax} is not greater than "
    f"baseline nconmax={baseline_cfg.sim.nconmax}"
  )


def test_crouch_to_lie_down_rl_cfg_differs_from_baseline() -> None:
  """Crouch-to-lie-down RL config should run shorter with a longer horizon."""
  baseline_task_id = "Mjlab-Tracking-Flat-Unitree-G1"
  task_id = "Mjlab-Tracking-Flat-Unitree-G1-CrouchToLieDown"

  baseline_cfg = load_rl_cfg(baseline_task_id)
  cfg = load_rl_cfg(task_id)

  assert cfg.experiment_name == "g1_tracking_crouch_to_lie_down", (
    f"Task {task_id} experiment_name={cfg.experiment_name}, "
    "expected g1_tracking_crouch_to_lie_down"
  )
  assert cfg.num_steps_per_env == 64, (
    f"Task {task_id} num_steps_per_env={cfg.num_steps_per_env}, expected 64"
  )
  assert cfg.max_iterations == 20_000, (
    f"Task {task_id} max_iterations={cfg.max_iterations}, expected 20000"
  )
  assert cfg.num_steps_per_env > baseline_cfg.num_steps_per_env, (
    f"Task {task_id} num_steps_per_env={cfg.num_steps_per_env} "
    f"is not longer than baseline {baseline_cfg.num_steps_per_env}"
  )
  assert cfg.max_iterations < baseline_cfg.max_iterations, (
    f"Task {task_id} max_iterations={cfg.max_iterations} "
    f"is not shorter than baseline {baseline_cfg.max_iterations}"
  )
