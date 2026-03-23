"""Tests for debug robot inventory and control-chain helpers."""

from unittest.mock import Mock

import pytest
import torch
from conftest import create_entity_from_fixture, get_test_device, initialize_entity

from mjlab.actuator.builtin_actuator import (
  BuiltinMotorActuatorCfg,
  BuiltinPositionActuatorCfg,
)
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.scripts.debug_robot import (
  HoldReferencePolicy,
  ManualDeltaPolicy,
  RobotDebugSession,
  apply_pose_delta,
  build_robot_mode_summary,
  build_task_mode_summary,
  build_manual_delta_policy,
)
from mjlab.viewer.viser.debug_panels import (
  build_actuator_inventory_from_cfg,
  build_control_chain_rows,
  build_joint_inventory,
)


@pytest.fixture
def device():
  return get_test_device()


def make_env(entity, name: str, device: str):
  """Create a minimal mock env for action-term construction."""
  env = Mock(spec=ManagerBasedRlEnv)
  env.num_envs = 1
  env.device = device
  env.scene = {name: entity}
  return env


def make_robot_session(device: str) -> RobotDebugSession:
  """Create a small initialized robot session for pose-browser tests."""
  entity = create_entity_from_fixture(
    "floating_base_articulated",
    actuator_cfg=BuiltinMotorActuatorCfg(
      target_names_expr=("joint.*",),
      effort_limit=10.0,
    ),
  )
  entity, sim = initialize_entity(entity, device)
  return RobotDebugSession(
    robot_name="fixture",
    scene=None,
    sim=sim,
    entity=entity,
    device=device,
  )


def test_build_actuator_inventory_from_g1_cfg() -> None:
  """G1 inventory rows should reflect grouped position actuators."""
  rows = build_actuator_inventory_from_cfg(get_g1_robot_cfg())

  assert len(rows) == 6
  assert any("ankle" in row.group_name.lower() for row in rows)
  assert all(row.control_type == "position" for row in rows)


def test_build_joint_inventory_matches_joint_order(device: str) -> None:
  """Joint inventory should preserve entity joint ordering."""
  entity = create_entity_from_fixture(
    "floating_base_articulated",
    actuator_cfg=BuiltinMotorActuatorCfg(
      target_names_expr=("joint.*",),
      effort_limit=10.0,
    ),
  )
  entity, _ = initialize_entity(entity, device)

  rows = build_joint_inventory(entity)

  assert [row.joint_name for row in rows] == list(entity.joint_names)
  assert all(row.actuator_group is not None for row in rows)


def test_build_control_chain_rows_aligns_action_and_joint_targets(device: str) -> None:
  """Control-chain rows should align action slices with joint/runtime fields."""
  entity = create_entity_from_fixture(
    "floating_base_articulated",
    actuator_cfg=BuiltinPositionActuatorCfg(
      target_names_expr=("joint.*",),
      stiffness=10.0,
      damping=0.5,
      effort_limit=20.0,
    ),
  )
  entity, sim = initialize_entity(entity, device)
  env = make_env(entity, "robot", device)
  action = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=("joint.*",),
    scale=2.0,
    offset=0.5,
    use_default_offset=False,
  ).build(env)

  raw = torch.tensor([[0.1, -0.2]], device=device)
  action.process_actions(raw)
  action.apply_actions()
  entity.write_data_to_sim()
  sim.forward()

  rows = build_control_chain_rows(entity, action, env_idx=0)

  assert [row.joint_name for row in rows] == list(action.target_names)
  assert rows[0].raw_action == pytest.approx(0.1)
  assert rows[0].processed_action == pytest.approx(0.7)
  assert rows[0].q_des == pytest.approx(0.7)
  assert rows[0].actuator_force is not None


def test_apply_pose_delta_updates_only_selected_joint(device: str) -> None:
  """Pose browser should move only the selected joint away from default."""
  session = make_robot_session(device)

  result = apply_pose_delta(session, joint_index=1, delta=0.25, clamp=True)

  assert result.joint_name == session.entity.joint_names[1]
  assert session.entity.data.joint_pos[0, 0].item() == pytest.approx(
    session.entity.data.default_joint_pos[0, 0].item()
  )
  assert session.entity.data.joint_pos[0, 1].item() == pytest.approx(result.q)


def test_apply_pose_delta_clamps_to_joint_limits(device: str) -> None:
  """Large deltas should clamp to the selected joint's kinematic limits."""
  session = make_robot_session(device)

  result = apply_pose_delta(session, joint_index=0, delta=10.0, clamp=True)

  low, high = result.joint_limit
  assert low <= result.q <= high
  assert result.q == pytest.approx(high)


def test_apply_pose_delta_zeros_selected_joint_velocity(device: str) -> None:
  """Pose browser should clear joint velocity before forwarding the new pose."""
  session = make_robot_session(device)
  session.entity.write_joint_velocity_to_sim(
    torch.tensor([[1.0, -2.0]], device=device)
  )
  session.sim.forward()

  apply_pose_delta(session, joint_index=1, delta=0.1, clamp=True)

  assert session.entity.data.joint_vel[0, 1].item() == pytest.approx(0.0)


def test_hold_reference_policy_returns_zero_actions(device: str) -> None:
  """Hold-reference should emit zero raw actions for every environment."""
  policy = HoldReferencePolicy(action_dim=4, device=device)

  action = policy(torch.ones((2, 3), device=device))

  assert action.shape == (2, 4)
  assert torch.count_nonzero(action).item() == 0


def test_manual_delta_policy_only_changes_selected_joint(device: str) -> None:
  """Manual-delta policy should only edit one action dimension."""
  policy = ManualDeltaPolicy(
    action_dim=6,
    action_index=2,
    raw_delta=0.1,
    device=device,
  )

  action = policy(torch.zeros((1, 1), device=device))

  assert torch.count_nonzero(action).item() == 1
  assert action[0, 2].item() == pytest.approx(0.1)


def test_build_manual_delta_policy_clamps_joint_delta(device: str) -> None:
  """Joint-space manual deltas should clamp before conversion to raw action."""
  entity = create_entity_from_fixture(
    "floating_base_articulated",
    actuator_cfg=BuiltinPositionActuatorCfg(
      target_names_expr=("joint.*",),
      stiffness=10.0,
      damping=0.5,
      effort_limit=20.0,
    ),
  )
  entity, _ = initialize_entity(entity, device)
  env = make_env(entity, "robot", device)
  action_term = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=("joint.*",),
    scale=2.0,
    offset=0.5,
    use_default_offset=False,
  ).build(env)

  policy = build_manual_delta_policy(
    action_term,
    selected_joint_index=1,
    joint_delta=1.0,
    joint_delta_limit=0.25,
    device=device,
  )
  action = policy(torch.zeros((1, 1), device=device))

  assert torch.count_nonzero(action).item() == 1
  assert action[0, 1].item() == pytest.approx(0.125)


def test_mode_summaries_explain_position_control() -> None:
  """UI summaries should explain the teaching path clearly."""
  robot_summary = build_robot_mode_summary()
  task_summary = build_task_mode_summary("manual-delta", "zero")

  assert "pose browser" in robot_summary
  assert "not a balance controller" in robot_summary
  assert "position targets" in task_summary
  assert "motor effort" in task_summary
