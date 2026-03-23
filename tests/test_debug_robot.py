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
