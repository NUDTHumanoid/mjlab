from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from mjlab.tasks.tracking.mdp.metrics import (
  compute_anchor_height_error,
  compute_anchor_planar_position_error,
  compute_mean_body_height_error,
)
from mjlab.tasks.tracking.mdp.rewards import (
  motion_global_anchor_planar_position_error_exp,
  motion_relative_body_height_error_exp,
)
from mjlab.tasks.tracking.mdp.terminations import (
  bad_anchor_pos_xy_only,
  bad_motion_body_mean_pos,
  bad_motion_body_mean_pos_z_only,
)


class _FakeCommandManager:
  def __init__(self, command: object) -> None:
    self._command = command

  def get_term(self, _name: str) -> object:
    return self._command


class _FakeEnv:
  def __init__(self, command: object) -> None:
    self.command_manager = _FakeCommandManager(command)


def _make_command() -> Any:
  return SimpleNamespace(
    anchor_pos_w=torch.tensor(
      [
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 2.0],
      ],
      dtype=torch.float32,
    ),
    robot_anchor_pos_w=torch.tensor(
      [
        [2.0, 4.0, 6.0],
        [5.0, 5.0, 3.0],
      ],
      dtype=torch.float32,
    ),
    body_pos_relative_w=torch.tensor(
      [
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [3.0, 4.0, 5.0]],
        [[2.0, 3.0, 4.0], [3.0, 5.0, 9.0], [4.0, 4.0, 4.5]],
      ],
      dtype=torch.float32,
    ),
    robot_body_pos_w=torch.tensor(
      [
        [[0.0, 0.0, 0.5], [1.0, 2.0, 2.0], [1.0, 1.0, 1.0]],
        [[2.0, 1.0, 1.0], [3.0, 1.0, 3.0], [4.0, 4.0, 3.0]],
      ],
      dtype=torch.float32,
    ),
    cfg=SimpleNamespace(body_names=("torso", "hand", "foot")),
    num_envs=2,
    device=torch.device("cpu"),
  )


def test_anchor_control_metrics_split_planar_vs_height_error() -> None:
  command: Any = _make_command()

  planar_error = compute_anchor_planar_position_error(command)
  height_error = compute_anchor_height_error(command)

  assert torch.allclose(
    planar_error,
    torch.tensor([5.0**0.5, 5.0], dtype=torch.float32),
  )
  assert torch.allclose(height_error, torch.tensor([3.0, 1.0], dtype=torch.float32))


def test_mean_body_height_metric_can_scope_to_selected_bodies() -> None:
  command: Any = _make_command()

  all_body_error = compute_mean_body_height_error(command)
  hand_only_error = compute_mean_body_height_error(command, ("hand",))

  assert torch.allclose(
    all_body_error,
    torch.tensor([11.0 / 6.0, 3.5], dtype=torch.float32),
  )
  assert torch.allclose(hand_only_error, torch.tensor([1.0, 6.0], dtype=torch.float32))


def test_env_contract_control_terminations_distinguish_any_vs_mean_thresholds() -> None:
  env: Any = _FakeEnv(_make_command())

  planar_failure = bad_anchor_pos_xy_only(env, "motion", threshold=3.0)
  body_mean_failure = bad_motion_body_mean_pos(env, "motion", threshold=2.4)
  body_mean_height_failure = bad_motion_body_mean_pos_z_only(
    env, "motion", threshold=2.0
  )

  assert torch.equal(planar_failure, torch.tensor([False, True]))
  assert torch.equal(body_mean_failure, torch.tensor([True, True]))
  assert torch.equal(body_mean_height_failure, torch.tensor([False, True]))


def test_env_contract_control_rewards_provide_smooth_planar_and_height_signals() -> (
  None
):
  env: Any = _FakeEnv(_make_command())

  planar_reward = motion_global_anchor_planar_position_error_exp(env, "motion", std=2.0)
  height_reward = motion_relative_body_height_error_exp(env, "motion", std=2.0)

  expected_planar = torch.exp(-torch.tensor([5.0 / 4.0, 25.0 / 4.0]))
  expected_height = torch.exp(-torch.tensor([5.75 / 4.0, 15.75 / 4.0]))

  assert torch.allclose(planar_reward, expected_planar)
  assert torch.allclose(height_reward, expected_height)
