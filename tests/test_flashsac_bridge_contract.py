from __future__ import annotations

import torch

from mjlab.flashsac.runtime import apply_tracking_evaluation_overrides
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.scripts.evaluate import (
  _reduce_ee_metric_traces,
  _reduce_metric_traces,
)


def test_tracking_evaluation_overrides_preserve_canonical_terminations() -> None:
  env_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=False)
  original_anchor_pos = float(env_cfg.terminations["anchor_pos"].params["threshold"])
  original_anchor_ori = float(env_cfg.terminations["anchor_ori"].params["threshold"])

  apply_tracking_evaluation_overrides(env_cfg)

  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  assert motion_cmd.sampling_mode == "start"
  assert motion_cmd.pose_range == {}
  assert motion_cmd.velocity_range == {}
  assert motion_cmd.joint_position_range == (0.0, 0.0)
  assert env_cfg.observations["actor"].enable_corruption is False
  assert env_cfg.observations["critic"].enable_corruption is False
  assert "push_robot" not in env_cfg.events
  assert all(cfg.mode != "startup" for cfg in env_cfg.events.values())
  assert "ee_body_pos" in env_cfg.terminations
  assert (
    float(env_cfg.terminations["anchor_pos"].params["threshold"]) == original_anchor_pos
  )
  assert (
    float(env_cfg.terminations["anchor_ori"].params["threshold"]) == original_anchor_ori
  )


def test_reduce_metric_traces_counts_zero_error_steps() -> None:
  metric_traces = [
    [
      torch.tensor([0.0, 2.0]),
      torch.tensor([4.0, 0.0]),
    ],
    [
      torch.tensor([0.0, 6.0]),
      torch.tensor([8.0, 0.0]),
    ],
  ]
  active_masks = [
    torch.tensor([True, True]),
    torch.tensor([True, True]),
  ]

  means = _reduce_metric_traces(metric_traces, active_masks)

  assert torch.allclose(means[0], torch.tensor([2.0, 1.0]))
  assert torch.allclose(means[1], torch.tensor([4.0, 3.0]))


def test_reduce_ee_metric_traces_counts_active_zero_error_steps() -> None:
  ee_pos_traces = [
    torch.tensor([0.0, 2.0]),
    torch.tensor([4.0, 0.0]),
  ]
  ee_ori_traces = [
    torch.tensor([1.0, 0.0]),
    torch.tensor([3.0, 4.0]),
  ]
  active_masks = [
    torch.tensor([True, True]),
    torch.tensor([True, True]),
  ]

  ee_pos_mean, ee_ori_mean = _reduce_ee_metric_traces(
    ee_pos_traces,
    ee_ori_traces,
    active_masks,
  )

  assert torch.allclose(ee_pos_mean, torch.tensor([2.0, 1.0]))
  assert torch.allclose(ee_ori_mean, torch.tensor([2.0, 2.0]))
