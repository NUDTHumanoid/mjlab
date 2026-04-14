"""Tests for raw-motion reporting and preview orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mjlab.scripts import report_and_preview_motion as report_script


def _write_raw_motion(path: Path) -> None:
  np.savez(
    path,
    trans=np.zeros((4, 3), dtype=np.float32),
    poses=np.zeros((4, 156), dtype=np.float32),
    betas=np.zeros((16,), dtype=np.float32),
    dmpls=np.zeros((4, 8), dtype=np.float32),
    mocap_framerate=np.array(120.0, dtype=np.float32),
    gender=np.array("female"),
  )


def test_summarize_raw_motion_reports_expected_fields(tmp_path: Path) -> None:
  raw_path = tmp_path / "handstand1.npz"
  _write_raw_motion(raw_path)

  report = report_script.summarize_raw_motion(raw_path)

  assert report["compatible_with_mjlab_raw_pipeline"] is True
  assert report["frame_count"] == 4
  assert report["mocap_fps"] == 120.0
  assert report["pose_dimension"] == 156
  assert report["betas_dim"] == 16
  assert report["has_dmpls"] is True


def test_report_and_preview_motion_writes_report_only(tmp_path: Path) -> None:
  raw_path = tmp_path / "handstand1.npz"
  _write_raw_motion(raw_path)

  result = report_script.report_and_preview_motion(
    input_file=raw_path,
    work_dir=tmp_path / "report",
  )

  report_path = Path(result["report_path"])
  assert report_path.is_file()
  report = json.loads(report_path.read_text(encoding="utf-8"))
  assert report["motion_name"] == "handstand1"
  assert report["requested_actions"]["build"] is False
  assert "build" not in report


def test_report_and_preview_motion_builds_and_previews_when_requested(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  raw_path = tmp_path / "handstand1.npz"
  _write_raw_motion(raw_path)
  built_motion = tmp_path / "built" / "motion.npz"
  built_motion.parent.mkdir(parents=True, exist_ok=True)
  built_motion.write_bytes(b"npz")
  recorded_preview: dict[str, object] = {}

  def fake_build_tracking_motion(**kwargs):
    return {
      "work_dir": tmp_path / "built",
      "raw_copy_path": tmp_path / "built" / "raw.npz",
      "keypoints_path": tmp_path / "built" / "keypoints.npy",
      "retargeted_motion_path": tmp_path / "built" / "retargeted.npz",
      "csv_path": tmp_path / "built" / "motion.csv",
      "motion_npz_path": built_motion,
      "manifest_path": tmp_path / "built" / "manifest.json",
    }

  def fake_run_play(task_id, cfg):
    recorded_preview["task_id"] = task_id
    recorded_preview["motion_file"] = cfg.motion_file
    recorded_preview["viewer"] = cfg.viewer
    recorded_preview["agent"] = cfg.agent

  monkeypatch.setattr(report_script, "build_tracking_motion", fake_build_tracking_motion)
  monkeypatch.setattr(report_script, "run_play", fake_run_play)

  result = report_script.report_and_preview_motion(
    input_file=raw_path,
    work_dir=tmp_path / "report",
    protomotions_root=tmp_path / "ProtoMotions",
    build=True,
    preview=True,
    viewer="viser",
  )

  report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
  assert report["build"]["artifacts"]["motion_npz_path"] == str(built_motion)
  assert recorded_preview["task_id"] == "Mjlab-Tracking-Flat-Unitree-G1"
  assert recorded_preview["motion_file"] == str(built_motion)
  assert recorded_preview["viewer"] == "viser"
  assert recorded_preview["agent"] == "zero"


def test_report_and_preview_motion_resolves_local_defaults(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  raw_path = tmp_path / "handstand1.npz"
  _write_raw_motion(raw_path)
  proto_root = tmp_path / ".external" / "ProtoMotions"
  proto_root.mkdir(parents=True)
  pyroki_python = tmp_path / "fake-pyroki" / "bin" / "python"
  pyroki_python.parent.mkdir(parents=True)
  pyroki_python.write_text("", encoding="utf-8")
  recorded_build_kwargs: dict[str, object] = {}

  def fake_build_tracking_motion(**kwargs):
    recorded_build_kwargs.update(kwargs)
    built_motion = tmp_path / "built" / "motion.npz"
    built_motion.parent.mkdir(parents=True, exist_ok=True)
    built_motion.write_bytes(b"npz")
    return {
      "work_dir": tmp_path / "built",
      "raw_copy_path": tmp_path / "built" / "raw.npz",
      "keypoints_path": tmp_path / "built" / "keypoints.npy",
      "retargeted_motion_path": tmp_path / "built" / "retargeted.npz",
      "csv_path": tmp_path / "built" / "motion.csv",
      "motion_npz_path": built_motion,
      "manifest_path": tmp_path / "built" / "manifest.json",
    }

  monkeypatch.chdir(tmp_path)
  monkeypatch.setattr(report_script, "build_tracking_motion", fake_build_tracking_motion)
  monkeypatch.setattr(report_script.Path, "home", lambda: tmp_path)

  result = report_script.report_and_preview_motion(
    input_file=raw_path,
    build=True,
    quality="quick",
  )

  report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
  assert report["resolved_protomotions_root"] == str(proto_root.resolve())
  assert recorded_build_kwargs["protomotions_root"] == proto_root.resolve()
  assert recorded_build_kwargs["retarget_subsample_factor"] == 2
  assert recorded_build_kwargs["retarget_target_raw_frames"] == 120


def test_report_and_preview_motion_requires_protomotions_for_build(tmp_path: Path) -> None:
  raw_path = tmp_path / "handstand1.npz"
  _write_raw_motion(raw_path)

  original_default = report_script._default_protomotions_root
  report_script._default_protomotions_root = lambda: None
  with pytest.raises(ValueError, match="protomotions_root"):
    report_script.report_and_preview_motion(
      input_file=raw_path,
      build=True,
    )
  report_script._default_protomotions_root = original_default
