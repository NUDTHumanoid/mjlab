"""Tests for the end-to-end tracking motion builder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mjlab.scripts import build_tracking_motion as build_script


def _write_raw_motion(path: Path) -> None:
  np.savez(
    path,
    trans=np.zeros((4, 3), dtype=np.float32),
    poses=np.zeros((4, 72), dtype=np.float32),
    betas=np.zeros((10,), dtype=np.float32),
    mocap_framerate=np.array(60.0, dtype=np.float32),
  )


def test_build_tracking_motion_writes_manifest_and_artifacts(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  raw_path = tmp_path / "raw_motion.npz"
  _write_raw_motion(raw_path)
  work_dir = tmp_path / "work"

  def fake_extract(**kwargs):
    output_dir = Path(kwargs["output_dir"])
    keypoints_path = output_dir / "raw_motion_keypoints.npy"
    manifest_path = output_dir / "raw_motion_manifest.json"
    motion_path = output_dir / "raw_motion.motion"
    np.save(
      keypoints_path,
      {
        "positions": np.zeros((4, 18, 3), dtype=np.float32),
        "orientations": np.tile(np.eye(3, dtype=np.float32), (4, 18, 1, 1)),
        "left_foot_contacts": np.zeros((4, 2), dtype=np.int32),
        "right_foot_contacts": np.zeros((4, 2), dtype=np.int32),
      },
    )
    manifest_path.write_text("{}\n", encoding="utf-8")
    motion_path.write_text("motion\n", encoding="utf-8")
    return {
      "keypoints_path": keypoints_path,
      "manifest_path": manifest_path,
      "motion_path": motion_path,
    }

  def fake_retarget(**kwargs):
    output_path = Path(kwargs["output_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
      output_path,
      base_frame_pos=np.zeros((4, 3), dtype=np.float32),
      base_frame_wxyz=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (4, 1)),
      joint_angles=np.zeros((4, 29), dtype=np.float32),
    )
    manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
    manifest_path.write_text("{}\n", encoding="utf-8")
    return {
      "retargeted_motion_path": output_path,
      "manifest_path": manifest_path,
    }

  def fake_pyroki_to_csv(input_file, output_file):
    del input_file
    csv_row = "0,0,0,0,0,0,1," + ",".join(["0"] * 29)
    csv_path = Path(output_file)
    csv_path.write_text(csv_row + "\n" + csv_row + "\n", encoding="utf-8")

  def fake_csv_to_motion_npz(**kwargs):
    output_path = Path(kwargs["output_file"])
    np.savez(
      output_path,
      joint_pos=np.zeros((2, 29), dtype=np.float32),
      joint_vel=np.zeros((2, 29), dtype=np.float32),
      body_pos_w=np.zeros((2, 4, 3), dtype=np.float32),
      body_quat_w=np.tile(np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32), (2, 4, 1)),
      body_lin_vel_w=np.zeros((2, 4, 3), dtype=np.float32),
      body_ang_vel_w=np.zeros((2, 4, 3), dtype=np.float32),
    )
    return output_path.resolve()

  monkeypatch.setattr(build_script, "extract_smpl_keypoints_from_raw_human_npz", fake_extract)
  monkeypatch.setattr(build_script, "retarget_smpl_keypoints_to_g1_npz", fake_retarget)
  monkeypatch.setattr(build_script, "convert_pyroki_npz_to_csv", fake_pyroki_to_csv)
  monkeypatch.setattr(build_script, "convert_csv_to_motion_npz", fake_csv_to_motion_npz)

  result = build_script.build_tracking_motion(
    input_file=raw_path,
    motion_name="demo_motion",
    work_dir=work_dir,
    protomotions_root=tmp_path / "ProtoMotions",
  )

  assert result["motion_npz_path"].is_file()
  manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))
  assert manifest["motion_name"] == "demo_motion"
  assert Path(manifest["artifacts"]["motion_npz_path"]) == result["motion_npz_path"]
  assert result["csv_path"].is_file()
  assert result["raw_copy_path"].is_file()


def test_build_tracking_motion_rejects_unsupported_raw_format(tmp_path: Path) -> None:
  raw_path = tmp_path / "raw_motion.npz"
  _write_raw_motion(raw_path)

  with pytest.raises(NotImplementedError, match="raw_format"):
    build_script.build_tracking_motion(
      input_file=raw_path,
      motion_name="demo_motion",
      work_dir=tmp_path / "work",
      protomotions_root=tmp_path / "ProtoMotions",
      raw_format="unknown",
    )
