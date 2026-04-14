"""Tests for the SMPL-keypoints-to-G1 retarget wrapper."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mjlab.scripts.smpl_keypoints_to_g1_npz import (
  retarget_smpl_keypoints_to_g1_npz,
)


def _write_keypoints(path: Path, *, frame_count: int = 4, keypoints: int = 18) -> None:
  orientations = np.tile(np.eye(3, dtype=np.float32), (frame_count, keypoints, 1, 1))
  np.save(
    path,
    {
      "positions": np.zeros((frame_count, keypoints, 3), dtype=np.float32),
      "orientations": orientations,
      "left_foot_contacts": np.zeros((frame_count, 2), dtype=np.int32),
      "right_foot_contacts": np.zeros((frame_count, 2), dtype=np.int32),
    },
  )


def _make_protomotions_root(root: Path) -> None:
  (root / "pyroki").mkdir(parents=True, exist_ok=True)
  (root / "protomotions" / "data" / "assets" / "urdf" / "for_retargeting").mkdir(
    parents=True, exist_ok=True
  )
  (root / "protomotions" / "data" / "assets" / "mesh" / "G1").mkdir(
    parents=True, exist_ok=True
  )
  (root / "pyroki" / "batch_retarget_to_g1_from_keypoints.py").write_text(
    "#!/usr/bin/env python3\n",
    encoding="utf-8",
  )
  (root / "protomotions" / "data" / "assets" / "urdf" / "for_retargeting" / "g1.urdf").write_text(
    "<robot />\n",
    encoding="utf-8",
  )


def test_retarget_smpl_keypoints_to_g1_npz_rejects_missing_required_key(tmp_path: Path) -> None:
  keypoint_path = tmp_path / "keypoints.npy"
  np.save(keypoint_path, {"positions": np.zeros((2, 18, 3), dtype=np.float32)})

  with pytest.raises(KeyError, match="orientations"):
    retarget_smpl_keypoints_to_g1_npz(
      input_file=keypoint_path,
      output_file=tmp_path / "retargeted.npz",
      protomotions_root=tmp_path / "ProtoMotions",
    )


def test_retarget_smpl_keypoints_to_g1_npz_rejects_missing_protomotions_files(
  tmp_path: Path,
) -> None:
  keypoint_path = tmp_path / "keypoints.npy"
  _write_keypoints(keypoint_path)
  protomotions_root = tmp_path / "ProtoMotions"
  protomotions_root.mkdir()

  with pytest.raises(FileNotFoundError, match="batch_retarget_to_g1_from_keypoints.py"):
    retarget_smpl_keypoints_to_g1_npz(
      input_file=keypoint_path,
      output_file=tmp_path / "retargeted.npz",
      protomotions_root=protomotions_root,
    )


def test_retarget_smpl_keypoints_to_g1_npz_runs_and_writes_manifest(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  keypoint_path = tmp_path / "keypoints.npy"
  _write_keypoints(keypoint_path)
  protomotions_root = tmp_path / "ProtoMotions"
  _make_protomotions_root(protomotions_root)
  output_path = tmp_path / "retargeted.npz"

  recorded_commands: list[list[str]] = []

  def fake_run(cmd, *, cwd=None, check=None, **kwargs):
    del cwd, check, kwargs
    recorded_commands.append(list(cmd))
    keypoints_dir = Path(cmd[cmd.index("--keypoints-folder-path") + 1])
    output_dir = Path(cmd[cmd.index("--output-dir") + 1])
    staged_input = next(keypoints_dir.glob("*.npy"))
    np.savez_compressed(
      output_dir / f"{staged_input.stem}_retargeted.npz",
      base_frame_pos=np.zeros((4, 3), dtype=np.float32),
      base_frame_wxyz=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (4, 1)),
      joint_angles=np.zeros((4, 29), dtype=np.float32),
    )
    return subprocess.CompletedProcess(cmd, 0)

  monkeypatch.setattr(subprocess, "run", fake_run)

  result = retarget_smpl_keypoints_to_g1_npz(
    input_file=keypoint_path,
    output_file=output_path,
    protomotions_root=protomotions_root,
  )

  assert result["retargeted_motion_path"] == output_path.resolve()
  assert result["manifest_path"].is_file()
  assert len(recorded_commands) == 1
  assert "--no-visualize" in recorded_commands[0]

  manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))
  assert Path(manifest["generated_output_path"]) == output_path.resolve()


def test_retarget_smpl_keypoints_to_g1_npz_reuses_cached_output(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  keypoint_path = tmp_path / "keypoints.npy"
  _write_keypoints(keypoint_path)
  protomotions_root = tmp_path / "ProtoMotions"
  _make_protomotions_root(protomotions_root)
  output_path = tmp_path / "retargeted.npz"
  np.savez_compressed(
    output_path,
    base_frame_pos=np.zeros((4, 3), dtype=np.float32),
    base_frame_wxyz=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (4, 1)),
    joint_angles=np.zeros((4, 29), dtype=np.float32),
  )
  manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
  manifest_path.write_text(
    json.dumps(
      {
        "source_path": str(keypoint_path.resolve()),
        "source_fingerprint": {
          "size": int(keypoint_path.stat().st_size),
          "mtime_ns": int(keypoint_path.stat().st_mtime_ns),
        },
        "generated_output_path": str(output_path.resolve()),
        "protomotions_root": str(protomotions_root.resolve()),
        "pyroki_python": "python",
        "source_type": "smpl",
        "subsample_factor": 1,
        "target_raw_frames": 450,
        "input_fps": 30.0,
      },
      indent=2,
      ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
  )

  def fail_run(*args, **kwargs):
    raise AssertionError("subprocess.run should not be called for cache hits")

  monkeypatch.setattr(subprocess, "run", fail_run)

  result = retarget_smpl_keypoints_to_g1_npz(
    input_file=keypoint_path,
    output_file=output_path,
    protomotions_root=protomotions_root,
    pyroki_python="python",
  )

  assert result["retargeted_motion_path"] == output_path.resolve()
