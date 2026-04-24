"""Build an mjlab tracking motion asset from raw human motion data."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import tyro

import mjlab
from mjlab.scripts.csv_to_npz import convert_csv_to_motion_npz
from mjlab.scripts.pyroki_npz_to_csv import convert_pyroki_npz_to_csv
from mjlab.scripts.raw_human_npz_to_smpl_keypoints import (
  extract_smpl_keypoints_from_raw_human_npz,
)
from mjlab.scripts.smpl_keypoints_to_g1_npz import (
  retarget_smpl_keypoints_to_g1_npz,
)


def _copy_if_needed(source_path: Path, target_path: Path) -> None:
  target_path.parent.mkdir(parents=True, exist_ok=True)
  if source_path.resolve() == target_path.resolve():
    return
  shutil.copy2(source_path, target_path)


def _source_fingerprint(input_path: Path) -> dict[str, int]:
  stat_result = input_path.stat()
  return {
    "size": int(stat_result.st_size),
    "mtime_ns": int(stat_result.st_mtime_ns),
  }


def build_tracking_motion(
  *,
  input_file: str | Path,
  motion_name: str,
  work_dir: str | Path,
  protomotions_root: str | Path,
  proto_python: str = sys.executable,
  pyroki_python: str = sys.executable,
  raw_format: str = "amass",
  keypoint_fps: int = 30,
  motion_fps: int = 50,
  retarget_subsample_factor: int = 1,
  retarget_target_raw_frames: int = 450,
  render: bool = False,
  force_remake: bool = False,
) -> dict[str, Path]:
  """Run the raw-human-motion to mjlab-tracking-asset pipeline."""
  if raw_format != "amass":
    raise NotImplementedError(
      f"Unsupported raw_format={raw_format!r}. Only 'amass' is supported currently."
    )
  if keypoint_fps <= 0:
    raise ValueError("keypoint_fps must be positive.")
  if motion_fps <= 0:
    raise ValueError("motion_fps must be positive.")
  if retarget_subsample_factor <= 0:
    raise ValueError("retarget_subsample_factor must be positive.")

  input_path = Path(input_file).expanduser().resolve()
  work_path = Path(work_dir).expanduser().resolve()
  work_path.mkdir(parents=True, exist_ok=True)

  raw_dir = work_path / "raw"
  keypoints_dir = work_path / "keypoints"
  retarget_dir = work_path / "retarget"
  mjlab_dir = work_path / "mjlab"
  for directory in (raw_dir, keypoints_dir, retarget_dir, mjlab_dir):
    directory.mkdir(parents=True, exist_ok=True)

  staged_raw_path = raw_dir / input_path.name
  _copy_if_needed(input_path, staged_raw_path)

  keypoint_result = extract_smpl_keypoints_from_raw_human_npz(
    input_file=staged_raw_path,
    output_dir=keypoints_dir,
    protomotions_root=protomotions_root,
    proto_python=proto_python,
    output_fps=keypoint_fps,
    force_remake=force_remake,
  )

  retarget_output_path = retarget_dir / f"{motion_name}_g1_retargeted.npz"
  retarget_result = retarget_smpl_keypoints_to_g1_npz(
    input_file=keypoint_result["keypoints_path"],
    output_file=retarget_output_path,
    protomotions_root=protomotions_root,
    pyroki_python=pyroki_python,
    source_type="smpl",
    subsample_factor=retarget_subsample_factor,
    target_raw_frames=retarget_target_raw_frames,
    input_fps=float(keypoint_fps),
    force_remake=force_remake,
  )

  csv_path = mjlab_dir / f"{motion_name}.csv"
  convert_pyroki_npz_to_csv(
    input_file=retarget_result["retargeted_motion_path"],
    output_file=csv_path,
  )

  csv_input_fps = float(keypoint_fps) / float(retarget_subsample_factor)
  motion_npz_path = convert_csv_to_motion_npz(
    input_file=str(csv_path),
    output_name=motion_name,
    input_fps=csv_input_fps,
    output_fps=float(motion_fps),
    render=render,
    output_file=str(mjlab_dir / "motion.npz"),
    skip_wandb=True,
  )

  manifest_path = work_path / "manifest.json"
  manifest = {
    "motion_name": motion_name,
    "robot": "unitree_g1",
    "raw_format": raw_format,
    "source_path": str(input_path),
    "source_fingerprint": _source_fingerprint(input_path),
    "artifacts": {
      "raw_copy_path": str(staged_raw_path),
      "keypoints_path": str(keypoint_result["keypoints_path"]),
      "keypoints_manifest_path": str(keypoint_result["manifest_path"]),
      "retargeted_motion_path": str(retarget_result["retargeted_motion_path"]),
      "retarget_manifest_path": str(retarget_result["manifest_path"]),
      "csv_path": str(csv_path),
      "motion_npz_path": str(motion_npz_path),
    },
    "params": {
      "protomotions_root": str(Path(protomotions_root).expanduser().resolve()),
      "proto_python": proto_python,
      "pyroki_python": pyroki_python,
      "keypoint_fps": keypoint_fps,
      "retarget_subsample_factor": retarget_subsample_factor,
      "retarget_target_raw_frames": retarget_target_raw_frames,
      "csv_input_fps": csv_input_fps,
      "motion_fps": motion_fps,
      "render": render,
    },
  }
  manifest_path.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
  )

  return {
    "work_dir": work_path,
    "raw_copy_path": staged_raw_path,
    "keypoints_path": keypoint_result["keypoints_path"],
    "retargeted_motion_path": retarget_result["retargeted_motion_path"],
    "csv_path": csv_path,
    "motion_npz_path": motion_npz_path,
    "manifest_path": manifest_path,
  }


def main(
  input_file: str,
  motion_name: str,
  work_dir: str,
  protomotions_root: str,
  proto_python: str = sys.executable,
  pyroki_python: str = sys.executable,
  raw_format: str = "amass",
  keypoint_fps: int = 30,
  motion_fps: int = 50,
  retarget_subsample_factor: int = 1,
  retarget_target_raw_frames: int = 450,
  render: bool = False,
  force_remake: bool = False,
) -> None:
  build_tracking_motion(
    input_file=input_file,
    motion_name=motion_name,
    work_dir=work_dir,
    protomotions_root=protomotions_root,
    proto_python=proto_python,
    pyroki_python=pyroki_python,
    raw_format=raw_format,
    keypoint_fps=keypoint_fps,
    motion_fps=motion_fps,
    retarget_subsample_factor=retarget_subsample_factor,
    retarget_target_raw_frames=retarget_target_raw_frames,
    render=render,
    force_remake=force_remake,
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
