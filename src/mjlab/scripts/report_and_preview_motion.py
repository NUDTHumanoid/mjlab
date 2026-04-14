"""Summarize raw motion data and optionally build/preview it on the robot."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

import mjlab
from mjlab.scripts.build_tracking_motion import build_tracking_motion
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.scripts.raw_human_npz_to_smpl_keypoints import _validate_raw_human_npz

_DEFAULT_TRACKING_TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1"


def _default_motion_name(input_path: Path) -> str:
  return input_path.stem.replace(" ", "_") or "motion"


def _default_work_dir(input_path: Path) -> Path:
  return (Path.cwd() / "artifacts" / "motion_reports" / _default_motion_name(input_path)).resolve()


def _default_protomotions_root() -> Path | None:
  env_value = os.environ.get("PROTOMOTIONS_ROOT")
  if env_value:
    candidate = Path(env_value).expanduser().resolve()
    if candidate.is_dir():
      return candidate

  candidate = (Path.cwd() / ".external" / "ProtoMotions").resolve()
  if candidate.is_dir():
    return candidate
  return None


def _default_pyroki_python() -> str:
  env_value = os.environ.get("PYROKI_PYTHON")
  if env_value:
    return env_value

  candidate = Path.home() / "anaconda3" / "envs" / "pyroki" / "bin" / "python"
  if candidate.is_file():
    return str(candidate)
  return sys.executable


def _retarget_preset(
  quality: Literal["quick", "high"],
) -> tuple[int, int]:
  if quality == "quick":
    return 2, 120
  return 1, 450


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def summarize_raw_motion(input_file: str | Path) -> dict[str, object]:
  """Generate a compact report for a raw AMASS-style motion file."""
  input_path = Path(input_file).expanduser().resolve()
  validated = _validate_raw_human_npz(input_path)

  with np.load(input_path, allow_pickle=True) as motion:
    trans = np.asarray(motion["trans"], dtype=np.float64)
    poses = np.asarray(motion["poses"], dtype=np.float64)
    betas = np.asarray(motion["betas"], dtype=np.float64)
    dmpls = np.asarray(motion["dmpls"], dtype=np.float64) if "dmpls" in motion else None
    gender = str(motion["gender"].item()) if "gender" in motion else None

  mocap_fps = float(validated["mocap_framerate"])
  frame_count = int(validated["frame_count"])
  duration_s = frame_count / mocap_fps

  report: dict[str, object] = {
    "source_path": str(input_path),
    "file_name": input_path.name,
    "file_size_bytes": int(input_path.stat().st_size),
    "sha256": _sha256(input_path),
    "raw_format": "amass",
    "frame_count": frame_count,
    "mocap_fps": mocap_fps,
    "duration_seconds": duration_s,
    "pose_dimension": int(validated["pose_dimension"]),
    "gender": gender,
    "has_dmpls": dmpls is not None,
    "betas_dim": int(betas.shape[0]),
    "trans_start": trans[0].tolist(),
    "trans_end": trans[-1].tolist(),
    "trans_min": trans.min(axis=0).tolist(),
    "trans_max": trans.max(axis=0).tolist(),
    "poses_shape": list(poses.shape),
    "dmpls_shape": list(dmpls.shape) if dmpls is not None else None,
    "recommended_pipeline": {
      "keypoint_fps": 30,
      "motion_fps": 50,
      "retarget_subsample_factor": 1,
    },
    "compatible_with_mjlab_raw_pipeline": True,
  }
  return report


def report_and_preview_motion(
  *,
  input_file: str | Path,
  motion_name: str | None = None,
  work_dir: str | Path | None = None,
  protomotions_root: str | Path | None = None,
  proto_python: str = sys.executable,
  pyroki_python: str | None = None,
  quality: Literal["quick", "high"] = "high",
  keypoint_fps: int = 30,
  motion_fps: int = 50,
  retarget_subsample_factor: int | None = None,
  retarget_target_raw_frames: int | None = None,
  build: bool = False,
  preview: bool = False,
  viewer: Literal["auto", "native", "viser"] = "viser",
  device: str | None = None,
  task_id: str = _DEFAULT_TRACKING_TASK_ID,
  no_terminations: bool = True,
  force_remake: bool = False,
) -> dict[str, object]:
  """Write a raw-motion report and optionally build/preview the robot motion."""
  input_path = Path(input_file).expanduser().resolve()
  resolved_motion_name = motion_name or _default_motion_name(input_path)
  resolved_protomotions_root = (
    Path(protomotions_root).expanduser().resolve()
    if protomotions_root is not None
    else _default_protomotions_root()
  )
  resolved_work_dir = (
    Path(work_dir).expanduser().resolve()
    if work_dir is not None
    else _default_work_dir(input_path)
  )
  resolved_work_dir.mkdir(parents=True, exist_ok=True)
  resolved_pyroki_python = pyroki_python or _default_pyroki_python()
  default_subsample_factor, default_target_raw_frames = _retarget_preset(quality)
  resolved_subsample_factor = retarget_subsample_factor or default_subsample_factor
  resolved_target_raw_frames = retarget_target_raw_frames or default_target_raw_frames

  report = summarize_raw_motion(input_path)
  report["motion_name"] = resolved_motion_name
  report["work_dir"] = str(resolved_work_dir)
  report["requested_actions"] = {
    "build": build,
    "preview": preview,
    "viewer": viewer,
    "quality": quality,
  }
  if resolved_protomotions_root is not None:
    report["resolved_protomotions_root"] = str(resolved_protomotions_root)
  report["resolved_pyroki_python"] = resolved_pyroki_python

  build_result: dict[str, str] | None = None
  if build or preview:
    if resolved_protomotions_root is None:
      raise ValueError(
        "`protomotions_root` is required when build or preview is requested, "
        "or place ProtoMotions under .external/ProtoMotions."
      )

    built = build_tracking_motion(
      input_file=input_path,
      motion_name=resolved_motion_name,
      work_dir=resolved_work_dir / "pipeline",
      protomotions_root=resolved_protomotions_root,
      proto_python=proto_python,
      pyroki_python=resolved_pyroki_python,
      keypoint_fps=keypoint_fps,
      motion_fps=motion_fps,
      retarget_subsample_factor=resolved_subsample_factor,
      retarget_target_raw_frames=resolved_target_raw_frames,
      force_remake=force_remake,
    )
    build_result = {k: str(v) for k, v in built.items()}
    report["build"] = {
      "quality": quality,
      "keypoint_fps": keypoint_fps,
      "motion_fps": motion_fps,
      "retarget_subsample_factor": resolved_subsample_factor,
      "retarget_target_raw_frames": resolved_target_raw_frames,
      "artifacts": build_result,
    }

    if preview:
      run_play(
        task_id,
        PlayConfig(
          agent="zero",
          motion_file=str(built["motion_npz_path"]),
          viewer=viewer,
          device=device,
          no_terminations=no_terminations,
        ),
      )

  report_path = resolved_work_dir / "report.json"
  report_path.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
  )
  return {
    "report_path": str(report_path),
    "build_artifacts": build_result,
  }


def main(
  input_file: str,
  motion_name: str | None = None,
  work_dir: str | None = None,
  protomotions_root: str | None = None,
  proto_python: str = sys.executable,
  pyroki_python: str | None = None,
  quality: Literal["quick", "high"] = "high",
  keypoint_fps: int = 30,
  motion_fps: int = 50,
  retarget_subsample_factor: int | None = None,
  retarget_target_raw_frames: int | None = None,
  build: bool = False,
  preview: bool = False,
  viewer: Literal["auto", "native", "viser"] = "viser",
  device: str | None = None,
  task_id: str = _DEFAULT_TRACKING_TASK_ID,
  no_terminations: bool = True,
  force_remake: bool = False,
) -> None:
  report_and_preview_motion(
    input_file=input_file,
    motion_name=motion_name,
    work_dir=work_dir,
    protomotions_root=protomotions_root,
    proto_python=proto_python,
    pyroki_python=pyroki_python,
    quality=quality,
    keypoint_fps=keypoint_fps,
    motion_fps=motion_fps,
    retarget_subsample_factor=retarget_subsample_factor,
    retarget_target_raw_frames=retarget_target_raw_frames,
    build=build,
    preview=preview,
    viewer=viewer,
    device=device,
    task_id=task_id,
    no_terminations=no_terminations,
    force_remake=force_remake,
  )


def cli() -> None:
  tyro.cli(main, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
  cli()
