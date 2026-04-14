"""Retarget ProtoMotions SMPL keypoints to Unitree G1 joint trajectories."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import tyro

import mjlab

_G1_MOTION_NUM_JOINTS = 29
_REQUIRED_KEYPOINT_KEYS = (
  "positions",
  "orientations",
  "left_foot_contacts",
  "right_foot_contacts",
)
_RETARGET_SCRIPT = Path("pyroki/batch_retarget_to_g1_from_keypoints.py")
_G1_URDF = Path("protomotions/data/assets/urdf/for_retargeting/g1.urdf")
_G1_MESH_DIR = Path("protomotions/data/assets/mesh/G1")


def _sanitize_stem(stem: str) -> str:
  sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
  return sanitized or "motion"


def _source_fingerprint(input_path: Path) -> dict[str, int]:
  stat_result = input_path.stat()
  return {
    "size": int(stat_result.st_size),
    "mtime_ns": int(stat_result.st_mtime_ns),
  }


def _manifest_path_for_output(output_path: Path) -> Path:
  return output_path.with_name(f"{output_path.stem}_manifest.json")


def _validate_keypoint_payload(input_path: Path) -> None:
  payload = np.load(input_path, allow_pickle=True)
  try:
    motion = payload.item()
  except ValueError as exc:
    raise ValueError(
      f"{input_path} must be a dict-like numpy payload created by ProtoMotions keypoint extraction."
    ) from exc

  missing = [key for key in _REQUIRED_KEYPOINT_KEYS if key not in motion]
  if missing:
    raise KeyError(f"Missing required keys in {input_path}: {', '.join(missing)}.")

  positions = np.asarray(motion["positions"], dtype=np.float32)
  orientations = np.asarray(motion["orientations"], dtype=np.float32)
  left_foot_contacts = np.asarray(motion["left_foot_contacts"])
  right_foot_contacts = np.asarray(motion["right_foot_contacts"])

  if positions.ndim != 3 or positions.shape[-1] != 3:
    raise ValueError(f"positions must have shape (frames, keypoints, 3), got {positions.shape}.")
  if orientations.ndim != 4 or orientations.shape[-2:] != (3, 3):
    raise ValueError(
      f"orientations must have shape (frames, keypoints, 3, 3), got {orientations.shape}."
    )
  if left_foot_contacts.ndim != 2 or left_foot_contacts.shape[1] != 2:
    raise ValueError(
      "left_foot_contacts must have shape (frames, 2), "
      f"got {left_foot_contacts.shape}."
    )
  if right_foot_contacts.ndim != 2 or right_foot_contacts.shape[1] != 2:
    raise ValueError(
      "right_foot_contacts must have shape (frames, 2), "
      f"got {right_foot_contacts.shape}."
    )

  frame_count = positions.shape[0]
  if frame_count == 0:
    raise ValueError("Keypoint payload must contain at least one frame.")
  if orientations.shape[0] != frame_count:
    raise ValueError("orientations must have the same number of frames as positions.")
  if left_foot_contacts.shape[0] != frame_count:
    raise ValueError("left_foot_contacts must have the same number of frames as positions.")
  if right_foot_contacts.shape[0] != frame_count:
    raise ValueError("right_foot_contacts must have the same number of frames as positions.")
  if not np.isfinite(positions).all():
    raise ValueError("positions must contain only finite values.")
  if not np.isfinite(orientations).all():
    raise ValueError("orientations must contain only finite values.")


def _validate_retarget_output(output_path: Path) -> None:
  with np.load(output_path) as motion:
    required = ("base_frame_pos", "base_frame_wxyz", "joint_angles")
    missing = [key for key in required if key not in motion]
    if missing:
      raise KeyError(
        f"Retarget output {output_path} is missing required keys: {', '.join(missing)}."
      )
    base_frame_pos = np.asarray(motion["base_frame_pos"], dtype=np.float32)
    base_frame_wxyz = np.asarray(motion["base_frame_wxyz"], dtype=np.float32)
    joint_angles = np.asarray(motion["joint_angles"], dtype=np.float32)

  if base_frame_pos.ndim != 2 or base_frame_pos.shape[1] != 3:
    raise ValueError(
      f"base_frame_pos must have shape (frames, 3), got {base_frame_pos.shape}."
    )
  if base_frame_wxyz.ndim != 2 or base_frame_wxyz.shape[1] != 4:
    raise ValueError(
      f"base_frame_wxyz must have shape (frames, 4), got {base_frame_wxyz.shape}."
    )
  if joint_angles.ndim != 2 or joint_angles.shape[1] != _G1_MOTION_NUM_JOINTS:
    raise ValueError(
      "joint_angles must have shape "
      f"(frames, {_G1_MOTION_NUM_JOINTS}), got {joint_angles.shape}."
    )

  frame_count = base_frame_pos.shape[0]
  if frame_count == 0:
    raise ValueError("Retarget output must contain at least one frame.")
  if base_frame_wxyz.shape[0] != frame_count:
    raise ValueError("base_frame_wxyz must have the same frame count as base_frame_pos.")
  if joint_angles.shape[0] != frame_count:
    raise ValueError("joint_angles must have the same frame count as base_frame_pos.")
  if not np.isfinite(base_frame_pos).all():
    raise ValueError("base_frame_pos must contain only finite values.")
  if not np.isfinite(base_frame_wxyz).all():
    raise ValueError("base_frame_wxyz must contain only finite values.")
  if not np.isfinite(joint_angles).all():
    raise ValueError("joint_angles must contain only finite values.")


def _validate_protomotions_root(protomotions_root: Path) -> Path:
  root = protomotions_root.expanduser().resolve()
  if not root.is_dir():
    raise FileNotFoundError(f"ProtoMotions root does not exist: {root}")

  required_paths = (
    root / _RETARGET_SCRIPT,
    root / _G1_URDF,
    root / _G1_MESH_DIR,
  )
  missing = [path for path in required_paths if not path.exists()]
  if missing:
    missing_rel = ", ".join(str(path.relative_to(root)) for path in missing)
    raise FileNotFoundError(
      f"ProtoMotions root is missing required files for G1 retargeting: {missing_rel}."
    )
  return root


def _run_retarget_step(command: list[str], *, cwd: Path) -> None:
  env = os.environ.copy()
  existing_pythonpath = env.get("PYTHONPATH")
  env["PYTHONPATH"] = (
    f"{cwd}:{existing_pythonpath}" if existing_pythonpath else str(cwd)
  )
  try:
    subprocess.run(command, cwd=str(cwd), check=True, env=env)
  except subprocess.CalledProcessError as exc:  # pragma: no cover
    raise RuntimeError(
      f"G1 retargeting failed with exit code {exc.returncode}: {' '.join(command)}"
    ) from exc


def _can_reuse_cached_outputs(
  *,
  input_path: Path,
  output_path: Path,
  manifest_path: Path,
  protomotions_root: Path,
  pyroki_python: str,
  source_type: str,
  subsample_factor: int,
  target_raw_frames: int,
  input_fps: float,
) -> bool:
  if not output_path.is_file() or not manifest_path.is_file():
    return False

  try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return False

  try:
    source_path = Path(manifest["source_path"]).expanduser().resolve()
    source_fingerprint = manifest["source_fingerprint"]
    generated_output_path = Path(manifest["generated_output_path"]).expanduser().resolve()
    manifest_root = Path(manifest["protomotions_root"]).expanduser().resolve()
    manifest_python = str(manifest["pyroki_python"])
    manifest_source_type = str(manifest["source_type"])
    manifest_subsample_factor = int(manifest["subsample_factor"])
    manifest_target_raw_frames = int(manifest["target_raw_frames"])
    manifest_input_fps = float(manifest["input_fps"])
  except (KeyError, TypeError, ValueError):
    return False

  if source_path != input_path:
    return False
  if source_fingerprint != _source_fingerprint(input_path):
    return False
  if generated_output_path != output_path.resolve():
    return False
  if manifest_root != protomotions_root.resolve():
    return False
  if manifest_python != pyroki_python:
    return False
  if manifest_source_type != source_type:
    return False
  if manifest_subsample_factor != subsample_factor:
    return False
  if manifest_target_raw_frames != target_raw_frames:
    return False
  if manifest_input_fps != input_fps:
    return False
  return True


def retarget_smpl_keypoints_to_g1_npz(
  *,
  input_file: str | Path,
  output_file: str | Path,
  protomotions_root: str | Path,
  pyroki_python: str = sys.executable,
  source_type: str = "smpl",
  subsample_factor: int = 1,
  target_raw_frames: int = 450,
  input_fps: float = 30.0,
  force_remake: bool = False,
) -> dict[str, Path]:
  """Retarget a single ProtoMotions SMPL keypoint file to a G1 motion npz."""
  if subsample_factor <= 0:
    raise ValueError("subsample_factor must be positive.")
  if target_raw_frames <= 0:
    raise ValueError("target_raw_frames must be positive.")
  if input_fps <= 0:
    raise ValueError("input_fps must be positive.")

  input_path = Path(input_file).expanduser().resolve()
  output_path = Path(output_file).expanduser().resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  manifest_path = _manifest_path_for_output(output_path)

  _validate_keypoint_payload(input_path)
  root = _validate_protomotions_root(Path(protomotions_root))

  if not force_remake and _can_reuse_cached_outputs(
    input_path=input_path,
    output_path=output_path,
    manifest_path=manifest_path,
    protomotions_root=root,
    pyroki_python=pyroki_python,
    source_type=source_type,
    subsample_factor=subsample_factor,
    target_raw_frames=target_raw_frames,
    input_fps=input_fps,
  ):
    _validate_retarget_output(output_path)
    return {
      "retargeted_motion_path": output_path,
      "manifest_path": manifest_path,
    }

  with tempfile.TemporaryDirectory(dir=output_path.parent) as tmp_dir:
    workspace = Path(tmp_dir)
    keypoints_dir = workspace / "keypoints"
    retarget_dir = workspace / "retargeted"
    keypoints_dir.mkdir(parents=True, exist_ok=True)
    retarget_dir.mkdir(parents=True, exist_ok=True)

    sanitized_stem = _sanitize_stem(input_path.stem)
    staged_input = keypoints_dir / f"{sanitized_stem}.npy"
    shutil.copy2(input_path, staged_input)

    command = [
      pyroki_python,
      str(root / _RETARGET_SCRIPT),
      "--no-visualize",
      "--keypoints-folder-path",
      str(keypoints_dir),
      "--output-dir",
      str(retarget_dir),
      "--source-type",
      source_type,
      "--input-fps",
      str(input_fps),
      "--subsample-factor",
      str(subsample_factor),
      "--target-raw-frames",
      str(target_raw_frames),
    ]
    _run_retarget_step(command, cwd=root)

    generated_output = retarget_dir / f"{sanitized_stem}_retargeted.npz"
    if not generated_output.is_file():
      raise FileNotFoundError(
        "PyRoki retargeting did not produce the expected output file: "
        f"{generated_output}"
      )

    _validate_retarget_output(generated_output)
    shutil.copy2(generated_output, output_path)

  manifest = {
    "source_path": str(input_path),
    "source_fingerprint": _source_fingerprint(input_path),
    "generated_output_path": str(output_path.resolve()),
    "protomotions_root": str(root),
    "pyroki_python": pyroki_python,
    "source_type": source_type,
    "subsample_factor": subsample_factor,
    "target_raw_frames": target_raw_frames,
    "input_fps": input_fps,
  }
  manifest_path.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
  )

  return {
    "retargeted_motion_path": output_path,
    "manifest_path": manifest_path,
  }


def main(
  input_file: str,
  output_file: str,
  protomotions_root: str,
  pyroki_python: str = sys.executable,
  source_type: str = "smpl",
  subsample_factor: int = 1,
  target_raw_frames: int = 450,
  input_fps: float = 30.0,
  force_remake: bool = False,
) -> None:
  retarget_smpl_keypoints_to_g1_npz(
    input_file=input_file,
    output_file=output_file,
    protomotions_root=protomotions_root,
    pyroki_python=pyroki_python,
    source_type=source_type,
    subsample_factor=subsample_factor,
    target_raw_frames=target_raw_frames,
    input_fps=input_fps,
    force_remake=force_remake,
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
