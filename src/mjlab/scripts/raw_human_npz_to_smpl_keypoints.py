"""Bridge raw AMASS-style human npz files into ProtoMotions keypoints."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import tyro

import mjlab


_REQUIRED_RAW_KEYS = ("trans", "poses", "betas", "mocap_framerate")
_SMPL_ASSET = Path("data/assets/mjcf/smpl_humanoid.xml")
_CONVERT_SCRIPT = Path("data/scripts/convert_amass_to_proto.py")
_EXTRACT_SCRIPT = Path("data/scripts/extract_keypoints_from_single_motion.py")


def _sanitize_stem(stem: str) -> str:
  sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
  return sanitized or "motion"


def _validate_2d_array(name: str, array: np.ndarray, expected_cols: int | None = None) -> np.ndarray:
  if array.ndim != 2:
    raise ValueError(f"{name} must be a 2D array, got shape {array.shape}.")
  if expected_cols is not None and array.shape[1] != expected_cols:
    raise ValueError(
      f"{name} must have shape (frames, {expected_cols}), got {array.shape}."
    )
  if not np.isfinite(array).all():
    raise ValueError(f"{name} must contain only finite values.")
  return np.asarray(array, dtype=np.float32)


def _validate_1d_array(name: str, array: np.ndarray) -> np.ndarray:
  if array.ndim != 1:
    raise ValueError(f"{name} must be a 1D array, got shape {array.shape}.")
  if not np.isfinite(array).all():
    raise ValueError(f"{name} must contain only finite values.")
  return np.asarray(array, dtype=np.float32)


def _validate_scalar(name: str, value: np.ndarray | float | int) -> float:
  scalar = np.asarray(value).reshape(-1)
  if scalar.size != 1:
    raise ValueError(f"{name} must be a scalar value, got shape {np.asarray(value).shape}.")
  scalar_value = float(scalar[0])
  if not np.isfinite(scalar_value):
    raise ValueError(f"{name} must be finite.")
  return scalar_value


def _validate_raw_human_npz(input_path: Path) -> dict[str, object]:
  with np.load(input_path) as motion:
    missing_keys = [key for key in _REQUIRED_RAW_KEYS if key not in motion]
    if missing_keys:
      raise KeyError(
        f"Missing required keys in {input_path}: {', '.join(missing_keys)}."
      )

    trans = _validate_2d_array("trans", motion["trans"], expected_cols=3)
    poses = _validate_2d_array("poses", motion["poses"])
    betas = _validate_1d_array("betas", motion["betas"])
    mocap_framerate = _validate_scalar("mocap_framerate", motion["mocap_framerate"])

  if trans.shape[0] == 0:
    raise ValueError("trans must contain at least one frame.")
  if poses.shape[0] != trans.shape[0]:
    raise ValueError("trans and poses must have the same number of frames.")

  return {
    "trans": trans,
    "poses": poses,
    "betas": betas,
    "mocap_framerate": mocap_framerate,
    "frame_count": int(trans.shape[0]),
    "pose_dimension": int(poses.shape[1]),
  }


def _source_fingerprint(input_path: Path) -> dict[str, int]:
  stat_result = input_path.stat()
  return {
    "size": int(stat_result.st_size),
    "mtime_ns": int(stat_result.st_mtime_ns),
  }


def _validate_protomotions_root(protomotions_root: Path) -> tuple[Path, Path, Path]:
  root = protomotions_root.expanduser().resolve()
  if not root.is_dir():
    raise FileNotFoundError(f"ProtoMotions root does not exist: {root}")

  convert_script = root / _CONVERT_SCRIPT
  extract_script = root / _EXTRACT_SCRIPT
  smpl_asset_candidates = (
    root / "protomotions" / _SMPL_ASSET,
    root / _SMPL_ASSET,
  )
  smpl_asset = next((path for path in smpl_asset_candidates if path.is_file()), None)

  missing = [path for path in (convert_script, extract_script) if not path.is_file()]
  if smpl_asset is None:
    missing.append(smpl_asset_candidates[0])
  if missing:
    missing_rel = ", ".join(str(path.relative_to(root)) for path in missing)
    raise FileNotFoundError(
      f"ProtoMotions root is missing required files: {missing_rel}."
    )

  return convert_script, extract_script, smpl_asset


def _run_proto_step(command: list[str], *, cwd: Path) -> None:
  try:
    subprocess.run(command, cwd=str(cwd), check=True)
  except subprocess.CalledProcessError as exc:  # pragma: no cover - subprocess failure path
    raise RuntimeError(
      f"ProtoMotions step failed with exit code {exc.returncode}: {' '.join(command)}"
    ) from exc


def _find_generated_keypoint_file(output_dir: Path, motion_file: Path) -> Path:
  expected = output_dir / f"{motion_file.parent.name}_{motion_file.stem}_keypoints.npy"
  if expected.is_file():
    return expected
  matches = sorted(output_dir.glob("*_keypoints.npy"))
  if len(matches) == 1:
    return matches[0]
  raise FileNotFoundError(
    f"Could not find keypoint output in {output_dir} after running extract_keypoints_from_single_motion.py."
  )


def _can_reuse_cached_outputs(
  *,
  manifest_path: Path,
  keypoint_path: Path,
  motion_path: Path,
  input_path: Path,
  protomotions_root: Path,
  output_fps: int,
) -> bool:
  if not (manifest_path.is_file() and keypoint_path.is_file() and motion_path.is_file()):
    return False

  try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return False

  try:
    manifest_source_path = Path(manifest["source_path"]).expanduser().resolve()
    manifest_output_fps = int(manifest["output_fps"])
    manifest_protomotions_root = Path(manifest["protomotions_root"]).expanduser().resolve()
    manifest_keypoint_path = Path(manifest["generated_keypoint_path"]).expanduser().resolve()
    manifest_motion_path = Path(manifest["generated_intermediate_motion_path"]).expanduser().resolve()
    manifest_source_fingerprint = manifest["source_fingerprint"]
  except (KeyError, TypeError, ValueError):
    return False

  if manifest_source_path != input_path:
    return False
  if manifest_output_fps != output_fps:
    return False
  if manifest_protomotions_root != protomotions_root.expanduser().resolve():
    return False
  if manifest_keypoint_path != keypoint_path.resolve():
    return False
  if manifest_motion_path != motion_path.resolve():
    return False
  if not isinstance(manifest_source_fingerprint, dict):
    return False
  if manifest_source_fingerprint != _source_fingerprint(input_path):
    return False
  if not manifest_keypoint_path.is_file() or not manifest_motion_path.is_file():
    return False
  return True


def extract_smpl_keypoints_from_raw_human_npz(
  *,
  input_file: str | Path,
  output_dir: str | Path,
  protomotions_root: str | Path,
  proto_python: str = sys.executable,
  output_fps: int = 30,
  force_remake: bool = False,
) -> dict[str, Path]:
  """Convert a raw AMASS-style npz into a ProtoMotions keypoint npy and manifest."""
  input_path = Path(input_file).expanduser().resolve()
  output_path = Path(output_dir).expanduser().resolve()
  output_path.mkdir(parents=True, exist_ok=True)

  raw_motion = _validate_raw_human_npz(input_path)
  convert_script, extract_script, _ = _validate_protomotions_root(Path(protomotions_root))

  sanitized_stem = _sanitize_stem(input_path.stem)
  keypoint_path = output_path / f"{sanitized_stem}_keypoints.npy"
  manifest_path = output_path / f"{sanitized_stem}_manifest.json"
  motion_path = output_path / f"{sanitized_stem}.motion"

  if not force_remake and _can_reuse_cached_outputs(
    manifest_path=manifest_path,
    keypoint_path=keypoint_path,
    motion_path=motion_path,
    input_path=input_path,
    protomotions_root=Path(protomotions_root),
    output_fps=output_fps,
  ):
    return {
      "keypoints_path": keypoint_path,
      "manifest_path": manifest_path,
      "motion_path": motion_path,
    }

  with tempfile.TemporaryDirectory(dir=output_path) as tmp_dir:
    workspace = Path(tmp_dir)
    amass_root = workspace / "amass"
    extract_output_dir = workspace / "keypoints"
    amass_root.mkdir(parents=True, exist_ok=True)
    extract_output_dir.mkdir(parents=True, exist_ok=True)

    staged_input = amass_root / f"{sanitized_stem}.npz"
    shutil.copy2(input_path, staged_input)

    convert_command = [
      proto_python,
      str(convert_script),
      str(amass_root),
      "--humanoid-type",
      "smpl",
      "--output-fps",
      str(output_fps),
    ]
    if force_remake:
      convert_command.append("--force-remake")
    _run_proto_step(convert_command, cwd=Path(protomotions_root))

    generated_motion_path = amass_root / f"{sanitized_stem}.motion"
    if not generated_motion_path.is_file():
      raise FileNotFoundError(
        f"ProtoMotions did not produce the expected motion file: {generated_motion_path}"
      )

    shutil.copy2(generated_motion_path, motion_path)

    extract_command = [
      proto_python,
      str(extract_script),
      str(generated_motion_path),
      "--skeleton-format",
      "smpl",
      "--output-path",
      str(extract_output_dir),
    ]
    if force_remake:
      extract_command.append("--force-remake")
    _run_proto_step(extract_command, cwd=Path(protomotions_root))

    generated_keypoint_path = _find_generated_keypoint_file(extract_output_dir, generated_motion_path)
    shutil.copy2(generated_keypoint_path, keypoint_path)

  manifest = {
    "source_path": str(input_path),
    "source_fingerprint": _source_fingerprint(input_path),
    "frame_count": raw_motion["frame_count"],
    "source_mocap_fps": raw_motion["mocap_framerate"],
    "pose_dimension": raw_motion["pose_dimension"],
    "output_fps": output_fps,
    "protomotions_root": str(Path(protomotions_root).expanduser().resolve()),
    "generated_keypoint_path": str(keypoint_path.resolve()),
    "generated_intermediate_motion_path": str(motion_path.resolve()),
  }
  manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

  return {
    "keypoints_path": keypoint_path,
    "manifest_path": manifest_path,
    "motion_path": motion_path,
  }


def main(
  input_file: str,
  output_dir: str,
  protomotions_root: str,
  proto_python: str = sys.executable,
  output_fps: int = 30,
  force_remake: bool = False,
) -> None:
  extract_smpl_keypoints_from_raw_human_npz(
    input_file=input_file,
    output_dir=output_dir,
    protomotions_root=protomotions_root,
    proto_python=proto_python,
    output_fps=output_fps,
    force_remake=force_remake,
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
