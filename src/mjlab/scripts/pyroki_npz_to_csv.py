"""Convert PyRoki-style retargeted G1 motion npz files into mjlab CSV."""

from pathlib import Path

import numpy as np
import tyro

import mjlab


def _validate_motion_array(name: str, array: np.ndarray, expected_cols: int) -> np.ndarray:
  if array.ndim != 2:
    raise ValueError(f"{name} must be a 2D array, got shape {array.shape}.")
  if array.shape[1] != expected_cols:
    raise ValueError(
      f"{name} must have shape (frames, {expected_cols}), got {array.shape}."
    )
  return np.asarray(array, dtype=np.float32)


def _ensure_finite(name: str, array: np.ndarray) -> None:
  if not np.isfinite(array).all():
    raise ValueError(f"{name} must contain only finite values.")


def convert_pyroki_npz_to_csv(input_file: str | Path, output_file: str | Path) -> None:
  """Convert a PyRoki-style retargeted G1 npz into an mjlab motion CSV."""
  input_path = Path(input_file)
  output_path = Path(output_file)

  with np.load(input_path) as motion:
    required_keys = ("base_frame_pos", "base_frame_wxyz", "joint_angles")
    missing_keys = [key for key in required_keys if key not in motion]
    if missing_keys:
      raise KeyError(
        f"Missing required keys in {input_path}: {', '.join(missing_keys)}."
      )

    base_frame_pos = _validate_motion_array(
      "base_frame_pos", motion["base_frame_pos"], expected_cols=3
    )
    base_frame_wxyz = _validate_motion_array(
      "base_frame_wxyz", motion["base_frame_wxyz"], expected_cols=4
    )
    joint_angles = _validate_motion_array(
      "joint_angles", motion["joint_angles"], expected_cols=29
    )

  _ensure_finite("base_frame_pos", base_frame_pos)
  _ensure_finite("base_frame_wxyz", base_frame_wxyz)
  _ensure_finite("joint_angles", joint_angles)

  frame_count = base_frame_pos.shape[0]
  if frame_count == 0:
    raise ValueError("Motion must contain at least one frame.")
  if frame_count == 1:
    raise ValueError(
      "Motion must contain at least two frames for csv_to_npz compatibility."
    )
  if base_frame_wxyz.shape[0] != frame_count:
    raise ValueError(
      "base_frame_pos and base_frame_wxyz must have the same number of frames."
    )
  if joint_angles.shape[0] != frame_count:
    raise ValueError(
      "joint_angles must have the same number of frames as base_frame_pos."
    )

  base_frame_xyzw = base_frame_wxyz[:, [1, 2, 3, 0]]
  motion_csv = np.concatenate((base_frame_pos, base_frame_xyzw, joint_angles), axis=1)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(output_path, motion_csv, delimiter=",", fmt="%.9f")


def main(input_file: str, output_file: str) -> None:
  """CLI entry point for the PyRoki-to-mjlab converter."""
  convert_pyroki_npz_to_csv(input_file=input_file, output_file=output_file)


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
