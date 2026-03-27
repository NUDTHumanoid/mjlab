"""Tests for the PyRoki-to-mjlab CSV converter."""

from pathlib import Path

import numpy as np

from mjlab.scripts.pyroki_npz_to_csv import convert_pyroki_npz_to_csv


def test_convert_pyroki_npz_to_csv_writes_expected_csv(tmp_path: Path):
  """Synthetic G1 retargeted motion is converted into mjlab CSV."""
  npz_path = tmp_path / "g1_retargeted.npz"
  csv_path = tmp_path / "motion.csv"

  base_frame_pos = np.array(
    [
      [0.1, 0.2, 0.3],
      [1.1, 1.2, 1.3],
      [2.1, 2.2, 2.3],
    ],
    dtype=np.float32,
  )
  base_frame_wxyz = np.array(
    [
      [0.0, 0.1, 0.2, 0.3],
      [1.0, 1.1, 1.2, 1.3],
      [2.0, 2.1, 2.2, 2.3],
    ],
    dtype=np.float32,
  )
  joint_angles = np.arange(3 * 29, dtype=np.float32).reshape(3, 29) + 10.0

  np.savez(
    npz_path,
    base_frame_pos=base_frame_pos,
    base_frame_wxyz=base_frame_wxyz,
    joint_angles=joint_angles,
  )

  convert_pyroki_npz_to_csv(npz_path, csv_path)

  motion = np.loadtxt(csv_path, delimiter=",")

  assert motion.shape == (3, 36)
  np.testing.assert_allclose(motion[:, :3], base_frame_pos)
  np.testing.assert_allclose(motion[:, 3:7], base_frame_wxyz[:, [1, 2, 3, 0]])
  np.testing.assert_allclose(motion[:, 7:], joint_angles)


def test_convert_pyroki_npz_to_csv_preserves_frame_count(tmp_path: Path):
  """Frame count in the output matches the input retargeted motion."""
  npz_path = tmp_path / "g1_retargeted.npz"
  csv_path = tmp_path / "motion.csv"

  np.savez(
    npz_path,
    base_frame_pos=np.zeros((5, 3), dtype=np.float32),
    base_frame_wxyz=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (5, 1)),
    joint_angles=np.zeros((5, 29), dtype=np.float32),
  )

  convert_pyroki_npz_to_csv(npz_path, csv_path)

  motion = np.loadtxt(csv_path, delimiter=",")
  assert motion.shape[0] == 5


def test_convert_pyroki_npz_to_csv_rejects_missing_key(tmp_path: Path):
  """Missing required input keys are rejected."""
  npz_path = tmp_path / "g1_retargeted.npz"
  csv_path = tmp_path / "motion.csv"

  np.savez(
    npz_path,
    base_frame_pos=np.zeros((1, 3), dtype=np.float32),
    joint_angles=np.zeros((1, 29), dtype=np.float32),
  )

  with np.testing.assert_raises_regex(KeyError, "base_frame_wxyz"):
    convert_pyroki_npz_to_csv(npz_path, csv_path)


def test_convert_pyroki_npz_to_csv_rejects_invalid_shape(tmp_path: Path):
  """Invalid input column counts are rejected."""
  npz_path = tmp_path / "g1_retargeted.npz"
  csv_path = tmp_path / "motion.csv"

  np.savez(
    npz_path,
    base_frame_pos=np.zeros((1, 2), dtype=np.float32),
    base_frame_wxyz=np.zeros((1, 4), dtype=np.float32),
    joint_angles=np.zeros((1, 29), dtype=np.float32),
  )

  with np.testing.assert_raises_regex(ValueError, "base_frame_pos"):
    convert_pyroki_npz_to_csv(npz_path, csv_path)


def test_convert_pyroki_npz_to_csv_rejects_mismatched_frames(tmp_path: Path):
  """Mismatched frame counts across arrays are rejected."""
  npz_path = tmp_path / "g1_retargeted.npz"
  csv_path = tmp_path / "motion.csv"

  np.savez(
    npz_path,
    base_frame_pos=np.zeros((2, 3), dtype=np.float32),
    base_frame_wxyz=np.zeros((3, 4), dtype=np.float32),
    joint_angles=np.zeros((2, 29), dtype=np.float32),
  )

  with np.testing.assert_raises_regex(ValueError, "same number of frames"):
    convert_pyroki_npz_to_csv(npz_path, csv_path)


def test_convert_pyroki_npz_to_csv_rejects_non_finite_values(tmp_path: Path):
  """NaN and Inf values are rejected before writing CSV."""
  npz_path = tmp_path / "g1_retargeted.npz"
  csv_path = tmp_path / "motion.csv"

  np.savez(
    npz_path,
    base_frame_pos=np.array([[0.0, 1.0, np.nan]], dtype=np.float32),
    base_frame_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    joint_angles=np.zeros((1, 29), dtype=np.float32),
  )

  with np.testing.assert_raises_regex(ValueError, "finite"):
    convert_pyroki_npz_to_csv(npz_path, csv_path)


def test_convert_pyroki_npz_to_csv_rejects_zero_frame_input(tmp_path: Path):
  """Zero-frame motions are rejected instead of writing an empty CSV."""
  npz_path = tmp_path / "g1_retargeted.npz"
  csv_path = tmp_path / "motion.csv"

  np.savez(
    npz_path,
    base_frame_pos=np.zeros((0, 3), dtype=np.float32),
    base_frame_wxyz=np.zeros((0, 4), dtype=np.float32),
    joint_angles=np.zeros((0, 29), dtype=np.float32),
  )

  with np.testing.assert_raises_regex(ValueError, "at least one frame"):
    convert_pyroki_npz_to_csv(npz_path, csv_path)


def test_convert_pyroki_npz_to_csv_rejects_single_frame_input(tmp_path: Path):
  """Single-frame motions are rejected because csv_to_npz expects 2D loadtxt output."""
  npz_path = tmp_path / "g1_retargeted.npz"
  csv_path = tmp_path / "motion.csv"

  np.savez(
    npz_path,
    base_frame_pos=np.zeros((1, 3), dtype=np.float32),
    base_frame_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    joint_angles=np.zeros((1, 29), dtype=np.float32),
  )

  with np.testing.assert_raises_regex(ValueError, "at least two frames"):
    convert_pyroki_npz_to_csv(npz_path, csv_path)
