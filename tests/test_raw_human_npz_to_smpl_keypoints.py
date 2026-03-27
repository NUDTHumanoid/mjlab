"""Tests for the raw-human-to-ProtoMotions keypoint bridge."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from mjlab.scripts.raw_human_npz_to_smpl_keypoints import (
  extract_smpl_keypoints_from_raw_human_npz,
)


def _write_raw_human_npz(path: Path, *, frame_count: int = 4, pose_dim: int = 72):
  trans = np.arange(frame_count * 3, dtype=np.float32).reshape(frame_count, 3)
  poses = np.arange(frame_count * pose_dim, dtype=np.float32).reshape(
    frame_count, pose_dim
  )
  betas = np.linspace(0.0, 1.0, 10, dtype=np.float32)
  np.savez(
    path,
    trans=trans,
    poses=poses,
    betas=betas,
    mocap_framerate=np.array(60.0, dtype=np.float32),
  )


def _make_protomotions_root(
  root: Path,
  *,
  include_convert: bool = True,
  include_extract: bool = True,
  nested_asset: bool = False,
):
  scripts_dir = root / "data" / "scripts"
  assets_dir = (
    root / "protomotions" / "data" / "assets" / "mjcf"
    if nested_asset
    else root / "data" / "assets" / "mjcf"
  )
  scripts_dir.mkdir(parents=True, exist_ok=True)
  assets_dir.mkdir(parents=True, exist_ok=True)
  (assets_dir / "smpl_humanoid.xml").write_text("<xml />\n", encoding="utf-8")
  if include_convert:
    (scripts_dir / "convert_amass_to_proto.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
  if include_extract:
    (scripts_dir / "extract_keypoints_from_single_motion.py").write_text(
      "#!/usr/bin/env python3\n", encoding="utf-8"
    )


def _write_cached_outputs(
  output_dir: Path,
  *,
  stem: str,
  source_path: Path,
  output_fps: int,
  protomotions_root: Path,
  frame_count: int = 4,
) -> tuple[Path, Path, Path]:
  keypoints_path = output_dir / f"{stem}_keypoints.npy"
  manifest_path = output_dir / f"{stem}_manifest.json"
  motion_path = output_dir / f"{stem}.motion"

  np.save(
    keypoints_path,
    {
      "positions": np.zeros((frame_count, 18, 3), dtype=np.float32),
      "orientations": np.zeros((frame_count, 18, 3, 3), dtype=np.float32),
      "left_foot_contacts": np.zeros((frame_count, 2), dtype=np.int32),
      "right_foot_contacts": np.zeros((frame_count, 2), dtype=np.int32),
    },
  )
  motion_path.write_text("cached motion\n", encoding="utf-8")
  manifest_path.write_text(
    json.dumps(
      {
        "source_path": str(source_path.resolve()),
        "source_fingerprint": {
          "size": int(source_path.stat().st_size),
          "mtime_ns": int(source_path.stat().st_mtime_ns),
        },
        "frame_count": frame_count,
        "source_mocap_fps": 60.0,
        "pose_dimension": 72,
        "output_fps": output_fps,
        "protomotions_root": str(protomotions_root.resolve()),
        "generated_keypoint_path": str(keypoints_path.resolve()),
        "generated_intermediate_motion_path": str(motion_path.resolve()),
      },
      indent=2,
      ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
  )
  return keypoints_path, manifest_path, motion_path


def test_extract_smpl_keypoints_from_raw_human_npz_validates_required_keys(tmp_path: Path):
  """Missing AMASS-style arrays are rejected with a clear error."""
  input_file = tmp_path / "raw_motion.npz"
  np.savez(
    input_file,
    trans=np.zeros((2, 3), dtype=np.float32),
    poses=np.zeros((2, 72), dtype=np.float32),
    betas=np.zeros((10,), dtype=np.float32),
  )

  with np.testing.assert_raises_regex(KeyError, "mocap_framerate"):
    extract_smpl_keypoints_from_raw_human_npz(
      input_file=input_file,
      output_dir=tmp_path / "output",
      protomotions_root=tmp_path / "protomotions",
    )


def test_extract_smpl_keypoints_from_raw_human_npz_rejects_missing_protomotions_scripts(
  tmp_path: Path,
):
  """A missing official ProtoMotions script produces an actionable error."""
  input_file = tmp_path / "raw_motion.npz"
  _write_raw_human_npz(input_file)

  protomotions_root = tmp_path / "protomotions"
  _make_protomotions_root(protomotions_root, include_convert=False, include_extract=True)

  with np.testing.assert_raises_regex(FileNotFoundError, "convert_amass_to_proto.py"):
    extract_smpl_keypoints_from_raw_human_npz(
      input_file=input_file,
      output_dir=tmp_path / "output",
      protomotions_root=protomotions_root,
    )


def test_extract_smpl_keypoints_from_raw_human_npz_rejects_missing_smpl_asset(
  tmp_path: Path,
):
  """The required SMPL MJCF asset must be present under the ProtoMotions root."""
  input_file = tmp_path / "raw_motion.npz"
  _write_raw_human_npz(input_file)

  protomotions_root = tmp_path / "protomotions"
  _make_protomotions_root(protomotions_root)
  (protomotions_root / "data" / "assets" / "mjcf" / "smpl_humanoid.xml").unlink()

  with np.testing.assert_raises_regex(FileNotFoundError, "smpl_humanoid.xml"):
    extract_smpl_keypoints_from_raw_human_npz(
      input_file=input_file,
      output_dir=tmp_path / "output",
      protomotions_root=protomotions_root,
    )


def test_extract_smpl_keypoints_from_raw_human_npz_accepts_real_protomotions_layout(
  tmp_path: Path, monkeypatch
):
  """The wrapper accepts the real ProtoMotions asset layout under protomotions/."""
  input_file = tmp_path / "raw_motion.npz"
  _write_raw_human_npz(input_file)

  protomotions_root = tmp_path / "ProtoMotions"
  _make_protomotions_root(protomotions_root, nested_asset=True)
  output_dir = tmp_path / "output"

  recorded_commands: list[list[str]] = []

  def fake_run(cmd, *, cwd=None, check=None, **kwargs):
    recorded_commands.append(list(cmd))
    if cmd[1].endswith("convert_amass_to_proto.py"):
      motion_dir = Path(cmd[2])
      (motion_dir / "raw_motion.motion").write_text("fake motion\n", encoding="utf-8")
    elif cmd[1].endswith("extract_keypoints_from_single_motion.py"):
      output_path = Path(cmd[cmd.index("--output-path") + 1])
      generated = output_path / "amass_raw_motion_keypoints.npy"
      np.save(
        generated,
        {
          "positions": np.zeros((4, 18, 3), dtype=np.float32),
          "orientations": np.zeros((4, 18, 3, 3), dtype=np.float32),
          "left_foot_contacts": np.zeros((4, 2), dtype=np.int32),
          "right_foot_contacts": np.zeros((4, 2), dtype=np.int32),
        },
      )
    return subprocess.CompletedProcess(cmd, 0)

  monkeypatch.setattr(subprocess, "run", fake_run)

  extract_smpl_keypoints_from_raw_human_npz(
    input_file=input_file,
    output_dir=output_dir,
    protomotions_root=protomotions_root,
  )

  assert len(recorded_commands) == 2


def test_extract_smpl_keypoints_from_raw_human_npz_reuses_matching_cached_outputs(
  tmp_path: Path, monkeypatch
):
  """Matching cached outputs short-circuit without invoking subprocess."""
  input_file = tmp_path / "raw motion.npz"
  _write_raw_human_npz(input_file)

  protomotions_root = tmp_path / "protomotions"
  _make_protomotions_root(protomotions_root)
  output_dir = tmp_path / "output"
  output_dir.mkdir()
  keypoints_path, manifest_path, motion_path = _write_cached_outputs(
    output_dir,
    stem="raw_motion",
    source_path=input_file,
    output_fps=30,
    protomotions_root=protomotions_root,
  )

  def fail_run(*args, **kwargs):
    raise AssertionError("subprocess.run should not be called for matching cache hits")

  monkeypatch.setattr(subprocess, "run", fail_run)

  result = extract_smpl_keypoints_from_raw_human_npz(
    input_file=input_file,
    output_dir=output_dir,
    protomotions_root=protomotions_root,
  )

  assert result["keypoints_path"] == keypoints_path
  assert result["manifest_path"] == manifest_path
  assert result["motion_path"] == motion_path


def test_extract_smpl_keypoints_from_raw_human_npz_reruns_when_cached_manifest_mismatches(
  tmp_path: Path, monkeypatch
):
  """A stale manifest does not get reused."""
  input_file = tmp_path / "raw motion.npz"
  _write_raw_human_npz(input_file)

  protomotions_root = tmp_path / "protomotions"
  _make_protomotions_root(protomotions_root)
  output_dir = tmp_path / "output"
  output_dir.mkdir()
  _write_cached_outputs(
    output_dir,
    stem="raw_motion",
    source_path=input_file,
    output_fps=24,
    protomotions_root=protomotions_root,
  )

  recorded_commands: list[list[str]] = []

  def fake_run(cmd, *, cwd=None, check=None, **kwargs):
    recorded_commands.append(list(cmd))
    if cmd[1].endswith("convert_amass_to_proto.py"):
      motion_dir = Path(cmd[2])
      (motion_dir / "raw_motion.motion").write_text("fake motion\n", encoding="utf-8")
    elif cmd[1].endswith("extract_keypoints_from_single_motion.py"):
      output_path = Path(cmd[cmd.index("--output-path") + 1])
      generated = output_path / "amass_raw_motion_keypoints.npy"
      np.save(
        generated,
        {
          "positions": np.zeros((4, 18, 3), dtype=np.float32),
          "orientations": np.zeros((4, 18, 3, 3), dtype=np.float32),
          "left_foot_contacts": np.zeros((4, 2), dtype=np.int32),
          "right_foot_contacts": np.zeros((4, 2), dtype=np.int32),
        },
      )
    return subprocess.CompletedProcess(cmd, 0)

  monkeypatch.setattr(subprocess, "run", fake_run)

  extract_smpl_keypoints_from_raw_human_npz(
    input_file=input_file,
    output_dir=output_dir,
    protomotions_root=protomotions_root,
    output_fps=30,
  )

  assert recorded_commands
  assert recorded_commands[0][1].endswith("convert_amass_to_proto.py")


def test_extract_smpl_keypoints_from_raw_human_npz_reruns_when_source_npz_changes(
  tmp_path: Path, monkeypatch
):
  """Editing the source npz in place invalidates the cache."""
  input_file = tmp_path / "raw motion.npz"
  _write_raw_human_npz(input_file)

  protomotions_root = tmp_path / "protomotions"
  _make_protomotions_root(protomotions_root)
  output_dir = tmp_path / "output"
  output_dir.mkdir()
  _write_cached_outputs(
    output_dir,
    stem="raw_motion",
    source_path=input_file,
    output_fps=30,
    protomotions_root=protomotions_root,
  )

  _write_raw_human_npz(input_file, frame_count=6)

  recorded_commands: list[list[str]] = []

  def fake_run(cmd, *, cwd=None, check=None, **kwargs):
    recorded_commands.append(list(cmd))
    if cmd[1].endswith("convert_amass_to_proto.py"):
      motion_dir = Path(cmd[2])
      (motion_dir / "raw_motion.motion").write_text("fake motion\n", encoding="utf-8")
    elif cmd[1].endswith("extract_keypoints_from_single_motion.py"):
      output_path = Path(cmd[cmd.index("--output-path") + 1])
      generated = output_path / "amass_raw_motion_keypoints.npy"
      np.save(
        generated,
        {
          "positions": np.zeros((6, 18, 3), dtype=np.float32),
          "orientations": np.zeros((6, 18, 3, 3), dtype=np.float32),
          "left_foot_contacts": np.zeros((6, 2), dtype=np.int32),
          "right_foot_contacts": np.zeros((6, 2), dtype=np.int32),
        },
      )
    return subprocess.CompletedProcess(cmd, 0)

  monkeypatch.setattr(subprocess, "run", fake_run)

  extract_smpl_keypoints_from_raw_human_npz(
    input_file=input_file,
    output_dir=output_dir,
    protomotions_root=protomotions_root,
    output_fps=30,
  )

  assert len(recorded_commands) == 2
  assert recorded_commands[0][1].endswith("convert_amass_to_proto.py")
  assert recorded_commands[1][1].endswith("extract_keypoints_from_single_motion.py")


def test_extract_smpl_keypoints_from_raw_human_npz_force_remake_reruns_cached_outputs(
  tmp_path: Path, monkeypatch
):
  """force_remake bypasses the cache even when outputs already exist."""
  input_file = tmp_path / "raw motion.npz"
  _write_raw_human_npz(input_file)

  protomotions_root = tmp_path / "protomotions"
  _make_protomotions_root(protomotions_root)
  output_dir = tmp_path / "output"
  output_dir.mkdir()
  _write_cached_outputs(
    output_dir,
    stem="raw_motion",
    source_path=input_file,
    output_fps=30,
    protomotions_root=protomotions_root,
  )

  recorded_commands: list[list[str]] = []

  def fake_run(cmd, *, cwd=None, check=None, **kwargs):
    recorded_commands.append(list(cmd))
    if cmd[1].endswith("convert_amass_to_proto.py"):
      motion_dir = Path(cmd[2])
      (motion_dir / "raw_motion.motion").write_text("fake motion\n", encoding="utf-8")
    elif cmd[1].endswith("extract_keypoints_from_single_motion.py"):
      output_path = Path(cmd[cmd.index("--output-path") + 1])
      generated = output_path / "amass_raw_motion_keypoints.npy"
      np.save(
        generated,
        {
          "positions": np.zeros((4, 18, 3), dtype=np.float32),
          "orientations": np.zeros((4, 18, 3, 3), dtype=np.float32),
          "left_foot_contacts": np.zeros((4, 2), dtype=np.int32),
          "right_foot_contacts": np.zeros((4, 2), dtype=np.int32),
        },
      )
    return subprocess.CompletedProcess(cmd, 0)

  monkeypatch.setattr(subprocess, "run", fake_run)

  extract_smpl_keypoints_from_raw_human_npz(
    input_file=input_file,
    output_dir=output_dir,
    protomotions_root=protomotions_root,
    force_remake=True,
  )

  assert len(recorded_commands) == 2
  assert recorded_commands[0][1].endswith("convert_amass_to_proto.py")
  assert recorded_commands[1][1].endswith("extract_keypoints_from_single_motion.py")


def test_extract_smpl_keypoints_from_raw_human_npz_preserves_manifest_metadata(
  tmp_path: Path, monkeypatch
):
  """The JSON manifest keeps the source metadata needed downstream."""
  input_file = tmp_path / "crouch lie down.npz"
  _write_raw_human_npz(input_file, frame_count=5, pose_dim=72)

  protomotions_root = tmp_path / "protomotions"
  _make_protomotions_root(protomotions_root)
  output_dir = tmp_path / "output"

  recorded_commands: list[list[str]] = []

  def fake_run(cmd, *, cwd=None, check=None, **kwargs):
    recorded_commands.append(list(cmd))
    if cmd[1].endswith("convert_amass_to_proto.py"):
      motion_dir = Path(cmd[2])
      (motion_dir / "crouch_lie_down.motion").write_text("fake motion\n", encoding="utf-8")
    elif cmd[1].endswith("extract_keypoints_from_single_motion.py"):
      motion_file = Path(cmd[2])
      output_path = Path(cmd[cmd.index("--output-path") + 1])
      generated = output_path / f"{motion_file.parent.name}_{motion_file.stem}_keypoints.npy"
      np.save(
        generated,
        {
          "positions": np.zeros((5, 18, 3), dtype=np.float32),
          "orientations": np.zeros((5, 18, 3, 3), dtype=np.float32),
          "left_foot_contacts": np.zeros((5, 2), dtype=np.int32),
          "right_foot_contacts": np.zeros((5, 2), dtype=np.int32),
        },
      )
    return subprocess.CompletedProcess(cmd, 0)

  monkeypatch.setattr(subprocess, "run", fake_run)

  result = extract_smpl_keypoints_from_raw_human_npz(
    input_file=input_file,
    output_dir=output_dir,
    protomotions_root=protomotions_root,
    output_fps=24,
  )

  manifest_path = output_dir / "crouch_lie_down_manifest.json"
  keypoints_path = output_dir / "crouch_lie_down_keypoints.npy"
  motion_path = output_dir / "crouch_lie_down.motion"

  assert result["manifest_path"] == manifest_path
  assert result["keypoints_path"] == keypoints_path
  assert manifest_path.is_file()
  assert keypoints_path.is_file()
  assert motion_path.is_file()

  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  assert manifest["source_path"] == str(input_file.resolve())
  assert manifest["frame_count"] == 5
  assert manifest["source_mocap_fps"] == 60.0
  assert manifest["pose_dimension"] == 72
  assert manifest["output_fps"] == 24
  assert manifest["protomotions_root"] == str(protomotions_root.resolve())
  assert manifest["generated_keypoint_path"] == str(keypoints_path.resolve())
  assert manifest["generated_intermediate_motion_path"] == str(motion_path.resolve())


def test_extract_smpl_keypoints_from_raw_human_npz_invokes_proto_steps_in_order(
  tmp_path: Path, monkeypatch
):
  """The wrapper runs the official ProtoMotions scripts in the expected order."""
  input_file = tmp_path / "raw_motion.npz"
  _write_raw_human_npz(input_file)

  protomotions_root = tmp_path / "protomotions"
  _make_protomotions_root(protomotions_root)
  output_dir = tmp_path / "output"

  recorded_commands: list[list[str]] = []

  def fake_run(cmd, *, cwd=None, check=None, **kwargs):
    recorded_commands.append(list(cmd))
    if cmd[1].endswith("convert_amass_to_proto.py"):
      motion_dir = Path(cmd[2])
      (motion_dir / "raw_motion.motion").write_text("fake motion\n", encoding="utf-8")
    elif cmd[1].endswith("extract_keypoints_from_single_motion.py"):
      motion_file = Path(cmd[2])
      output_path = Path(cmd[cmd.index("--output-path") + 1])
      generated = output_path / f"{motion_file.parent.name}_{motion_file.stem}_keypoints.npy"
      np.save(
        generated,
        {
          "positions": np.zeros((4, 18, 3), dtype=np.float32),
          "orientations": np.zeros((4, 18, 3, 3), dtype=np.float32),
          "left_foot_contacts": np.zeros((4, 2), dtype=np.int32),
          "right_foot_contacts": np.zeros((4, 2), dtype=np.int32),
        },
      )
    return subprocess.CompletedProcess(cmd, 0)

  monkeypatch.setattr(subprocess, "run", fake_run)

  extract_smpl_keypoints_from_raw_human_npz(
    input_file=input_file,
    output_dir=output_dir,
    protomotions_root=protomotions_root,
  )

  assert recorded_commands[0][1].endswith("convert_amass_to_proto.py")
  assert recorded_commands[1][1].endswith("extract_keypoints_from_single_motion.py")
