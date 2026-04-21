from __future__ import annotations

from pathlib import Path

import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.flashsac.runtime import (
  apply_flashsac_checkpoint_env_parity,
  maybe_load_flashsac_checkpoint_env_parity,
  render_flashsac_checkpoint_env_parity_audit,
  resolve_tracking_motion_file,
)


def _write_flashsac_env_yaml(
  tmp_path: Path,
  *,
  motion_file: str = "/tmp/flashsac-motion.npz",
  include_ee_body_pos: bool = False,
  include_sampling_mode: bool = True,
) -> Path:
  run_dir = tmp_path / "2026-04-15_20-14-10_flashsac-1024env-100m-seed42"
  params_dir = run_dir / "params"
  checkpoint_dir = run_dir / "step_29298"
  params_dir.mkdir(parents=True)
  checkpoint_dir.mkdir()
  (checkpoint_dir / "actor.pt").write_bytes(b"actor")
  ee_body_block = (
    """
  ee_body_pos:
    params:
      threshold: 0.35
""".rstrip()
    if include_ee_body_pos
    else ""
  )
  sampling_mode_block = (
    """
    sampling_mode: uniform
""".rstrip()
    if include_sampling_mode
    else ""
  )
  (params_dir / "env.yaml").write_text(
    (
      f"""
scene:
  num_envs: 4096
observations:
  actor:
    enable_corruption: false
  critic:
    enable_corruption: false
commands:
  motion:
    motion_file: {motion_file}
{sampling_mode_block}
events:
  base_com:
    mode: startup
  encoder_bias:
    mode: startup
episode_length_s: 15.0
terminations:
  anchor_pos:
    params:
      threshold: 0.6
  anchor_ori:
    params:
      threshold: 1.2
{ee_body_block}
""".strip()
      + "\n"
    ),
    encoding="utf-8",
  )
  return checkpoint_dir


def test_maybe_load_flashsac_checkpoint_env_parity_reads_saved_scalars(
  tmp_path: Path,
) -> None:
  checkpoint_dir = _write_flashsac_env_yaml(tmp_path)

  parity = maybe_load_flashsac_checkpoint_env_parity(checkpoint_dir)

  assert parity is not None
  assert parity.run_dir == checkpoint_dir.parent
  assert parity.sampling_mode == "uniform"
  assert parity.motion_file == "/tmp/flashsac-motion.npz"
  assert parity.actor_enable_corruption is False
  assert parity.critic_enable_corruption is False
  assert parity.startup_event_names == ("base_com", "encoder_bias")
  assert parity.push_robot_enabled is False
  assert parity.episode_length_s == 15.0
  assert parity.num_envs == 4096
  assert parity.anchor_pos_threshold == 0.6
  assert parity.anchor_ori_threshold == 1.2
  assert parity.has_ee_body_pos is False


def test_apply_flashsac_checkpoint_env_parity_updates_tracking_env_without_num_env_override(
  tmp_path: Path,
) -> None:
  motion_file = tmp_path / "flashsac-motion.npz"
  motion_file.write_bytes(b"motion")
  checkpoint_dir = _write_flashsac_env_yaml(tmp_path, motion_file=str(motion_file))
  env_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=True)

  original_num_envs = env_cfg.scene.num_envs
  audit = apply_flashsac_checkpoint_env_parity(env_cfg, checkpoint_dir)

  assert audit is not None
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  assert motion_cmd.motion_file == str(motion_file)
  assert motion_cmd.sampling_mode == "uniform"
  assert env_cfg.observations["actor"].enable_corruption is False
  assert env_cfg.observations["critic"].enable_corruption is False
  assert env_cfg.episode_length_s == 15.0
  assert env_cfg.scene.num_envs == original_num_envs
  assert env_cfg.terminations["anchor_pos"].params["threshold"] == 0.6
  assert env_cfg.terminations["anchor_ori"].params["threshold"] == 1.2
  assert "ee_body_pos" not in env_cfg.terminations
  assert audit.num_envs_source == "audit-only"
  assert "num_envs" in audit.skipped_fields
  rendered = render_flashsac_checkpoint_env_parity_audit(audit)
  assert "restored=" in rendered
  assert "skipped=" in rendered
  assert "num_envs_source=audit-only" in rendered


def test_apply_flashsac_checkpoint_env_parity_skips_ambiguous_ee_body_pos(
  tmp_path: Path,
) -> None:
  motion_file = tmp_path / "acro-motion.npz"
  motion_file.write_bytes(b"motion")
  checkpoint_dir = _write_flashsac_env_yaml(
    tmp_path,
    motion_file=str(motion_file),
    include_ee_body_pos=True,
  )
  env_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1-Acrobatics", play=False)

  assert "ee_body_pos" not in env_cfg.terminations
  audit = apply_flashsac_checkpoint_env_parity(env_cfg, checkpoint_dir)

  assert audit is not None
  assert "ee_body_pos" not in env_cfg.terminations
  assert "ee_body_pos" in audit.skipped_fields


def test_apply_flashsac_checkpoint_env_parity_skips_missing_fields(
  tmp_path: Path,
) -> None:
  motion_file = tmp_path / "missing-fields-motion.npz"
  motion_file.write_bytes(b"motion")
  checkpoint_dir = _write_flashsac_env_yaml(
    tmp_path,
    motion_file=str(motion_file),
    include_sampling_mode=False,
  )
  env_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=False)

  audit = apply_flashsac_checkpoint_env_parity(env_cfg, checkpoint_dir)

  assert audit is not None
  assert "sampling_mode" in audit.skipped_fields


def test_resolve_tracking_motion_file_prefers_cli_over_preseeded_checkpoint_value(
  tmp_path: Path,
) -> None:
  checkpoint_motion = tmp_path / "checkpoint-motion.npz"
  checkpoint_motion.write_bytes(b"checkpoint")
  cli_motion = tmp_path / "cli-motion.npz"
  cli_motion.write_bytes(b"cli")

  env_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=False)
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = str(checkpoint_motion)

  resolve_tracking_motion_file(
    motion_cmd,
    motion_file=str(cli_motion),
    registry_name=None,
    wandb_run_path=None,
    checkpoint_file="/tmp/fake-checkpoint",
  )

  assert motion_cmd.motion_file == str(cli_motion)


def test_resolve_tracking_motion_file_accepts_preseeded_motion_file(
  tmp_path: Path,
) -> None:
  motion_file = tmp_path / "seeded-motion.npz"
  motion_file.write_bytes(b"seeded")
  env_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=False)
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = str(motion_file)

  resolve_tracking_motion_file(
    motion_cmd,
    motion_file=None,
    registry_name=None,
    wandb_run_path=None,
    checkpoint_file="/tmp/fake-checkpoint",
  )

  assert motion_cmd.motion_file == str(motion_file)
