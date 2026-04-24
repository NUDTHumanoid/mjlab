from __future__ import annotations

from pathlib import Path

import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.scripts.evaluate import (
  EvaluateConfig,
  _resolve_rsl_rl_motion_file,
)
from mjlab.utils.os import (
  apply_rsl_rl_checkpoint_env_parity,
  maybe_load_rsl_rl_checkpoint_env_parity,
)


def _write_checkpoint_env_yaml(
  tmp_path: Path,
  *,
  motion_file: str = "/tmp/authoritative-motion.npz",
) -> Path:
  run_dir = tmp_path / "2026-04-14_12-19-21_handstand1_acrobatics_ft_40000"
  params_dir = run_dir / "params"
  params_dir.mkdir(parents=True)
  checkpoint_file = run_dir / "model_31500.pt"
  checkpoint_file.write_bytes(b"checkpoint")
  (params_dir / "env.yaml").write_text(
    f"""
scene:
  num_envs: 4096
observations:
  actor:
    enable_corruption: true
  critic:
    enable_corruption: false
commands:
  motion:
    sampling_mode: start
    motion_file: {motion_file}
events:
  base_com:
    mode: startup
  encoder_bias:
    mode: startup
episode_length_s: 15.0
""".strip()
    + "\n",
    encoding="utf-8",
  )
  return checkpoint_file


def test_maybe_load_rsl_rl_checkpoint_env_parity_reads_saved_scalars(
  tmp_path: Path,
) -> None:
  checkpoint_file = _write_checkpoint_env_yaml(tmp_path)

  parity = maybe_load_rsl_rl_checkpoint_env_parity(checkpoint_file)

  assert parity is not None
  assert parity.run_dir == checkpoint_file.parent
  assert parity.sampling_mode == "start"
  assert parity.motion_file == "/tmp/authoritative-motion.npz"
  assert parity.actor_enable_corruption is True
  assert parity.critic_enable_corruption is False
  assert parity.startup_event_names == ("base_com", "encoder_bias")
  assert parity.push_robot_enabled is False
  assert parity.episode_length_s == 15.0
  assert parity.num_envs == 4096


def test_apply_rsl_rl_checkpoint_env_parity_updates_tracking_env(
  tmp_path: Path,
) -> None:
  checkpoint_file = _write_checkpoint_env_yaml(tmp_path)
  env_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=True)

  assert env_cfg.observations["actor"].enable_corruption is False
  assert env_cfg.episode_length_s == 1_000_000_000

  parity = apply_rsl_rl_checkpoint_env_parity(env_cfg, checkpoint_file)

  assert parity is not None
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  assert motion_cmd.sampling_mode == "start"
  assert motion_cmd.motion_file == "/tmp/authoritative-motion.npz"
  assert env_cfg.observations["actor"].enable_corruption is True
  assert env_cfg.observations["critic"].enable_corruption is False
  assert env_cfg.episode_length_s == 15.0
  startup_events = tuple(
    name
    for name, cfg in env_cfg.events.items()
    if getattr(cfg, "mode", None) == "startup"
  )
  assert startup_events == ("base_com", "encoder_bias")
  assert "push_robot" not in env_cfg.events


def test_resolve_rsl_rl_motion_file_accepts_checkpoint_seeded_motion(
  tmp_path: Path,
) -> None:
  motion_file = tmp_path / "authoritative-motion.npz"
  motion_file.write_bytes(b"motion")
  checkpoint_file = _write_checkpoint_env_yaml(tmp_path, motion_file=str(motion_file))
  env_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=False)
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)

  parity = apply_rsl_rl_checkpoint_env_parity(env_cfg, checkpoint_file)
  assert parity is not None

  _resolve_rsl_rl_motion_file(
    motion_cmd,
    EvaluateConfig(
      checkpoint_file=str(checkpoint_file),
      motion_file=None,
      wandb_run_path=None,
    ),
  )

  assert motion_cmd.motion_file == str(motion_file)
