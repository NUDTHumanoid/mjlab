from __future__ import annotations

from pathlib import Path

from mjlab.flashsac_verification import (
  AUTHORITATIVE_PPO_GOLD_CHECKPOINT,
  AUTHORITATIVE_PPO_GOLD_RUN_DIR,
)
from mjlab.rl.checkpoint_restore import (
  load_local_rsl_rl_checkpoint_params,
  load_rsl_rl_runtime_configs,
  resolve_rsl_rl_run_dir,
)
from mjlab.tasks.tracking.mdp import MotionCommandCfg


def test_resolve_rsl_rl_run_dir_accepts_checkpoint_file() -> None:
  assert resolve_rsl_rl_run_dir(AUTHORITATIVE_PPO_GOLD_CHECKPOINT) == (
    AUTHORITATIVE_PPO_GOLD_RUN_DIR.resolve()
  )


def test_load_local_rsl_rl_checkpoint_params_restores_saved_configs() -> None:
  restored = load_local_rsl_rl_checkpoint_params(
    AUTHORITATIVE_PPO_GOLD_CHECKPOINT,
    task_id="Mjlab-Tracking-Flat-Unitree-G1",
    play=False,
  )

  assert restored is not None
  env_cfg, agent_cfg, run_dir = restored
  motion_cmd = env_cfg.commands["motion"]

  assert run_dir == AUTHORITATIVE_PPO_GOLD_RUN_DIR.resolve()
  assert isinstance(motion_cmd, MotionCommandCfg)
  assert env_cfg.scene.num_envs == 4096
  assert motion_cmd.sampling_mode == "start"
  assert motion_cmd.pose_range == {}
  assert motion_cmd.velocity_range == {}
  assert motion_cmd.joint_position_range == (0.0, 0.0)
  assert float(env_cfg.terminations["anchor_pos"].params["threshold"]) == 0.6
  assert float(env_cfg.terminations["anchor_ori"].params["threshold"]) == 1.2
  assert "ee_body_pos" not in env_cfg.terminations
  assert agent_cfg["experiment_name"] == "g1_tracking_handstand1"
  assert agent_cfg["max_iterations"] == 40000
  assert agent_cfg["algorithm"]["learning_rate"] == 0.0005
  assert agent_cfg["algorithm"]["entropy_coef"] == 0.002


def test_load_rsl_rl_runtime_configs_uses_checkpoint_params_when_available() -> None:
  env_cfg, agent_cfg, restored, run_dir = load_rsl_rl_runtime_configs(
    "Mjlab-Tracking-Flat-Unitree-G1",
    checkpoint_file=str(AUTHORITATIVE_PPO_GOLD_CHECKPOINT),
    play=False,
  )

  motion_cmd = env_cfg.commands["motion"]
  assert restored is True
  assert run_dir == AUTHORITATIVE_PPO_GOLD_RUN_DIR.resolve()
  assert isinstance(motion_cmd, MotionCommandCfg)
  assert motion_cmd.sampling_mode == "start"
  assert agent_cfg["experiment_name"] == "g1_tracking_handstand1"


def test_load_rsl_rl_runtime_configs_fall_back_without_saved_params(
  tmp_path: Path,
) -> None:
  checkpoint_file = tmp_path / "model_1.pt"
  checkpoint_file.write_bytes(b"")

  env_cfg, agent_cfg, restored, run_dir = load_rsl_rl_runtime_configs(
    "Mjlab-Tracking-Flat-Unitree-G1",
    checkpoint_file=str(checkpoint_file),
    play=False,
  )

  motion_cmd = env_cfg.commands["motion"]
  assert restored is False
  assert run_dir is None
  assert isinstance(motion_cmd, MotionCommandCfg)
  assert motion_cmd.sampling_mode == "adaptive"
  assert agent_cfg["experiment_name"] == "g1_tracking"
