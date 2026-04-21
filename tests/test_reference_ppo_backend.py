from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml

import mjlab.scripts.train as train_mod
from mjlab.flashsac.reference_ppo import (
  ReferencePpoAgent,
  ReferencePpoRunnerCfg,
  ReferencePpoTrainConfig,
  launch_reference_ppo_training,
)


def test_reference_ppo_train_config_from_task_uses_existing_env_cfg() -> None:
  cfg = ReferencePpoTrainConfig.from_task("Mjlab-Cartpole-Swingup")

  assert cfg.agent.experiment_name == "cartpole_swingup_reference_ppo"
  assert cfg.agent.asymmetric_observation is True
  assert "actor" in cfg.env.observations
  assert "critic" in cfg.env.observations


def test_reference_ppo_train_config_from_task_tracks_native_ppo_defaults() -> None:
  cfg = ReferencePpoTrainConfig.from_task("Mjlab-Tracking-Flat-Unitree-G1")

  assert cfg.agent.rollout_length == 24
  assert cfg.agent.num_epochs == 5
  assert cfg.agent.num_minibatches == 4
  assert cfg.agent.learning_rate == pytest.approx(1e-3)
  assert cfg.agent.value_loss_coef == pytest.approx(1.0)
  assert cfg.agent.entropy_coef == pytest.approx(0.005)
  assert cfg.agent.clip_coef == pytest.approx(0.2)
  assert cfg.agent.normalize_observation is True
  assert cfg.agent.activation == "elu"
  assert cfg.agent.init_std == pytest.approx(1.0)
  assert cfg.agent.std_type == "scalar"
  assert cfg.agent.use_clipped_value_loss is True
  assert cfg.agent.schedule == "adaptive"
  assert cfg.agent.desired_kl == pytest.approx(0.01)
  assert cfg.agent.actor_hidden_dims == (512, 256, 128)
  assert cfg.agent.critic_hidden_dims == (512, 256, 128)


def test_reference_ppo_agent_keeps_actor_and_critic_observations_separate() -> None:
  agent = ReferencePpoAgent(
    actor_observation_dim=3,
    critic_observation_dim=5,
    action_dim=2,
    cfg=ReferencePpoRunnerCfg(
      normalize_observation=False,
      actor_hidden_dims=(16,),
      critic_hidden_dims=(16,),
      device_type="cpu",
    ),
    device=torch.device("cpu"),
  )

  actor_linear = next(
    layer for layer in agent.actor.backbone if isinstance(layer, torch.nn.Linear)
  )
  critic_linear = next(
    layer for layer in agent.critic.backbone if isinstance(layer, torch.nn.Linear)
  )

  assert actor_linear.in_features == 3
  assert critic_linear.in_features == 5


def test_train_main_routes_reference_ppo_backend(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  routed: dict[str, object] = {}

  def fake_reference_ppo_launch(task_id: str, args: object) -> None:
    routed["backend"] = "reference_ppo"
    routed["task_id"] = task_id
    routed["args"] = args

  def fail_other_launch(*args, **kwargs) -> None:
    raise AssertionError("other backends should not be called for --backend reference_ppo")

  monkeypatch.setattr(
    train_mod, "launch_reference_ppo_training", fake_reference_ppo_launch
  )
  monkeypatch.setattr(train_mod, "launch_flashsac_training", fail_other_launch)
  monkeypatch.setattr(train_mod, "launch_training", fail_other_launch)
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "train.py",
      "Mjlab-Cartpole-Swingup",
      "--backend",
      "reference_ppo",
      "--agent.num-env-steps",
      "16",
      "--env.scene.num-envs",
      "2",
      "--gpu-ids",
      "None",
    ],
  )

  train_mod.main()

  assert routed["backend"] == "reference_ppo"
  assert routed["task_id"] == "Mjlab-Cartpole-Swingup"
  assert isinstance(routed["args"], ReferencePpoTrainConfig)


def test_reference_ppo_cartpole_smoke_writes_summary(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.chdir(tmp_path)
  cfg = ReferencePpoTrainConfig.from_task("Mjlab-Cartpole-Swingup")
  cfg.env.scene.num_envs = 4
  cfg.agent.device_type = "cpu"
  cfg.agent.num_env_steps = 128
  cfg.agent.rollout_length = 8
  cfg.agent.num_epochs = 1
  cfg.agent.num_minibatches = 1
  cfg.agent.logging_per_interaction_step = 4
  cfg.agent.save_checkpoint_per_interaction_step = 16
  cfg.agent.logger = "tensorboard"

  launch_reference_ppo_training("Mjlab-Cartpole-Swingup", cfg)

  log_root = tmp_path / "logs" / "reference_ppo" / cfg.agent.experiment_name
  run_dirs = sorted(log_root.iterdir())
  assert run_dirs
  run_dir = run_dirs[-1]
  metrics = yaml.safe_load((run_dir / "summary" / "metrics.json").read_text())
  runtime = yaml.safe_load((run_dir / "params" / "runtime.yaml").read_text())

  assert metrics["backend"] == "reference_ppo"
  assert metrics["task_id"] == "Mjlab-Cartpole-Swingup"
  assert runtime["num_env_steps"] == 128
  assert runtime["num_envs"] == 4
  assert (run_dir / "step_16" / "agent_state.pt").exists()
