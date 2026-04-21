from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
import yaml

import mjlab.scripts.play as play_mod
import mjlab.scripts.train as train_mod
import mjlab.tasks.tracking.scripts.evaluate as evaluate_mod
from mjlab.envs import ManagerBasedRlEnv
from mjlab.flashsac.adapter import MjlabFlashSACEnvAdapter
from mjlab.flashsac.agent import FlashSACAgent
from mjlab.flashsac.config import (
  FLASHSAC_TRACKING_CHECKPOINT_COUNT,
  FLASHSAC_TRACKING_DEFAULT_CHECKPOINT_INTERVAL,
  FLASHSAC_TRACKING_NUM_ENVS,
  FLASHSAC_TRACKING_TOTAL_ENV_STEPS,
  FLASHSAC_TRACKING_UPSTREAM_INTERACTION_STEPS,
  FlashSACRunnerCfg,
  FlashSACTrainConfig,
  apply_flashsac_tracking_train_overrides,
  maybe_recompute_flashsac_tracking_checkpoint_cadence,
)
from mjlab.flashsac.runtime import (
  apply_flashsac_tracking_inference_overrides,
  load_flashsac_runner_cfg,
  load_flashsac_saved_runner_cfg,
  make_flashsac_inference_cfg,
  resolve_flashsac_checkpoint_dir,
)
from mjlab.flashsac.trainer import (
  _apply_resume_agent_contract,
  _checkpoint_summary_entry,
  _randomize_episode_horizons,
  _resolve_replay_next_observations,
  _write_training_audit_artifacts,
)
from mjlab.flashsac.utils import NetworkBundle, ObservationNormalizer
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.scripts.evaluate import EvaluateConfig, run_evaluate
from mjlab.utils.training_steps import (
  checkpoint_interval_from_total_env_steps,
  interaction_steps_from_total_env_steps,
  total_env_steps_from_interaction_steps,
)


def test_flashsac_train_config_from_task_uses_existing_env_cfg() -> None:
  cfg = FlashSACTrainConfig.from_task("Mjlab-Cartpole-Swingup")

  assert cfg.agent.experiment_name == "cartpole_swingup_flashsac"
  assert cfg.agent.asymmetric_observation is True
  assert cfg.agent.save_final_replay_buffer is False
  assert "actor" in cfg.env.observations
  assert "critic" in cfg.env.observations


def test_flashsac_tracking_train_config_uses_stronger_defaults() -> None:
  cfg = FlashSACTrainConfig.from_task("Mjlab-Tracking-Flat-Unitree-G1")
  baseline_env_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=False)

  assert cfg.env.scene.num_envs == FLASHSAC_TRACKING_NUM_ENVS
  assert cfg.agent.num_env_steps == FLASHSAC_TRACKING_TOTAL_ENV_STEPS
  assert cfg.agent.updates_per_interaction_step == 2.0
  assert cfg.agent.n_step == 3
  assert cfg.agent.buffer_min_length == 100_000
  assert cfg.agent.save_buffer_per_interaction_step is None
  assert cfg.agent.save_final_replay_buffer is False
  assert cfg.agent.normalize_observation is False
  assert cfg.agent.asymmetric_observation is False
  assert (
    cfg.agent.save_checkpoint_per_interaction_step
    == checkpoint_interval_from_total_env_steps(
      FLASHSAC_TRACKING_TOTAL_ENV_STEPS,
      num_envs=FLASHSAC_TRACKING_NUM_ENVS,
      checkpoint_count=FLASHSAC_TRACKING_CHECKPOINT_COUNT,
    )
  )
  motion_cmd = cfg.env.commands["motion"]
  baseline_motion_cmd = baseline_env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  assert isinstance(baseline_motion_cmd, MotionCommandCfg)
  assert motion_cmd.sampling_mode == baseline_motion_cmd.sampling_mode
  assert motion_cmd.pose_range == baseline_motion_cmd.pose_range
  assert motion_cmd.velocity_range == baseline_motion_cmd.velocity_range
  assert motion_cmd.joint_position_range == baseline_motion_cmd.joint_position_range
  assert (
    cfg.env.observations["actor"].enable_corruption
    == baseline_env_cfg.observations["actor"].enable_corruption
  )
  assert set(cfg.env.events) == set(baseline_env_cfg.events)
  assert set(cfg.env.terminations) == set(baseline_env_cfg.terminations)
  assert (
    cfg.env.terminations["anchor_pos"].params["threshold"]
    == baseline_env_cfg.terminations["anchor_pos"].params["threshold"]
  )
  assert (
    cfg.env.terminations["anchor_ori"].params["threshold"]
    == baseline_env_cfg.terminations["anchor_ori"].params["threshold"]
  )


def test_train_main_routes_flashsac_backend(monkeypatch: pytest.MonkeyPatch) -> None:
  routed: dict[str, object] = {}

  def fake_flashsac_launch(task_id: str, args: object) -> None:
    routed["backend"] = "flashsac"
    routed["task_id"] = task_id
    routed["args"] = args

  def fail_default_launch(*args, **kwargs) -> None:
    raise AssertionError(
      "default PPO launch should not be called for --backend flashsac"
    )

  monkeypatch.setattr(train_mod, "launch_flashsac_training", fake_flashsac_launch)
  monkeypatch.setattr(train_mod, "launch_training", fail_default_launch)
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "train.py",
      "Mjlab-Cartpole-Swingup",
      "--backend",
      "flashsac",
      "--agent.num-env-steps",
      "8",
      "--env.scene.num-envs",
      "2",
      "--gpu-ids",
      "None",
    ],
  )

  train_mod.main()

  assert routed["backend"] == "flashsac"
  assert routed["task_id"] == "Mjlab-Cartpole-Swingup"
  assert isinstance(routed["args"], FlashSACTrainConfig)


def test_train_main_routes_default_backend(monkeypatch: pytest.MonkeyPatch) -> None:
  routed: dict[str, object] = {}

  def fake_default_launch(task_id: str, args: object) -> None:
    routed["backend"] = "rsl_rl"
    routed["task_id"] = task_id
    routed["args"] = args

  def fail_flashsac_launch(*args, **kwargs) -> None:
    raise AssertionError(
      "FlashSAC launch should not be called without --backend flashsac"
    )

  monkeypatch.setattr(train_mod, "launch_training", fake_default_launch)
  monkeypatch.setattr(train_mod, "launch_flashsac_training", fail_flashsac_launch)
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "train.py",
      "Mjlab-Cartpole-Swingup",
      "--agent.max-iterations",
      "1",
      "--env.scene.num-envs",
      "2",
      "--gpu-ids",
      "None",
    ],
  )

  train_mod.main()

  assert routed["backend"] == "rsl_rl"
  assert routed["task_id"] == "Mjlab-Cartpole-Swingup"


def test_flashsac_adapter_exposes_final_obs_on_timeout(
  tmp_path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "warp-cache"))
  env_cfg = load_env_cfg("Mjlab-Cartpole-Swingup")
  env_cfg.scene.num_envs = 2
  env_cfg.episode_length_s = env_cfg.sim.mujoco.timestep * env_cfg.decimation

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  try:
    adapter = MjlabFlashSACEnvAdapter(env)
    obs, info = adapter.reset()
    assert obs.shape == (2, adapter.observation_space.shape[-1])
    assert adapter.observation_space.shape[-1] == 10
    assert info["actor_observation_size"] == (5,)

    next_obs, rewards, terminateds, truncateds, step_info = adapter.step(
      torch.zeros((2, 1), dtype=torch.float32).numpy()
    )

    assert next_obs.shape == (2, adapter.observation_space.shape[-1])
    assert rewards.shape == (2,)
    assert terminateds.shape == (2,)
    assert truncateds.shape == (2,)
    assert truncateds.all()
    assert "final_obs" in step_info
    assert step_info["final_obs"].shape == next_obs.shape
  finally:
    env.close()


def test_flashsac_adapter_policy_observation_dim_respects_asymmetry(
  tmp_path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "warp-cache"))
  env_cfg = load_env_cfg("Mjlab-Cartpole-Swingup")
  env_cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  try:
    adapter = MjlabFlashSACEnvAdapter(env)

    assert adapter.actor_dim == 5
    assert adapter.observation_space.shape[-1] == 10
    assert adapter.policy_observation_dim(asymmetric_observation=True) == 5
    assert adapter.policy_observation_dim(asymmetric_observation=False) == 10
  finally:
    env.close()


def test_flashsac_multi_gpu_launch_rejected() -> None:
  cfg = FlashSACTrainConfig.from_task("Mjlab-Cartpole-Swingup")
  cfg = FlashSACTrainConfig(
    env=cfg.env, agent=cfg.agent, registry_name=None, gpu_ids=[0, 1]
  )

  with pytest.raises(ValueError, match="Multi-GPU launch is disabled"):
    train_mod.launch_flashsac_training("Mjlab-Cartpole-Swingup", cfg)


def test_flashsac_tracking_inference_overrides_remove_randomization() -> None:
  env_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=False)
  baseline_env_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=False)

  apply_flashsac_tracking_inference_overrides(env_cfg)

  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  assert motion_cmd.sampling_mode == "start"
  assert env_cfg.observations["actor"].enable_corruption is False
  assert env_cfg.observations["critic"].enable_corruption is False
  assert "push_robot" not in env_cfg.events
  assert "base_com" not in env_cfg.events
  assert "encoder_bias" not in env_cfg.events
  assert "foot_friction" not in env_cfg.events
  assert "ee_body_pos" in env_cfg.terminations
  assert (
    env_cfg.terminations["anchor_pos"].params["threshold"]
    == baseline_env_cfg.terminations["anchor_pos"].params["threshold"]
  )
  assert (
    env_cfg.terminations["anchor_ori"].params["threshold"]
    == baseline_env_cfg.terminations["anchor_ori"].params["threshold"]
  )


def test_apply_flashsac_tracking_train_overrides_preserve_tracking_env_semantics() -> (
  None
):
  env_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=False)
  baseline_env_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=False)

  apply_flashsac_tracking_train_overrides(env_cfg)

  motion_cmd = env_cfg.commands["motion"]
  baseline_motion_cmd = baseline_env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  assert isinstance(baseline_motion_cmd, MotionCommandCfg)
  assert motion_cmd.sampling_mode == baseline_motion_cmd.sampling_mode
  assert motion_cmd.pose_range == baseline_motion_cmd.pose_range
  assert motion_cmd.velocity_range == baseline_motion_cmd.velocity_range
  assert motion_cmd.joint_position_range == baseline_motion_cmd.joint_position_range
  assert (
    env_cfg.observations["actor"].enable_corruption
    == baseline_env_cfg.observations["actor"].enable_corruption
  )
  assert set(env_cfg.events) == set(baseline_env_cfg.events)
  assert set(env_cfg.terminations) == set(baseline_env_cfg.terminations)


def test_flashsac_step_helpers_match_upstream_equivalent_budget() -> None:
  assert (
    interaction_steps_from_total_env_steps(50_000_896, num_envs=1024)
    == FLASHSAC_TRACKING_UPSTREAM_INTERACTION_STEPS
  )
  assert (
    interaction_steps_from_total_env_steps(
      FLASHSAC_TRACKING_TOTAL_ENV_STEPS,
      num_envs=FLASHSAC_TRACKING_NUM_ENVS,
    )
    == FLASHSAC_TRACKING_UPSTREAM_INTERACTION_STEPS
  )
  assert (
    total_env_steps_from_interaction_steps(
      FLASHSAC_TRACKING_UPSTREAM_INTERACTION_STEPS,
      num_envs=FLASHSAC_TRACKING_NUM_ENVS,
    )
    == FLASHSAC_TRACKING_TOTAL_ENV_STEPS
  )
  assert (
    checkpoint_interval_from_total_env_steps(
      FLASHSAC_TRACKING_TOTAL_ENV_STEPS,
      num_envs=FLASHSAC_TRACKING_NUM_ENVS,
      checkpoint_count=FLASHSAC_TRACKING_CHECKPOINT_COUNT,
    )
    == 4883
  )


def test_flashsac_checkpoint_cadence_recomputes_after_budget_override() -> None:
  cfg = FlashSACTrainConfig.from_task("Mjlab-Tracking-Flat-Unitree-G1")

  assert (
    cfg.agent.save_checkpoint_per_interaction_step
    == FLASHSAC_TRACKING_DEFAULT_CHECKPOINT_INTERVAL
  )
  cfg.agent.num_env_steps = 10_000_000
  cfg.env.scene.num_envs = 4096

  maybe_recompute_flashsac_tracking_checkpoint_cadence(cfg.env, cfg.agent)

  assert (
    cfg.agent.save_checkpoint_per_interaction_step
    == checkpoint_interval_from_total_env_steps(
      total_env_steps=10_000_000,
      num_envs=4096,
      checkpoint_count=FLASHSAC_TRACKING_CHECKPOINT_COUNT,
    )
  )


def test_resolve_flashsac_checkpoint_dir_accepts_dir_and_file(tmp_path: Path) -> None:
  checkpoint_dir = tmp_path / "step_10"
  checkpoint_dir.mkdir()
  actor_file = checkpoint_dir / "actor.pt"
  actor_file.write_bytes(b"")

  assert resolve_flashsac_checkpoint_dir(checkpoint_dir) == checkpoint_dir.resolve()
  assert resolve_flashsac_checkpoint_dir(actor_file) == checkpoint_dir.resolve()


def test_make_flashsac_inference_cfg_cpu_shrinks_runtime_state() -> None:
  cfg = FlashSACRunnerCfg(
    device_type="cuda",
    buffer_device_type="cuda",
    buffer_max_length=123,
    buffer_min_length=12,
    sample_batch_size=32,
    use_amp=True,
    load_optimizer=True,
  )

  inference_cfg = make_flashsac_inference_cfg(cfg, device="cpu")

  assert inference_cfg.device_type == "cpu"
  assert inference_cfg.buffer_device_type == "cpu"
  assert inference_cfg.buffer_max_length == 1
  assert inference_cfg.buffer_min_length == 1
  assert inference_cfg.sample_batch_size == 1
  assert inference_cfg.use_compile is False
  assert inference_cfg.use_amp is False
  assert inference_cfg.load_optimizer is False


def test_make_flashsac_inference_cfg_preserves_explicit_cuda_device() -> None:
  cfg = FlashSACRunnerCfg(device_type="cuda", buffer_device_type="cuda")

  inference_cfg = make_flashsac_inference_cfg(cfg, device="cuda:1")

  assert inference_cfg.device_type == "cuda:1"
  assert inference_cfg.buffer_device_type == "cuda:1"


def test_observation_normalizer_roundtrip(tmp_path: Path) -> None:
  normalizer = ObservationNormalizer(shape=(2,), device=torch.device("cpu"))
  normalizer.update(torch.tensor([[1.0, 3.0], [3.0, 5.0]], dtype=torch.float32))
  normalized = normalizer.normalize(torch.tensor([[2.0, 4.0]], dtype=torch.float32))
  save_path = tmp_path / "obs_norm.pt"
  normalizer.save(str(save_path))

  reloaded = ObservationNormalizer(shape=(2,), device=torch.device("cpu"))
  reloaded.load(str(save_path))
  reloaded_normalized = reloaded.normalize(
    torch.tensor([[2.0, 4.0]], dtype=torch.float32)
  )

  assert torch.allclose(normalized, reloaded_normalized)
  assert torch.allclose(reloaded.obs_rms.mean, torch.tensor([2.0, 4.0]))


def test_load_flashsac_runner_cfg_reads_yaml_and_normalizes_for_inference(
  tmp_path: Path,
) -> None:
  run_dir = tmp_path / "run"
  checkpoint_dir = run_dir / "step_20"
  (run_dir / "params").mkdir(parents=True)
  checkpoint_dir.mkdir()
  with (run_dir / "params" / "agent.yaml").open("w", encoding="utf-8") as fh:
    yaml.safe_dump(
      {
        "device_type": "cuda",
        "buffer_device_type": "cuda",
        "buffer_max_length": 999,
        "buffer_min_length": 111,
        "sample_batch_size": 64,
        "use_amp": True,
      },
      fh,
    )

  cfg = load_flashsac_runner_cfg(checkpoint_dir, device="cpu")

  assert cfg.device_type == "cpu"
  assert cfg.buffer_device_type == "cpu"
  assert cfg.buffer_max_length == 1
  assert cfg.sample_batch_size == 1


def test_load_flashsac_saved_runner_cfg_preserves_training_shape_flags(
  tmp_path: Path,
) -> None:
  run_dir = tmp_path / "run"
  checkpoint_dir = run_dir / "step_20"
  (run_dir / "params").mkdir(parents=True)
  checkpoint_dir.mkdir()
  with (run_dir / "params" / "agent.yaml").open("w", encoding="utf-8") as fh:
    yaml.safe_dump(
      {
        "normalize_observation": True,
        "observation_clip_value": 5.0,
        "normalized_G_max": 7.0,
        "asymmetric_observation": True,
        "actor_hidden_dim": 64,
        "critic_hidden_dim": 128,
        "critic_num_bins": 51,
      },
      fh,
    )

  cfg = load_flashsac_saved_runner_cfg(checkpoint_dir)

  assert cfg.normalize_observation is True
  assert cfg.observation_clip_value == 5.0
  assert cfg.normalized_G_max == 7.0
  assert cfg.asymmetric_observation is True
  assert cfg.actor_hidden_dim == 64
  assert cfg.critic_hidden_dim == 128
  assert cfg.critic_num_bins == 51


def test_apply_resume_agent_contract_restores_architecture_shaping_flags(
  tmp_path: Path,
) -> None:
  run_dir = tmp_path / "run"
  checkpoint_dir = run_dir / "step_20"
  (run_dir / "params").mkdir(parents=True)
  checkpoint_dir.mkdir()
  with (run_dir / "params" / "agent.yaml").open("w", encoding="utf-8") as fh:
    yaml.safe_dump(
      {
        "normalize_observation": True,
        "load_observation_normalizer": True,
        "observation_clip_value": 3.0,
        "normalized_G_max": 9.0,
        "asymmetric_observation": True,
        "actor_num_blocks": 3,
        "actor_hidden_dim": 96,
        "critic_num_blocks": 4,
        "critic_hidden_dim": 192,
        "critic_num_bins": 77,
      },
      fh,
    )

  runtime_cfg = FlashSACRunnerCfg(
    normalize_observation=False,
    observation_clip_value=10.0,
    normalized_G_max=5.0,
    asymmetric_observation=False,
    actor_num_blocks=2,
    actor_hidden_dim=128,
    critic_num_blocks=2,
    critic_hidden_dim=256,
    critic_num_bins=101,
  )

  resumed_cfg = _apply_resume_agent_contract(
    runtime_cfg,
    checkpoint_path=checkpoint_dir,
  )

  assert resumed_cfg.normalize_observation is True
  assert resumed_cfg.load_observation_normalizer is True
  assert resumed_cfg.observation_clip_value == 3.0
  assert resumed_cfg.normalized_G_max == 9.0
  assert resumed_cfg.asymmetric_observation is True
  assert resumed_cfg.actor_num_blocks == 3
  assert resumed_cfg.actor_hidden_dim == 96
  assert resumed_cfg.critic_num_blocks == 4
  assert resumed_cfg.critic_hidden_dim == 192
  assert resumed_cfg.critic_num_bins == 77


def test_flashsac_viewer_env_wrapper_tracks_latest_obs() -> None:
  class FakeEnv:
    def __init__(self) -> None:
      self.unwrapped = self
      self.cfg = object()
      self.device = "cpu"
      self.num_envs = 1
      self.render_mode = None
      self._obs = {"actor": torch.tensor([[1.0]])}

    def reset(self):
      self._obs = {"actor": torch.tensor([[1.0]])}
      return self._obs, {}

    def step(self, actions):
      del actions
      self._obs = {"actor": torch.tensor([[2.0]])}
      return (
        self._obs,
        torch.tensor([0.0]),
        torch.tensor([False]),
        torch.tensor([False]),
        {},
      )

    def close(self) -> None:
      return None

  wrapped = play_mod.FlashSACViewerEnvWrapper(FakeEnv())

  assert wrapped.get_observations()["actor"].item() == 1.0
  wrapped.step(torch.zeros((1, 1)))
  assert wrapped.get_observations()["actor"].item() == 2.0


def test_randomize_episode_horizons_spreads_vector_timeouts() -> None:
  class FakeEnv:
    def __init__(self) -> None:
      self.num_envs = 8
      self.max_episode_length = 100
      self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)

  env = FakeEnv()
  _randomize_episode_horizons(env)

  assert env.episode_length_buf.shape == (8,)
  assert torch.all(env.episode_length_buf >= 0)
  assert torch.all(env.episode_length_buf < 100)


def test_resolve_replay_next_observations_prefers_final_obs_only_for_done_envs() -> (
  None
):
  next_observations = np.array(
    [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
    dtype=np.float32,
  )
  final_obs = np.array(
    [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]],
    dtype=np.float32,
  )
  terminateds = np.array([False, True, False], dtype=np.bool_)
  truncateds = np.array([True, False, False], dtype=np.bool_)

  replay_next_observations = _resolve_replay_next_observations(
    next_observations,
    terminateds,
    truncateds,
    {"final_obs": final_obs},
  )

  assert np.allclose(
    replay_next_observations,
    np.array([[10.0, 10.0], [20.0, 20.0], [3.0, 3.0]], dtype=np.float32),
  )
  assert np.allclose(
    next_observations,
    np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
  )


def test_checkpoint_summary_entry_tracks_env_step_and_checkpoint_name() -> None:
  checkpoint_dir = Path("/tmp/flashsac-run/step_13")

  summary = _checkpoint_summary_entry(
    interaction_step=13,
    num_envs=4096,
    checkpoint_dir=checkpoint_dir,
    kind="periodic",
  )

  assert summary == {
    "kind": "periodic",
    "interaction_step": 13,
    "env_step": total_env_steps_from_interaction_steps(13, num_envs=4096),
    "checkpoint_dir": str(checkpoint_dir),
    "checkpoint_name": "step_13",
  }


def test_write_training_audit_artifacts_emits_summary_bundle(tmp_path: Path) -> None:
  log_dir = tmp_path / "run"
  params_dir = log_dir / "params"
  params_dir.mkdir(parents=True)
  for name in ("env.yaml", "agent.yaml", "runtime.yaml"):
    (params_dir / name).write_text(f"{name}: true\n", encoding="utf-8")

  runtime_metadata = {
    "task_id": "Mjlab-Tracking-Flat-Unitree-G1",
    "seed": 42,
    "device": "cpu",
    "buffer_device_type": "cpu",
    "use_amp": False,
    "cuda_visible_devices": "",
    "num_envs": 4,
    "num_env_steps": 50_000,
    "num_interaction_steps": 13,
    "target_update_budget": 26.0,
    "actual_update_steps": 11,
    "actual_updates_per_interaction_step": 11 / 13,
    "final_env_steps": 52_000,
    "final_interaction_steps": 13,
    "final_replay_size": 512,
    "final_replay_fill_ratio": 0.512,
    "checkpoint_count": 2,
    "final_checkpoint_dir": str(log_dir / "step_13"),
  }
  checkpoint_summaries = [
    _checkpoint_summary_entry(
      interaction_step=13,
      num_envs=4,
      checkpoint_dir=log_dir / "step_13",
      kind="final",
    )
  ]
  log_history = [
    {
      "step": 40_960.0,
      "Perf/effective_updates_per_interaction_step": 1.0,
      "Perf/replay_fill_ratio": 0.4,
    }
  ]

  outputs = _write_training_audit_artifacts(
    log_dir=log_dir,
    runtime_metadata=runtime_metadata,
    checkpoint_summaries=checkpoint_summaries,
    log_history=log_history,
  )

  metrics_path = Path(outputs["summary_metrics_file"])
  checkpoints_path = Path(outputs["checkpoint_summary_file"])
  history_path = Path(outputs["log_history_file"])
  artifact_capture_path = Path(outputs["artifact_capture_file"])

  metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
  checkpoints_payload = json.loads(checkpoints_path.read_text(encoding="utf-8"))
  history_payload = json.loads(history_path.read_text(encoding="utf-8"))
  artifact_capture_payload = json.loads(
    artifact_capture_path.read_text(encoding="utf-8")
  )

  assert metrics_payload["actual_update_steps"] == 11
  assert metrics_payload["final_replay_fill_ratio"] == 0.512
  assert metrics_payload["last_logged_metrics"] == log_history[-1]
  assert checkpoints_payload["checkpoints"] == checkpoint_summaries
  assert history_payload == {"entries": log_history}
  assert artifact_capture_payload["metrics_summary_path"] == str(metrics_path)
  assert artifact_capture_payload["runtime_yaml_path"] == str(
    log_dir / "params" / "runtime.yaml"
  )
  assert (
    "actual_updates_per_interaction_step"
    in artifact_capture_payload["required_audit_fields"]
  )


def test_network_bundle_load_accepts_compiled_prefix_checkpoint(tmp_path: Path) -> None:
  model = torch.nn.Linear(2, 3)
  prefixed_state = {
    f"_orig_mod.{key}": value.clone() for key, value in model.state_dict().items()
  }
  checkpoint_path = tmp_path / "prefixed.pt"
  torch.save(
    {
      "network_state_dict": prefixed_state,
      "optimizer_state_dict": None,
      "scheduler_state_dict": None,
    },
    checkpoint_path,
  )

  loaded_model = torch.nn.Linear(2, 3)
  loaded_bundle = NetworkBundle(loaded_model)
  loaded_bundle.load(str(checkpoint_path), load_optimizer=False)

  for key, value in model.state_dict().items():
    assert torch.equal(value, loaded_model.state_dict()[key])


def test_network_bundle_load_accepts_portable_checkpoint_for_compiled_wrapper(
  tmp_path: Path,
) -> None:
  class FakeCompiledModule(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module) -> None:
      super().__init__()
      self._orig_mod = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self._orig_mod(x)

  model = torch.nn.Linear(2, 3)
  checkpoint_path = tmp_path / "portable.pt"
  torch.save(
    {
      "network_state_dict": model.state_dict(),
      "optimizer_state_dict": None,
      "scheduler_state_dict": None,
    },
    checkpoint_path,
  )

  loaded_model = torch.nn.Linear(2, 3)
  compiled_wrapper = FakeCompiledModule(loaded_model)
  loaded_bundle = NetworkBundle(compiled_wrapper)
  loaded_bundle.load(str(checkpoint_path), load_optimizer=False)

  for key, value in model.state_dict().items():
    assert torch.equal(value, loaded_model.state_dict()[key])


def test_run_play_routes_flashsac_backend(monkeypatch: pytest.MonkeyPatch) -> None:
  routed: dict[str, object] = {}

  def fake_flashsac(task_id: str, cfg: object) -> None:
    routed["backend"] = "flashsac"
    routed["task_id"] = task_id
    routed["cfg"] = cfg

  def fail_default(*args, **kwargs) -> None:
    raise AssertionError("default play path should not run for --backend flashsac")

  monkeypatch.setattr(play_mod, "_run_flashsac_play", fake_flashsac)
  monkeypatch.setattr(play_mod, "_run_rsl_rl_play", fail_default)

  play_mod.run_play(
    "Mjlab-Cartpole-Swingup",
    play_mod.PlayConfig(backend="flashsac", checkpoint_file="."),
  )

  assert routed["backend"] == "flashsac"
  assert routed["task_id"] == "Mjlab-Cartpole-Swingup"


def test_run_evaluate_routes_flashsac_backend(monkeypatch: pytest.MonkeyPatch) -> None:
  routed: dict[str, object] = {}

  def fake_flashsac(task_id: str, cfg: object, device: str) -> dict[str, float]:
    routed["backend"] = "flashsac"
    routed["task_id"] = task_id
    routed["cfg"] = cfg
    routed["device"] = device
    return {"success_rate": 1.0}

  def fail_default(*args, **kwargs) -> dict[str, float]:
    raise AssertionError("default evaluate path should not run for flashsac")

  monkeypatch.setattr(
    "mjlab.tasks.tracking.scripts.evaluate._run_flashsac_evaluate",
    fake_flashsac,
  )
  monkeypatch.setattr(
    "mjlab.tasks.tracking.scripts.evaluate._run_rsl_rl_evaluate",
    fail_default,
  )

  metrics = run_evaluate(
    "Mjlab-Tracking-Flat-Unitree-G1",
    EvaluateConfig(backend="flashsac", checkpoint_file="."),
  )

  assert metrics["success_rate"] == 1.0
  assert routed["backend"] == "flashsac"
  assert routed["task_id"] == "Mjlab-Tracking-Flat-Unitree-G1"


def test_flashsac_evaluate_preserves_checkpoint_semantics_when_parity_exists(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  motion_file = tmp_path / "motion.npz"
  motion_file.write_bytes(b"motion")
  captured: dict[str, object] = {}

  class StopEval(RuntimeError):
    pass

  class FakeEnv:
    def __init__(self, *, cfg, device: str) -> None:
      del device
      captured["env_cfg"] = cfg
      raise StopEval()

  def fake_checkpoint_parity(env_cfg, checkpoint_file):
    del checkpoint_file
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "adaptive"
    env_cfg.observations["actor"].enable_corruption = True
    env_cfg.observations["critic"].enable_corruption = False
    return object()

  monkeypatch.setattr(evaluate_mod, "ManagerBasedRlEnv", FakeEnv)
  monkeypatch.setattr(
    evaluate_mod,
    "apply_flashsac_checkpoint_env_parity",
    fake_checkpoint_parity,
  )

  with pytest.raises(StopEval):
    evaluate_mod._run_flashsac_evaluate(
      "Mjlab-Tracking-Flat-Unitree-G1",
      EvaluateConfig(
        backend="flashsac",
        checkpoint_file="/tmp/fake-step",
        motion_file=str(motion_file),
        num_envs=4,
        device="cpu",
      ),
      "cpu",
    )

  env_cfg = cast(object, captured["env_cfg"])
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  assert motion_cmd.sampling_mode == "adaptive"
  assert env_cfg.observations["actor"].enable_corruption is True


def test_flashsac_evaluate_uses_deterministic_fallback_without_parity(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  motion_file = tmp_path / "motion.npz"
  motion_file.write_bytes(b"motion")
  captured: dict[str, object] = {}

  class StopEval(RuntimeError):
    pass

  class FakeEnv:
    def __init__(self, *, cfg, device: str) -> None:
      del device
      captured["env_cfg"] = cfg
      raise StopEval()

  monkeypatch.setattr(evaluate_mod, "ManagerBasedRlEnv", FakeEnv)
  monkeypatch.setattr(
    evaluate_mod,
    "apply_flashsac_checkpoint_env_parity",
    lambda env_cfg, checkpoint_file: None,
  )

  with pytest.raises(StopEval):
    evaluate_mod._run_flashsac_evaluate(
      "Mjlab-Tracking-Flat-Unitree-G1",
      EvaluateConfig(
        backend="flashsac",
        checkpoint_file="/tmp/fake-step",
        motion_file=str(motion_file),
        num_envs=4,
        device="cpu",
      ),
      "cpu",
    )

  env_cfg = cast(object, captured["env_cfg"])
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  assert motion_cmd.sampling_mode == "start"
  assert env_cfg.observations["actor"].enable_corruption is False
  assert env_cfg.observations["critic"].enable_corruption is False


def test_resolve_rsl_rl_checkpoint_path_accepts_local_checkpoint(
  tmp_path: Path,
) -> None:
  checkpoint_path = tmp_path / "model_123.pt"
  checkpoint_path.write_bytes(b"checkpoint")

  resolved = evaluate_mod._resolve_rsl_rl_checkpoint_path(
    "g1_tracking",
    EvaluateConfig(checkpoint_file=str(checkpoint_path)),
  )

  assert resolved == checkpoint_path


def test_resolve_rsl_rl_checkpoint_path_requires_wandb_without_checkpoint() -> None:
  with pytest.raises(
    ValueError,
    match="RSL-RL evaluation requires `wandb_run_path` when `checkpoint_file` is not provided.",
  ):
    evaluate_mod._resolve_rsl_rl_checkpoint_path("g1_tracking", EvaluateConfig())


def test_resolve_rsl_rl_motion_file_uses_local_motion_without_wandb() -> None:
  motion_cfg = cast(MotionCommandCfg, SimpleNamespace(motion_file=None))

  evaluate_mod._resolve_rsl_rl_motion_file(
    motion_cfg,
    EvaluateConfig(
      checkpoint_file="/tmp/model_123.pt",
      motion_file="/tmp/motion.npz",
    ),
  )

  assert motion_cfg.motion_file == "/tmp/motion.npz"


def test_resolve_rsl_rl_motion_file_requires_motion_or_wandb_for_local_checkpoint() -> (
  None
):
  motion_cfg = cast(MotionCommandCfg, SimpleNamespace(motion_file=None))

  with pytest.raises(
    ValueError,
    match="Tracking evaluation requires `motion_file` when using `checkpoint_file`, or provide `wandb_run_path` so the motion artifact can be resolved.",
  ):
    evaluate_mod._resolve_rsl_rl_motion_file(
      motion_cfg,
      EvaluateConfig(checkpoint_file="/tmp/model_123.pt"),
    )


def test_resolve_rsl_rl_motion_file_can_resolve_from_wandb(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  motion_cfg = cast(MotionCommandCfg, SimpleNamespace(motion_file=None))
  calls: dict[str, object] = {}

  def fake_resolve_tracking_motion_file(
    motion_cfg_arg,
    *,
    motion_file: str | None,
    registry_name: str | None,
    wandb_run_path: str | None,
    checkpoint_file: str | None = None,
  ) -> None:
    calls["motion_file"] = motion_file
    calls["registry_name"] = registry_name
    calls["wandb_run_path"] = wandb_run_path
    calls["checkpoint_file"] = checkpoint_file
    motion_cfg_arg.motion_file = "/tmp/downloaded-motion.npz"

  monkeypatch.setattr(
    evaluate_mod,
    "resolve_tracking_motion_file",
    fake_resolve_tracking_motion_file,
  )

  evaluate_mod._resolve_rsl_rl_motion_file(
    motion_cfg,
    EvaluateConfig(wandb_run_path="entity/project/run123"),
  )

  assert motion_cfg.motion_file == "/tmp/downloaded-motion.npz"
  assert calls == {
    "motion_file": None,
    "registry_name": None,
    "wandb_run_path": "entity/project/run123",
    "checkpoint_file": None,
  }


def test_flashsac_final_obs_with_history_differs_from_reset_obs(
  tmp_path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "warp-cache"))
  env_cfg = load_env_cfg("Mjlab-Cartpole-Swingup")
  env_cfg.scene.num_envs = 1
  env_cfg.episode_length_s = env_cfg.sim.mujoco.timestep * env_cfg.decimation
  env_cfg.observations["actor"].history_length = 2
  env_cfg.observations["critic"].history_length = 2

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  try:
    adapter = MjlabFlashSACEnvAdapter(env)
    reset_obs, _ = adapter.reset()
    _next_obs, _rewards, _terminateds, truncateds, step_info = adapter.step(
      np.array([[1.0]], dtype=np.float32)
    )

    assert truncateds.all()
    final_obs = step_info["final_obs"]
    assert final_obs.shape == reset_obs.shape
    assert not np.allclose(final_obs, reset_obs)
  finally:
    env.close()


@pytest.mark.slow
def test_flashsac_adapter_g1_velocity_final_obs_smoke(
  tmp_path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "warp-cache"))
  env_cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
  env_cfg.scene.num_envs = 1
  env_cfg.episode_length_s = env_cfg.sim.mujoco.timestep * env_cfg.decimation

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  try:
    adapter = MjlabFlashSACEnvAdapter(env)
    reset_obs, _ = adapter.reset()
    next_obs, _rewards, _terminateds, truncateds, step_info = adapter.step(
      np.zeros((1, adapter.action_space.shape[-1]), dtype=np.float32)
    )

    assert adapter.observation_space.shape[-1] == 210
    assert truncateds.all()
    assert "final_obs" in step_info
    final_obs = step_info["final_obs"]
    assert final_obs.shape == reset_obs.shape == next_obs.shape
    assert not np.allclose(final_obs, reset_obs)
    assert not np.allclose(final_obs, next_obs)
  finally:
    env.close()


def test_flashsac_replay_buffer_roundtrip(tmp_path) -> None:
  cfg = FlashSACTrainConfig.from_task("Mjlab-Cartpole-Swingup")
  cfg.agent.device_type = "cpu"
  cfg.agent.buffer_device_type = "cpu"
  cfg.agent.use_compile = False
  cfg.agent.use_amp = False
  cfg.agent.buffer_min_length = 1
  cfg.agent.sample_batch_size = 2

  agent = FlashSACAgent(
    observation_dim=10,
    action_dim=1,
    actor_observation_dim=5,
    cfg=cfg.agent,
  )
  transition = {
    "observation": np.zeros((2, 10), dtype=np.float32),
    "action": np.zeros((2, 1), dtype=np.float32),
    "reward": np.ones((2,), dtype=np.float32),
    "terminated": np.zeros((2,), dtype=np.bool_),
    "truncated": np.zeros((2,), dtype=np.bool_),
    "next_observation": np.ones((2, 10), dtype=np.float32),
  }
  agent.process_transition(transition)
  assert len(agent.replay_buffer) == 2

  save_dir = tmp_path / "flashsac-buffer"
  agent.save_replay_buffer(str(save_dir))

  reloaded = FlashSACAgent(
    observation_dim=10,
    action_dim=1,
    actor_observation_dim=5,
    cfg=cfg.agent,
  )
  reloaded.load_replay_buffer(str(save_dir))

  assert len(reloaded.replay_buffer) == 2
  assert reloaded.can_start_training()
