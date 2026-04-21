from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.flashsac.adapter import MjlabFlashSACEnvAdapter
from mjlab.flashsac.agent import FlashSACAgent
from mjlab.flashsac.config import (
  FlashSACRunnerCfg,
  FlashSACTrainConfig,
  maybe_recompute_flashsac_tracking_checkpoint_cadence,
)
from mjlab.flashsac.runtime import (
  apply_flashsac_checkpoint_env_parity,
  load_flashsac_saved_runner_cfg,
  render_flashsac_checkpoint_env_parity_audit,
)
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.training_steps import (
  interaction_steps_from_total_env_steps,
  total_env_steps_from_interaction_steps,
)


class AverageMeter:
  def __init__(self) -> None:
    self.sum = 0.0
    self.count = 0

  def update(self, value: float, n: int = 1) -> None:
    self.sum += value * n
    self.count += n

  def average(self) -> float:
    return self.sum / max(self.count, 1)


class FlashSACLogger:
  def __init__(self, cfg: FlashSACTrainConfig, log_dir: Path):
    self.cfg = cfg
    self.log_dir = log_dir
    self.scalars: dict[str, AverageMeter] = defaultdict(AverageMeter)
    self.last_payload: dict[str, float] = {}
    self.history: list[dict[str, float]] = []
    self.kind = cfg.agent.logger
    self.writer = None
    self.wandb = None
    if self.kind == "tensorboard":
      from torch.utils.tensorboard import SummaryWriter

      self.writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
      self.writer.add_text("config", str(cfg))
    else:
      import wandb

      self.wandb = wandb
      wandb.init(
        project=cfg.agent.wandb_project,
        entity=cfg.agent.wandb_entity,
        group=cfg.agent.wandb_group,
        name=cfg.agent.run_name or None,
        dir=str(log_dir),
        config={
          "backend": "flashsac",
          "task": cfg.agent.experiment_name,
          "num_env_steps": cfg.agent.num_env_steps,
          "updates_per_interaction_step": cfg.agent.updates_per_interaction_step,
        },
        tags=list(cfg.agent.wandb_tags),
      )

  def update_metric(self, **kwargs: Any) -> None:
    for key, value in kwargs.items():
      if isinstance(value, (float, int)):
        self.scalars[key].update(float(value))

  def log_metric(self, step: int, step_metrics: dict[str, float] | None = None) -> None:
    payload = {key: meter.average() for key, meter in self.scalars.items()}
    if step_metrics is not None:
      payload.update(step_metrics)
    history_entry = {
      "step": float(step),
      **{key: float(value) for key, value in payload.items()},
    }
    self.last_payload = history_entry
    self.history.append(history_entry)
    if self.writer is not None:
      for key, value in payload.items():
        self.writer.add_scalar(key, value, global_step=step)  # type: ignore[no-untyped-call]
      self.writer.flush()  # type: ignore[no-untyped-call]
    elif self.wandb is not None:
      self.wandb.log(payload, step=step)

  def reset(self) -> None:
    self.scalars = defaultdict(AverageMeter)


def _resolve_motion_tracking_registry(cfg: FlashSACTrainConfig) -> str | None:
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )
  if not is_tracking_task:
    return None
  motion_cmd = cfg.env.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
    return None
  if cfg.registry_name:
    registry_name = cast(str, cfg.registry_name)
    if ":" not in registry_name:
      registry_name = registry_name + ":latest"
    import wandb

    artifact = wandb.Api().artifact(registry_name)
    motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    return registry_name
  raise ValueError(
    "For tracking tasks, provide either --registry-name your-org/motions/name or "
    "--env.commands.motion.motion-file /path/to/motion.npz"
  )


def _default_logging_interval(num_interaction_steps: int) -> int:
  return max(1, math.ceil(num_interaction_steps / 100))


def _normalize_runtime_agent_cfg(
  agent_cfg: FlashSACRunnerCfg, *, device: str, seed: int
) -> FlashSACRunnerCfg:
  if device == "cpu":
    buffer_device_type = "cpu"
  elif agent_cfg.buffer_device_type.startswith("cuda"):
    buffer_device_type = device
  else:
    buffer_device_type = agent_cfg.buffer_device_type
  return replace(
    agent_cfg,
    seed=seed,
    device_type=device,
    buffer_device_type=buffer_device_type,
    use_amp=device.startswith("cuda") and agent_cfg.use_amp,
  )


def _apply_resume_agent_contract(
  agent_cfg: FlashSACRunnerCfg,
  *,
  checkpoint_path: Path,
) -> FlashSACRunnerCfg:
  saved_cfg = load_flashsac_saved_runner_cfg(checkpoint_path)
  return replace(
    agent_cfg,
    normalize_observation=saved_cfg.normalize_observation,
    load_observation_normalizer=saved_cfg.load_observation_normalizer,
    observation_clip_value=saved_cfg.observation_clip_value,
    normalized_G_max=saved_cfg.normalized_G_max,
    asymmetric_observation=saved_cfg.asymmetric_observation,
    actor_num_blocks=saved_cfg.actor_num_blocks,
    actor_hidden_dim=saved_cfg.actor_hidden_dim,
    critic_num_blocks=saved_cfg.critic_num_blocks,
    critic_hidden_dim=saved_cfg.critic_hidden_dim,
    critic_num_bins=saved_cfg.critic_num_bins,
  )


def _trainer_step_metrics(
  *,
  env_step: int,
  interaction_step: int,
  num_envs: int,
  num_updates: int,
  replay_size: int,
  buffer_min_length: int,
) -> dict[str, float]:
  return {
    "Perf/env_steps": float(env_step),
    "Perf/interaction_steps": float(interaction_step),
    "Perf/num_envs": float(num_envs),
    "Perf/update_steps": float(num_updates),
    "Perf/effective_updates_per_interaction_step": float(
      num_updates / max(interaction_step, 1)
    ),
    "Perf/replay_size": float(replay_size),
    "Perf/replay_fill_ratio": float(replay_size / max(buffer_min_length, 1)),
  }


def _checkpoint_summary_entry(
  *,
  interaction_step: int,
  num_envs: int,
  checkpoint_dir: Path,
  kind: str,
) -> dict[str, Any]:
  return {
    "kind": kind,
    "interaction_step": interaction_step,
    "env_step": total_env_steps_from_interaction_steps(
      interaction_step, num_envs=num_envs
    ),
    "checkpoint_dir": str(checkpoint_dir),
    "checkpoint_name": checkpoint_dir.name,
  }


def _write_json(path: Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )


def _write_training_audit_artifacts(
  *,
  log_dir: Path,
  runtime_metadata: dict[str, Any],
  checkpoint_summaries: list[dict[str, Any]],
  log_history: list[dict[str, float]],
) -> dict[str, str]:
  summary_dir = log_dir / "summary"
  metrics_path = summary_dir / "metrics.json"
  checkpoint_path = summary_dir / "checkpoints.json"
  log_history_path = summary_dir / "log-history.json"
  artifact_capture_path = summary_dir / "artifact-capture.json"
  params_dir = log_dir / "params"
  runtime_yaml_path = params_dir / "runtime.yaml"

  metrics_payload = {
    "task_id": runtime_metadata["task_id"],
    "seed": runtime_metadata["seed"],
    "device": runtime_metadata["device"],
    "buffer_device_type": runtime_metadata["buffer_device_type"],
    "use_amp": runtime_metadata["use_amp"],
    "cuda_visible_devices": runtime_metadata["cuda_visible_devices"],
    "num_envs": runtime_metadata["num_envs"],
    "num_env_steps": runtime_metadata["num_env_steps"],
    "num_interaction_steps": runtime_metadata["num_interaction_steps"],
    "target_update_budget": runtime_metadata["target_update_budget"],
    "actual_update_steps": runtime_metadata["actual_update_steps"],
    "actual_updates_per_interaction_step": runtime_metadata[
      "actual_updates_per_interaction_step"
    ],
    "final_env_steps": runtime_metadata["final_env_steps"],
    "final_interaction_steps": runtime_metadata["final_interaction_steps"],
    "final_replay_size": runtime_metadata["final_replay_size"],
    "final_replay_fill_ratio": runtime_metadata["final_replay_fill_ratio"],
    "checkpoint_count": runtime_metadata["checkpoint_count"],
    "final_checkpoint_dir": runtime_metadata["final_checkpoint_dir"],
    "log_history_points": len(log_history),
    "last_logged_metrics": log_history[-1] if log_history else None,
    "runtime_yaml_path": str(runtime_yaml_path),
    "env_yaml_path": str(params_dir / "env.yaml"),
    "agent_yaml_path": str(params_dir / "agent.yaml"),
    "checkpoint_summary_path": str(checkpoint_path),
    "log_history_path": str(log_history_path),
  }
  checkpoint_payload = {
    "task_id": runtime_metadata["task_id"],
    "seed": runtime_metadata["seed"],
    "num_envs": runtime_metadata["num_envs"],
    "checkpoint_count": runtime_metadata["checkpoint_count"],
    "final_checkpoint_dir": runtime_metadata["final_checkpoint_dir"],
    "checkpoints": checkpoint_summaries,
  }
  artifact_capture_payload = {
    "runtime_yaml_path": str(runtime_yaml_path),
    "env_yaml_path": str(params_dir / "env.yaml"),
    "agent_yaml_path": str(params_dir / "agent.yaml"),
    "metrics_summary_path": str(metrics_path),
    "checkpoint_summary_path": str(checkpoint_path),
    "log_history_path": str(log_history_path),
    "required_audit_fields": [
      "actual_update_steps",
      "actual_updates_per_interaction_step",
      "final_replay_size",
      "final_replay_fill_ratio",
      "checkpoint_count",
      "final_checkpoint_dir",
      "device",
      "buffer_device_type",
      "use_amp",
      "cuda_visible_devices",
    ],
  }

  _write_json(metrics_path, metrics_payload)
  _write_json(checkpoint_path, checkpoint_payload)
  _write_json(log_history_path, {"entries": log_history})
  _write_json(artifact_capture_path, artifact_capture_payload)
  return {
    "summary_metrics_file": str(metrics_path),
    "checkpoint_summary_file": str(checkpoint_path),
    "log_history_file": str(log_history_path),
    "artifact_capture_file": str(artifact_capture_path),
  }


def _randomize_episode_horizons(env: Any) -> None:
  """Decorrelate timeouts across vector envs like upstream IsaacLab FlashSAC."""
  if env.num_envs <= 1:
    return
  env.episode_length_buf = torch.randint_like(
    env.episode_length_buf, high=max(int(env.max_episode_length), 1)
  )


def _resolve_replay_next_observations(
  next_observations: np.ndarray,
  terminateds: np.ndarray,
  truncateds: np.ndarray,
  env_infos: dict[str, Any],
) -> np.ndarray:
  next_buffer_observations = next_observations.copy()
  final_obs = env_infos.get("final_obs")
  if final_obs is None:
    return next_buffer_observations
  for env_idx in range(len(next_buffer_observations)):
    if terminateds[env_idx] or truncateds[env_idx]:
      next_buffer_observations[env_idx] = final_obs[env_idx]
  return next_buffer_observations


def run_flashsac_train(task_id: str, cfg: FlashSACTrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    seed = cfg.agent.seed + local_rank
  configure_torch_backends()
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

  cfg.env.seed = seed
  agent_cfg = _normalize_runtime_agent_cfg(cfg.agent, device=device, seed=seed)
  resume_path: Path | None = None
  checkpoint_parity = None
  if agent_cfg.resume:
    resume_path = get_checkpoint_path(
      log_dir.parent, agent_cfg.load_run, agent_cfg.load_checkpoint
    )
    agent_cfg = _apply_resume_agent_contract(
      agent_cfg,
      checkpoint_path=resume_path,
    )
    checkpoint_parity = apply_flashsac_checkpoint_env_parity(
      cfg.env,
      resume_path,
    )
    if checkpoint_parity is not None:
      print(render_flashsac_checkpoint_env_parity_audit(checkpoint_parity))
  runtime_cfg = FlashSACTrainConfig(
    env=cfg.env,
    agent=agent_cfg,
    registry_name=cfg.registry_name,
    gpu_ids=cfg.gpu_ids,
  )
  _resolve_motion_tracking_registry(runtime_cfg)

  env = ManagerBasedRlEnv(cfg=cfg.env, device=device)
  adapter = MjlabFlashSACEnvAdapter(env)
  observations, env_info = adapter.reset()
  _randomize_episode_horizons(env)
  num_interaction_steps = max(
    1,
    interaction_steps_from_total_env_steps(
      total_env_steps=agent_cfg.num_env_steps,
      num_envs=env.num_envs,
    ),
  )
  logging_every = agent_cfg.logging_per_interaction_step or _default_logging_interval(
    num_interaction_steps
  )
  checkpoint_every = (
    agent_cfg.save_checkpoint_per_interaction_step or num_interaction_steps
  )
  runtime_metadata: dict[str, Any] = {
    "task_id": task_id,
    "seed": seed,
    "device": device,
    "buffer_device_type": agent_cfg.buffer_device_type,
    "use_amp": agent_cfg.use_amp,
    "cuda_visible_devices": cuda_visible,
    "num_envs": env.num_envs,
    "num_env_steps": agent_cfg.num_env_steps,
    "num_interaction_steps": num_interaction_steps,
    "updates_per_interaction_step": agent_cfg.updates_per_interaction_step,
    "target_update_budget": float(
      num_interaction_steps * agent_cfg.updates_per_interaction_step
    ),
    "logging_per_interaction_step": logging_every,
    "save_checkpoint_per_interaction_step": checkpoint_every,
  }
  dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))
  dump_yaml(log_dir / "params" / "runtime.yaml", runtime_metadata)
  print(
    "[INFO] FlashSAC training with "
    f"device={device}, buffer_device={agent_cfg.buffer_device_type}, "
    f"seed={seed}, use_amp={agent_cfg.use_amp}, num_envs={env.num_envs}"
  )

  agent = FlashSACAgent(
    observation_dim=adapter.observation_space.shape[-1],
    action_dim=adapter.action_space.shape[-1],
    actor_observation_dim=adapter.policy_observation_dim(
      asymmetric_observation=agent_cfg.asymmetric_observation
    ),
    cfg=agent_cfg,
  )

  if agent_cfg.resume:
    assert resume_path is not None
    agent.load(str(resume_path))
    replay_buffer_path = resume_path / "replay_buffer.pt"
    if agent_cfg.load_replay_buffer and replay_buffer_path.exists():
      agent.load_replay_buffer(str(resume_path))
    elif agent_cfg.load_replay_buffer:
      print(
        f"[WARN] Replay buffer checkpoint not found at {replay_buffer_path}; resuming weights only."
      )

  logger = FlashSACLogger(runtime_cfg, log_dir)
  transition: Optional[dict[str, Any]] = None
  update_counter = 0.0
  num_updates = 0
  checkpoint_count = 0
  checkpoint_summaries: list[dict[str, Any]] = []

  for interaction_step in range(1, num_interaction_steps + 1):
    env_step = total_env_steps_from_interaction_steps(
      interaction_step, num_envs=env.num_envs
    )
    if agent.can_start_training() and transition is not None:
      actions = agent.sample_actions(
        interaction_step, prev_transition=transition, training=True
      )
    else:
      actions = adapter.sample_random_actions()
    next_observations, rewards, terminateds, truncateds, env_infos = adapter.step(
      actions
    )
    next_buffer_observations = _resolve_replay_next_observations(
      next_observations,
      terminateds,
      truncateds,
      env_infos,
    )

    if "episode_info" in env_infos:
      logger.update_metric(**env_infos["episode_info"])

    transition = {
      "observation": observations,
      "action": actions,
      "reward": rewards,
      "terminated": terminateds,
      "truncated": truncateds,
      "next_observation": next_buffer_observations,
    }
    agent.process_transition(transition)
    transition["next_observation"] = next_observations
    observations = next_observations

    if agent.can_start_training():
      update_counter += agent_cfg.updates_per_interaction_step
      while update_counter >= 1.0:
        logger.update_metric(**agent.update())
        update_counter -= 1.0
        num_updates += 1
      if interaction_step % logging_every == 0:
        logger.log_metric(
          step=env_step,
          step_metrics=_trainer_step_metrics(
            env_step=env_step,
            interaction_step=interaction_step,
            num_envs=env.num_envs,
            num_updates=num_updates,
            replay_size=len(agent.replay_buffer),
            buffer_min_length=agent_cfg.buffer_min_length,
          ),
        )
        logger.reset()
      if checkpoint_every and interaction_step % checkpoint_every == 0:
        save_dir = log_dir / f"step_{interaction_step}"
        agent.save(str(save_dir))
        checkpoint_count += 1
        checkpoint_summaries.append(
          _checkpoint_summary_entry(
            interaction_step=interaction_step,
            num_envs=env.num_envs,
            checkpoint_dir=save_dir,
            kind="periodic",
          )
        )
        if (
          agent_cfg.save_buffer_per_interaction_step
          and interaction_step % agent_cfg.save_buffer_per_interaction_step == 0
        ):
          agent.save_replay_buffer(str(save_dir))

  final_save_dir = log_dir / f"step_{num_interaction_steps}"
  agent.save(str(final_save_dir))
  checkpoint_count += 1
  checkpoint_summaries.append(
    _checkpoint_summary_entry(
      interaction_step=num_interaction_steps,
      num_envs=env.num_envs,
      checkpoint_dir=final_save_dir,
      kind="final",
    )
  )
  if agent_cfg.save_final_replay_buffer:
    agent.save_replay_buffer(str(final_save_dir))
  final_env_step = total_env_steps_from_interaction_steps(
    num_interaction_steps, num_envs=env.num_envs
  )
  runtime_metadata.update(
    {
      "final_env_steps": final_env_step,
      "final_interaction_steps": num_interaction_steps,
      "actual_update_steps": num_updates,
      "actual_updates_per_interaction_step": float(
        num_updates / max(num_interaction_steps, 1)
      ),
      "final_replay_size": len(agent.replay_buffer),
      "final_replay_fill_ratio": float(
        len(agent.replay_buffer) / max(agent_cfg.buffer_min_length, 1)
      ),
      "checkpoint_count": checkpoint_count,
      "final_checkpoint_dir": str(final_save_dir),
      "checkpoint_summaries": checkpoint_summaries,
    }
  )
  logger.log_metric(
    step=final_env_step,
    step_metrics=_trainer_step_metrics(
      env_step=final_env_step,
      interaction_step=num_interaction_steps,
      num_envs=env.num_envs,
      num_updates=num_updates,
      replay_size=len(agent.replay_buffer),
      buffer_min_length=agent_cfg.buffer_min_length,
    ),
  )
  runtime_metadata.update(
    _write_training_audit_artifacts(
      log_dir=log_dir,
      runtime_metadata=runtime_metadata,
      checkpoint_summaries=checkpoint_summaries,
      log_history=logger.history,
    )
  )
  dump_yaml(log_dir / "params" / "runtime.yaml", runtime_metadata)
  adapter.close()


def launch_flashsac_training(
  task_id: str, args: FlashSACTrainConfig | None = None
) -> None:
  args = args or FlashSACTrainConfig.from_task(task_id)
  if "motion" in args.env.commands and isinstance(
    args.env.commands["motion"], MotionCommandCfg
  ):
    maybe_recompute_flashsac_tracking_checkpoint_cadence(args.env, args.agent)
  log_root_path = Path("logs") / "flashsac" / args.agent.experiment_name
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name
  if args.gpu_ids == "all" or (
    isinstance(args.gpu_ids, list) and len(args.gpu_ids) > 1
  ):
    raise ValueError(
      "FlashSAC backend currently supports only single-process CPU/single-GPU training. "
      "Multi-GPU launch is disabled until parameter/replay synchronization is implemented."
    )
  selected_gpus, num_gpus = select_gpus(
    args.gpu_ids if args.agent.device_type.startswith("cuda") else None
  )
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"
  if num_gpus <= 1:
    run_flashsac_train(task_id, args, log_dir)
    return
  raise ValueError(
    "FlashSAC backend currently supports only single-process CPU/single-GPU training. "
    "Multi-GPU launch is disabled until parameter/replay synchronization is implemented."
  )
