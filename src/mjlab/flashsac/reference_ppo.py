from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from rsl_rl.modules.normalization import EmpiricalNormalization

from mjlab.envs import ManagerBasedRlEnv
from mjlab.flashsac.trainer import (
  FlashSACLogger,
  _checkpoint_summary_entry,
  _default_logging_interval,
  _randomize_episode_horizons,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.training_steps import (
  interaction_steps_from_total_env_steps,
  total_env_steps_from_interaction_steps,
)


def _write_json(path: Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )


def _write_reference_ppo_audit_artifacts(
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
  params_dir = log_dir / "params"
  runtime_yaml_path = params_dir / "runtime.yaml"

  metrics_payload = {
    "backend": "reference_ppo",
    "task_id": runtime_metadata["task_id"],
    "seed": runtime_metadata["seed"],
    "device": runtime_metadata["device"],
    "num_envs": runtime_metadata["num_envs"],
    "num_env_steps": runtime_metadata["num_env_steps"],
    "num_interaction_steps": runtime_metadata["num_interaction_steps"],
    "actual_update_steps": runtime_metadata["actual_update_steps"],
    "final_env_steps": runtime_metadata["final_env_steps"],
    "final_interaction_steps": runtime_metadata["final_interaction_steps"],
    "checkpoint_count": runtime_metadata["checkpoint_count"],
    "final_checkpoint_dir": runtime_metadata["final_checkpoint_dir"],
    "runtime_yaml_path": str(runtime_yaml_path),
    "env_yaml_path": str(params_dir / "env.yaml"),
    "agent_yaml_path": str(params_dir / "agent.yaml"),
    "log_history_points": len(log_history),
    "last_logged_metrics": log_history[-1] if log_history else None,
  }
  checkpoint_payload = {
    "backend": "reference_ppo",
    "task_id": runtime_metadata["task_id"],
    "checkpoint_count": runtime_metadata["checkpoint_count"],
    "final_checkpoint_dir": runtime_metadata["final_checkpoint_dir"],
    "checkpoints": checkpoint_summaries,
  }

  _write_json(metrics_path, metrics_payload)
  _write_json(checkpoint_path, checkpoint_payload)
  _write_json(log_history_path, {"entries": log_history})
  return {
    "summary_metrics_file": str(metrics_path),
    "checkpoint_summary_file": str(checkpoint_path),
    "log_history_file": str(log_history_path),
  }


def _resolve_motion_tracking_registry(cfg: "ReferencePpoTrainConfig") -> str | None:
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
    registry_name = cfg.registry_name
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


def _build_mlp(
  input_dim: int,
  hidden_dims: tuple[int, ...],
  output_dim: int,
  *,
  activation: str,
) -> nn.Sequential:
  layers: list[nn.Module] = []
  prev = input_dim
  for hidden_dim in hidden_dims:
    layers.append(nn.Linear(prev, hidden_dim))
    if activation == "elu":
      layers.append(nn.ELU())
    elif activation == "tanh":
      layers.append(nn.Tanh())
    else:
      raise ValueError(f"Unsupported activation for reference_ppo: {activation!r}")
    prev = hidden_dim
  layers.append(nn.Linear(prev, output_dim))
  return nn.Sequential(*layers)


class GaussianActor(nn.Module):
  def __init__(
    self,
    observation_dim: int,
    action_dim: int,
    hidden_dims: tuple[int, ...],
    *,
    activation: str,
    normalize_observation: bool,
    init_std: float,
    std_type: Literal["scalar", "log"],
  ):
    super().__init__()
    self.obs_normalization = normalize_observation
    if normalize_observation:
      self.obs_normalizer: nn.Module = EmpiricalNormalization(observation_dim)
    else:
      self.obs_normalizer = nn.Identity()
    self.backbone = _build_mlp(
      observation_dim,
      hidden_dims,
      action_dim,
      activation=activation,
    )
    self.std_type = std_type
    if std_type == "scalar":
      self.std_param = nn.Parameter(torch.full((action_dim,), init_std))
    elif std_type == "log":
      self.log_std_param = nn.Parameter(
        torch.log(torch.full((action_dim,), init_std))
      )
    else:
      raise ValueError(f"Unsupported std_type for reference_ppo: {std_type!r}")

  def update_normalization(self, observations: torch.Tensor) -> None:
    if self.obs_normalization:
      assert isinstance(self.obs_normalizer, EmpiricalNormalization)
      self.obs_normalizer.update(observations)

  def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = self.obs_normalizer(observations)
    mean = self.backbone(normalized)
    if self.std_type == "scalar":
      std = torch.clamp(self.std_param, min=1e-6).expand_as(mean)
      log_std = torch.log(std)
    else:
      log_std = self.log_std_param.expand_as(mean)
    return mean, log_std


class ValueCritic(nn.Module):
  def __init__(
    self,
    observation_dim: int,
    hidden_dims: tuple[int, ...],
    *,
    activation: str,
    normalize_observation: bool,
  ):
    super().__init__()
    self.obs_normalization = normalize_observation
    if normalize_observation:
      self.obs_normalizer: nn.Module = EmpiricalNormalization(observation_dim)
    else:
      self.obs_normalizer = nn.Identity()
    self.backbone = _build_mlp(
      observation_dim,
      hidden_dims,
      1,
      activation=activation,
    )

  def update_normalization(self, observations: torch.Tensor) -> None:
    if self.obs_normalization:
      assert isinstance(self.obs_normalizer, EmpiricalNormalization)
      self.obs_normalizer.update(observations)

  def forward(self, observations: torch.Tensor) -> torch.Tensor:
    normalized = self.obs_normalizer(observations)
    return self.backbone(normalized).squeeze(-1)


def _gaussian_log_prob(
  mean: torch.Tensor,
  log_std: torch.Tensor,
  action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  std = log_std.exp()
  normal = torch.distributions.Normal(mean, std)
  log_prob = normal.log_prob(action).sum(dim=-1)
  entropy = normal.entropy().sum(dim=-1)
  return log_prob, entropy


@dataclass
class ReferencePpoRollout:
  actor_observation: list[np.ndarray] = field(default_factory=list)
  critic_observation: list[np.ndarray] = field(default_factory=list)
  next_critic_observation: list[np.ndarray] = field(default_factory=list)
  action: list[np.ndarray] = field(default_factory=list)
  log_prob: list[np.ndarray] = field(default_factory=list)
  mean: list[np.ndarray] = field(default_factory=list)
  std: list[np.ndarray] = field(default_factory=list)
  reward: list[np.ndarray] = field(default_factory=list)
  done: list[np.ndarray] = field(default_factory=list)
  value: list[np.ndarray] = field(default_factory=list)

  def clear(self) -> None:
    self.actor_observation.clear()
    self.critic_observation.clear()
    self.next_critic_observation.clear()
    self.action.clear()
    self.log_prob.clear()
    self.mean.clear()
    self.std.clear()
    self.reward.clear()
    self.done.clear()
    self.value.clear()

  def __len__(self) -> int:
    return len(self.reward)


@dataclass
class ReferencePpoRunnerCfg:
  seed: int = 42
  num_env_steps: int = 1_000_000
  rollout_length: int = 24
  num_epochs: int = 5
  num_minibatches: int = 4
  learning_rate: float = 1e-3
  gamma: float = 0.99
  gae_lambda: float = 0.95
  clip_coef: float = 0.2
  value_loss_coef: float = 1.0
  entropy_coef: float = 0.005
  max_grad_norm: float = 1.0
  desired_kl: float | None = 0.01
  schedule: Literal["adaptive", "fixed"] = "adaptive"
  use_clipped_value_loss: bool = True
  logging_per_interaction_step: int | None = None
  save_checkpoint_per_interaction_step: int | None = None
  experiment_name: str = "reference_ppo"
  run_name: str = ""
  logger: Literal["wandb", "tensorboard"] = "tensorboard"
  wandb_project: str = "mjlab"
  wandb_entity: str | None = None
  wandb_group: str = "reference_ppo"
  wandb_tags: tuple[str, ...] = ()
  asymmetric_observation: bool = True
  normalize_observation: bool = False
  actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
  critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
  activation: Literal["elu", "tanh"] = "elu"
  init_std: float = 1.0
  std_type: Literal["scalar", "log"] = "scalar"
  device_type: str = "cuda"


@dataclass(frozen=True)
class ReferencePpoTrainConfig:
  env: Any
  agent: ReferencePpoRunnerCfg
  registry_name: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "ReferencePpoTrainConfig":
    env_cfg = load_env_cfg(task_id)
    rl_cfg = load_rl_cfg(task_id)
    has_actor = "actor" in env_cfg.observations
    has_critic = "critic" in env_cfg.observations
    agent_cfg = ReferencePpoRunnerCfg(
      asymmetric_observation=has_actor and has_critic,
      experiment_name=task_id.replace("Mjlab-", "").replace("-", "_").lower()
      + "_reference_ppo",
    )
    actor_cfg = getattr(rl_cfg, "actor", None)
    critic_cfg = getattr(rl_cfg, "critic", None)
    algorithm_cfg = getattr(rl_cfg, "algorithm", None)
    if getattr(rl_cfg, "num_steps_per_env", None) is not None:
      agent_cfg.rollout_length = int(rl_cfg.num_steps_per_env)
    if actor_cfg is not None:
      hidden_dims = getattr(actor_cfg, "hidden_dims", None)
      if hidden_dims is not None:
        agent_cfg.actor_hidden_dims = tuple(hidden_dims)
      activation = getattr(actor_cfg, "activation", None)
      if activation in ("elu", "tanh"):
        agent_cfg.activation = activation
      obs_normalization = getattr(actor_cfg, "obs_normalization", None)
      if obs_normalization is not None:
        agent_cfg.normalize_observation = bool(obs_normalization)
      distribution_cfg = getattr(actor_cfg, "distribution_cfg", None) or {}
      init_std = distribution_cfg.get("init_std")
      if init_std is not None:
        agent_cfg.init_std = float(init_std)
      std_type = distribution_cfg.get("std_type")
      if std_type in ("scalar", "log"):
        agent_cfg.std_type = std_type
    if critic_cfg is not None:
      hidden_dims = getattr(critic_cfg, "hidden_dims", None)
      if hidden_dims is not None:
        agent_cfg.critic_hidden_dims = tuple(hidden_dims)
    if algorithm_cfg is not None:
      for attr_name, cfg_name in (
        ("num_learning_epochs", "num_epochs"),
        ("num_mini_batches", "num_minibatches"),
        ("learning_rate", "learning_rate"),
        ("gamma", "gamma"),
        ("lam", "gae_lambda"),
        ("clip_param", "clip_coef"),
        ("value_loss_coef", "value_loss_coef"),
        ("entropy_coef", "entropy_coef"),
        ("max_grad_norm", "max_grad_norm"),
      ):
        value = getattr(algorithm_cfg, attr_name, None)
        if value is not None:
          setattr(agent_cfg, cfg_name, value)
      desired_kl = getattr(algorithm_cfg, "desired_kl", None)
      if desired_kl is not None:
        agent_cfg.desired_kl = float(desired_kl)
      schedule = getattr(algorithm_cfg, "schedule", None)
      if schedule in ("adaptive", "fixed"):
        agent_cfg.schedule = schedule
      use_clipped_value_loss = getattr(
        algorithm_cfg, "use_clipped_value_loss", None
      )
      if use_clipped_value_loss is not None:
        agent_cfg.use_clipped_value_loss = bool(use_clipped_value_loss)
    return ReferencePpoTrainConfig(env=env_cfg, agent=agent_cfg)


class ReferencePpoAgent:
  def __init__(
    self,
    *,
    actor_observation_dim: int,
    critic_observation_dim: int,
    action_dim: int,
    cfg: ReferencePpoRunnerCfg,
    device: torch.device,
  ) -> None:
    self.cfg = cfg
    self.device = device
    self.actor_observation_dim = actor_observation_dim
    self.critic_observation_dim = critic_observation_dim
    self.action_dim = action_dim
    self.actor = GaussianActor(
      actor_observation_dim,
      action_dim,
      cfg.actor_hidden_dims,
      activation=cfg.activation,
      normalize_observation=cfg.normalize_observation,
      init_std=cfg.init_std,
      std_type=cfg.std_type,
    ).to(device)
    self.critic = ValueCritic(
      critic_observation_dim,
      cfg.critic_hidden_dims,
      activation=cfg.activation,
      normalize_observation=cfg.normalize_observation,
    ).to(device)
    self.optimizer = torch.optim.Adam(
      list(self.actor.parameters()) + list(self.critic.parameters()),
      lr=cfg.learning_rate,
    )
    self.rollout = ReferencePpoRollout()
    self.update_step = 0
    self.learning_rate = cfg.learning_rate

  def update_normalization(
    self, actor_observation: np.ndarray, critic_observation: np.ndarray
  ) -> None:
    actor_observation_t = torch.as_tensor(
      actor_observation, dtype=torch.float32, device=self.device
    )
    critic_observation_t = torch.as_tensor(
      critic_observation, dtype=torch.float32, device=self.device
    )
    self.actor.update_normalization(actor_observation_t)
    self.critic.update_normalization(critic_observation_t)

  def act(
    self, actor_observation: np.ndarray, critic_observation: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actor_observation_t = torch.as_tensor(
      actor_observation, dtype=torch.float32, device=self.device
    )
    critic_observation_t = torch.as_tensor(
      critic_observation, dtype=torch.float32, device=self.device
    )
    with torch.no_grad():
      mean, log_std = self.actor(actor_observation_t)
      std = log_std.exp()
      normal = torch.distributions.Normal(mean, std)
      action = normal.sample()
      log_prob, _ = _gaussian_log_prob(mean, log_std, action)
      value = self.critic(critic_observation_t)
    return (
      action.cpu().numpy(),
      log_prob.cpu().numpy(),
      value.cpu().numpy(),
      mean.cpu().numpy(),
      std.cpu().numpy(),
    )

  def add_transition(
    self,
    *,
    actor_observation: np.ndarray,
    critic_observation: np.ndarray,
    next_critic_observation: np.ndarray,
    action: np.ndarray,
    log_prob: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    reward: np.ndarray,
    done: np.ndarray,
    value: np.ndarray,
  ) -> None:
    self.rollout.actor_observation.append(actor_observation.copy())
    self.rollout.critic_observation.append(critic_observation.copy())
    self.rollout.next_critic_observation.append(next_critic_observation.copy())
    self.rollout.action.append(action.copy())
    self.rollout.log_prob.append(log_prob.copy())
    self.rollout.mean.append(mean.copy())
    self.rollout.std.append(std.copy())
    self.rollout.reward.append(reward.copy())
    self.rollout.done.append(done.astype(np.float32, copy=True))
    self.rollout.value.append(value.copy())

  def _apply_adaptive_lr(self, kl_mean: torch.Tensor) -> None:
    if self.cfg.desired_kl is None or self.cfg.schedule != "adaptive":
      return
    kl_value = float(kl_mean.item())
    if kl_value > self.cfg.desired_kl * 2.0:
      self.learning_rate = max(1e-5, self.learning_rate / 1.5)
    elif 0.0 < kl_value < self.cfg.desired_kl / 2.0:
      self.learning_rate = min(1e-2, self.learning_rate * 1.5)
    for param_group in self.optimizer.param_groups:
      param_group["lr"] = self.learning_rate

  def ready_to_update(self) -> bool:
    return len(self.rollout) >= self.cfg.rollout_length

  def _compute_advantages(self) -> tuple[torch.Tensor, torch.Tensor]:
    rewards = torch.as_tensor(
      np.stack(self.rollout.reward), dtype=torch.float32, device=self.device
    )
    dones = torch.as_tensor(
      np.stack(self.rollout.done), dtype=torch.float32, device=self.device
    )
    values = torch.as_tensor(
      np.stack(self.rollout.value), dtype=torch.float32, device=self.device
    )
    next_critic_observations = torch.as_tensor(
      np.stack(self.rollout.next_critic_observation),
      dtype=torch.float32,
      device=self.device,
    )
    with torch.no_grad():
      next_values = self.critic(
        next_critic_observations.reshape(-1, self.critic_observation_dim)
      )
      next_values = next_values.view_as(values)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[1], dtype=torch.float32, device=self.device)
    for idx in reversed(range(rewards.shape[0])):
      delta = (
        rewards[idx]
        + self.cfg.gamma * (1.0 - dones[idx]) * next_values[idx]
        - values[idx]
      )
      gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1.0 - dones[idx]) * gae
      advantages[idx] = gae
    returns = advantages + values
    return advantages, returns

  def update(self) -> dict[str, float]:
    actor_observations = torch.as_tensor(
      np.stack(self.rollout.actor_observation),
      dtype=torch.float32,
      device=self.device,
    )
    critic_observations = torch.as_tensor(
      np.stack(self.rollout.critic_observation),
      dtype=torch.float32,
      device=self.device,
    )
    actions = torch.as_tensor(
      np.stack(self.rollout.action), dtype=torch.float32, device=self.device
    )
    old_log_probs = torch.as_tensor(
      np.stack(self.rollout.log_prob), dtype=torch.float32, device=self.device
    )
    old_means = torch.as_tensor(
      np.stack(self.rollout.mean), dtype=torch.float32, device=self.device
    )
    old_stds = torch.as_tensor(
      np.stack(self.rollout.std), dtype=torch.float32, device=self.device
    )
    old_values = torch.as_tensor(
      np.stack(self.rollout.value), dtype=torch.float32, device=self.device
    )
    advantages, returns = self._compute_advantages()
    actor_observations = actor_observations.reshape(-1, self.actor_observation_dim)
    critic_observations = critic_observations.reshape(-1, self.critic_observation_dim)
    actions = actions.reshape(-1, self.action_dim)
    old_log_probs = old_log_probs.reshape(-1)
    old_means = old_means.reshape(-1, self.action_dim)
    old_stds = old_stds.reshape(-1, self.action_dim)
    old_values = old_values.reshape(-1)
    advantages = advantages.reshape(-1)
    returns = returns.reshape(-1)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batch_size = critic_observations.shape[0]
    minibatch_size = max(1, batch_size // max(self.cfg.num_minibatches, 1))
    value_loss_total = 0.0
    policy_loss_total = 0.0
    entropy_total = 0.0
    updates = 0

    for _ in range(self.cfg.num_epochs):
      permutation = torch.randperm(batch_size, device=self.device)
      for start in range(0, batch_size, minibatch_size):
        index = permutation[start : start + minibatch_size]
        mean, log_std = self.actor(actor_observations[index])
        std = log_std.exp()
        new_log_prob, entropy = _gaussian_log_prob(mean, log_std, actions[index])
        if self.cfg.desired_kl is not None and self.cfg.schedule == "adaptive":
          with torch.no_grad():
            old_mean = old_means[index]
            old_std = old_stds[index]
            kl = torch.log(std / old_std) + (
              old_std.pow(2) + (old_mean - mean).pow(2)
            ) / (2.0 * std.pow(2)) - 0.5
            self._apply_adaptive_lr(kl.sum(dim=-1).mean())
        ratio = (new_log_prob - old_log_probs[index]).exp()
        unclipped = -advantages[index] * ratio
        clipped = -advantages[index] * torch.clamp(
          ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef
        )
        policy_loss = torch.maximum(unclipped, clipped).mean()
        value = self.critic(critic_observations[index])
        if self.cfg.use_clipped_value_loss:
          value_clipped = old_values[index] + (
            value - old_values[index]
          ).clamp(-self.cfg.clip_coef, self.cfg.clip_coef)
          value_losses = (value - returns[index]).pow(2)
          value_losses_clipped = (value_clipped - returns[index]).pow(2)
          value_loss = torch.maximum(value_losses, value_losses_clipped).mean()
        else:
          value_loss = F.mse_loss(value, returns[index])
        entropy_loss = entropy.mean()
        loss = (
          policy_loss
          + self.cfg.value_loss_coef * value_loss
          - self.cfg.entropy_coef * entropy_loss
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
          list(self.actor.parameters()) + list(self.critic.parameters()),
          self.cfg.max_grad_norm,
        )
        self.optimizer.step()
        policy_loss_total += float(policy_loss.item())
        value_loss_total += float(value_loss.item())
        entropy_total += float(entropy_loss.item())
        updates += 1

    self.rollout.clear()
    self.update_step += 1
    denom = max(updates, 1)
    return {
      "reference_ppo/policy_loss": policy_loss_total / denom,
      "reference_ppo/value_loss": value_loss_total / denom,
      "reference_ppo/entropy": entropy_total / denom,
      "reference_ppo/learning_rate": float(self.learning_rate),
      "reference_ppo/update_steps": float(self.update_step),
    }

  def save(self, path: str) -> None:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    torch.save(self.actor.state_dict(), path_obj / "actor.pt")
    torch.save(self.critic.state_dict(), path_obj / "critic.pt")
    torch.save(
      {
        "optimizer": self.optimizer.state_dict(),
        "update_step": self.update_step,
        "cfg": asdict(self.cfg),
      },
      path_obj / "agent_state.pt",
    )


def run_reference_ppo_train(
  task_id: str, cfg: ReferencePpoTrainConfig, log_dir: Path
) -> None:
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

  cfg.env.seed = seed
  runtime_cfg = ReferencePpoTrainConfig(
    env=cfg.env,
    agent=cfg.agent,
    registry_name=cfg.registry_name,
    gpu_ids=cfg.gpu_ids,
  )
  _resolve_motion_tracking_registry(runtime_cfg)

  env = ManagerBasedRlEnv(cfg=cfg.env, device=device)
  observation_space = env.single_observation_space
  actor_key = "actor" if "actor" in observation_space.spaces else next(
    iter(observation_space.spaces)
  )
  critic_key = "critic" if "critic" in observation_space.spaces else actor_key
  actor_dim = observation_space.spaces[actor_key].shape[-1]
  critic_dim = observation_space.spaces[critic_key].shape[-1]

  def to_numpy_observations(
    obs_dict: dict[str, Any],
  ) -> tuple[np.ndarray, np.ndarray]:
    actor_obs = obs_dict[actor_key]
    critic_obs = obs_dict[critic_key]
    assert isinstance(actor_obs, torch.Tensor)
    assert isinstance(critic_obs, torch.Tensor)
    return (
      actor_obs.cpu().numpy().astype(np.float32, copy=False),
      critic_obs.cpu().numpy().astype(np.float32, copy=False),
    )

  def extract_episode_info(extras: dict[str, Any]) -> dict[str, Any]:
    if "log" not in extras:
      return {}
    episode_info = {
      key: scalar
      for key, value in extras["log"].items()
      if isinstance((scalar := value), (float, int))
      or (
        isinstance(value, torch.Tensor)
        and (
          scalar := (
            float(value.item())
            if value.numel() == 1
            else float(value.float().mean().item())
          )
        )
      )
    }
    return episode_info

  obs_dict, reset_extras = env.reset()
  actor_observations, critic_observations = to_numpy_observations(obs_dict)
  env_info: dict[str, Any] = {"actor_observation_size": (actor_dim,)}
  episode_info = extract_episode_info(reset_extras)
  if episode_info:
    env_info["episode_info"] = episode_info
  _randomize_episode_horizons(env)

  num_interaction_steps = max(
    1,
    interaction_steps_from_total_env_steps(
      total_env_steps=cfg.agent.num_env_steps,
      num_envs=env.num_envs,
    ),
  )
  logging_every = cfg.agent.logging_per_interaction_step or _default_logging_interval(
    num_interaction_steps
  )
  checkpoint_every = cfg.agent.save_checkpoint_per_interaction_step or num_interaction_steps

  runtime_metadata: dict[str, Any] = {
    "backend": "reference_ppo",
    "task_id": task_id,
    "seed": seed,
    "device": device,
    "cuda_visible_devices": cuda_visible,
    "num_envs": env.num_envs,
    "num_env_steps": cfg.agent.num_env_steps,
    "num_interaction_steps": num_interaction_steps,
    "rollout_length": cfg.agent.rollout_length,
    "num_epochs": cfg.agent.num_epochs,
    "num_minibatches": cfg.agent.num_minibatches,
    "logging_per_interaction_step": logging_every,
    "save_checkpoint_per_interaction_step": checkpoint_every,
  }
  dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(cfg.agent))
  dump_yaml(log_dir / "params" / "runtime.yaml", runtime_metadata)
  print(
    "[INFO] Reference PPO training with "
    f"device={device}, seed={seed}, num_envs={env.num_envs}"
  )

  agent = ReferencePpoAgent(
    actor_observation_dim=actor_dim,
    critic_observation_dim=critic_dim,
    action_dim=env.single_action_space.shape[-1],
    cfg=cfg.agent,
    device=torch.device(device),
  )
  agent.update_normalization(actor_observations, critic_observations)
  logger = FlashSACLogger(cast(Any, runtime_cfg), log_dir)
  checkpoint_count = 0
  checkpoint_summaries: list[dict[str, Any]] = []
  num_updates = 0

  for interaction_step in range(1, num_interaction_steps + 1):
    env_step = total_env_steps_from_interaction_steps(
      interaction_step, num_envs=env.num_envs
    )
    actions, log_probs, values, means, stds = agent.act(
      actor_observations, critic_observations
    )
    (
      next_obs_dict,
      rewards_t,
      terminated_t,
      truncated_t,
      step_extras,
    ) = env.step(torch.as_tensor(actions, dtype=torch.float32, device=env.device))
    next_actor_observations, next_critic_observations = to_numpy_observations(
      next_obs_dict
    )
    step_episode_info = extract_episode_info(step_extras)
    if step_episode_info:
      logger.update_metric(**step_episode_info)
    rewards = rewards_t.cpu().numpy().astype(np.float32, copy=False)
    terminateds = terminated_t.cpu().numpy().astype(np.bool_, copy=False)
    truncateds = truncated_t.cpu().numpy().astype(np.bool_, copy=False)
    done = np.logical_or(terminateds, truncateds)
    rewards = rewards + cfg.agent.gamma * values * truncateds.astype(
      np.float32, copy=False
    )
    agent.add_transition(
      actor_observation=actor_observations,
      critic_observation=critic_observations,
      next_critic_observation=next_critic_observations,
      action=actions,
      log_prob=log_probs,
      mean=means,
      std=stds,
      reward=rewards,
      done=done,
      value=values,
    )
    agent.update_normalization(next_actor_observations, next_critic_observations)
    actor_observations = next_actor_observations
    critic_observations = next_critic_observations

    if agent.ready_to_update():
      logger.update_metric(**agent.update())
      num_updates += 1
    if interaction_step % logging_every == 0:
      logger.log_metric(
        step=env_step,
        step_metrics={
          "Perf/env_steps": float(env_step),
          "Perf/interaction_steps": float(interaction_step),
          "Perf/num_envs": float(env.num_envs),
          "Perf/update_steps": float(num_updates),
        },
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
  final_env_step = total_env_steps_from_interaction_steps(
    num_interaction_steps, num_envs=env.num_envs
  )
  logger.log_metric(
    step=final_env_step,
    step_metrics={
      "Perf/env_steps": float(final_env_step),
      "Perf/interaction_steps": float(num_interaction_steps),
      "Perf/num_envs": float(env.num_envs),
      "Perf/update_steps": float(num_updates),
    },
  )
  runtime_metadata.update(
    {
      "final_env_steps": final_env_step,
      "final_interaction_steps": num_interaction_steps,
      "actual_update_steps": num_updates,
      "checkpoint_count": checkpoint_count,
      "final_checkpoint_dir": str(final_save_dir),
      "checkpoint_summaries": checkpoint_summaries,
    }
  )
  runtime_metadata.update(
    _write_reference_ppo_audit_artifacts(
      log_dir=log_dir,
      runtime_metadata=runtime_metadata,
      checkpoint_summaries=checkpoint_summaries,
      log_history=logger.history,
    )
  )
  dump_yaml(log_dir / "params" / "runtime.yaml", runtime_metadata)
  env.close()


def launch_reference_ppo_training(
  task_id: str, args: ReferencePpoTrainConfig | None = None
) -> None:
  args = args or ReferencePpoTrainConfig.from_task(task_id)
  log_root_path = Path("logs") / "reference_ppo" / args.agent.experiment_name
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name
  if args.gpu_ids == "all" or (
    isinstance(args.gpu_ids, list) and len(args.gpu_ids) > 1
  ):
    raise ValueError("Reference PPO debug backend supports only CPU/single-GPU training.")
  selected_gpus, num_gpus = select_gpus(
    args.gpu_ids if args.agent.device_type.startswith("cuda") else None
  )
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"
  if num_gpus <= 1:
    run_reference_ppo_train(task_id, args, log_dir)
    return
  raise ValueError("Reference PPO debug backend supports only CPU/single-GPU training.")
