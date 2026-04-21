from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Any, MutableMapping, cast

import numpy as np
import torch
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler

from mjlab.flashsac.buffer import TorchUniformBuffer
from mjlab.flashsac.config import FlashSACRunnerCfg
from mjlab.flashsac.network import (
  FlashSACActor,
  FlashSACDoubleCritic,
  FlashSACTemperature,
)
from mjlab.flashsac.utils import (
  NetworkBundle,
  ObservationNormalizer,
  RewardNormalizer,
  warmup_cosine_decay_scheduler,
)
from mjlab.utils.training_steps import interaction_steps_from_total_env_steps


def _build_truncated_zeta_cdf(mu: float, max_n: int) -> torch.Tensor:
  ns = torch.arange(1, max_n + 1, dtype=torch.float32)
  pmf = ns ** (-mu)
  pmf = pmf / torch.sum(pmf)
  return torch.cumsum(pmf, dim=0)


def _sample_integer_from_cdf(cdf: torch.Tensor) -> torch.Tensor:
  uniform = torch.rand((), device=cdf.device)
  idx = torch.argmax((uniform < cdf).to(torch.int32))
  return (idx + 1).to(torch.int32)


def _select_min_q_log_probs(
  next_qs: torch.Tensor, next_q_log_probs: torch.Tensor
) -> torch.Tensor:
  num_bins = next_q_log_probs.shape[-1]
  min_indices = next_qs.argmin(dim=0)
  return torch.gather(
    next_q_log_probs,
    dim=0,
    index=min_indices[None, :, None].expand(1, -1, num_bins),
  )[0]


def _compute_categorical_td_target(
  target_log_probs: torch.Tensor,
  reward: torch.Tensor,
  done: torch.Tensor,
  actor_entropy: torch.Tensor,
  gamma: float,
  num_bins: int,
  min_v: float,
  max_v: float,
) -> torch.Tensor:
  batch_size = reward.shape[0]
  reward = reward.reshape(-1, 1)
  done = done.reshape(-1, 1)
  actor_entropy = actor_entropy.reshape(-1, 1)
  bin_width = (max_v - min_v) / (num_bins - 1)
  bin_values = torch.linspace(
    min_v, max_v, num_bins, device=target_log_probs.device, dtype=target_log_probs.dtype
  ).view(1, -1)
  target_bin_values = reward + gamma * (bin_values - actor_entropy) * (1.0 - done)
  target_bin_values = torch.clamp(target_bin_values, min_v, max_v)
  bucket = (target_bin_values - min_v) / bin_width
  lower = torch.floor(bucket).long()
  upper = torch.clamp(lower + 1, 0, num_bins - 1)
  frac = bucket - lower.float()
  target_probs_exp = target_log_probs.exp()
  lower_mass = target_probs_exp * (1.0 - frac)
  upper_mass = target_probs_exp * frac
  target_probs = torch.zeros(
    batch_size, num_bins, dtype=target_probs_exp.dtype, device=target_probs_exp.device
  )
  target_probs.scatter_add_(1, lower, lower_mass)
  target_probs.scatter_add_(1, upper, upper_mass)
  return target_probs


def _add_prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
  return {f"{prefix}/{key}": value for key, value in d.items()}


def _resolve_compile_mode(mode: str) -> str:
  if mode != "auto":
    return mode
  major, minor = (int(x) for x in torch.__version__.split(".")[:2])
  return "max-autotune" if (major, minor) >= (2, 9) else "reduce-overhead"


def _autocast_kwargs(device: torch.device) -> dict[str, Any]:
  if device.type == "cuda":
    return {"device_type": "cuda", "dtype": torch.float16}
  return {"device_type": "cpu", "dtype": torch.bfloat16}


def _compute_lr_schedule_steps(
  num_env_steps: int,
  updates_per_interaction_step: float,
  learning_rate_warmup_rate: float,
  learning_rate_decay_rate: float,
  num_envs: int,
) -> tuple[int, int]:
  num_interaction_steps = interaction_steps_from_total_env_steps(num_env_steps, num_envs)
  total_update_steps = num_interaction_steps * updates_per_interaction_step
  warmup_steps = int(learning_rate_warmup_rate * total_update_steps)
  decay_steps = int(learning_rate_decay_rate * total_update_steps)
  return warmup_steps, max(decay_steps, 1)


def _build_lr_scheduler(
  optimizer: optim.Optimizer,
  cfg: FlashSACRunnerCfg,
  num_envs: int,
  scheduler_state: dict[str, Any] | None = None,
) -> torch.optim.lr_scheduler.LambdaLR:
  warmup_steps, decay_steps = _compute_lr_schedule_steps(
    num_env_steps=cfg.num_env_steps,
    updates_per_interaction_step=cfg.updates_per_interaction_step,
    learning_rate_warmup_rate=cfg.learning_rate_warmup_rate,
    learning_rate_decay_rate=cfg.learning_rate_decay_rate,
    num_envs=num_envs,
  )
  scheduler_fn = warmup_cosine_decay_scheduler(
    init_value=cfg.learning_rate_init,
    peak_value=cfg.learning_rate_peak,
    end_value=cfg.learning_rate_end,
    warmup_steps=warmup_steps,
    decay_steps=decay_steps,
  )
  scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: scheduler_fn(step) / cfg.learning_rate_peak,
  )
  if scheduler_state is not None:
    scheduler.load_state_dict(scheduler_state)
  return scheduler


@dataclass
class FlashSACAgentState:
  update_step: int = 0


class FlashSACAgent:
  def __init__(
    self,
    observation_dim: int,
    action_dim: int,
    actor_observation_dim: int,
    cfg: FlashSACRunnerCfg,
  ):
    target_entropy = (
      0.5 * action_dim * math.log(2 * math.pi * math.e * cfg.temp_target_sigma**2)
    )
    self.cfg = replace(cfg, compile_mode=_resolve_compile_mode(cfg.compile_mode))
    self.target_entropy = target_entropy
    device_type = cfg.device_type
    device_type = (
      device_type
      if device_type.startswith("cuda:")
      else ("cuda:0" if device_type.startswith("cuda") else "cpu")
    )
    self.device = torch.device(device_type)
    self.observation_dim = observation_dim
    self.action_dim = action_dim
    self.actor_observation_dim = actor_observation_dim

    self._lr_schedule_num_envs = 1
    use_fused = self.device.type == "cuda" and torch.cuda.is_available()

    actor_net = FlashSACActor(
      num_blocks=self.cfg.actor_num_blocks,
      input_dim=self.actor_observation_dim,
      hidden_dim=self.cfg.actor_hidden_dim,
      action_dim=self.action_dim,
    ).to(self.device)
    actor_optimizer = optim.Adam(
      actor_net.parameters(), lr=self.cfg.learning_rate_peak, fused=use_fused
    )
    actor_scheduler = _build_lr_scheduler(
      actor_optimizer,
      self.cfg,
      num_envs=self._lr_schedule_num_envs,
    )
    self.actor = NetworkBundle(
      network=actor_net,
      optimizer=actor_optimizer,
      scheduler=actor_scheduler,
      compile_network=self.cfg.use_compile,
      compile_mode=self.cfg.compile_mode,
      use_weight_normalization=True,
    )
    if self.cfg.use_compile:
      actor_network = cast(Any, self.actor.network)
      actor_network.get_mean_and_std = torch.compile(
        actor_network.get_mean_and_std, mode=self.cfg.compile_mode
      )  # type: ignore[attr-defined]

    critic_net = FlashSACDoubleCritic(
      num_blocks=self.cfg.critic_num_blocks,
      input_dim=self.observation_dim + self.action_dim,
      hidden_dim=self.cfg.critic_hidden_dim,
      num_bins=self.cfg.critic_num_bins,
      min_v=-self.cfg.normalized_G_max,
      max_v=self.cfg.normalized_G_max,
    ).to(self.device)
    critic_optimizer = optim.Adam(
      critic_net.parameters(), lr=self.cfg.learning_rate_peak, fused=use_fused
    )
    critic_scheduler = _build_lr_scheduler(
      critic_optimizer,
      self.cfg,
      num_envs=self._lr_schedule_num_envs,
    )
    self.critic = NetworkBundle(
      network=critic_net,
      optimizer=critic_optimizer,
      scheduler=critic_scheduler,
      compile_network=self.cfg.use_compile,
      compile_mode=self.cfg.compile_mode,
      use_weight_normalization=True,
    )

    target_critic_net = FlashSACDoubleCritic(
      num_blocks=self.cfg.critic_num_blocks,
      input_dim=self.observation_dim + self.action_dim,
      hidden_dim=self.cfg.critic_hidden_dim,
      num_bins=self.cfg.critic_num_bins,
      min_v=-self.cfg.normalized_G_max,
      max_v=self.cfg.normalized_G_max,
    ).to(self.device)
    target_critic_net.load_state_dict(critic_net.state_dict())
    self.target_critic = NetworkBundle(
      network=target_critic_net,
      optimizer=None,
      scheduler=None,
      compile_network=self.cfg.use_compile,
      compile_mode=self.cfg.compile_mode,
      use_weight_normalization=True,
      ema_source=self.critic,
      ema_tau=self.cfg.critic_target_update_tau,
    )

    temp_net = FlashSACTemperature(self.cfg.temp_initial_value).to(self.device)
    temp_optimizer = optim.Adam(
      temp_net.parameters(), lr=self.cfg.learning_rate_peak, fused=use_fused
    )
    temp_scheduler = _build_lr_scheduler(
      temp_optimizer,
      self.cfg,
      num_envs=self._lr_schedule_num_envs,
    )
    self.temperature = NetworkBundle(
      network=temp_net,
      optimizer=temp_optimizer,
      scheduler=temp_scheduler,
      compile_network=self.cfg.use_compile,
      compile_mode=self.cfg.compile_mode,
      use_weight_normalization=False,
    )

    self.actor.normalize_parameters()
    self.critic.normalize_parameters()
    self.target_critic.normalize_parameters()

    self.grad_scaler = GradScaler(device=self.device.type, enabled=self.cfg.use_amp)
    self.zeta_cdf = _build_truncated_zeta_cdf(
      self.cfg.actor_noise_zeta_mu, self.cfg.actor_noise_zeta_max
    ).to(self.device)
    self.cur_noise_repeat_n = torch.tensor(1, dtype=torch.int32, device=self.device)
    self.cur_noise_repeat_count = torch.tensor(0, dtype=torch.int32, device=self.device)
    self.cached_noise = torch.randn((self.action_dim,), device=self.device)

    self.reward_normalizer = None
    if self.cfg.normalize_reward:
      self.reward_normalizer = RewardNormalizer(
        gamma=self.cfg.gamma,
        G_max=self.cfg.normalized_G_max,
        load_rms=self.cfg.load_reward_normalizer,
        device=self.device,
      )

    self.observation_normalizer = None
    self.actor_observation_normalizer = None
    if self.cfg.normalize_observation:
      self.observation_normalizer = ObservationNormalizer(
        shape=(self.observation_dim,),
        device=self.device,
        clip_value=self.cfg.observation_clip_value,
      )
      if self.cfg.asymmetric_observation:
        self.actor_observation_normalizer = ObservationNormalizer(
          shape=(self.actor_observation_dim,),
          device=self.device,
          clip_value=self.cfg.observation_clip_value,
        )
      else:
        self.actor_observation_normalizer = self.observation_normalizer

    self.replay_buffer = TorchUniformBuffer(
      observation_shape=(self.observation_dim,),
      action_shape=(self.action_dim,),
      n_step=self.cfg.n_step,
      gamma=self.cfg.gamma,
      max_length=self.cfg.buffer_max_length,
      min_length=self.cfg.buffer_min_length,
      sample_batch_size=self.cfg.sample_batch_size,
      device_type=self.cfg.buffer_device_type,
    )
    self.state = FlashSACAgentState()

  def _reconfigure_lr_schedulers(self, num_envs: int) -> None:
    if num_envs <= 0 or num_envs == self._lr_schedule_num_envs:
      return
    for bundle in (self.actor, self.critic, self.temperature):
      if bundle.optimizer is None:
        continue
      scheduler_state = bundle.scheduler.state_dict() if bundle.scheduler is not None else None
      bundle.scheduler = _build_lr_scheduler(
        bundle.optimizer,
        self.cfg,
        num_envs=num_envs,
        scheduler_state=scheduler_state,
      )
    self._lr_schedule_num_envs = num_envs

  def _infer_num_envs(self, transition: MutableMapping[str, Any]) -> int | None:
    for key in ("reward", "terminated", "truncated", "observation", "next_observation"):
      value = transition.get(key)
      if value is None:
        continue
      shape = getattr(value, "shape", None)
      if shape and len(shape) > 0:
        return int(shape[0])
      if isinstance(value, (list, tuple)) and value:
        return len(value)
    return None

  def _normalize_critic_observation(self, observation: torch.Tensor) -> torch.Tensor:
    if self.observation_normalizer is None:
      return observation
    return self.observation_normalizer.normalize(observation)

  def _normalize_actor_observation(self, observation: torch.Tensor) -> torch.Tensor:
    if self.actor_observation_normalizer is None:
      return observation
    return self.actor_observation_normalizer.normalize(observation)

  def _sample_actions(
    self,
    observations: torch.Tensor,
    temperature: float,
  ) -> torch.Tensor:
    mean, std = self.actor.apply(
      "get_mean_and_std", observations=observations, training=False
    )
    if temperature == 0.0:
      return torch.tanh(mean)
    reinit = (self.cur_noise_repeat_count == 0) | (
      self.cur_noise_repeat_count >= self.cur_noise_repeat_n
    )
    new_noise = torch.randn_like(mean)
    new_n = _sample_integer_from_cdf(self.zeta_cdf)
    self.cached_noise = torch.where(reinit, new_noise, self.cached_noise)
    self.cur_noise_repeat_n = torch.where(reinit, new_n, self.cur_noise_repeat_n)
    self.cur_noise_repeat_count = torch.where(
      reinit, torch.zeros_like(self.cur_noise_repeat_count), self.cur_noise_repeat_count
    )
    actions = torch.tanh(mean + std * self.cached_noise * temperature)
    self.cur_noise_repeat_count = self.cur_noise_repeat_count + 1
    return actions

  def sample_actions(
    self,
    interaction_step: int,
    prev_transition: MutableMapping[str, Any],
    training: bool,
  ) -> np.ndarray:
    del interaction_step
    temperature = 1.0 if training else 0.0
    observations = torch.as_tensor(
      prev_transition["next_observation"], dtype=torch.float32, device=self.device
    )
    actor_observations = (
      observations[:, : self.actor_observation_dim]
      if self.cfg.asymmetric_observation
      else observations
    )
    actor_observations = self._normalize_actor_observation(actor_observations)
    with torch.no_grad():
      actions = self._sample_actions(actor_observations, temperature)
    return actions.cpu().numpy()

  def process_transition(self, transition: MutableMapping[str, Any]) -> None:
    num_envs = self._infer_num_envs(transition)
    if num_envs is not None:
      self._reconfigure_lr_schedulers(num_envs)
    if self.observation_normalizer is not None:
      observations = torch.as_tensor(
        transition["observation"], dtype=torch.float32, device=self.device
      )
      next_observations = torch.as_tensor(
        transition["next_observation"], dtype=torch.float32, device=self.device
      )
      self.observation_normalizer.update(observations)
      self.observation_normalizer.update(next_observations)
      if self.actor_observation_normalizer is not None:
        if self.cfg.asymmetric_observation:
          self.actor_observation_normalizer.update(
            observations[:, : self.actor_observation_dim]
          )
          self.actor_observation_normalizer.update(
            next_observations[:, : self.actor_observation_dim]
          )
        else:
          self.actor_observation_normalizer.update(observations)
          self.actor_observation_normalizer.update(next_observations)
    self.replay_buffer.add(dict(transition))
    if self.reward_normalizer is not None:
      self.reward_normalizer.update_reward_stats(
        reward=torch.as_tensor(transition["reward"], device=self.device),
        terminated=torch.as_tensor(transition["terminated"], device=self.device),
        truncated=torch.as_tensor(transition["truncated"], device=self.device),
      )

  def can_start_training(self) -> bool:
    return self.replay_buffer.can_sample()

  def update(self) -> dict[str, float]:
    batch = self.replay_buffer.sample()
    for key, value in batch.items():
      batch[key] = value.to(self.device, non_blocking=True)

    batch["observation"] = self._normalize_critic_observation(batch["observation"])
    batch["next_observation"] = self._normalize_critic_observation(
      batch["next_observation"]
    )

    if self.cfg.asymmetric_observation:
      batch["actor_observation"] = self._normalize_actor_observation(
        batch["observation"][:, : self.actor_observation_dim]
      )
      batch["actor_next_observation"] = self._normalize_actor_observation(
        batch["next_observation"][:, : self.actor_observation_dim]
      )
    else:
      batch["actor_observation"] = batch["observation"]
      batch["actor_next_observation"] = batch["next_observation"]

    if self.reward_normalizer is not None:
      batch["reward"] = self.reward_normalizer.normalize_rewards(batch["reward"])

    do_actor_update = self.state.update_step % self.cfg.actor_update_period == 0
    update_info = {}
    if do_actor_update:
      update_info.update(self._update_actor(batch))
      update_info.update(self._update_temperature(update_info["actor/entropy"]))
    update_info.update(self._update_critic(batch))
    self.target_critic.ema_update_parameters()
    self.state.update_step += 1
    return {
      key: float(value.item()) if isinstance(value, torch.Tensor) else float(value)
      for key, value in update_info.items()
    }

  def _update_actor(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    autocast_kwargs = _autocast_kwargs(self.device)
    with torch.autocast(enabled=self.cfg.use_amp, **autocast_kwargs):
      actor_obs_all = torch.cat(
        [batch["actor_observation"], batch["actor_next_observation"]], dim=0
      )
      actions_all, info = self.actor(observations=actor_obs_all, training=True)
      log_probs_all = info["log_prob"]
      actions = torch.chunk(actions_all, 2, dim=0)[0]
      log_probs = torch.chunk(log_probs_all, 2, dim=0)[0]
      self.critic.network.requires_grad_(False)
      qs, _ = self.critic(
        observations=batch["observation"], actions=actions, training=False
      )
      q = torch.minimum(qs[0], qs[1])
      self.critic.network.requires_grad_(True)
      temp_value = self.temperature().detach()
      actor_loss = (log_probs * temp_value - q).mean()
      if self.cfg.actor_bc_alpha > 0:
        q_abs = torch.abs(q).mean().detach()
        bc_loss = ((actions - batch["action"]) ** 2).mean()
        actor_loss = actor_loss + self.cfg.actor_bc_alpha * q_abs * bc_loss
      entropy = -log_probs.mean()
      mean_action = actions.mean()
    assert self.actor.optimizer is not None
    self.actor.optimizer.zero_grad(set_to_none=True)
    if self.cfg.use_amp:
      self.grad_scaler.scale(actor_loss).backward()
      self.grad_scaler.step(self.actor.optimizer)
      self.grad_scaler.update()
    else:
      actor_loss.backward()
      self.actor.optimizer.step()
    if self.actor.scheduler is not None:
      self.actor.scheduler.step()
    self.actor.normalize_parameters()
    return _add_prefix(
      {"loss": actor_loss, "entropy": entropy, "mean_action": mean_action}, "actor"
    )

  def _update_critic(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    autocast_kwargs = _autocast_kwargs(self.device)
    min_v = -self.cfg.normalized_G_max
    max_v = self.cfg.normalized_G_max
    with torch.autocast(enabled=self.cfg.use_amp, **autocast_kwargs):
      with torch.no_grad():
        next_actions, info = self.actor(
          observations=batch["actor_next_observation"], training=False
        )
        next_actions = next_actions.clone()
        next_actor_log_probs = info["log_prob"].clone()
        temp_value = self.temperature()
        next_actor_entropy = temp_value * next_actor_log_probs
        obs_all = torch.cat([batch["observation"], batch["next_observation"]], dim=0)
        act_all = torch.cat([batch["action"], next_actions], dim=0)
        qs_all, q_infos_all = self.target_critic(
          observations=obs_all, actions=act_all, training=True
        )
        next_qs = qs_all.chunk(2, dim=1)[1]
        next_q_log_probs = q_infos_all["log_prob"].chunk(2, dim=1)[1]
        next_q_log_probs = _select_min_q_log_probs(next_qs, next_q_log_probs)
        target_probs = _compute_categorical_td_target(
          target_log_probs=next_q_log_probs,
          reward=batch["reward"],
          done=batch["terminated"].float(),
          actor_entropy=next_actor_entropy,
          gamma=self.cfg.gamma**self.cfg.n_step,
          num_bins=self.cfg.critic_num_bins,
          min_v=min_v,
          max_v=max_v,
        )
        max_entropy_bonus = next_actor_entropy.max()
      pred_qs_all, pred_q_infos = self.critic(
        observations=obs_all, actions=act_all, training=True
      )
      pred_log_probs = torch.chunk(pred_q_infos["log_prob"], 2, dim=1)[0]
      ce_loss = -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1)
      critic_loss = ce_loss.mean()
    assert self.critic.optimizer is not None
    self.critic.optimizer.zero_grad(set_to_none=True)
    if self.cfg.use_amp:
      self.grad_scaler.scale(critic_loss).backward()
      self.grad_scaler.step(self.critic.optimizer)
      self.grad_scaler.update()
    else:
      critic_loss.backward()
      self.critic.optimizer.step()
    if self.critic.scheduler is not None:
      self.critic.scheduler.step()
    self.critic.normalize_parameters()
    return _add_prefix(
      {"loss": critic_loss, "max_entropy_bonus": max_entropy_bonus}, "critic"
    )

  def _update_temperature(self, entropy: torch.Tensor) -> dict[str, torch.Tensor]:
    temperature_value = self.temperature().clone()
    temperature_loss = (
      temperature_value * (entropy.detach() - self.target_entropy).mean()
    )
    assert self.temperature.optimizer is not None
    self.temperature.optimizer.zero_grad(set_to_none=True)
    temperature_loss.backward()
    self.temperature.optimizer.step()
    if self.temperature.scheduler is not None:
      self.temperature.scheduler.step()
    return _add_prefix(
      {"value": temperature_value, "loss": temperature_loss}, "temperature"
    )

  def save(self, path: str) -> None:
    os.makedirs(path, exist_ok=True)
    self.actor.save(os.path.join(path, "actor.pt"))
    self.critic.save(os.path.join(path, "critic.pt"))
    self.target_critic.save(os.path.join(path, "target_critic.pt"))
    self.temperature.save(os.path.join(path, "temperature.pt"))
    if self.reward_normalizer is not None:
      self.reward_normalizer.save(os.path.join(path, "reward_normalizer.pt"))
    if self.observation_normalizer is not None:
      self.observation_normalizer.save(os.path.join(path, "observation_normalizer.pt"))
    if (
      self.actor_observation_normalizer is not None
      and self.actor_observation_normalizer is not self.observation_normalizer
    ):
      self.actor_observation_normalizer.save(
        os.path.join(path, "actor_observation_normalizer.pt")
      )
    torch.save(
      {
        "update_step": self.state.update_step,
        "grad_scaler_state_dict": self.grad_scaler.state_dict(),
      },
      os.path.join(path, "agent_state.pt"),
    )

  def save_replay_buffer(self, path: str) -> None:
    self.replay_buffer.save(os.path.join(path, "replay_buffer.pt"))

  def load(self, path: str) -> None:
    self.actor.load(
      os.path.join(path, "actor.pt"), load_optimizer=self.cfg.load_optimizer
    )
    self.critic.load(
      os.path.join(path, "critic.pt"), load_optimizer=self.cfg.load_optimizer
    )
    self.target_critic.load(
      os.path.join(path, "target_critic.pt"), load_optimizer=False
    )
    self.temperature.load(
      os.path.join(path, "temperature.pt"), load_optimizer=self.cfg.load_optimizer
    )
    if self.cfg.load_optimizer:
      agent_state = torch.load(
        os.path.join(path, "agent_state.pt"), map_location=self.device
      )
      self.state.update_step = agent_state["update_step"]
      self.grad_scaler.load_state_dict(agent_state["grad_scaler_state_dict"])
    if self.cfg.load_reward_normalizer and self.reward_normalizer is not None:
      self.reward_normalizer.load(os.path.join(path, "reward_normalizer.pt"))
    if (
      self.cfg.load_observation_normalizer
      and self.observation_normalizer is not None
      and os.path.exists(os.path.join(path, "observation_normalizer.pt"))
    ):
      self.observation_normalizer.load(os.path.join(path, "observation_normalizer.pt"))
    if (
      self.cfg.load_observation_normalizer
      and self.actor_observation_normalizer is not None
      and self.actor_observation_normalizer is not self.observation_normalizer
      and os.path.exists(os.path.join(path, "actor_observation_normalizer.pt"))
    ):
      self.actor_observation_normalizer.load(
        os.path.join(path, "actor_observation_normalizer.pt")
      )

  def load_replay_buffer(self, path: str) -> None:
    self.replay_buffer.load(os.path.join(path, "replay_buffer.pt"))
