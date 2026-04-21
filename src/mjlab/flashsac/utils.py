from __future__ import annotations

import math
import os
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


def _portable_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
  if hasattr(module, "_orig_mod"):
    orig_mod = module._orig_mod  # type: ignore[attr-defined]
    if isinstance(orig_mod, nn.Module):
      return orig_mod.state_dict()
  return module.state_dict()


def _strip_orig_mod_prefix(
  state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
  prefix = "_orig_mod."
  if not any(key.startswith(prefix) for key in state_dict):
    return state_dict
  return {
    key[len(prefix) :] if key.startswith(prefix) else key: value
    for key, value in state_dict.items()
  }


def _load_portable_state_dict(
  module: nn.Module, state_dict: dict[str, torch.Tensor]
) -> None:
  prefix = "_orig_mod."
  orig_mod = getattr(module, "_orig_mod", None)
  has_prefixed_keys = any(key.startswith(prefix) for key in state_dict)
  if isinstance(orig_mod, nn.Module):
    if has_prefixed_keys:
      module.load_state_dict(state_dict)
      return
    orig_mod.load_state_dict(state_dict)
    return
  module.load_state_dict(_strip_orig_mod_prefix(state_dict))


def safe_tanh_log_det_jacobian(x: torch.Tensor) -> torch.Tensor:
  return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


def warmup_cosine_decay_scheduler(
  init_value: float,
  peak_value: float,
  end_value: float,
  warmup_steps: int,
  decay_steps: int,
):
  def scheduler(step: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
      return init_value + (peak_value - init_value) * (step / warmup_steps)
    if step < decay_steps:
      denom = max(decay_steps - warmup_steps, 1)
      progress = (step - warmup_steps) / denom
      return end_value + (peak_value - end_value) * 0.5 * (
        1 + math.cos(math.pi * progress)
      )
    return end_value

  return scheduler


class NetworkBundle:
  def __init__(
    self,
    network: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    compile_network: bool = False,
    compile_mode: str = "default",
    use_weight_normalization: bool = False,
    ema_source: NetworkBundle | None = None,
    ema_tau: float | None = None,
  ):
    self.network = network
    self.optimizer = optimizer
    self.scheduler = scheduler

    if compile_network:
      self.network = cast(
        nn.Module,
        torch.compile(self.network, mode=compile_mode),  # type: ignore[arg-type]
      )

    self._weight_normalize_fn = None
    if use_weight_normalization:
      modules = [
        mod for mod in network.modules() if hasattr(mod, "normalize_parameters")
      ]

      def _weight_normalize_fn() -> None:
        for mod in modules:
          mod.normalize_parameters()  # type: ignore[attr-defined]

      self._weight_normalize_fn = _weight_normalize_fn
      if compile_network:
        self._weight_normalize_fn = torch.compile(
          self._weight_normalize_fn, mode=compile_mode
        )

    self._ema_update_fn = None
    if ema_source is not None:
      assert ema_tau is not None
      target_params: list[torch.Tensor] = list(self.network.parameters())
      source_params: list[torch.Tensor] = list(ema_source.network.parameters())

      def _ema_update_fn() -> None:
        torch._foreach_lerp_(target_params, source_params, ema_tau)

      self._ema_update_fn = _ema_update_fn
      if compile_network:
        self._ema_update_fn = torch.compile(self._ema_update_fn, mode=compile_mode)

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    return self.network(*args, **kwargs)

  def apply(self, method: str, *args: Any, **kwargs: Any) -> Any:
    fn = getattr(self.network, method)
    return fn(*args, **kwargs)

  @torch.no_grad()
  def normalize_parameters(self) -> None:
    if self._weight_normalize_fn is not None:
      self._weight_normalize_fn()

  @torch.no_grad()
  def ema_update_parameters(self) -> None:
    if self._ema_update_fn is not None:
      self._ema_update_fn()

  def save(self, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
      {
        "network_state_dict": _portable_state_dict(cast(nn.Module, self.network)),
        "optimizer_state_dict": self.optimizer.state_dict()
        if self.optimizer is not None
        else None,
        "scheduler_state_dict": self.scheduler.state_dict()
        if self.scheduler is not None
        else None,
      },
      path,
    )

  def load(self, path: str, load_optimizer: bool = True) -> None:
    network = cast(nn.Module, self.network)
    checkpoint = torch.load(path, map_location=next(network.parameters()).device)
    state_dict = checkpoint["network_state_dict"]
    _load_portable_state_dict(network, state_dict)
    if (
      load_optimizer
      and self.optimizer is not None
      and checkpoint["optimizer_state_dict"] is not None
    ):
      self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if (
      load_optimizer
      and self.scheduler is not None
      and checkpoint["scheduler_state_dict"] is not None
    ):
      self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])


def _update_reward_stats(
  reward: torch.Tensor,
  terminated: torch.Tensor,
  truncated: torch.Tensor,
  returns: torch.Tensor,
  returns_max: torch.Tensor,
  gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  done = torch.logical_or(terminated, truncated).float()
  new_returns = gamma * (1.0 - done) * returns + reward
  new_returns_max = torch.maximum(returns_max, torch.max(torch.abs(new_returns)))
  return new_returns, new_returns_max


def _scale_reward(
  rewards: torch.Tensor,
  returns_var: torch.Tensor,
  returns_max: torch.Tensor,
  G_max: float,
  eps: float,
) -> torch.Tensor:
  var_denominator = torch.sqrt(returns_var + eps)
  min_required_denominator = returns_max / G_max
  denominator = torch.maximum(var_denominator, min_required_denominator)
  return rewards / denominator


def _update_mean_var_count_from_moments(
  samples: torch.Tensor,
  running_mean: torch.Tensor,
  running_var: torch.Tensor,
  running_count: torch.Tensor,
  epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  sample_mean = torch.mean(samples, dim=0)
  sample_var = torch.var(samples, dim=0, unbiased=False)
  sample_count = float(samples.shape[0])
  delta = sample_mean - running_mean
  total_count = running_count + sample_count
  ratio = sample_count / total_count
  new_mean = running_mean + delta * ratio
  m_a = running_var * (running_count + epsilon)
  m_b = sample_var * sample_count
  M2 = m_a + m_b + torch.square(delta) * running_count * ratio
  new_var = M2 / total_count
  return new_mean, new_var, total_count


class RunningMeanStd:
  def __init__(
    self,
    device: torch.device,
    epsilon: float = 1e-4,
    shape: tuple[int, ...] = (),
    dtype: torch.dtype = torch.float32,
  ):
    self.mean = torch.zeros(shape, dtype=dtype, device=device)
    self.var = torch.ones(shape, dtype=dtype, device=device)
    self.count = torch.tensor(0.0, dtype=dtype, device=device)
    self.epsilon = epsilon
    self.device = device

  def update(self, x: torch.Tensor) -> None:
    self.mean, self.var, self.count = _update_mean_var_count_from_moments(
      samples=x,
      running_mean=self.mean,
      running_var=self.var,
      running_count=self.count,
      epsilon=self.epsilon,
    )


class ObservationNormalizer:
  def __init__(
    self,
    shape: tuple[int, ...],
    device: torch.device,
    epsilon: float = 1e-8,
    clip_value: float | None = 10.0,
  ):
    self.obs_rms = RunningMeanStd(shape=shape, device=device, dtype=torch.float32)
    self.epsilon = epsilon
    self.clip_value = clip_value
    self.device = device

  def update(self, observations: torch.Tensor) -> None:
    self.obs_rms.update(observations.to(dtype=torch.float32))

  def normalize(self, observations: torch.Tensor) -> torch.Tensor:
    normalized = (observations - self.obs_rms.mean) / torch.sqrt(
      self.obs_rms.var + self.epsilon
    )
    if self.clip_value is not None:
      normalized = torch.clamp(normalized, -self.clip_value, self.clip_value)
    return normalized

  def save(self, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
      {
        "mean": self.obs_rms.mean,
        "var": self.obs_rms.var,
        "count": self.obs_rms.count,
      },
      path,
    )

  def load(self, path: str) -> None:
    state = torch.load(path, map_location=self.device)
    self.obs_rms.mean = state["mean"]
    self.obs_rms.var = state["var"]
    self.obs_rms.count = state["count"]


class RewardNormalizer:
  def __init__(
    self,
    gamma: float,
    G_max: float,
    load_rms: bool,
    device: torch.device,
    epsilon: float = 1e-8,
  ):
    self.gamma = gamma
    self.returns = torch.zeros(1, dtype=torch.float32, device=device)
    self.returns_max = torch.zeros(1, dtype=torch.float32, device=device)
    self.returns_rms = RunningMeanStd(shape=(1,), device=device, dtype=torch.float32)
    self.G_max = G_max
    self.load_rms = load_rms
    self.epsilon = epsilon
    self.device = device

  def update_reward_stats(
    self,
    reward: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
  ) -> None:
    self.returns, self.returns_max = _update_reward_stats(
      reward=reward,
      terminated=terminated,
      truncated=truncated,
      returns=self.returns,
      returns_max=self.returns_max,
      gamma=self.gamma,
    )
    self.returns_rms.update(self.returns)

  def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
    return _scale_reward(
      rewards=rewards,
      returns_var=self.returns_rms.var,
      returns_max=self.returns_max,
      G_max=self.G_max,
      eps=self.epsilon,
    )

  def save(self, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
      {
        "returns": self.returns,
        "returns_max": self.returns_max,
        "returns_rms_mean": self.returns_rms.mean,
        "returns_rms_var": self.returns_rms.var,
        "returns_rms_count": self.returns_rms.count,
      },
      path,
    )

  def load(self, path: str) -> None:
    state = torch.load(path, map_location=self.device)
    self.returns = state["returns"]
    self.returns_max = state["returns_max"]
    self.returns_rms.mean = state["returns_rms_mean"]
    self.returns_rms.var = state["returns_rms_var"]
    self.returns_rms.count = state["returns_rms_count"]
