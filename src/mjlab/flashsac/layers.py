from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mjlab.flashsac.utils import safe_tanh_log_det_jacobian


class UnitLinear(nn.Module):
  def __init__(self, input_dim: int, output_dim: int):
    super().__init__()
    self.w = nn.Linear(input_dim, output_dim, bias=False)
    nn.init.orthogonal_(self.w.weight, gain=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.w(x)

  def normalize_parameters(self) -> None:
    self.w.weight.copy_(F.normalize(self.w.weight, dim=-1, eps=1e-8))


class UnitBatchNorm(nn.Module):
  running_mean: torch.Tensor
  running_var: torch.Tensor

  def __init__(self, input_dim: int, momentum: float = 0.01, eps: float = 1e-5):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(input_dim))
    self.bias = nn.Parameter(torch.zeros(input_dim))
    self.register_buffer("running_mean", torch.zeros(input_dim))
    self.register_buffer("running_var", torch.ones(input_dim))
    self.momentum = momentum
    self.eps = eps

  def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
    return F.batch_norm(
      x,
      self.running_mean,
      self.running_var,
      self.weight,
      self.bias,
      training=training,
      momentum=self.momentum,
      eps=self.eps,
    )

  def normalize_parameters(self) -> None:
    scale, bias = self.weight.data, self.bias.data
    ndim = scale.shape[-1]
    sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
    norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
    self.weight.data.copy_(scale * norm_factor)
    self.bias.data.copy_(bias * norm_factor)


class UnitRMSNorm(nn.Module):
  def __init__(self, input_dim: int, eps: float = 1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(input_dim))
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, self.weight.shape, self.weight, eps=self.eps)

  def normalize_parameters(self) -> None:
    scale = self.weight.data
    ndim = scale.shape[-1]
    sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
    norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
    self.weight.data.copy_(scale * norm_factor)


class FlashSACEmbedder(nn.Module):
  def __init__(self, input_dim: int, hidden_dim: int):
    super().__init__()
    self.norm = UnitBatchNorm(input_dim)
    self.w = UnitLinear(input_dim, hidden_dim)

  def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
    return self.w(self.norm(x, training=training))


class FlashSACBlock(nn.Module):
  def __init__(self, hidden_dim: int, expansion: int = 4):
    super().__init__()
    self.w1 = UnitLinear(hidden_dim, hidden_dim * expansion)
    self.w2 = UnitLinear(hidden_dim * expansion, hidden_dim)
    self.norm1 = UnitBatchNorm(hidden_dim * expansion)
    self.norm2 = UnitBatchNorm(hidden_dim)

  def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
    residual = x
    x = F.relu(self.norm1(self.w1(x), training=training))
    x = F.relu(self.norm2(self.w2(x), training=training))
    return x + residual


class NormalTanhPolicy(nn.Module):
  def __init__(
    self,
    hidden_dim: int,
    action_dim: int,
    log_std_min: float = -10.0,
    log_std_max: float = 2.0,
  ):
    super().__init__()
    self.mean_w = UnitLinear(hidden_dim, action_dim)
    self.mean_bias = nn.Parameter(torch.zeros(action_dim))
    self.std_w = UnitLinear(hidden_dim, action_dim)
    self.std_bias = nn.Parameter(torch.zeros(action_dim))
    self.log_std_min = log_std_min
    self.log_std_max = log_std_max

  def get_mean_and_std(
    self, x: torch.Tensor, training: bool
  ) -> tuple[torch.Tensor, torch.Tensor]:
    del training
    mean = F.linear(x, self.mean_w.w.weight, self.mean_bias)
    raw_log_std = F.linear(x, self.std_w.w.weight, self.std_bias)
    log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (
      1 + torch.tanh(raw_log_std)
    )
    return mean, torch.exp(log_std)

  def forward(
    self, x: torch.Tensor, training: bool
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mean, std = self.get_mean_and_std(x, training)
    dist = torch.distributions.Normal(mean, std)
    raw_action = dist.rsample()
    tanh_action = torch.tanh(raw_action)
    log_prob = dist.log_prob(raw_action) - safe_tanh_log_det_jacobian(raw_action)
    return tanh_action, {"log_prob": log_prob.sum(1)}


class EnsembleUnitLinear(nn.Module):
  def __init__(self, num_ensemble: int, input_dim: int, output_dim: int):
    super().__init__()
    self.weight = nn.Parameter(torch.empty(num_ensemble, output_dim, input_dim))
    for idx in range(num_ensemble):
      nn.init.orthogonal_(self.weight.data[idx], gain=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nbi,noi->nbo", x, self.weight)

  def normalize_parameters(self) -> None:
    self.weight.copy_(F.normalize(self.weight, dim=-1, eps=1e-8))


class EnsembleUnitBatchNorm(nn.Module):
  running_mean: torch.Tensor
  running_var: torch.Tensor

  def __init__(
    self, num_ensemble: int, input_dim: int, momentum: float = 0.01, eps: float = 1e-5
  ):
    super().__init__()
    self.momentum = momentum
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
    self.bias = nn.Parameter(torch.zeros(num_ensemble, input_dim))
    self.register_buffer("running_mean", torch.zeros(num_ensemble, input_dim))
    self.register_buffer("running_var", torch.ones(num_ensemble, input_dim))

  def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
    if training:
      mean = x.mean(dim=1, keepdim=True)
      var = x.var(dim=1, correction=0, keepdim=True)
      with torch.no_grad():
        batch_size = x.shape[1]
        self.running_mean.lerp_(mean.squeeze(1).float(), self.momentum)
        self.running_var.lerp_(
          (var.squeeze(1) * (batch_size / max(batch_size - 1, 1))).float(),
          self.momentum,
        )
      x = (x - mean) * torch.rsqrt(var + self.eps)
    else:
      x = (x - self.running_mean.unsqueeze(1)) * torch.rsqrt(
        self.running_var.unsqueeze(1) + self.eps
      )
    return x * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)

  def normalize_parameters(self) -> None:
    scale, bias = self.weight.data, self.bias.data
    ndim = scale.shape[-1]
    sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
    norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
    self.weight.data.copy_(scale * norm_factor)
    self.bias.data.copy_(bias * norm_factor)


class EnsembleUnitRMSNorm(nn.Module):
  def __init__(self, num_ensemble: int, input_dim: int, eps: float = 1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
    return (x / rms) * self.weight.unsqueeze(1)

  def normalize_parameters(self) -> None:
    scale = self.weight.data
    ndim = scale.shape[-1]
    sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
    norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
    self.weight.data.copy_(scale * norm_factor)


class EnsembleFlashSACEmbedder(nn.Module):
  def __init__(self, num_ensemble: int, input_dim: int, hidden_dim: int):
    super().__init__()
    self.norm = EnsembleUnitBatchNorm(num_ensemble, input_dim)
    self.w = EnsembleUnitLinear(num_ensemble, input_dim, hidden_dim)

  def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
    return self.w(self.norm(x, training=training))


class EnsembleFlashSACBlock(nn.Module):
  def __init__(self, num_ensemble: int, hidden_dim: int, expansion: int = 4):
    super().__init__()
    self.w1 = EnsembleUnitLinear(num_ensemble, hidden_dim, hidden_dim * expansion)
    self.w2 = EnsembleUnitLinear(num_ensemble, hidden_dim * expansion, hidden_dim)
    self.norm1 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim * expansion)
    self.norm2 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim)

  def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
    residual = x
    x = F.relu(self.norm1(self.w1(x), training=training))
    x = F.relu(self.norm2(self.w2(x), training=training))
    return x + residual


class EnsembleCategoricalValue(nn.Module):
  bin_values: torch.Tensor

  def __init__(
    self, num_ensemble: int, hidden_dim: int, num_bins: int, min_v: float, max_v: float
  ):
    super().__init__()
    self.w = EnsembleUnitLinear(num_ensemble, hidden_dim, num_bins)
    self.bias = nn.Parameter(torch.zeros(num_ensemble, num_bins))
    self.register_buffer(
      "bin_values",
      torch.linspace(
        start=min_v, end=max_v, steps=num_bins, dtype=torch.float32
      ).reshape(1, 1, -1),
    )

  def forward(
    self, x: torch.Tensor, training: bool
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    del training
    value = self.w(x) + self.bias.unsqueeze(1)
    log_prob = F.log_softmax(value, dim=-1)
    q_value = torch.sum(torch.exp(log_prob) * self.bin_values, dim=-1)
    return q_value, {"log_prob": log_prob}
