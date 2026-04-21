from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.spaces import Box
from mjlab.utils.spaces import Dict as DictSpace


def _tensor_to_float(value: Any) -> float | None:
  if isinstance(value, torch.Tensor):
    if value.numel() == 1:
      return float(value.item())
    return float(value.float().mean().item())
  if isinstance(value, (float, int)):
    return float(value)
  return None


@dataclass
class FlashSACActionSpace:
  shape: tuple[int, ...]
  dtype: str = "float32"
  low: float = -1.0
  high: float = 1.0


class MjlabFlashSACEnvAdapter:
  def __init__(self, env: ManagerBasedRlEnv):
    self.env = env
    observation_space = env.single_observation_space
    assert isinstance(observation_space, DictSpace)
    assert observation_space.spaces
    self.actor_key = (
      "actor"
      if "actor" in observation_space.spaces
      else next(iter(observation_space.spaces))
    )
    self.critic_key = (
      "critic" if "critic" in observation_space.spaces else self.actor_key
    )
    actor_space = observation_space.spaces[self.actor_key]
    critic_space = observation_space.spaces[self.critic_key]
    assert isinstance(actor_space, Box)
    assert isinstance(critic_space, Box)
    self.actor_dim = actor_space.shape[-1]
    self.critic_dim = critic_space.shape[-1]
    self.has_critic_obs = "critic" in observation_space.spaces
    obs_dim = (
      self.actor_dim + self.critic_dim if self.has_critic_obs else self.actor_dim
    )
    self.observation_space = Box(shape=(obs_dim,), low=-np.inf, high=np.inf)
    self.action_space = FlashSACActionSpace(shape=env.single_action_space.shape)
    self.num_envs = env.num_envs

  def policy_observation_dim(self, *, asymmetric_observation: bool) -> int:
    if asymmetric_observation:
      return self.actor_dim
    return self.observation_space.shape[-1]

  def _flatten_obs(self, obs_dict: dict[str, Any]) -> torch.Tensor:
    actor_obs = obs_dict[self.actor_key]
    assert isinstance(actor_obs, torch.Tensor)
    if not self.has_critic_obs:
      return actor_obs
    critic_obs = obs_dict[self.critic_key]
    assert isinstance(critic_obs, torch.Tensor)
    return torch.cat([actor_obs, critic_obs], dim=-1)

  def sample_random_actions(self) -> np.ndarray:
    return np.random.uniform(
      -1.0, 1.0, size=(self.num_envs,) + self.action_space.shape
    ).astype(np.float32)

  def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
    obs_dict, extras = self.env.reset()
    flat_obs = self._flatten_obs(obs_dict).cpu().numpy().astype(np.float32, copy=False)
    env_info = {"actor_observation_size": (self.actor_dim,)}
    if "log" in extras:
      episode_info = {
        key: scalar
        for key, value in extras["log"].items()
        if (scalar := _tensor_to_float(value)) is not None
      }
      if episode_info:
        env_info["episode_info"] = episode_info
    return flat_obs, env_info

  def step(
    self,
    actions: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    obs_dict, rewards, terminated, truncated, extras = self.env.step(
      torch.as_tensor(actions, dtype=torch.float32, device=self.env.device)
    )
    next_obs = self._flatten_obs(obs_dict).cpu().numpy().astype(np.float32, copy=False)
    info: dict[str, Any] = {}
    if "final_obs" in extras:
      final_obs = self._flatten_obs(extras["final_obs"])
      info["final_obs"] = final_obs.cpu().numpy().astype(np.float32, copy=False)
    info["time_outs"] = truncated.cpu().numpy().astype(np.float32, copy=False)
    if "log" in extras:
      episode_info = {
        key: scalar
        for key, value in extras["log"].items()
        if (scalar := _tensor_to_float(value)) is not None
      }
      if episode_info:
        info["episode_info"] = episode_info
    return (
      next_obs,
      rewards.cpu().numpy().astype(np.float32, copy=False),
      terminated.cpu().numpy().astype(np.bool_, copy=False),
      truncated.cpu().numpy().astype(np.bool_, copy=False),
      info,
    )

  def close(self) -> None:
    self.env.close()
