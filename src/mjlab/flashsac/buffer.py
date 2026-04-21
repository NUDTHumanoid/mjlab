from __future__ import annotations

import os
from collections import deque
from typing import Any

import numpy as np
import torch

Batch = dict[str, Any]


class TorchUniformBuffer:
  def __init__(
    self,
    observation_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    n_step: int,
    gamma: float,
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    device_type: str,
  ):
    self._observation_shape = observation_shape
    self._action_shape = action_shape
    self._n_step = n_step
    self._gamma = gamma
    self._max_length = max_length
    self._min_length = min_length
    self._sample_batch_size = sample_batch_size
    self._device = torch.device(
      device_type
      if device_type.startswith("cuda:")
      else ("cuda:0" if device_type.startswith("cuda") else "cpu")
    )
    self.reset()

  def __len__(self) -> int:
    return self._num_in_buffer

  def reset(self) -> None:
    pin_memory = self._device.type == "cpu" and torch.cuda.is_available()
    max_length = self._max_length
    self._observations = torch.empty(
      (max_length,) + self._observation_shape,
      dtype=torch.float32,
      device=self._device,
      pin_memory=pin_memory,
    )
    self._next_observations = torch.empty(
      (max_length,) + self._observation_shape,
      dtype=torch.float32,
      device=self._device,
      pin_memory=pin_memory,
    )
    self._actions = torch.empty(
      (max_length,) + self._action_shape,
      dtype=torch.float32,
      device=self._device,
      pin_memory=pin_memory,
    )
    self._rewards = torch.empty(
      (max_length,), dtype=torch.float32, device=self._device, pin_memory=pin_memory
    )
    self._terminateds = torch.empty(
      (max_length,), dtype=torch.bool, device=self._device, pin_memory=pin_memory
    )
    self._truncateds = torch.empty(
      (max_length,), dtype=torch.bool, device=self._device, pin_memory=pin_memory
    )
    self._n_step_transitions: deque[dict[str, torch.Tensor]] = deque(
      maxlen=self._n_step
    )
    self._num_in_buffer = 0
    self._current_idx = 0

  def _to_tensor(self, value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
      return value.detach().to(self._device, copy=True)
    if isinstance(value, np.ndarray):
      return torch.as_tensor(value, device=self._device).clone()
    return torch.tensor(value, device=self._device)

  def _get_n_step_prev_transition(self) -> dict[str, torch.Tensor]:
    prev_transition = self._n_step_transitions[0]
    curr_transition = self._n_step_transitions[-1]
    n_step_reward = curr_transition["reward"].clone()
    n_step_terminated = curr_transition["terminated"].clone()
    n_step_truncated = curr_transition["truncated"].clone()
    n_step_next_observation = curr_transition["next_observation"].clone()

    for idx in reversed(range(self._n_step - 1)):
      transition = self._n_step_transitions[idx]
      reward = transition["reward"]
      terminated = transition["terminated"]
      truncated = transition["truncated"]
      next_observation = transition["next_observation"]
      done = terminated | truncated
      n_step_reward = reward + self._gamma * n_step_reward * (~done).float()
      n_step_terminated[done] = terminated[done]
      n_step_truncated[done] = truncated[done]
      n_step_next_observation[done] = next_observation[done]

    prev_transition["reward"] = n_step_reward
    prev_transition["terminated"] = n_step_terminated
    prev_transition["truncated"] = n_step_truncated
    prev_transition["next_observation"] = n_step_next_observation
    return prev_transition

  def add(self, transition: Batch) -> None:
    self._n_step_transitions.append(
      {key: self._to_tensor(value) for key, value in transition.items()}
    )
    if len(self._n_step_transitions) < self._n_step:
      return
    prev_transition = self._get_n_step_prev_transition()
    batch_size = len(prev_transition["observation"])
    end_idx = self._current_idx + batch_size
    if end_idx <= self._max_length:
      idxs: Any = slice(self._current_idx, end_idx)
    else:
      idxs = (
        torch.arange(batch_size, device=self._device) + self._current_idx
      ) % self._max_length
    self._observations[idxs] = prev_transition["observation"].to(torch.float32)
    self._next_observations[idxs] = prev_transition["next_observation"].to(
      torch.float32
    )
    self._actions[idxs] = prev_transition["action"].to(torch.float32)
    self._rewards[idxs] = prev_transition["reward"].to(torch.float32)
    self._terminateds[idxs] = prev_transition["terminated"].to(torch.bool)
    self._truncateds[idxs] = prev_transition["truncated"].to(torch.bool)
    self._num_in_buffer = min(self._num_in_buffer + batch_size, self._max_length)
    self._current_idx = (self._current_idx + batch_size) % self._max_length

  def can_sample(self) -> bool:
    return self._num_in_buffer >= self._min_length

  def sample(self) -> dict[str, torch.Tensor]:
    idxs = torch.randint(
      0, self._num_in_buffer, (self._sample_batch_size,), device=self._device
    )
    return {
      "observation": self._observations[idxs],
      "action": self._actions[idxs],
      "reward": self._rewards[idxs],
      "terminated": self._terminateds[idxs],
      "truncated": self._truncateds[idxs],
      "next_observation": self._next_observations[idxs],
    }

  def save(self, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = self._num_in_buffer
    torch.save(
      {
        "observation": self._observations[:count],
        "action": self._actions[:count],
        "reward": self._rewards[:count],
        "terminated": self._terminateds[:count],
        "truncated": self._truncateds[:count],
        "next_observation": self._next_observations[:count],
        "num_in_buffer": self._num_in_buffer,
        "current_idx": self._current_idx,
      },
      path,
    )

  def load(self, path: str) -> None:
    dataset = torch.load(path, map_location=self._device)
    count = dataset["num_in_buffer"]
    self._observations[:count] = dataset["observation"]
    self._next_observations[:count] = dataset["next_observation"]
    self._actions[:count] = dataset["action"]
    self._rewards[:count] = dataset["reward"]
    self._terminateds[:count] = dataset["terminated"]
    self._truncateds[:count] = dataset["truncated"]
    self._num_in_buffer = count
    self._current_idx = dataset["current_idx"]
    self._n_step_transitions.clear()
