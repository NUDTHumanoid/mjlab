from __future__ import annotations

import math


def _require_positive(name: str, value: int) -> None:
  if value <= 0:
    raise ValueError(f"{name} must be positive, got {value}.")


def interaction_steps_from_total_env_steps(total_env_steps: int, num_envs: int) -> int:
  """Convert a total env-step budget into vector-env interaction steps."""
  _require_positive("num_envs", num_envs)
  if total_env_steps < 0:
    raise ValueError(f"total_env_steps must be non-negative, got {total_env_steps}.")
  return math.ceil(total_env_steps / num_envs)


def total_env_steps_from_interaction_steps(
  interaction_steps: int,
  num_envs: int,
  world_size: int = 1,
) -> int:
  """Convert interaction steps into total env steps across envs/worlds."""
  if interaction_steps < 0:
    raise ValueError(
      f"interaction_steps must be non-negative, got {interaction_steps}."
    )
  _require_positive("num_envs", num_envs)
  _require_positive("world_size", world_size)
  return interaction_steps * num_envs * world_size


def checkpoint_interval_from_total_env_steps(
  total_env_steps: int,
  num_envs: int,
  checkpoint_count: int,
) -> int:
  """Derive an interaction-step checkpoint cadence from a total env-step budget."""
  _require_positive("checkpoint_count", checkpoint_count)
  total_interaction_steps = interaction_steps_from_total_env_steps(
    total_env_steps, num_envs
  )
  return max(1, math.ceil(total_interaction_steps / checkpoint_count))


def on_policy_iteration_env_steps(
  num_envs: int,
  num_steps_per_env: int,
  world_size: int = 1,
) -> int:
  """Total env steps collected by one on-policy learning iteration."""
  _require_positive("num_steps_per_env", num_steps_per_env)
  return total_env_steps_from_interaction_steps(
    num_steps_per_env,
    num_envs=num_envs,
    world_size=world_size,
  )
