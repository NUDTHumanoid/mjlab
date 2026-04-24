from __future__ import annotations

import pytest
import torch

from mjlab.scripts.benchmark_flashsac_ppo import (
  DEFAULT_TRACKING_ACRO_TOTAL_ENV_STEPS,
  DEFAULT_TRACKING_TOTAL_ENV_STEPS,
  DEFAULT_VELOCITY_TOTAL_ENV_STEPS,
  BenchmarkConfig,
  _build_train_command,
  build_cases,
)
from mjlab.tasks.tracking.scripts.evaluate import _summarize_episode_metrics


def test_build_cases_emits_full_matrix() -> None:
  cases = build_cases(BenchmarkConfig())

  assert len(cases) == 36
  assert {case.task_key for case in cases} == {
    "tracking",
    "tracking_acro",
    "velocity",
  }
  assert {case.backend for case in cases} == {"rsl_rl", "flashsac"}
  assert {case.num_envs for case in cases} == {1024, 4096}
  assert {case.seed for case in cases} == {42, 43, 44}


def test_velocity_ppo_budget_stays_fixed_across_env_counts() -> None:
  cases = {
    (case.backend, case.num_envs): case
    for case in build_cases(BenchmarkConfig(tasks=("velocity",), seeds=(42,)))
  }

  ppo_4096 = cases[("rsl_rl", 4096)]
  ppo_1024 = cases[("rsl_rl", 1024)]
  flashsac_4096 = cases[("flashsac", 4096)]
  flashsac_1024 = cases[("flashsac", 1024)]

  assert ppo_4096.total_env_steps == DEFAULT_VELOCITY_TOTAL_ENV_STEPS
  assert ppo_1024.total_env_steps == DEFAULT_VELOCITY_TOTAL_ENV_STEPS
  assert ppo_4096.max_iterations == 509
  assert ppo_1024.max_iterations == 2036
  assert flashsac_4096.total_env_steps == DEFAULT_VELOCITY_TOTAL_ENV_STEPS
  assert flashsac_1024.total_env_steps == DEFAULT_VELOCITY_TOTAL_ENV_STEPS


def test_tracking_ppo_budget_stays_fixed_across_env_counts() -> None:
  cases = {
    (case.backend, case.num_envs): case
    for case in build_cases(BenchmarkConfig(tasks=("tracking",), seeds=(42,)))
  }

  ppo_4096 = cases[("rsl_rl", 4096)]
  ppo_1024 = cases[("rsl_rl", 1024)]
  flashsac_4096 = cases[("flashsac", 4096)]
  flashsac_1024 = cases[("flashsac", 1024)]

  assert ppo_4096.total_env_steps == DEFAULT_TRACKING_TOTAL_ENV_STEPS
  assert ppo_1024.total_env_steps == DEFAULT_TRACKING_TOTAL_ENV_STEPS
  assert ppo_4096.max_iterations == 4069
  assert ppo_1024.max_iterations == 16276
  assert flashsac_4096.total_env_steps == DEFAULT_TRACKING_TOTAL_ENV_STEPS
  assert flashsac_1024.total_env_steps == DEFAULT_TRACKING_TOTAL_ENV_STEPS


def test_tracking_acro_budget_stays_fixed_across_env_counts() -> None:
  cases = {
    (case.backend, case.num_envs): case
    for case in build_cases(BenchmarkConfig(tasks=("tracking_acro",), seeds=(42,)))
  }

  ppo_4096 = cases[("rsl_rl", 4096)]
  ppo_1024 = cases[("rsl_rl", 1024)]
  flashsac_4096 = cases[("flashsac", 4096)]
  flashsac_1024 = cases[("flashsac", 1024)]

  assert ppo_4096.total_env_steps == DEFAULT_TRACKING_ACRO_TOTAL_ENV_STEPS
  assert ppo_1024.total_env_steps == DEFAULT_TRACKING_ACRO_TOTAL_ENV_STEPS
  assert ppo_4096.max_iterations == 381
  assert ppo_1024.max_iterations == 1524
  assert flashsac_4096.total_env_steps == DEFAULT_TRACKING_ACRO_TOTAL_ENV_STEPS
  assert flashsac_1024.total_env_steps == DEFAULT_TRACKING_ACRO_TOTAL_ENV_STEPS


def test_flashsac_train_commands_use_task_specific_runtime_recipe() -> None:
  cfg = BenchmarkConfig(tasks=("velocity", "tracking", "tracking_acro"), seeds=(42,))
  cases = {
    (case.task_key, case.backend, case.num_envs): case for case in build_cases(cfg)
  }

  velocity_cmd = _build_train_command(cases[("velocity", "flashsac", 1024)], cfg)
  tracking_cmd = _build_train_command(cases[("tracking", "flashsac", 1024)], cfg)
  tracking_acro_cmd = _build_train_command(
    cases[("tracking_acro", "flashsac", 1024)], cfg
  )

  assert velocity_cmd[velocity_cmd.index("--agent.use-compile") + 1] == "True"
  assert velocity_cmd[velocity_cmd.index("--agent.use-amp") + 1] == "True"
  assert (
    velocity_cmd[velocity_cmd.index("--agent.logging-per-interaction-step") + 1] == "49"
  )
  assert (
    velocity_cmd[velocity_cmd.index("--agent.save-checkpoint-per-interaction-step") + 1]
    == "4883"
  )

  assert tracking_cmd[tracking_cmd.index("--agent.use-compile") + 1] == "True"
  assert tracking_cmd[tracking_cmd.index("--agent.use-amp") + 1] == "True"

  assert tracking_acro_cmd[tracking_acro_cmd.index("--agent.use-compile") + 1] == "True"
  assert tracking_acro_cmd[tracking_acro_cmd.index("--agent.use-amp") + 1] == "True"


def test_summarize_episode_metrics_includes_reward_and_length_stats() -> None:
  metrics = _summarize_episode_metrics(
    episode_returns=torch.tensor([1.0, 3.0], dtype=torch.float32),
    episode_lengths=torch.tensor([10, 20], dtype=torch.int64),
    step_dt=0.02,
    success=torch.tensor([True, False]),
    terminated_rate=0.5,
    time_out_rate=0.5,
    termination_counts={"anchor_pos": 1, "time_out": 1},
  )

  assert metrics["success_rate"] == pytest.approx(0.5)
  assert metrics["episode_return_mean"] == pytest.approx(2.0)
  assert metrics["episode_return_std"] == pytest.approx(1.0)
  assert metrics["episode_length_steps_mean"] == pytest.approx(15.0)
  assert metrics["episode_length_seconds_mean"] == pytest.approx(0.3)
  assert metrics["terminated_rate"] == pytest.approx(0.5)
  assert metrics["time_out_rate"] == pytest.approx(0.5)
  assert metrics["termination_rate_anchor_pos"] == pytest.approx(0.5)
  assert metrics["termination_rate_time_out"] == pytest.approx(0.5)
