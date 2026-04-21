from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.training_steps import checkpoint_interval_from_total_env_steps

FLASHSAC_TRACKING_UPSTREAM_INTERACTION_STEPS = 48_829
FLASHSAC_TRACKING_NUM_ENVS = 4096
FLASHSAC_TRACKING_TOTAL_ENV_STEPS = (
  FLASHSAC_TRACKING_UPSTREAM_INTERACTION_STEPS * FLASHSAC_TRACKING_NUM_ENVS
)
FLASHSAC_TRACKING_UPDATES_PER_INTERACTION_STEP = 2.0
FLASHSAC_TRACKING_N_STEP = 3
FLASHSAC_TRACKING_BUFFER_MIN_LENGTH = 100_000
FLASHSAC_TRACKING_CHECKPOINT_COUNT = 10
FLASHSAC_TRACKING_DEFAULT_CHECKPOINT_INTERVAL = (
  checkpoint_interval_from_total_env_steps(
    total_env_steps=FLASHSAC_TRACKING_TOTAL_ENV_STEPS,
    num_envs=FLASHSAC_TRACKING_NUM_ENVS,
    checkpoint_count=FLASHSAC_TRACKING_CHECKPOINT_COUNT,
  )
)


def apply_flashsac_tracking_train_overrides(env_cfg: ManagerBasedRlEnvCfg) -> None:
  """Preserve the registered tracking task semantics for FlashSAC training."""
  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    return


def apply_flashsac_tracking_runner_defaults(
  env_cfg: ManagerBasedRlEnvCfg, agent_cfg: "FlashSACRunnerCfg"
) -> None:
  """Formal upstream-equivalent recipe for mjlab tracking with FlashSAC."""
  env_cfg.scene.num_envs = max(env_cfg.scene.num_envs, FLASHSAC_TRACKING_NUM_ENVS)
  agent_cfg.num_env_steps = max(
    agent_cfg.num_env_steps, FLASHSAC_TRACKING_TOTAL_ENV_STEPS
  )
  agent_cfg.updates_per_interaction_step = max(
    agent_cfg.updates_per_interaction_step,
    FLASHSAC_TRACKING_UPDATES_PER_INTERACTION_STEP,
  )
  agent_cfg.n_step = max(agent_cfg.n_step, FLASHSAC_TRACKING_N_STEP)
  agent_cfg.buffer_min_length = max(
    agent_cfg.buffer_min_length, FLASHSAC_TRACKING_BUFFER_MIN_LENGTH
  )
  # Upstream FlashSAC tracking recipes do not force observation normalization and
  # default to symmetric observations unless a task explicitly opts into asymmetry.
  agent_cfg.normalize_observation = False
  agent_cfg.asymmetric_observation = False
  agent_cfg.save_buffer_per_interaction_step = None
  agent_cfg.save_final_replay_buffer = False
  if agent_cfg.save_checkpoint_per_interaction_step is None:
    agent_cfg.save_checkpoint_per_interaction_step = (
      checkpoint_interval_from_total_env_steps(
        total_env_steps=agent_cfg.num_env_steps,
        num_envs=max(env_cfg.scene.num_envs, 1),
        checkpoint_count=FLASHSAC_TRACKING_CHECKPOINT_COUNT,
      )
    )


def maybe_recompute_flashsac_tracking_checkpoint_cadence(
  env_cfg: ManagerBasedRlEnvCfg, agent_cfg: "FlashSACRunnerCfg"
) -> None:
  """Recompute tracking checkpoint cadence after user budget/env-count overrides.

  Tyro overrides mutate the default config after `from_task()` has already
  materialized the formal 200M/4096 tracking cadence. If the user later shrinks
  `num_env_steps` or `num_envs`, the inherited cadence can become far larger than
  the actual interaction budget, eliminating intermediate checkpoints. We only
  recompute when the cadence still matches the untouched formal default.
  """
  if (
    agent_cfg.save_checkpoint_per_interaction_step
    != FLASHSAC_TRACKING_DEFAULT_CHECKPOINT_INTERVAL
  ):
    return
  agent_cfg.save_checkpoint_per_interaction_step = (
    checkpoint_interval_from_total_env_steps(
      total_env_steps=agent_cfg.num_env_steps,
      num_envs=max(env_cfg.scene.num_envs, 1),
      checkpoint_count=FLASHSAC_TRACKING_CHECKPOINT_COUNT,
    )
  )


@dataclass
class FlashSACRunnerCfg:
  seed: int = 42
  num_env_steps: int = 1_000_000
  updates_per_interaction_step: float = 1.0
  logging_per_interaction_step: int | None = None
  save_checkpoint_per_interaction_step: int | None = None
  save_buffer_per_interaction_step: int | None = None
  experiment_name: str = "flashsac"
  run_name: str = ""
  logger: Literal["wandb", "tensorboard"] = "wandb"
  wandb_project: str = "mjlab"
  wandb_entity: str | None = None
  wandb_group: str = "flashsac"
  wandb_tags: tuple[str, ...] = ()
  resume: bool = False
  load_run: str = ".*"
  load_checkpoint: str = "step_.*"
  upload_model: bool = True
  load_replay_buffer: bool = True
  save_final_replay_buffer: bool = False
  normalize_observation: bool = False
  load_observation_normalizer: bool = True
  observation_clip_value: float | None = 10.0
  normalize_reward: bool = True
  normalized_G_max: float = 5.0
  asymmetric_observation: bool = True
  device_type: str = "cuda"
  buffer_max_length: int = 1_000_000
  buffer_min_length: int = 10_000
  buffer_device_type: str = "cuda"
  sample_batch_size: int = 2048
  learning_rate_init: float = 3e-4
  learning_rate_peak: float = 3e-4
  learning_rate_end: float = 1.5e-4
  learning_rate_warmup_rate: float = 1e-6
  learning_rate_decay_rate: float = 1.0
  actor_num_blocks: int = 2
  actor_hidden_dim: int = 128
  actor_bc_alpha: float = 0.0
  actor_noise_zeta_mu: float = 2.0
  actor_noise_zeta_max: int = 16
  actor_update_period: int = 2
  critic_num_blocks: int = 2
  critic_hidden_dim: int = 256
  critic_num_bins: int = 101
  critic_target_update_tau: float = 0.01
  temp_initial_value: float = 0.01
  temp_target_sigma: float = 0.15
  gamma: float = 0.99
  n_step: int = 1
  use_compile: bool = True
  compile_mode: str = "auto"
  use_amp: bool = True
  load_optimizer: bool = True
  load_reward_normalizer: bool = True


@dataclass(frozen=True)
class FlashSACTrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: FlashSACRunnerCfg
  registry_name: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "FlashSACTrainConfig":
    env_cfg = load_env_cfg(task_id)
    has_actor = "actor" in env_cfg.observations
    has_critic = "critic" in env_cfg.observations
    is_tracking_task = "motion" in env_cfg.commands and isinstance(
      env_cfg.commands["motion"], MotionCommandCfg
    )
    agent_cfg = FlashSACRunnerCfg(
      asymmetric_observation=has_actor and has_critic,
      experiment_name=task_id.replace("Mjlab-", "").replace("-", "_").lower()
      + "_flashsac",
    )
    if is_tracking_task:
      apply_flashsac_tracking_train_overrides(env_cfg)
      apply_flashsac_tracking_runner_defaults(env_cfg, agent_cfg)
    return FlashSACTrainConfig(env=env_cfg, agent=agent_cfg)
