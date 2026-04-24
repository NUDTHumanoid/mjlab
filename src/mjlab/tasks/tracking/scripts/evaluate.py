"""Evaluate a trained tracking policy and compute metrics."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.flashsac import load_flashsac_policy
from mjlab.flashsac.runtime import (
  apply_flashsac_checkpoint_env_parity,
  apply_tracking_evaluation_overrides,
  render_flashsac_checkpoint_env_parity_audit,
  resolve_tracking_motion_file,
)
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.mdp.metrics import (
  compute_ee_orientation_error,
  compute_ee_position_error,
  compute_joint_velocity_error,
  compute_mpkpe,
  compute_root_relative_mpkpe,
)
from mjlab.utils.os import (
  apply_rsl_rl_checkpoint_env_parity,
  get_wandb_checkpoint_path,
)
from mjlab.utils.torch import configure_torch_backends


def _resolve_ee_body_names(env_cfg) -> tuple[str, ...]:
  ee_term = env_cfg.terminations.get("ee_body_pos")
  if ee_term is None:
    return ()
  return tuple(ee_term.params["body_names"])


def _reduce_metric_traces(
  traces: list[list[torch.Tensor]],
  active_masks: list[torch.Tensor],
) -> list[torch.Tensor]:
  active_steps = torch.stack(active_masks, dim=0).sum(dim=0).float().clamp(min=1)
  return [torch.stack(trace, dim=0).sum(dim=0) / active_steps for trace in traces]


def _reduce_ee_metric_traces(
  ee_pos_traces: list[torch.Tensor],
  ee_ori_traces: list[torch.Tensor],
  active_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
  active_steps = torch.stack(active_masks, dim=0).sum(dim=0).float().clamp(min=1)
  ee_pos_mean = torch.stack(ee_pos_traces, dim=0).sum(dim=0) / active_steps
  ee_ori_mean = torch.stack(ee_ori_traces, dim=0).sum(dim=0) / active_steps
  return ee_pos_mean, ee_ori_mean


def _termination_term_names(env_cfg) -> tuple[str, ...]:
  return tuple(env_cfg.terminations.keys())


def _summarize_episode_metrics(
  *,
  episode_returns: torch.Tensor,
  episode_lengths: torch.Tensor,
  step_dt: float,
  success: torch.Tensor,
  terminated_rate: float,
  time_out_rate: float,
  termination_counts: dict[str, int],
) -> dict[str, float]:
  episode_lengths_f = episode_lengths.to(dtype=torch.float32)
  episode_seconds = episode_lengths_f * float(step_dt)
  metrics: dict[str, float] = {
    "success_rate": success.float().mean().item(),
    "episode_return_mean": episode_returns.mean().item(),
    "episode_return_std": episode_returns.std(unbiased=False).item(),
    "episode_return_min": episode_returns.min().item(),
    "episode_return_max": episode_returns.max().item(),
    "episode_length_steps_mean": episode_lengths_f.mean().item(),
    "episode_length_steps_std": episode_lengths_f.std(unbiased=False).item(),
    "episode_length_seconds_mean": episode_seconds.mean().item(),
    "episode_length_seconds_std": episode_seconds.std(unbiased=False).item(),
    "terminated_rate": terminated_rate,
    "time_out_rate": time_out_rate,
  }
  num_episodes = max(int(episode_returns.numel()), 1)
  for term_name, count in termination_counts.items():
    metrics[f"termination_rate_{term_name}"] = float(count) / float(num_episodes)
  return metrics


@dataclass(frozen=True)
class EvaluateConfig:
  """Configuration for policy evaluation."""

  backend: str = "rsl_rl"
  wandb_run_path: str | None = None
  """W&B run path in format 'entity/project/run_id'."""
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int = 1024
  """Number of parallel environments (= number of episodes to evaluate)."""
  device: str | None = None
  """Device to run on. Defaults to CUDA if available."""
  output_file: str | None = None
  """Optional path to save metrics as JSON."""


def run_evaluate(task_id: str, cfg: EvaluateConfig) -> dict[str, float]:
  """Run policy evaluation and compute metrics."""
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  if cfg.backend == "flashsac":
    return _run_flashsac_evaluate(task_id, cfg, device)
  return _run_rsl_rl_evaluate(task_id, cfg, device)


def _resolve_rsl_rl_checkpoint_path(experiment_name: str, cfg: EvaluateConfig) -> Path:
  if cfg.checkpoint_file is not None:
    checkpoint_path = Path(cfg.checkpoint_file)
    if not checkpoint_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    return checkpoint_path
  if cfg.wandb_run_path is None:
    raise ValueError(
      "RSL-RL evaluation requires `wandb_run_path` when `checkpoint_file` is not "
      "provided."
    )
  log_root_path = (Path("logs") / "rsl_rl" / experiment_name).resolve()
  resume_path, _ = get_wandb_checkpoint_path(
    log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
  )
  return resume_path


def _resolve_rsl_rl_motion_file(
  motion_cfg: MotionCommandCfg, cfg: EvaluateConfig
) -> None:
  if cfg.motion_file is not None:
    motion_cfg.motion_file = cfg.motion_file
    return
  if motion_cfg.motion_file:
    return
  if cfg.wandb_run_path is None:
    raise ValueError(
      "Tracking evaluation requires `motion_file` when using `checkpoint_file`, "
      "or provide `wandb_run_path` so the motion artifact can be resolved."
    )
  resolve_tracking_motion_file(
    motion_cfg,
    motion_file=None,
    registry_name=None,
    wandb_run_path=cfg.wandb_run_path,
    checkpoint_file=None,
  )


def _run_rsl_rl_evaluate(
  task_id: str, cfg: EvaluateConfig, device: str
) -> dict[str, float]:
  # Load configs.
  env_cfg = load_env_cfg(task_id, play=False)
  agent_cfg = load_rl_cfg(task_id)
  resume_path = _resolve_rsl_rl_checkpoint_path(agent_cfg.experiment_name, cfg)

  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise ValueError(f"Task {task_id} is not a tracking task.")

  checkpoint_parity = apply_rsl_rl_checkpoint_env_parity(env_cfg, resume_path)
  _resolve_rsl_rl_motion_file(motion_cmd, cfg)

  # Evaluation config.
  if checkpoint_parity is None:
    motion_cmd.sampling_mode = "start"
    env_cfg.observations["actor"].enable_corruption = True
    env_cfg.events.pop("push_robot", None)
  env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  print(f"[INFO] Loading checkpoint: {resume_path}")
  if checkpoint_parity is not None:
    print(
      "[INFO] Applied checkpoint env parity from "
      f"{checkpoint_parity.env_yaml_path}: "
      f"sampling_mode={checkpoint_parity.sampling_mode}, "
      f"actor_corruption={checkpoint_parity.actor_enable_corruption}, "
      f"critic_corruption={checkpoint_parity.critic_enable_corruption}, "
      f"episode_length_s={checkpoint_parity.episode_length_s}, "
      f"startup_events={checkpoint_parity.startup_event_names}, "
      f"saved_motion_file={checkpoint_parity.motion_file}, "
      f"saved_num_envs={checkpoint_parity.num_envs}"
    )

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(resume_path), map_location=device)
  policy = runner.get_inference_policy(device=device)

  command = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))
  ee_body_names = _resolve_ee_body_names(env_cfg)
  print(f"[INFO] End effector bodies: {ee_body_names}")

  # Metric accumulators.
  all_mpkpe: list[torch.Tensor] = []
  all_r_mpkpe: list[torch.Tensor] = []
  all_joint_vel_error: list[torch.Tensor] = []
  all_ee_pos_error: list[torch.Tensor] = []
  all_ee_ori_error: list[torch.Tensor] = []
  active_masks: list[torch.Tensor] = []
  termination_counts = {term_name: 0 for term_name in _termination_term_names(env_cfg)}
  episode_returns = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)
  episode_lengths = torch.zeros(cfg.num_envs, dtype=torch.int64, device=device)
  completed_returns = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)
  completed_lengths = torch.zeros(cfg.num_envs, dtype=torch.int64, device=device)
  terminated_episode_count = 0
  time_out_episode_count = 0

  done_envs = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
  success = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)

  obs = env.get_observations()
  env.unwrapped.command_manager.compute(dt=env.unwrapped.step_dt)

  print(f"[INFO] Running {cfg.num_envs} evaluation episodes...")

  step = 0
  while not done_envs.all():
    with torch.no_grad():
      actions = policy(obs)
    obs, rewards, dones, _ = env.step(actions)

    # Compute metrics for active envs.
    active = ~done_envs
    if active.any():
      episode_returns[active] += rewards[active]
      episode_lengths[active] += 1
      active_masks.append(active.float())
      all_mpkpe.append(torch.where(active, compute_mpkpe(command), 0.0))
      all_r_mpkpe.append(torch.where(active, compute_root_relative_mpkpe(command), 0.0))
      all_joint_vel_error.append(
        torch.where(active, compute_joint_velocity_error(command), 0.0)
      )
      if ee_body_names:
        all_ee_pos_error.append(
          torch.where(active, compute_ee_position_error(command, ee_body_names), 0.0)
        )
        all_ee_ori_error.append(
          torch.where(active, compute_ee_orientation_error(command, ee_body_names), 0.0)
        )

    # Track completions.
    terminated = env.unwrapped.termination_manager.terminated
    truncated = env.unwrapped.termination_manager.time_outs
    newly_done = dones.bool() & ~done_envs

    if newly_done.any():
      success = success | (newly_done & truncated & ~terminated)
      completed_returns[newly_done] = episode_returns[newly_done]
      completed_lengths[newly_done] = episode_lengths[newly_done]
      terminated_episode_count += int((newly_done & terminated).sum().item())
      time_out_episode_count += int((newly_done & truncated).sum().item())
      for term_name in termination_counts:
        termination_counts[term_name] += int(
          (newly_done & env.unwrapped.termination_manager.get_term(term_name))
          .sum()
          .item()
        )
      done_envs = done_envs | newly_done
      print(
        f"[INFO] {done_envs.sum().item()}/{cfg.num_envs} episodes completed "
        f"(step {step}, truncated={(newly_done & truncated).sum().item()}, "
        f"terminated={(newly_done & terminated).sum().item()})"
      )
    step += 1

  # Compute mean metrics.
  means = _reduce_metric_traces(
    [all_mpkpe, all_r_mpkpe, all_joint_vel_error],
    active_masks,
  )

  metrics = {
    "mpkpe": means[0].mean().item(),
    "r_mpkpe": means[1].mean().item(),
    "joint_vel_error": means[2].mean().item(),
    "ee_pos_error": float("nan"),
    "ee_ori_error": float("nan"),
  }
  metrics.update(
    _summarize_episode_metrics(
      episode_returns=completed_returns,
      episode_lengths=completed_lengths,
      step_dt=env.unwrapped.step_dt,
      success=success,
      terminated_rate=float(terminated_episode_count) / float(cfg.num_envs),
      time_out_rate=float(time_out_episode_count) / float(cfg.num_envs),
      termination_counts=termination_counts,
    )
  )
  if ee_body_names and all_ee_pos_error and all_ee_ori_error:
    ee_pos_mean, ee_ori_mean = _reduce_ee_metric_traces(
      all_ee_pos_error,
      all_ee_ori_error,
      active_masks,
    )
    metrics["ee_pos_error"] = ee_pos_mean.mean().item()
    metrics["ee_ori_error"] = ee_ori_mean.mean().item()

  print("\n" + "=" * 50)
  print("Evaluation Results")
  print("=" * 50)
  for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")
  print("=" * 50)

  if cfg.output_file:
    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
      json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to {output_path}")

  env.close()
  return metrics


def _run_flashsac_evaluate(
  task_id: str, cfg: EvaluateConfig, device: str
) -> dict[str, float]:
  if cfg.checkpoint_file is None:
    raise ValueError("FlashSAC evaluation requires `checkpoint_file`.")

  env_cfg = load_env_cfg(task_id, play=False)
  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise ValueError(f"Task {task_id} is not a tracking task.")
  checkpoint_parity = apply_flashsac_checkpoint_env_parity(
    env_cfg,
    cfg.checkpoint_file,
  )
  ee_body_names = _resolve_ee_body_names(env_cfg)

  if checkpoint_parity is None:
    apply_tracking_evaluation_overrides(env_cfg)
  resolve_tracking_motion_file(
    motion_cmd,
    motion_file=cfg.motion_file,
    registry_name=None,
    wandb_run_path=cfg.wandb_run_path,
    checkpoint_file=cfg.checkpoint_file,
  )
  env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  policy = load_flashsac_policy(
    env=env, checkpoint_path=cfg.checkpoint_file, device=device
  )

  command = cast(MotionCommand, env.command_manager.get_term("motion"))
  print(f"[INFO] Loading FlashSAC checkpoint: {cfg.checkpoint_file}")
  if checkpoint_parity is not None:
    print(render_flashsac_checkpoint_env_parity_audit(checkpoint_parity))
  print(f"[INFO] End effector bodies: {ee_body_names}")

  all_mpkpe: list[torch.Tensor] = []
  all_r_mpkpe: list[torch.Tensor] = []
  all_joint_vel_error: list[torch.Tensor] = []
  all_ee_pos_error: list[torch.Tensor] = []
  all_ee_ori_error: list[torch.Tensor] = []
  active_masks: list[torch.Tensor] = []
  termination_counts = {term_name: 0 for term_name in _termination_term_names(env_cfg)}
  episode_returns = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)
  episode_lengths = torch.zeros(cfg.num_envs, dtype=torch.int64, device=device)
  completed_returns = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)
  completed_lengths = torch.zeros(cfg.num_envs, dtype=torch.int64, device=device)
  terminated_episode_count = 0
  time_out_episode_count = 0

  done_envs = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
  success = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)

  obs, _ = env.reset()
  env.command_manager.compute(dt=env.step_dt)

  print(f"[INFO] Running {cfg.num_envs} evaluation episodes...")

  step = 0
  while not done_envs.all():
    with torch.no_grad():
      actions = policy(obs)
    obs, rewards, terminated, truncated, _ = env.step(actions)
    dones = terminated | truncated

    active = ~done_envs
    if active.any():
      episode_returns[active] += rewards[active]
      episode_lengths[active] += 1
      active_masks.append(active.float())
      all_mpkpe.append(torch.where(active, compute_mpkpe(command), 0.0))
      all_r_mpkpe.append(torch.where(active, compute_root_relative_mpkpe(command), 0.0))
      all_joint_vel_error.append(
        torch.where(active, compute_joint_velocity_error(command), 0.0)
      )
      if ee_body_names:
        all_ee_pos_error.append(
          torch.where(active, compute_ee_position_error(command, ee_body_names), 0.0)
        )
        all_ee_ori_error.append(
          torch.where(active, compute_ee_orientation_error(command, ee_body_names), 0.0)
        )

    newly_done = dones.bool() & ~done_envs
    if newly_done.any():
      success = success | (newly_done & truncated & ~terminated)
      completed_returns[newly_done] = episode_returns[newly_done]
      completed_lengths[newly_done] = episode_lengths[newly_done]
      terminated_episode_count += int((newly_done & terminated).sum().item())
      time_out_episode_count += int((newly_done & truncated).sum().item())
      for term_name in termination_counts:
        termination_counts[term_name] += int(
          (newly_done & env.termination_manager.get_term(term_name)).sum().item()
        )
      done_envs = done_envs | newly_done
      print(
        f"[INFO] {done_envs.sum().item()}/{cfg.num_envs} episodes completed "
        f"(step {step}, truncated={(newly_done & truncated).sum().item()}, "
        f"terminated={(newly_done & terminated).sum().item()})"
      )
    step += 1

  means = _reduce_metric_traces(
    [all_mpkpe, all_r_mpkpe, all_joint_vel_error],
    active_masks,
  )
  metrics = {
    "mpkpe": means[0].mean().item(),
    "r_mpkpe": means[1].mean().item(),
    "joint_vel_error": means[2].mean().item(),
    "ee_pos_error": float("nan"),
    "ee_ori_error": float("nan"),
  }
  metrics.update(
    _summarize_episode_metrics(
      episode_returns=completed_returns,
      episode_lengths=completed_lengths,
      step_dt=env.step_dt,
      success=success,
      terminated_rate=float(terminated_episode_count) / float(cfg.num_envs),
      time_out_rate=float(time_out_episode_count) / float(cfg.num_envs),
      termination_counts=termination_counts,
    )
  )
  if ee_body_names and all_ee_pos_error and all_ee_ori_error:
    ee_pos_mean, ee_ori_mean = _reduce_ee_metric_traces(
      all_ee_pos_error,
      all_ee_ori_error,
      active_masks,
    )
    metrics["ee_pos_error"] = ee_pos_mean.mean().item()
    metrics["ee_ori_error"] = ee_ori_mean.mean().item()

  print("\n" + "=" * 50)
  print("Evaluation Results")
  print("=" * 50)
  for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")
  print("=" * 50)

  if cfg.output_file:
    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
      json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to {output_path}")

  env.close()
  return metrics


def main():
  import mjlab.tasks  # noqa: F401

  tracking_tasks = [t for t in list_tasks() if "Tracking" in t]
  if not tracking_tasks:
    print("No tracking tasks found.")
    sys.exit(1)

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(tracking_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    EvaluateConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )

  run_evaluate(chosen_task, args)


if __name__ == "__main__":
  main()
