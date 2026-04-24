from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Mapping, TypedDict, cast

import tyro
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from mjlab.tasks.registry import load_rl_cfg

Backend = Literal["rsl_rl", "flashsac"]
TaskKey = Literal["tracking", "tracking_acro", "velocity"]

DEFAULT_VELOCITY_TOTAL_ENV_STEPS = 50_036_736
DEFAULT_TRACKING_TOTAL_ENV_STEPS = 399_998_976
DEFAULT_TRACKING_ACRO_TOTAL_ENV_STEPS = 49_938_432
DEFAULT_TRACKING_MOTION_FILE = (
  "artifacts/motion_runs/handstand1_high/pipeline/mjlab/motion.npz"
)
SELECTED_TRAINING_TAGS: dict[str, tuple[str, ...]] = {
  "tracking": (
    "Episode/length_seconds",
    "Episode/length_steps",
    "Episode_Reward/total",
    "Episode_Termination/anchor_pos",
    "Episode_Termination/anchor_ori",
    "Episode_Termination/ee_body_pos",
    "Metrics/motion/error_anchor_pos",
    "Metrics/motion/error_body_pos",
    "Metrics/motion/error_joint_pos",
  ),
  "tracking_acro": (
    "Episode/length_seconds",
    "Episode/length_steps",
    "Episode_Reward/total",
    "Episode_Termination/anchor_pos",
    "Episode_Termination/anchor_ori",
    "Metrics/motion/error_anchor_pos",
    "Metrics/motion/error_body_pos",
    "Metrics/motion/error_joint_pos",
  ),
  "velocity": (
    "Episode/length_seconds",
    "Episode/length_steps",
    "Episode_Reward/total",
    "Episode_Termination/fell_over",
    "Episode_Termination/time_out",
    "Metrics/twist/error_vel_xy",
    "Metrics/twist/error_vel_yaw",
  ),
}


@dataclass(frozen=True)
class TaskSpec:
  key: TaskKey
  task_id: str
  total_env_steps: int
  tracking_motion_file: str | None = None


@dataclass(frozen=True)
class RunCase:
  case_id: str
  task_key: TaskKey
  task_id: str
  backend: Backend
  seed: int
  num_envs: int
  total_env_steps: int
  run_name: str
  experiment_name: str
  rollout_length: int | None = None
  max_iterations: int | None = None
  tracking_motion_file: str | None = None


class ScalarOverviewRow(TypedDict):
  tag: str
  points: int
  last_step: float
  last_value: float
  min_value: float
  max_value: float


@dataclass
class BenchmarkConfig:
  suite_name: str = "flashsac_vs_ppo_serial"
  benchmark_root: str = "logs/benchmark"
  backends: tuple[Backend, ...] = ("rsl_rl", "flashsac")
  seeds: tuple[int, ...] = (42, 43, 44)
  num_envs: tuple[int, ...] = (4096, 1024)
  tasks: tuple[TaskKey, ...] = ("velocity", "tracking", "tracking_acro")
  velocity_total_env_steps: int = DEFAULT_VELOCITY_TOTAL_ENV_STEPS
  tracking_total_env_steps: int = DEFAULT_TRACKING_TOTAL_ENV_STEPS
  tracking_acro_total_env_steps: int = DEFAULT_TRACKING_ACRO_TOTAL_ENV_STEPS
  tracking_motion_file: str = DEFAULT_TRACKING_MOTION_FILE
  flashsac_tracking_task_id: str = "Mjlab-Tracking-Flat-Unitree-G1"
  flashsac_tracking_use_compile: bool = True
  flashsac_tracking_use_amp: bool = True
  flashsac_tracking_logging_per_interaction_step: int = 500
  flashsac_tracking_save_checkpoint_per_interaction_step: int = 5000
  flashsac_tracking_acro_use_compile: bool = True
  flashsac_tracking_acro_use_amp: bool = True
  flashsac_tracking_acro_logging_per_interaction_step: int = 489
  flashsac_tracking_acro_save_checkpoint_per_interaction_step: int = 4883
  flashsac_velocity_use_compile: bool = True
  flashsac_velocity_use_amp: bool = True
  flashsac_velocity_logging_per_interaction_step: int = 49
  flashsac_velocity_save_checkpoint_per_interaction_step: int = 4883
  tracking_eval_num_envs: int = 1024
  cuda_visible_devices: str = "0"
  dry_run: bool = False
  skip_completed: bool = True
  continue_on_error: bool = False


def _utc_now() -> str:
  return datetime.now(tz=timezone.utc).isoformat()


def _duration_seconds(started_at: str | None, finished_at: str | None) -> float | None:
  if not started_at or not finished_at:
    return None
  try:
    start = datetime.fromisoformat(started_at)
    end = datetime.fromisoformat(finished_at)
  except ValueError:
    return None
  return max((end - start).total_seconds(), 0.0)


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[3]


def _sanitize_token(value: str) -> str:
  sanitized = []
  for char in value.lower():
    if char.isalnum():
      sanitized.append(char)
    else:
      sanitized.append("-")
  return "".join(sanitized).strip("-")


def _task_specs(config: BenchmarkConfig, repo_root: Path) -> list[TaskSpec]:
  tracking_motion = str((repo_root / config.tracking_motion_file).resolve())
  return [
    TaskSpec(
      key="tracking",
      task_id="Mjlab-Tracking-Flat-Unitree-G1",
      total_env_steps=config.tracking_total_env_steps,
      tracking_motion_file=tracking_motion,
    ),
    TaskSpec(
      key="tracking_acro",
      task_id="Mjlab-Tracking-Flat-Unitree-G1-Acrobatics",
      total_env_steps=config.tracking_acro_total_env_steps,
      tracking_motion_file=tracking_motion,
    ),
    TaskSpec(
      key="velocity",
      task_id="Mjlab-Velocity-Flat-Unitree-G1",
      total_env_steps=config.velocity_total_env_steps,
    ),
  ]


def _flashsac_experiment_name(task_id: str) -> str:
  return task_id.replace("Mjlab-", "").replace("-", "_").lower() + "_flashsac"


def _rsl_rl_experiment_name(task_id: str) -> str:
  rl_cfg = load_rl_cfg(task_id)
  return rl_cfg.experiment_name


def _rollout_length_for_task(task_id: str) -> int:
  rl_cfg = load_rl_cfg(task_id)
  rollout_length = getattr(rl_cfg, "num_steps_per_env", None)
  if rollout_length is None:
    raise ValueError(f"Task {task_id} does not define num_steps_per_env.")
  return int(rollout_length)


def _derive_max_iterations(task_id: str, total_env_steps: int, num_envs: int) -> int:
  rollout_length = _rollout_length_for_task(task_id)
  per_iteration_env_steps = num_envs * rollout_length
  if total_env_steps % per_iteration_env_steps != 0:
    raise ValueError(
      "Budget is not divisible by PPO iteration size: "
      f"task={task_id}, total_env_steps={total_env_steps}, num_envs={num_envs}, "
      f"rollout_length={rollout_length}"
    )
  return total_env_steps // per_iteration_env_steps


def _resolve_task_id(
  config: BenchmarkConfig, task_key: TaskKey, backend: Backend
) -> str:
  if task_key == "tracking" and backend == "flashsac":
    return config.flashsac_tracking_task_id
  if task_key == "tracking":
    return "Mjlab-Tracking-Flat-Unitree-G1"
  if task_key == "tracking_acro":
    return "Mjlab-Tracking-Flat-Unitree-G1-Acrobatics"
  return "Mjlab-Velocity-Flat-Unitree-G1"


def build_cases(config: BenchmarkConfig) -> list[RunCase]:
  import mjlab.tasks  # noqa: F401

  repo_root = _repo_root()
  selected_specs = {
    spec.key: spec
    for spec in _task_specs(config, repo_root)
    if spec.key in config.tasks
  }
  suite_token = _sanitize_token(config.suite_name)
  cases: list[RunCase] = []
  for task_key in config.tasks:
    spec = selected_specs[task_key]
    for num_envs in config.num_envs:
      for seed in config.seeds:
        for backend in config.backends:
          task_id = _resolve_task_id(config, task_key, backend)
          run_name = f"{suite_token}-{task_key}-{backend}-env{num_envs}-seed{seed}"
          case_id = f"{task_key}__{backend}__env{num_envs}__seed{seed}"
          if backend == "rsl_rl":
            rollout_length = _rollout_length_for_task(task_id)
            max_iterations = _derive_max_iterations(
              task_id, spec.total_env_steps, num_envs
            )
            experiment_name = _rsl_rl_experiment_name(task_id)
          else:
            rollout_length = None
            max_iterations = None
            experiment_name = _flashsac_experiment_name(task_id)
          cases.append(
            RunCase(
              case_id=case_id,
              task_key=task_key,
              task_id=task_id,
              backend=backend,
              seed=seed,
              num_envs=num_envs,
              total_env_steps=spec.total_env_steps,
              run_name=run_name,
              experiment_name=experiment_name,
              rollout_length=rollout_length,
              max_iterations=max_iterations,
              tracking_motion_file=spec.tracking_motion_file,
            )
          )
  return cases


def _backend_log_root(case: RunCase, repo_root: Path) -> Path:
  backend_dir = "rsl_rl" if case.backend == "rsl_rl" else "flashsac"
  return repo_root / "logs" / backend_dir / case.experiment_name


def _build_train_command(case: RunCase, config: BenchmarkConfig) -> list[str]:
  command = [
    "uv",
    "run",
    "train",
    case.task_id,
    "--backend",
    case.backend,
    "--agent.logger",
    "tensorboard",
    "--agent.seed",
    str(case.seed),
    "--env.scene.num-envs",
    str(case.num_envs),
    "--agent.run-name",
    case.run_name,
  ]
  if case.backend == "rsl_rl":
    assert case.max_iterations is not None
    command.extend(["--agent.max-iterations", str(case.max_iterations)])
  else:
    if case.task_key == "velocity":
      use_compile = config.flashsac_velocity_use_compile
      use_amp = config.flashsac_velocity_use_amp
      logging_every = config.flashsac_velocity_logging_per_interaction_step
      checkpoint_every = config.flashsac_velocity_save_checkpoint_per_interaction_step
    elif case.task_key == "tracking_acro":
      use_compile = config.flashsac_tracking_acro_use_compile
      use_amp = config.flashsac_tracking_acro_use_amp
      logging_every = config.flashsac_tracking_acro_logging_per_interaction_step
      checkpoint_every = (
        config.flashsac_tracking_acro_save_checkpoint_per_interaction_step
      )
    else:
      use_compile = config.flashsac_tracking_use_compile
      use_amp = config.flashsac_tracking_use_amp
      logging_every = config.flashsac_tracking_logging_per_interaction_step
      checkpoint_every = config.flashsac_tracking_save_checkpoint_per_interaction_step
    command.extend(
      [
        "--agent.num-env-steps",
        str(case.total_env_steps),
        "--agent.use-compile",
        "True" if use_compile else "False",
        "--agent.use-amp",
        "True" if use_amp else "False",
        "--agent.logging-per-interaction-step",
        str(logging_every),
        "--agent.save-checkpoint-per-interaction-step",
        str(checkpoint_every),
      ]
    )
  if case.tracking_motion_file is not None:
    command.extend(
      ["--env.commands.motion.motion-file", str(case.tracking_motion_file)]
    )
  return command


def _build_tracking_eval_command(
  case: RunCase,
  checkpoint_path: Path,
  config: BenchmarkConfig,
  metrics_path: Path,
) -> list[str]:
  command = [
    "uv",
    "run",
    "evaluate-tracking",
    case.task_id,
    "--backend",
    case.backend,
    "--checkpoint-file",
    str(checkpoint_path),
    "--num-envs",
    str(config.tracking_eval_num_envs),
    "--output-file",
    str(metrics_path),
  ]
  if case.tracking_motion_file is not None:
    command.extend(["--motion-file", str(case.tracking_motion_file)])
  return command


def _find_new_run_dir(
  log_root: Path,
  run_name: str,
  before: set[Path],
) -> Path | None:
  if not log_root.exists():
    return None
  candidates = sorted(
    (
      path
      for path in log_root.iterdir()
      if path.is_dir() and path not in before and path.name.endswith(f"_{run_name}")
    ),
    key=lambda path: path.stat().st_mtime,
  )
  if candidates:
    return candidates[-1]
  fallback = sorted(
    (
      path
      for path in log_root.iterdir()
      if path.is_dir() and path.name.endswith(f"_{run_name}")
    ),
    key=lambda path: path.stat().st_mtime,
  )
  if fallback:
    return fallback[-1]
  return None


def _run_with_live_log(
  command: list[str],
  *,
  cwd: Path,
  env: dict[str, str],
  log_path: Path,
  dry_run: bool,
) -> int:
  log_path.parent.mkdir(parents=True, exist_ok=True)
  with log_path.open("w", encoding="utf-8") as handle:
    handle.write("$ " + " ".join(command) + "\n\n")
    handle.flush()
    if dry_run:
      return 0
    process = subprocess.Popen(
      command,
      cwd=str(cwd),
      env=env,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
      sys.stdout.write(line)
      handle.write(line)
    process.wait()
    return int(process.returncode)


def _copy_file_if_exists(src: Path, dst: Path) -> None:
  if src.exists():
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree_events(run_dir: Path, case_dir: Path) -> None:
  event_files = sorted(run_dir.rglob("events.out.tfevents.*"))
  if not event_files:
    return
  tensorboard_dir = case_dir / "artifacts" / "tensorboard"
  tensorboard_dir.mkdir(parents=True, exist_ok=True)
  for event_file in event_files:
    shutil.copy2(event_file, tensorboard_dir / event_file.name)


def _tensorboard_source_dir(run_dir: Path) -> Path | None:
  event_files = sorted(run_dir.rglob("events.out.tfevents.*"))
  if not event_files:
    return None
  parents = {event_file.parent for event_file in event_files}
  if len(parents) == 1:
    return next(iter(parents))
  return run_dir


def _export_tensorboard_summaries(
  run_dir: Path, case_dir: Path
) -> dict[str, dict[str, float]]:
  source_dir = _tensorboard_source_dir(run_dir)
  if source_dir is None:
    return {}
  accumulator = EventAccumulator(str(source_dir))
  accumulator.Reload()
  scalar_tags = accumulator.Tags().get("scalars", [])
  last_metrics: dict[str, dict[str, float]] = {}
  overview_rows: list[ScalarOverviewRow] = []
  for tag in scalar_tags:
    values = accumulator.Scalars(tag)
    if not values:
      continue
    last = values[-1]
    numeric_values = [float(item.value) for item in values]
    last_metrics[tag] = {
      "step": float(last.step),
      "wall_time": float(last.wall_time),
      "value": float(last.value),
    }
    overview_rows.append(
      ScalarOverviewRow(
        tag=tag,
        points=len(values),
        last_step=float(last.step),
        last_value=float(last.value),
        min_value=min(numeric_values),
        max_value=max(numeric_values),
      )
    )
  metrics_dir = case_dir / "metrics"
  metrics_dir.mkdir(parents=True, exist_ok=True)
  (metrics_dir / "training_scalars_last.json").write_text(
    json.dumps(last_metrics, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  with (metrics_dir / "training_scalars_overview.csv").open(
    "w", encoding="utf-8", newline=""
  ) as handle:
    writer = csv.writer(handle)
    writer.writerow(
      ("tag", "points", "last_step", "last_value", "min_value", "max_value")
    )
    for row in overview_rows:
      writer.writerow(
        (
          row["tag"],
          row["points"],
          row["last_step"],
          row["last_value"],
          row["min_value"],
          row["max_value"],
        )
      )
  return last_metrics


def _latest_rsl_rl_checkpoint(run_dir: Path) -> Path:
  checkpoints = sorted(
    run_dir.glob("model_*.pt"),
    key=lambda path: int(path.stem.split("_")[1]),
  )
  if not checkpoints:
    raise FileNotFoundError(f"No RSL-RL checkpoints found in {run_dir}")
  return checkpoints[-1]


def _latest_flashsac_checkpoint(run_dir: Path) -> Path:
  checkpoints = sorted(
    (path for path in run_dir.glob("step_*") if path.is_dir()),
    key=lambda path: int(path.name.split("_")[1]),
  )
  if not checkpoints:
    raise FileNotFoundError(f"No FlashSAC checkpoints found in {run_dir}")
  return checkpoints[-1]


def _latest_checkpoint(case: RunCase, run_dir: Path) -> Path:
  if case.backend == "rsl_rl":
    return _latest_rsl_rl_checkpoint(run_dir)
  return _latest_flashsac_checkpoint(run_dir)


def _write_case_status(case_dir: Path, payload: dict[str, object]) -> None:
  (case_dir / "status.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True),
    encoding="utf-8",
  )


def _link_training_run(case_dir: Path, run_dir: Path) -> None:
  target = case_dir / "artifacts" / "training_run"
  target.parent.mkdir(parents=True, exist_ok=True)
  if target.exists() or target.is_symlink():
    target.unlink()
  target.symlink_to(run_dir)


def _copy_run_params(run_dir: Path, case_dir: Path) -> None:
  params_dir = run_dir / "params"
  if not params_dir.exists():
    return
  for name in ("agent.yaml", "env.yaml", "runtime.yaml"):
    _copy_file_if_exists(params_dir / name, case_dir / "artifacts" / "params" / name)


def _summary_value(
  scalars_last: Mapping[str, object],
  tag: str,
) -> float | str:
  metric = scalars_last.get(tag)
  if not isinstance(metric, dict):
    return ""
  metric_values = cast(dict[str, object], metric)
  value = metric_values.get("value")
  return value if isinstance(value, (int, float)) else ""


def _load_json(path: Path) -> dict[str, object]:
  return json.loads(path.read_text(encoding="utf-8"))


def _write_case_metrics_file(
  case_dir: Path,
  *,
  case: RunCase,
  started_at: str | None,
  finished_at: str | None,
  selected_training_metrics: dict[str, float],
  tracking_eval: dict[str, object] | None = None,
) -> None:
  duration_seconds = _duration_seconds(started_at, finished_at)
  payload: dict[str, object] = {
    "case_id": case.case_id,
    "task_key": case.task_key,
    "task_id": case.task_id,
    "backend": case.backend,
    "seed": case.seed,
    "num_envs": case.num_envs,
    "total_env_steps": case.total_env_steps,
    "started_at": started_at,
    "finished_at": finished_at,
    "training_duration_seconds": duration_seconds,
    "training_duration_hours": (
      None if duration_seconds is None else duration_seconds / 3600.0
    ),
    "selected_training_metrics": selected_training_metrics,
  }
  if tracking_eval is not None:
    payload["tracking_eval_metrics"] = tracking_eval
  metrics_path = case_dir / "metrics" / "case_metrics.json"
  metrics_path.parent.mkdir(parents=True, exist_ok=True)
  metrics_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True),
    encoding="utf-8",
  )


def _write_suite_readme(
  suite_dir: Path,
  config: BenchmarkConfig,
  cases: list[RunCase],
) -> None:
  lines = [
    "# FlashSAC vs PPO Benchmark Suite",
    "",
    f"- suite_name: `{config.suite_name}`",
    f"- seeds: `{list(config.seeds)}`",
    f"- num_envs: `{list(config.num_envs)}`",
    f"- tracking_total_env_steps: `{config.tracking_total_env_steps}`",
    f"- velocity_total_env_steps: `{config.velocity_total_env_steps}`",
    f"- tracking_motion_file: `{config.tracking_motion_file}`",
    f"- tracking_eval_num_envs: `{config.tracking_eval_num_envs}`",
    "",
    "## Cases",
    "",
    "| case_id | task | backend | num_envs | seed | total_env_steps | max_iterations |",
    "| --- | --- | --- | --- | --- | --- | --- |",
  ]
  for case in cases:
    lines.append(
      "| "
      + " | ".join(
        (
          case.case_id,
          case.task_key,
          case.backend,
          str(case.num_envs),
          str(case.seed),
          str(case.total_env_steps),
          "" if case.max_iterations is None else str(case.max_iterations),
        )
      )
      + " |"
    )
  (suite_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_suite_summary(suite_dir: Path) -> None:
  case_dirs = sorted((suite_dir / "cases").glob("*"))
  rows: list[dict[str, object]] = []
  fieldnames = [
    "case_id",
    "status",
    "task_key",
    "backend",
    "num_envs",
    "seed",
    "total_env_steps",
    "started_at",
    "finished_at",
    "training_duration_seconds",
    "training_duration_hours",
    "train_run_dir",
    "checkpoint_path",
    "tracking_eval_success_rate",
    "tracking_eval_episode_return_mean",
    "tracking_eval_episode_return_std",
    "tracking_eval_episode_length_seconds_mean",
    "tracking_eval_episode_length_steps_mean",
    "tracking_eval_terminated_rate",
    "tracking_eval_time_out_rate",
    "tracking_eval_termination_rate_anchor_pos",
    "tracking_eval_termination_rate_anchor_ori",
    "tracking_eval_termination_rate_ee_body_pos",
    "tracking_eval_mpkpe",
    "tracking_eval_r_mpkpe",
    "tracking_eval_joint_vel_error",
    "tracking_eval_ee_pos_error",
    "tracking_eval_ee_ori_error",
    "train_episode_length_seconds",
    "train_episode_reward_total",
    "train_tracking_anchor_pos",
    "train_tracking_ee_body_pos",
    "train_tracking_error_anchor_pos",
    "train_velocity_fell_over",
    "train_velocity_error_vel_xy",
    "train_velocity_error_vel_yaw",
  ]
  for case_dir in case_dirs:
    status_path = case_dir / "status.json"
    if not status_path.exists():
      continue
    status = _load_json(status_path)
    scalars_last_path = case_dir / "metrics" / "training_scalars_last.json"
    scalars_last = _load_json(scalars_last_path) if scalars_last_path.exists() else {}
    tracking_eval_path = case_dir / "metrics" / "tracking_eval_metrics.json"
    tracking_eval = (
      _load_json(tracking_eval_path) if tracking_eval_path.exists() else {}
    )
    row = {
      "case_id": status.get("case_id", case_dir.name),
      "status": status.get("status", ""),
      "task_key": status.get("task_key", ""),
      "backend": status.get("backend", ""),
      "num_envs": status.get("num_envs", ""),
      "seed": status.get("seed", ""),
      "total_env_steps": status.get("total_env_steps", ""),
      "started_at": status.get("started_at", ""),
      "finished_at": status.get("finished_at", ""),
      "training_duration_seconds": status.get("training_duration_seconds", ""),
      "training_duration_hours": status.get("training_duration_hours", ""),
      "train_run_dir": status.get("train_run_dir", ""),
      "checkpoint_path": status.get("checkpoint_path", ""),
      "tracking_eval_success_rate": tracking_eval.get("success_rate", ""),
      "tracking_eval_episode_return_mean": tracking_eval.get("episode_return_mean", ""),
      "tracking_eval_episode_return_std": tracking_eval.get("episode_return_std", ""),
      "tracking_eval_episode_length_seconds_mean": tracking_eval.get(
        "episode_length_seconds_mean", ""
      ),
      "tracking_eval_episode_length_steps_mean": tracking_eval.get(
        "episode_length_steps_mean", ""
      ),
      "tracking_eval_terminated_rate": tracking_eval.get("terminated_rate", ""),
      "tracking_eval_time_out_rate": tracking_eval.get("time_out_rate", ""),
      "tracking_eval_termination_rate_anchor_pos": tracking_eval.get(
        "termination_rate_anchor_pos", ""
      ),
      "tracking_eval_termination_rate_anchor_ori": tracking_eval.get(
        "termination_rate_anchor_ori", ""
      ),
      "tracking_eval_termination_rate_ee_body_pos": tracking_eval.get(
        "termination_rate_ee_body_pos", ""
      ),
      "tracking_eval_mpkpe": tracking_eval.get("mpkpe", ""),
      "tracking_eval_r_mpkpe": tracking_eval.get("r_mpkpe", ""),
      "tracking_eval_joint_vel_error": tracking_eval.get("joint_vel_error", ""),
      "tracking_eval_ee_pos_error": tracking_eval.get("ee_pos_error", ""),
      "tracking_eval_ee_ori_error": tracking_eval.get("ee_ori_error", ""),
      "train_episode_length_seconds": _summary_value(
        scalars_last, "Episode/length_seconds"
      ),
      "train_episode_reward_total": _summary_value(
        scalars_last, "Episode_Reward/total"
      ),
      "train_tracking_anchor_pos": _summary_value(
        scalars_last, "Episode_Termination/anchor_pos"
      ),
      "train_tracking_ee_body_pos": _summary_value(
        scalars_last, "Episode_Termination/ee_body_pos"
      ),
      "train_tracking_error_anchor_pos": _summary_value(
        scalars_last, "Metrics/motion/error_anchor_pos"
      ),
      "train_velocity_fell_over": _summary_value(
        scalars_last, "Episode_Termination/fell_over"
      ),
      "train_velocity_error_vel_xy": _summary_value(
        scalars_last, "Metrics/twist/error_vel_xy"
      ),
      "train_velocity_error_vel_yaw": _summary_value(
        scalars_last, "Metrics/twist/error_vel_yaw"
      ),
    }
    rows.append(row)
  summary_path = suite_dir / "suite_summary.csv"
  with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def _run_case(case: RunCase, config: BenchmarkConfig, suite_dir: Path) -> None:
  repo_root = _repo_root()
  case_dir = suite_dir / "cases" / case.case_id
  case_dir.mkdir(parents=True, exist_ok=True)
  status_path = case_dir / "status.json"
  if status_path.exists() and config.skip_completed:
    status = _load_json(status_path)
    if status.get("status") == "completed":
      print(f"[SKIP] {case.case_id} already completed.")
      return

  env = os.environ.copy()
  env["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

  train_command = _build_train_command(case, config)
  (case_dir / "train.command.txt").write_text(
    " ".join(train_command) + "\n",
    encoding="utf-8",
  )
  log_root = _backend_log_root(case, repo_root)
  before = set(log_root.iterdir()) if log_root.exists() else set()

  running_status = {
    "case_id": case.case_id,
    "status": "running",
    "started_at": _utc_now(),
    "task_key": case.task_key,
    "task_id": case.task_id,
    "backend": case.backend,
    "seed": case.seed,
    "num_envs": case.num_envs,
    "total_env_steps": case.total_env_steps,
    "run_name": case.run_name,
    "experiment_name": case.experiment_name,
    "train_command": train_command,
  }
  _write_case_status(case_dir, running_status)

  train_exit_code = _run_with_live_log(
    train_command,
    cwd=repo_root,
    env=env,
    log_path=case_dir / "train.log",
    dry_run=config.dry_run,
  )
  (case_dir / "train.exit_code").write_text(f"{train_exit_code}\n", encoding="utf-8")

  run_dir = _find_new_run_dir(log_root, case.run_name, before)
  if run_dir is None and not config.dry_run:
    raise FileNotFoundError(
      f"Unable to locate run directory for case {case.case_id} under {log_root}"
    )

  status_payload = dict(running_status)
  status_payload["train_exit_code"] = train_exit_code
  status_payload["train_run_dir"] = "" if run_dir is None else str(run_dir)

  if train_exit_code != 0:
    status_payload["status"] = "failed"
    status_payload["finished_at"] = _utc_now()
    duration_seconds = _duration_seconds(
      status_payload.get("started_at"), status_payload.get("finished_at")
    )
    status_payload["training_duration_seconds"] = duration_seconds
    status_payload["training_duration_hours"] = (
      None if duration_seconds is None else duration_seconds / 3600.0
    )
    _write_case_status(case_dir, status_payload)
    return

  if config.dry_run:
    status_payload["status"] = "completed"
    status_payload["finished_at"] = _utc_now()
    duration_seconds = _duration_seconds(
      status_payload.get("started_at"), status_payload.get("finished_at")
    )
    status_payload["training_duration_seconds"] = duration_seconds
    status_payload["training_duration_hours"] = (
      None if duration_seconds is None else duration_seconds / 3600.0
    )
    _write_case_status(case_dir, status_payload)
    return

  assert run_dir is not None
  _link_training_run(case_dir, run_dir)
  _copy_run_params(run_dir, case_dir)
  _copy_tree_events(run_dir, case_dir)
  scalars_last = _export_tensorboard_summaries(run_dir, case_dir)

  checkpoint_path = _latest_checkpoint(case, run_dir)
  status_payload["checkpoint_path"] = str(checkpoint_path)

  if case.task_key == "tracking":
    metrics_path = case_dir / "metrics" / "tracking_eval_metrics.json"
    eval_command = _build_tracking_eval_command(
      case,
      checkpoint_path,
      config,
      metrics_path,
    )
    (case_dir / "evaluate.command.txt").write_text(
      " ".join(eval_command) + "\n",
      encoding="utf-8",
    )
    eval_exit_code = _run_with_live_log(
      eval_command,
      cwd=repo_root,
      env=env,
      log_path=case_dir / "evaluate.log",
      dry_run=False,
    )
    (case_dir / "evaluate.exit_code").write_text(
      f"{eval_exit_code}\n", encoding="utf-8"
    )
    status_payload["evaluate_exit_code"] = eval_exit_code
    if eval_exit_code != 0:
      status_payload["status"] = "failed"
      status_payload["finished_at"] = _utc_now()
      duration_seconds = _duration_seconds(
        status_payload.get("started_at"), status_payload.get("finished_at")
      )
      status_payload["training_duration_seconds"] = duration_seconds
      status_payload["training_duration_hours"] = (
        None if duration_seconds is None else duration_seconds / 3600.0
      )
      _write_case_status(case_dir, status_payload)
      return
    if metrics_path.exists():
      status_payload["tracking_eval_metrics"] = _load_json(metrics_path)

  selected_training_metrics = {
    tag: scalars_last[tag]["value"]
    for tag in SELECTED_TRAINING_TAGS[case.task_key]
    if tag in scalars_last
  }
  status_payload["selected_training_metrics"] = selected_training_metrics
  status_payload["status"] = "completed"
  status_payload["finished_at"] = _utc_now()
  duration_seconds = _duration_seconds(
    status_payload.get("started_at"), status_payload.get("finished_at")
  )
  status_payload["training_duration_seconds"] = duration_seconds
  status_payload["training_duration_hours"] = (
    None if duration_seconds is None else duration_seconds / 3600.0
  )
  _write_case_status(case_dir, status_payload)
  _write_case_metrics_file(
    case_dir,
    case=case,
    started_at=status_payload.get("started_at"),
    finished_at=status_payload.get("finished_at"),
    selected_training_metrics=selected_training_metrics,
    tracking_eval=(
      status_payload.get("tracking_eval_metrics")
      if isinstance(status_payload.get("tracking_eval_metrics"), dict)
      else None
    ),
  )


def main() -> None:
  import mjlab.tasks  # noqa: F401

  config = tyro.cli(BenchmarkConfig, config=mjlab.TYRO_FLAGS)
  repo_root = _repo_root()
  suite_dir = (repo_root / config.benchmark_root / config.suite_name).resolve()
  suite_dir.mkdir(parents=True, exist_ok=True)
  cases = build_cases(config)

  (suite_dir / "suite_config.json").write_text(
    json.dumps(asdict(config), indent=2, sort_keys=True),
    encoding="utf-8",
  )
  _write_suite_readme(suite_dir, config, cases)

  failures: list[str] = []
  for case in cases:
    print(
      f"[CASE] {case.case_id} task={case.task_id} backend={case.backend} "
      f"num_envs={case.num_envs} seed={case.seed}"
    )
    try:
      _run_case(case, config, suite_dir)
    except Exception as exc:  # noqa: BLE001
      failures.append(case.case_id)
      case_dir = suite_dir / "cases" / case.case_id
      case_dir.mkdir(parents=True, exist_ok=True)
      _write_case_status(
        case_dir,
        {
          "case_id": case.case_id,
          "status": "failed",
          "task_key": case.task_key,
          "task_id": case.task_id,
          "backend": case.backend,
          "seed": case.seed,
          "num_envs": case.num_envs,
          "total_env_steps": case.total_env_steps,
          "error": repr(exc),
          "finished_at": _utc_now(),
        },
      )
      if not config.continue_on_error:
        _write_suite_summary(suite_dir)
        raise
    finally:
      _write_suite_summary(suite_dir)

  if failures:
    raise SystemExit(f"Benchmark finished with failures: {', '.join(failures)}")


if __name__ == "__main__":
  main()
