import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


def dump_yaml(filename: Path, data: Dict, sort_keys: bool = False) -> None:
  """Saves data to a YAML file.

  Args:
      filename: The path to the YAML file.
      data: The data to save. Must be a dictionary.
      sort_keys: Whether to sort the keys in the YAML file.
  """
  if not filename.suffix:
    filename = filename.with_suffix(".yaml")
  filename.parent.mkdir(parents=True, exist_ok=True)
  with open(filename, "w") as f:
    yaml.dump(data, f, sort_keys=sort_keys)


@dataclass(frozen=True)
class RslRlCheckpointEnvParity:
  run_dir: Path
  env_yaml_path: Path
  sampling_mode: str | None
  motion_file: str | None
  actor_enable_corruption: bool | None
  critic_enable_corruption: bool | None
  startup_event_names: tuple[str, ...]
  push_robot_enabled: bool | None
  episode_length_s: float | None
  num_envs: int | None


def _resolve_run_dir_from_checkpoint_path(checkpoint_path: str | Path) -> Path:
  path = Path(checkpoint_path).expanduser().resolve()
  if not path.exists():
    raise FileNotFoundError(f"Checkpoint path not found: {path}")
  return path if path.is_dir() else path.parent


def _base_yaml_mapping(path: Path) -> Mapping[str, Any]:
  with path.open("r", encoding="utf-8") as handle:
    data = yaml.load(handle, Loader=yaml.BaseLoader) or {}
  if not isinstance(data, Mapping):
    raise TypeError(f"Expected YAML mapping at {path}")
  return data


def _mapping_get(mapping: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
  current: Mapping[str, Any] | Any = mapping
  for key in keys:
    if not isinstance(current, Mapping):
      return {}
    current = current.get(key, {})
  return current if isinstance(current, Mapping) else {}


def _optional_bool(value: Any) -> bool | None:
  if value is None:
    return None
  if isinstance(value, bool):
    return value
  if isinstance(value, str):
    lowered = value.strip().lower()
    if lowered == "true":
      return True
    if lowered == "false":
      return False
  raise TypeError(f"Expected boolean-compatible value, got {value!r}")


def _optional_float(value: Any) -> float | None:
  if value in (None, ""):
    return None
  if isinstance(value, (float, int)):
    return float(value)
  if isinstance(value, str):
    return float(value)
  raise TypeError(f"Expected float-compatible value, got {value!r}")


def _optional_int(value: Any) -> int | None:
  if value in (None, ""):
    return None
  if isinstance(value, int):
    return value
  if isinstance(value, str):
    return int(value)
  raise TypeError(f"Expected int-compatible value, got {value!r}")


def maybe_load_rsl_rl_checkpoint_env_parity(
  checkpoint_path: str | Path,
) -> RslRlCheckpointEnvParity | None:
  run_dir = _resolve_run_dir_from_checkpoint_path(checkpoint_path)
  env_yaml_path = run_dir / "params" / "env.yaml"
  if not env_yaml_path.is_file():
    return None

  data = _base_yaml_mapping(env_yaml_path)
  observations = _mapping_get(data, "observations")
  actor_cfg = _mapping_get(observations, "actor")
  critic_cfg = _mapping_get(observations, "critic")
  motion_cfg = _mapping_get(data, "commands", "motion")
  events = _mapping_get(data, "events")
  startup_event_names = tuple(
    name
    for name, cfg in events.items()
    if isinstance(cfg, Mapping) and cfg.get("mode") == "startup"
  )
  motion_file = motion_cfg.get("motion_file")
  if motion_file == "":
    motion_file = None

  return RslRlCheckpointEnvParity(
    run_dir=run_dir,
    env_yaml_path=env_yaml_path,
    sampling_mode=motion_cfg.get("sampling_mode"),
    motion_file=motion_file if isinstance(motion_file, str) else None,
    actor_enable_corruption=_optional_bool(actor_cfg.get("enable_corruption")),
    critic_enable_corruption=_optional_bool(critic_cfg.get("enable_corruption")),
    startup_event_names=startup_event_names,
    push_robot_enabled="push_robot" in events,
    episode_length_s=_optional_float(data.get("episode_length_s")),
    num_envs=_optional_int(_mapping_get(data, "scene").get("num_envs")),
  )


def apply_rsl_rl_checkpoint_env_parity(
  env_cfg: Any,
  checkpoint_path: str | Path,
) -> RslRlCheckpointEnvParity | None:
  parity = maybe_load_rsl_rl_checkpoint_env_parity(checkpoint_path)
  if parity is None:
    return None

  motion_cfg = getattr(env_cfg, "commands", {}).get("motion")
  if motion_cfg is not None:
    if parity.sampling_mode is not None:
      motion_cfg.sampling_mode = parity.sampling_mode
    if parity.motion_file:
      motion_cfg.motion_file = parity.motion_file

  actor_cfg = getattr(env_cfg, "observations", {}).get("actor")
  if actor_cfg is not None and parity.actor_enable_corruption is not None:
    actor_cfg.enable_corruption = parity.actor_enable_corruption

  critic_cfg = getattr(env_cfg, "observations", {}).get("critic")
  if critic_cfg is not None and parity.critic_enable_corruption is not None:
    critic_cfg.enable_corruption = parity.critic_enable_corruption

  if parity.startup_event_names:
    startup_names = set(parity.startup_event_names)
    env_cfg.events = {
      name: cfg
      for name, cfg in env_cfg.events.items()
      if getattr(cfg, "mode", None) != "startup" or name in startup_names
    }
  else:
    env_cfg.events = {
      name: cfg
      for name, cfg in env_cfg.events.items()
      if getattr(cfg, "mode", None) != "startup"
    }

  if parity.push_robot_enabled is False:
    env_cfg.events.pop("push_robot", None)

  if parity.episode_length_s is not None:
    env_cfg.episode_length_s = parity.episode_length_s

  return parity


def get_checkpoint_path(
  log_path: Path,
  run_dir: str = ".*",
  checkpoint: str = ".*",
  sort_alpha: bool = True,
) -> Path:
  """Get path to model checkpoint in input directory.

  The checkpoint file is resolved as: `<log_path>/<run_dir>/<checkpoint>`.

  If `run_dir` and `checkpoint` are regex expressions, then the most recent
  (highest alphabetical order) run and checkpoint are selected. To disable this
  behavior, set `sort_alpha` to `False`.
  """
  if not log_path.exists():
    raise ValueError(f"Log path does not exist: {log_path}")
  # Exclude wandb_checkpoints directory which is used for caching downloaded checkpoints.
  runs = [
    log_path / run.name
    for run in log_path.iterdir()
    if run.is_dir() and run.name != "wandb_checkpoints" and re.match(run_dir, run.name)
  ]
  if len(runs) == 0:
    raise ValueError(f"No run directories found in {log_path} matching '{run_dir}'")
  if sort_alpha:
    runs.sort()
  else:
    runs = sorted(runs, key=lambda p: p.stat().st_mtime)
  run_path = runs[-1]

  model_checkpoints = [
    f.name for f in run_path.iterdir() if re.match(checkpoint, f.name)
  ]
  if len(model_checkpoints) == 0:
    raise ValueError(f"No checkpoint found in {run_path} matching {checkpoint}")
  model_checkpoints.sort(key=lambda m: f"{m:0>15}")
  checkpoint_file = model_checkpoints[-1]
  return run_path / checkpoint_file


def get_wandb_checkpoint_path(
  log_path: Path, run_path: Path, checkpoint_name: str | None = None
) -> tuple[Path, bool]:
  """Get checkpoint path from wandb, downloading if needed.

  Returns:
    Tuple of (checkpoint_path, was_cached)
  """
  import wandb

  # Extract run_id from path (e.g., "entity/project/run_id" -> "run_id").
  run_id = str(run_path).split("/")[-1]
  download_dir = log_path / "wandb_checkpoints" / run_id

  # Query wandb API to find the latest checkpoint.
  api = wandb.Api()
  wandb_run = api.run(str(run_path))
  files = [
    file.name for file in wandb_run.files() if re.match(r"^model_\d+\.pt$", file.name)
  ]
  if checkpoint_name is None:
    checkpoint_file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
  else:
    if checkpoint_name not in files:
      raise ValueError(
        f"Checkpoint '{checkpoint_name}' not found in run {run_path}."
        f" Available: {files}"
      )
    checkpoint_file = checkpoint_name

  checkpoint_path = download_dir / checkpoint_file

  # If this checkpoint is not cached locally, download it.
  was_cached = checkpoint_path.exists()
  if not was_cached:
    download_dir.mkdir(parents=True, exist_ok=True)
    wandb_file = wandb_run.file(str(checkpoint_file))
    wandb_file.download(str(download_dir), replace=True)

  return checkpoint_path, was_cached
