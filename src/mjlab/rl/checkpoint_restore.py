from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml
from mujoco import MjSpec

from mjlab.entity import entity as entity_module
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


def _install_yaml_compat_aliases() -> None:
  # Older dumped env.yaml files can contain python/name tags for the nested
  # default lambda used by EntityCfg.spec_fn. PyYAML resolves that tag as the
  # attribute name ``<lambda>``, so we recreate a stable alias before loading.
  if not hasattr(entity_module, "<lambda>"):
    setattr(entity_module, "<lambda>", lambda: MjSpec())


def _restore_like(template: Any, data: Any) -> Any:
  if is_dataclass(template) and not isinstance(template, type) and isinstance(data, dict):
    restored_obj = deepcopy(template)
    for field in fields(restored_obj):
      if field.name in data:
        setattr(
          restored_obj,
          field.name,
          _restore_like(getattr(restored_obj, field.name), data[field.name]),
        )
    return restored_obj
  if isinstance(template, dict) and isinstance(data, dict):
    restored_dict: dict[Any, Any] = {}
    for key, value in data.items():
      if key in template:
        restored_dict[key] = _restore_like(template[key], value)
      else:
        restored_dict[key] = value
    return restored_dict
  if isinstance(template, tuple) and isinstance(data, (list, tuple)):
    if len(template) == len(data):
      return tuple(
        _restore_like(item_template, value)
        for item_template, value in zip(template, data, strict=False)
      )
    return tuple(data)
  if isinstance(template, list) and isinstance(data, list):
    if len(template) == len(data):
      return [
        _restore_like(item_template, value)
        for item_template, value in zip(template, data, strict=False)
      ]
    return list(data)
  return data


def resolve_rsl_rl_run_dir(checkpoint_path: str | Path) -> Path:
  path = Path(checkpoint_path).expanduser().resolve()
  if path.is_dir():
    return path
  if path.is_file():
    return path.parent
  raise FileNotFoundError(f"Checkpoint path not found: {path}")


def load_local_rsl_rl_checkpoint_params(
  checkpoint_path: str | Path,
  *,
  task_id: str,
  play: bool,
) -> tuple[ManagerBasedRlEnvCfg, dict[str, Any], Path] | None:
  run_dir = resolve_rsl_rl_run_dir(checkpoint_path)
  params_dir = run_dir / "params"
  env_yaml = params_dir / "env.yaml"
  agent_yaml = params_dir / "agent.yaml"
  if not env_yaml.exists() or not agent_yaml.exists():
    return None

  import mjlab.tasks  # noqa: F401
  _install_yaml_compat_aliases()

  with env_yaml.open("r", encoding="utf-8") as handle:
    env_data = yaml.unsafe_load(handle)
  with agent_yaml.open("r", encoding="utf-8") as handle:
    agent_data = yaml.full_load(handle) or {}

  if not isinstance(env_data, dict):
    raise TypeError(f"Expected dict in {env_yaml}, got {type(env_data)!r}")
  if not isinstance(agent_data, dict):
    raise TypeError(f"Expected dict in {agent_yaml}, got {type(agent_data)!r}")
  env_cfg = _restore_like(load_env_cfg(task_id, play=play), env_data)
  agent_cfg = _restore_like(asdict(load_rl_cfg(task_id)), agent_data)
  if not isinstance(env_cfg, ManagerBasedRlEnvCfg):
    raise TypeError(
      f"Expected ManagerBasedRlEnvCfg after restore from {env_yaml}, got {type(env_cfg)!r}"
    )
  if not isinstance(agent_cfg, dict):
    raise TypeError(
      f"Expected dict after restore from {agent_yaml}, got {type(agent_cfg)!r}"
    )
  return env_cfg, agent_cfg, run_dir


def load_rsl_rl_runtime_configs(
  task_id: str,
  *,
  checkpoint_file: str | None,
  play: bool,
) -> tuple[ManagerBasedRlEnvCfg, dict[str, Any], bool, Path | None]:
  if checkpoint_file is not None:
    restored = load_local_rsl_rl_checkpoint_params(
      checkpoint_file,
      task_id=task_id,
      play=play,
    )
    if restored is not None:
      env_cfg, agent_cfg, run_dir = restored
      return env_cfg, agent_cfg, True, run_dir

  return load_env_cfg(task_id, play=play), asdict(load_rl_cfg(task_id)), False, None
