from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.flashsac.adapter import MjlabFlashSACEnvAdapter
from mjlab.flashsac.agent import FlashSACAgent
from mjlab.flashsac.config import FlashSACRunnerCfg
from mjlab.flashsac.config import (
  apply_flashsac_tracking_train_overrides as apply_flashsac_tracking_train_overrides,
)
from mjlab.tasks.tracking.mdp import MotionCommandCfg


@dataclass(frozen=True)
class FlashSACCheckpointEnvParity:
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
  anchor_pos_threshold: float | None
  anchor_ori_threshold: float | None
  has_ee_body_pos: bool
  ee_body_pos_threshold: float | None


@dataclass(frozen=True)
class FlashSACCheckpointEnvParityAudit:
  parity: FlashSACCheckpointEnvParity
  restored_fields: tuple[str, ...]
  skipped_fields: tuple[str, ...]
  num_envs_source: str


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


def _normalize_device_type(device: str | torch.device) -> str:
  if isinstance(device, torch.device):
    if device.type != "cuda":
      return "cpu"
    return f"cuda:{device.index}" if device.index is not None else "cuda"
  value = str(device)
  return value if value.startswith("cuda") else "cpu"


def resolve_flashsac_checkpoint_dir(checkpoint_path: str | Path) -> Path:
  path = Path(checkpoint_path).expanduser().resolve()
  if not path.exists():
    raise FileNotFoundError(f"FlashSAC checkpoint path not found: {path}")
  if path.is_dir():
    return path
  if path.is_file():
    return path.parent
  raise ValueError(f"Unsupported FlashSAC checkpoint path: {path}")


def make_flashsac_inference_cfg(
  cfg: FlashSACRunnerCfg,
  device: str | torch.device,
) -> FlashSACRunnerCfg:
  device_type = _normalize_device_type(device)
  return replace(
    cfg,
    device_type=device_type,
    buffer_device_type=device_type,
    buffer_max_length=1,
    buffer_min_length=1,
    sample_batch_size=1,
    use_compile=False,
    use_amp=device_type == "cuda" and cfg.use_amp,
    load_optimizer=False,
    load_replay_buffer=False,
    save_final_replay_buffer=False,
    load_reward_normalizer=cfg.normalize_reward,
  )


def load_flashsac_runner_cfg(
  checkpoint_path: str | Path,
  device: str | torch.device,
) -> FlashSACRunnerCfg:
  return make_flashsac_inference_cfg(
    load_flashsac_saved_runner_cfg(checkpoint_path),
    device=device,
  )


def load_flashsac_saved_runner_cfg(
  checkpoint_path: str | Path,
) -> FlashSACRunnerCfg:
  checkpoint_dir = resolve_flashsac_checkpoint_dir(checkpoint_path)
  config_path = checkpoint_dir.parent / "params" / "agent.yaml"
  if not config_path.exists():
    raise FileNotFoundError(
      f"FlashSAC agent config not found next to checkpoint: {config_path}"
    )
  with config_path.open("r", encoding="utf-8") as fh:
    data = yaml.full_load(fh) or {}
  return FlashSACRunnerCfg(**data)


def maybe_load_flashsac_checkpoint_env_parity(
  checkpoint_path: str | Path,
) -> FlashSACCheckpointEnvParity | None:
  checkpoint_dir = resolve_flashsac_checkpoint_dir(checkpoint_path)
  run_dir = checkpoint_dir.parent
  env_yaml_path = run_dir / "params" / "env.yaml"
  if not env_yaml_path.is_file():
    return None

  data = _base_yaml_mapping(env_yaml_path)
  observations = _mapping_get(data, "observations")
  actor_cfg = _mapping_get(observations, "actor")
  critic_cfg = _mapping_get(observations, "critic")
  motion_cfg = _mapping_get(data, "commands", "motion")
  events = _mapping_get(data, "events")
  terminations = _mapping_get(data, "terminations")
  anchor_pos_cfg = _mapping_get(terminations, "anchor_pos", "params")
  anchor_ori_cfg = _mapping_get(terminations, "anchor_ori", "params")
  ee_body_cfg = _mapping_get(terminations, "ee_body_pos", "params")
  startup_event_names = tuple(
    name
    for name, cfg in events.items()
    if isinstance(cfg, Mapping) and cfg.get("mode") == "startup"
  )
  motion_file = motion_cfg.get("motion_file")
  if motion_file == "":
    motion_file = None

  return FlashSACCheckpointEnvParity(
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
    anchor_pos_threshold=_optional_float(anchor_pos_cfg.get("threshold")),
    anchor_ori_threshold=_optional_float(anchor_ori_cfg.get("threshold")),
    has_ee_body_pos="ee_body_pos" in terminations,
    ee_body_pos_threshold=_optional_float(ee_body_cfg.get("threshold")),
  )


def apply_flashsac_checkpoint_env_parity(
  env_cfg: ManagerBasedRlEnvCfg,
  checkpoint_path: str | Path,
  *,
  restore_num_envs: bool = False,
) -> FlashSACCheckpointEnvParityAudit | None:
  parity = maybe_load_flashsac_checkpoint_env_parity(checkpoint_path)
  if parity is None:
    return None

  restored_fields: list[str] = []
  skipped_fields: list[str] = []

  motion_cfg = env_cfg.commands.get("motion")
  if isinstance(motion_cfg, MotionCommandCfg):
    if parity.sampling_mode is not None:
      motion_cfg.sampling_mode = parity.sampling_mode
      restored_fields.append("sampling_mode")
    else:
      skipped_fields.append("sampling_mode")
    if parity.motion_file:
      motion_cfg.motion_file = parity.motion_file
      restored_fields.append("motion_file")
    else:
      skipped_fields.append("motion_file")
  else:
    skipped_fields.extend(("sampling_mode", "motion_file"))

  actor_cfg = env_cfg.observations.get("actor")
  if actor_cfg is not None and parity.actor_enable_corruption is not None:
    actor_cfg.enable_corruption = parity.actor_enable_corruption
    restored_fields.append("actor_enable_corruption")
  else:
    skipped_fields.append("actor_enable_corruption")

  critic_cfg = env_cfg.observations.get("critic")
  if critic_cfg is not None and parity.critic_enable_corruption is not None:
    critic_cfg.enable_corruption = parity.critic_enable_corruption
    restored_fields.append("critic_enable_corruption")
  else:
    skipped_fields.append("critic_enable_corruption")

  if parity.startup_event_names:
    startup_names = set(parity.startup_event_names)
    env_cfg.events = {
      name: cfg
      for name, cfg in env_cfg.events.items()
      if getattr(cfg, "mode", None) != "startup" or name in startup_names
    }
    restored_fields.append("startup_event_names")
  else:
    env_cfg.events = {
      name: cfg
      for name, cfg in env_cfg.events.items()
      if getattr(cfg, "mode", None) != "startup"
    }
    restored_fields.append("startup_event_names")

  if parity.push_robot_enabled is False:
    env_cfg.events.pop("push_robot", None)
    restored_fields.append("push_robot_enabled")
  elif parity.push_robot_enabled is True:
    skipped_fields.append("push_robot_enabled")
  else:
    skipped_fields.append("push_robot_enabled")

  if parity.episode_length_s is not None:
    env_cfg.episode_length_s = parity.episode_length_s
    restored_fields.append("episode_length_s")
  else:
    skipped_fields.append("episode_length_s")

  if restore_num_envs and parity.num_envs is not None:
    env_cfg.scene.num_envs = parity.num_envs
    restored_fields.append("num_envs")
    num_envs_source = "checkpoint"
  else:
    skipped_fields.append("num_envs")
    num_envs_source = "audit-only"

  if "anchor_pos" in env_cfg.terminations and parity.anchor_pos_threshold is not None:
    env_cfg.terminations["anchor_pos"].params["threshold"] = parity.anchor_pos_threshold
    restored_fields.append("anchor_pos_threshold")
  else:
    skipped_fields.append("anchor_pos_threshold")

  if "anchor_ori" in env_cfg.terminations and parity.anchor_ori_threshold is not None:
    env_cfg.terminations["anchor_ori"].params["threshold"] = parity.anchor_ori_threshold
    restored_fields.append("anchor_ori_threshold")
  else:
    skipped_fields.append("anchor_ori_threshold")

  if parity.has_ee_body_pos:
    if (
      "ee_body_pos" in env_cfg.terminations and parity.ee_body_pos_threshold is not None
    ):
      env_cfg.terminations["ee_body_pos"].params["threshold"] = (
        parity.ee_body_pos_threshold
      )
      restored_fields.append("ee_body_pos")
    else:
      skipped_fields.append("ee_body_pos")
  else:
    env_cfg.terminations.pop("ee_body_pos", None)
    restored_fields.append("ee_body_pos")

  return FlashSACCheckpointEnvParityAudit(
    parity=parity,
    restored_fields=tuple(restored_fields),
    skipped_fields=tuple(skipped_fields),
    num_envs_source=num_envs_source,
  )


def render_flashsac_checkpoint_env_parity_audit(
  audit: FlashSACCheckpointEnvParityAudit,
) -> str:
  parity = audit.parity
  return (
    "[INFO] Applied FlashSAC checkpoint env parity from "
    f"{parity.env_yaml_path}: restored={audit.restored_fields}, "
    f"skipped={audit.skipped_fields}, "
    f"sampling_mode={parity.sampling_mode}, "
    f"actor_corruption={parity.actor_enable_corruption}, "
    f"critic_corruption={parity.critic_enable_corruption}, "
    f"episode_length_s={parity.episode_length_s}, "
    f"saved_motion_file={parity.motion_file}, "
    f"saved_num_envs={parity.num_envs}, "
    f"num_envs_source={audit.num_envs_source}, "
    f"anchor_pos={parity.anchor_pos_threshold}, "
    f"anchor_ori={parity.anchor_ori_threshold}, "
    f"has_ee_body_pos={parity.has_ee_body_pos}, "
    f"ee_body_pos_threshold={parity.ee_body_pos_threshold}"
  )


def apply_tracking_evaluation_overrides(env_cfg: ManagerBasedRlEnvCfg) -> None:
  """Make tracking evaluation/play deterministic without relaxing task success rules."""
  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    return
  motion_cmd.sampling_mode = "start"
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.joint_position_range = (0.0, 0.0)
  env_cfg.events = {
    name: cfg for name, cfg in env_cfg.events.items() if cfg.mode != "startup"
  }
  for obs_group in env_cfg.observations.values():
    obs_group.enable_corruption = False
  env_cfg.events.pop("push_robot", None)


def apply_flashsac_tracking_inference_overrides(env_cfg: ManagerBasedRlEnvCfg) -> None:
  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    return
  apply_tracking_evaluation_overrides(env_cfg)


def resolve_tracking_motion_file(
  motion_cfg: MotionCommandCfg,
  *,
  motion_file: str | None,
  registry_name: str | None,
  wandb_run_path: str | None,
  checkpoint_file: str | None = None,
) -> None:
  if motion_file is not None and Path(motion_file).exists():
    motion_cfg.motion_file = motion_file
    return
  if motion_cfg.motion_file and Path(motion_cfg.motion_file).exists():
    return
  if registry_name:
    artifact_name = registry_name if ":" in registry_name else registry_name + ":latest"
    import wandb

    artifact = wandb.Api().artifact(artifact_name)
    motion_cfg.motion_file = str(Path(artifact.download()) / "motion.npz")
    return
  if wandb_run_path:
    import wandb

    run = wandb.Api().run(str(wandb_run_path))
    art = next((a for a in run.used_artifacts() if a.type == "motions"), None)
    if art is None:
      raise RuntimeError("No motion artifact found in the run.")
    motion_cfg.motion_file = str(Path(art.download()) / "motion.npz")
    return
  if checkpoint_file is not None:
    raise ValueError(
      "Tracking tasks require `motion_file` when using a local FlashSAC checkpoint, "
      "or provide `wandb_run_path` / `registry_name` so the motion asset can be resolved."
    )
  raise ValueError(
    "Tracking tasks require either a local `motion_file`, `registry_name`, or "
    "`wandb_run_path`."
  )


class FlashSACInferencePolicy:
  def __init__(self, agent: FlashSACAgent, adapter: MjlabFlashSACEnvAdapter):
    self.agent = agent
    self.adapter = adapter
    self.device = agent.device

  def __call__(self, obs: dict[str, Any]) -> torch.Tensor:
    flat_obs = (
      self.adapter._flatten_obs(obs)
      .detach()
      .cpu()
      .numpy()
      .astype("float32", copy=False)
    )
    actions = self.agent.sample_actions(
      interaction_step=0,
      prev_transition={"next_observation": flat_obs},
      training=False,
    )
    return torch.as_tensor(actions, dtype=torch.float32, device=self.device)


def load_flashsac_policy(
  env: Any,
  checkpoint_path: str | Path,
  device: str | torch.device,
) -> FlashSACInferencePolicy:
  adapter = MjlabFlashSACEnvAdapter(env.unwrapped)
  agent_cfg = load_flashsac_runner_cfg(checkpoint_path, device=device)
  agent = FlashSACAgent(
    observation_dim=adapter.observation_space.shape[-1],
    action_dim=adapter.action_space.shape[-1],
    actor_observation_dim=adapter.policy_observation_dim(
      asymmetric_observation=agent_cfg.asymmetric_observation
    ),
    cfg=agent_cfg,
  )
  agent.load(str(resolve_flashsac_checkpoint_dir(checkpoint_path)))
  return FlashSACInferencePolicy(agent=agent, adapter=adapter)
