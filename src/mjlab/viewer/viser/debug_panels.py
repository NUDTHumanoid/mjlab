"""Helpers for building debug inventory and control-chain snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity, EntityCfg

if TYPE_CHECKING:
  import torch


@dataclass(frozen=True)
class JointInventoryRow:
  joint_index: int
  joint_name: str
  actuator_group: str | None
  q: float
  q_des: float
  dq: float
  q_default: float
  joint_limit: tuple[float, float]


@dataclass(frozen=True)
class ActuatorInventoryRow:
  group_name: str
  target_expr: tuple[str, ...]
  matched_joints: tuple[str, ...]
  control_type: str
  stiffness: float | None
  damping: float | None
  effort_limit: float | None
  armature: float
  transmission_type: str


@dataclass(frozen=True)
class ControlChainRow:
  action_index: int
  joint_index: int
  joint_name: str
  raw_action: float
  processed_action: float
  q_des: float
  q: float
  dq: float
  actuator_force: float | None
  qfrc_actuator: float


def _format_group_name(target_expr: tuple[str, ...]) -> str:
  return ", ".join(target_expr)


def _control_type_name(actuator_cfg: Any) -> str:
  cls_name = actuator_cfg.__class__.__name__.lower()
  if "position" in cls_name or "pd" in cls_name:
    return "position"
  if "velocity" in cls_name:
    return "velocity"
  if "motor" in cls_name:
    return "effort"
  if "muscle" in cls_name:
    return "muscle"
  return cls_name.removesuffix("cfg")


def _joint_group_map(entity: Entity) -> dict[str, str]:
  mapping: dict[str, str] = {}
  for act in entity.actuators:
    group_name = _format_group_name(tuple(str(x) for x in act.cfg.target_names_expr))
    for joint_name in act.target_names:
      mapping[joint_name] = group_name
  return mapping


def _joint_ctrl_index_map(entity: Entity) -> dict[str, int]:
  mapping: dict[str, int] = {}
  for act in entity.actuators:
    if act.transmission_type != TransmissionType.JOINT:
      continue
    for joint_name, ctrl_idx in zip(act.target_names, act.ctrl_ids.tolist()):
      mapping[joint_name] = int(ctrl_idx)
  return mapping


def build_actuator_inventory_from_cfg(entity_cfg: EntityCfg) -> list[ActuatorInventoryRow]:
  """Build grouped actuator inventory rows from an entity config."""
  entity = Entity(entity_cfg)
  rows: list[ActuatorInventoryRow] = []
  for act in entity.actuators:
    cfg = act.cfg
    rows.append(
      ActuatorInventoryRow(
        group_name=_format_group_name(tuple(str(x) for x in cfg.target_names_expr)),
        target_expr=tuple(str(x) for x in cfg.target_names_expr),
        matched_joints=tuple(act.target_names),
        control_type=_control_type_name(cfg),
        stiffness=getattr(cfg, "stiffness", None),
        damping=getattr(cfg, "damping", None),
        effort_limit=getattr(cfg, "effort_limit", None),
        armature=float(cfg.armature),
        transmission_type=cfg.transmission_type.value,
      )
    )
  return rows


def build_joint_inventory(entity: Entity, env_idx: int = 0) -> list[JointInventoryRow]:
  """Build per-joint snapshot rows for the selected environment."""
  joint_group_map = _joint_group_map(entity)
  q = entity.data.joint_pos[env_idx]
  q_des = entity.data.joint_pos_target[env_idx]
  dq = entity.data.joint_vel[env_idx]
  q_default = entity.data.default_joint_pos[env_idx]
  limits = entity.data.joint_pos_limits[env_idx]

  rows: list[JointInventoryRow] = []
  for idx, joint_name in enumerate(entity.joint_names):
    rows.append(
      JointInventoryRow(
        joint_index=idx,
        joint_name=joint_name,
        actuator_group=joint_group_map.get(joint_name),
        q=float(q[idx].item()),
        q_des=float(q_des[idx].item()),
        dq=float(dq[idx].item()),
        q_default=float(q_default[idx].item()),
        joint_limit=(float(limits[idx, 0].item()), float(limits[idx, 1].item())),
      )
    )
  return rows


def build_control_chain_rows(
  entity: Entity, action_term: Any, env_idx: int = 0
) -> list[ControlChainRow]:
  """Build action-to-joint control-chain rows for joint-based actions."""
  transmission_type = getattr(action_term.cfg, "transmission_type", None)
  if transmission_type != TransmissionType.JOINT:
    raise ValueError("build_control_chain_rows only supports joint-based actions.")

  joint_ctrl_map = _joint_ctrl_index_map(entity)
  target_ids = action_term.target_ids.tolist()
  target_names = list(action_term.target_names)
  raw_action: torch.Tensor = action_term.raw_action[env_idx]
  processed_action: torch.Tensor = action_term.processed_action[env_idx]

  rows: list[ControlChainRow] = []
  for action_idx, (joint_idx, joint_name) in enumerate(zip(target_ids, target_names)):
    ctrl_idx = joint_ctrl_map.get(joint_name)
    actuator_force = None
    if ctrl_idx is not None:
      actuator_force = float(entity.data.actuator_force[env_idx, ctrl_idx].item())

    rows.append(
      ControlChainRow(
        action_index=action_idx,
        joint_index=int(joint_idx),
        joint_name=joint_name,
        raw_action=float(raw_action[action_idx].item()),
        processed_action=float(processed_action[action_idx].item()),
        q_des=float(entity.data.joint_pos_target[env_idx, joint_idx].item()),
        q=float(entity.data.joint_pos[env_idx, joint_idx].item()),
        dq=float(entity.data.joint_vel[env_idx, joint_idx].item()),
        actuator_force=actuator_force,
        qfrc_actuator=float(entity.data.qfrc_actuator[env_idx, joint_idx].item()),
      )
    )
  return rows
