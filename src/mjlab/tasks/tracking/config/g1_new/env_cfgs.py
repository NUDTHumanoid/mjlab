"""Tracking env wrappers that swap in the isolated mode_15-aligned G1 asset."""

from mjlab.asset_zoo.robots.unitree_g1.g1_constants_new import (
  G1_NEW_ACTION_SCALE,
  get_g1_new_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import (
  unitree_g1_flat_late_phase_dr_finetune_env_cfg as _base_flat_late_phase_dr_finetune_env_cfg,
)
from mjlab.tasks.tracking.config.g1.env_cfgs import (
  unitree_g1_flat_tracking_env_cfg as _base_flat_tracking_env_cfg,
)
from mjlab.tasks.tracking.config.g1.env_cfgs import (
  unitree_g1_rough_late_phase_dr_finetune_env_cfg as _base_rough_late_phase_dr_finetune_env_cfg,
)
from mjlab.tasks.tracking.config.g1.env_cfgs import (
  unitree_g1_rough_tracking_env_cfg as _base_rough_tracking_env_cfg,
)


def _swap_to_g1_new(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  cfg.scene.entities = {"robot": get_g1_new_robot_cfg()}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_NEW_ACTION_SCALE
  return cfg


def unitree_g1_new_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return _swap_to_g1_new(
    _base_flat_tracking_env_cfg(
      has_state_estimation=has_state_estimation,
      play=play,
    )
  )


def unitree_g1_new_flat_late_phase_dr_finetune_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return _swap_to_g1_new(
    _base_flat_late_phase_dr_finetune_env_cfg(
      has_state_estimation=has_state_estimation,
      play=play,
    )
  )


def unitree_g1_new_rough_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return _swap_to_g1_new(
    _base_rough_tracking_env_cfg(
      has_state_estimation=has_state_estimation,
      play=play,
    )
  )


def unitree_g1_new_rough_late_phase_dr_finetune_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return _swap_to_g1_new(
    _base_rough_late_phase_dr_finetune_env_cfg(
      has_state_estimation=has_state_estimation,
      play=play,
    )
  )
