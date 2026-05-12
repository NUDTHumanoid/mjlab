"""Tracking env wrappers that swap in the Z2 asset with G1-New-like tuning."""

from mjlab.asset_zoo.robots.nubot_z2.z2_constants import (
  Z2_ACTION_SCALE,
  Z2_FLAT_TRACKING_EE_BODY_NAMES,
  Z2_FOOT_GEOM_PATTERN,
  Z2_ROOT_BODY_NAME,
  Z2_TRACKING_BODY_NAMES,
  Z2_WAIST_ROLL_BODY_NAME,
  get_z2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import (
  _apply_tracking_late_phase_dr_finetune_overrides,
  _apply_tracking_play_overrides,
  _apply_tracking_rough_overrides,
)
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

_Z2_TRACKING_NCONMAX = 48
_Z2_TRACKING_NJMAX = 320
_Z2_TRACKING_CONTACT_SENSOR_MAXMATCH = 64
_Z2_TRACKING_CCD_ITERATIONS = 96


def _configure_z2_tracking_cfg(
  cfg: ManagerBasedRlEnvCfg,
  has_state_estimation: bool,
) -> MotionCommandCfg:
  cfg.scene.entities = {"robot": get_z2_robot_cfg()}
  # Z2 exposes many more simultaneous contacts than the G1 defaults, especially
  # when mesh collisions are enabled. We keep only foot sole primitives active,
  # so a modest but explicit budget is enough and much cheaper than the earlier
  # all-mesh setup.
  cfg.sim.nconmax = _Z2_TRACKING_NCONMAX
  cfg.sim.njmax = _Z2_TRACKING_NJMAX
  cfg.sim.contact_sensor_maxmatch = _Z2_TRACKING_CONTACT_SENSOR_MAXMATCH
  cfg.sim.mujoco.ccd_iterations = _Z2_TRACKING_CCD_ITERATIONS

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=Z2_ROOT_BODY_NAME, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=Z2_ROOT_BODY_NAME, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = Z2_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = Z2_ROOT_BODY_NAME
  motion_cmd.body_names = Z2_TRACKING_BODY_NAMES

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = Z2_FOOT_GEOM_PATTERN
  cfg.events["base_com"].params["asset_cfg"].body_names = (Z2_WAIST_ROLL_BODY_NAME,)
  cfg.terminations["ee_body_pos"].params["body_names"] = Z2_FLAT_TRACKING_EE_BODY_NAMES
  cfg.viewer.body_name = Z2_WAIST_ROLL_BODY_NAME

  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  return motion_cmd


def _apply_z2_contact_budget(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.sim.nconmax = _Z2_TRACKING_NCONMAX
  cfg.sim.njmax = _Z2_TRACKING_NJMAX
  cfg.sim.contact_sensor_maxmatch = _Z2_TRACKING_CONTACT_SENSOR_MAXMATCH
  cfg.sim.mujoco.ccd_iterations = _Z2_TRACKING_CCD_ITERATIONS


def nubot_z2_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()
  motion_cmd = _configure_z2_tracking_cfg(cfg, has_state_estimation)
  if play:
    _apply_tracking_play_overrides(cfg, motion_cmd)
  return cfg


def nubot_z2_rough_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()
  motion_cmd = _configure_z2_tracking_cfg(cfg, has_state_estimation)
  _apply_tracking_rough_overrides(cfg, motion_cmd, play=play)
  _apply_z2_contact_budget(cfg)
  return cfg


def nubot_z2_flat_late_phase_dr_finetune_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()
  motion_cmd = _configure_z2_tracking_cfg(cfg, has_state_estimation)
  _apply_tracking_late_phase_dr_finetune_overrides(cfg, motion_cmd, play=play)
  return cfg


def nubot_z2_rough_late_phase_dr_finetune_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()
  motion_cmd = _configure_z2_tracking_cfg(cfg, has_state_estimation)
  _apply_tracking_rough_overrides(cfg, motion_cmd, play=play)
  _apply_z2_contact_budget(cfg)
  _apply_tracking_late_phase_dr_finetune_overrides(cfg, motion_cmd, play=play)
  return cfg
