"""Zhixing tracking environment configurations."""

from copy import deepcopy

from mjlab.asset_zoo.robots import (
  ZHIXING_ACTION_SCALE,
  get_zhixing_robot_cfg,
)
from mjlab.asset_zoo.robots.zhixing.zhixing_constants import ZHIXING_FOOT_GEOM_PATTERN
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG

_TRACKING_BODY_NAMES = (
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "waist_yaw_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_roll_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_roll_link",
)
_FLAT_TRACKING_EE_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_roll_link",
  "right_wrist_roll_link",
)

_TRACKING_ROUGH_STAGE_ITERATION_OFFSET: int | None = None
_TRACKING_ROUGH_STAGE_STEP_OFFSET = (
  None
  if _TRACKING_ROUGH_STAGE_ITERATION_OFFSET is None
  else _TRACKING_ROUGH_STAGE_ITERATION_OFFSET * 24
)

_TRACKING_ROUGH_TERRAIN_STAGES = [
  {
    "step": 0,
    "max_terrain_level": 0,
    "terrain_type_probs": {
      "flat": 0.70,
      "random_rough": 0.20,
      "wave_terrain": 0.10,
    },
  },
  {
    "step": 3_000 * 24,
    "max_terrain_level": 2,
    "terrain_type_probs": {
      "flat": 0.50,
      "random_rough": 0.20,
      "wave_terrain": 0.30,
    },
  },
  {
    "step": 6_000 * 24,
    "max_terrain_level": 4,
    "terrain_type_probs": {
      "flat": 0.30,
      "random_rough": 0.25,
      "wave_terrain": 0.45,
    },
  },
  {
    "step": 9_000 * 24,
    "max_terrain_level": 5,
    "terrain_type_probs": {
      "flat": 0.15,
      "random_rough": 0.25,
      "wave_terrain": 0.60,
    },
  },
]


def _configure_zhixing_tracking_cfg(
  cfg: ManagerBasedRlEnvCfg,
  has_state_estimation: bool,
) -> MotionCommandCfg:
  """Apply Zhixing-specific robot, sensor, and observation settings."""
  cfg.scene.entities = {"robot": get_zhixing_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = ZHIXING_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "waist_yaw_link"
  motion_cmd.body_names = _TRACKING_BODY_NAMES

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = ZHIXING_FOOT_GEOM_PATTERN
  cfg.events["base_com"].params["asset_cfg"].body_names = ("waist_yaw_link",)
  cfg.terminations["ee_body_pos"].params["body_names"] = _FLAT_TRACKING_EE_BODY_NAMES
  cfg.viewer.body_name = "waist_yaw_link"

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


def _apply_tracking_play_overrides(
  cfg: ManagerBasedRlEnvCfg,
  motion_cmd: MotionCommandCfg,
) -> None:
  """Use deterministic play-time settings for easier qualitative evaluation."""
  cfg.episode_length_s = int(1e9)
  cfg.observations["actor"].enable_corruption = False
  cfg.events.pop("push_robot", None)
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.sampling_mode = "start"


def _apply_staged_terrain_play_overrides(
  cfg: ManagerBasedRlEnvCfg,
  motion_cmd: MotionCommandCfg,
  *,
  num_rows: int,
  num_cols: int,
  border_width: float = 10.0,
) -> None:
  _apply_tracking_play_overrides(cfg, motion_cmd)
  cfg.events["staged_terrain_sampling"] = EventTermCfg(
    func=mdp.staged_tracking_terrain_sampling,
    mode="reset",
    params={
      "stages": _TRACKING_ROUGH_TERRAIN_STAGES,
      "rough_stage_step_offset": _TRACKING_ROUGH_STAGE_STEP_OFFSET,
      "auto_capture_offset": True,
    },
  )

  if cfg.scene.terrain is not None:
    terrain_generator = cfg.scene.terrain.terrain_generator
    if terrain_generator is not None:
      terrain_generator.curriculum = True
      terrain_generator.num_rows = num_rows
      terrain_generator.num_cols = num_cols
      terrain_generator.border_width = border_width


def _make_tracking_rough_terrain_cfg():
  terrain_cfg = deepcopy(ROUGH_TERRAINS_CFG)
  terrain_cfg.curriculum = True
  terrain_cfg.num_rows = 6
  terrain_cfg.num_cols = 12

  terrain_cfg.sub_terrains.pop("pyramid_stairs", None)
  terrain_cfg.sub_terrains.pop("pyramid_stairs_inv", None)
  terrain_cfg.sub_terrains.pop("hf_pyramid_slope", None)
  terrain_cfg.sub_terrains.pop("hf_pyramid_slope_inv", None)

  terrain_cfg.sub_terrains["flat"].proportion = 0.25
  terrain_cfg.sub_terrains["random_rough"].proportion = 0.25
  terrain_cfg.sub_terrains["random_rough"].noise_range = (0.01, 0.04)
  terrain_cfg.sub_terrains["wave_terrain"].proportion = 0.50
  terrain_cfg.sub_terrains["wave_terrain"].amplitude_range = (0.0, 0.08)
  return terrain_cfg


def zhixing_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Zhixing flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()
  motion_cmd = _configure_zhixing_tracking_cfg(cfg, has_state_estimation)

  if play:
    _apply_tracking_play_overrides(cfg, motion_cmd)

  return cfg


def zhixing_rough_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Zhixing rough terrain tracking configuration."""
  cfg = make_tracking_env_cfg()
  motion_cmd = _configure_zhixing_tracking_cfg(cfg, has_state_estimation)

  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=_make_tracking_rough_terrain_cfg(),
    max_init_terrain_level=0,
  )
  cfg.scene.extent = 2.0

  if not play:
    cfg.events["staged_terrain_sampling"] = EventTermCfg(
      func=mdp.staged_tracking_terrain_sampling,
      mode="reset",
      params={
        "stages": _TRACKING_ROUGH_TERRAIN_STAGES,
        "rough_stage_step_offset": _TRACKING_ROUGH_STAGE_STEP_OFFSET,
        "auto_capture_offset": True,
      },
    )

  cfg.sim.nconmax = 60
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.mujoco.ccd_iterations = 200

  cfg.rewards["motion_global_root_pos"] = RewardTermCfg(
    func=mdp.motion_global_anchor_xy_position_error_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 0.4},
  )
  cfg.rewards["motion_global_root_z_vel"] = RewardTermCfg(
    func=mdp.motion_global_anchor_z_velocity_error_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 1.0},
  )
  cfg.rewards["motion_global_root_z_pos"] = RewardTermCfg(
    func=mdp.motion_global_anchor_z_position_error_exp,
    weight=0.75,
    params={"command_name": "motion", "std": 0.2},
  )

  cfg.rewards["motion_global_root_ori"].weight = 1.0
  cfg.rewards["motion_body_ori"].weight = 1.5

  cfg.events.pop("push_robot", None)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.015, 0.015),
    1: (-0.03, 0.03),
    2: (-0.03, 0.03),
  }
  cfg.events["encoder_bias"].params["bias_range"] = (-0.005, 0.005)
  cfg.events["foot_friction"].params["ranges"] = (0.5, 1.0)

  cfg.terminations.pop("anchor_pos", None)
  cfg.terminations.pop("ee_body_pos", None)
  cfg.terminations["anchor_ori"].params["threshold"] = 1.2

  if play:
    _apply_staged_terrain_play_overrides(
      cfg,
      motion_cmd,
      num_rows=6,
      num_cols=6,
      border_width=10.0,
    )

  return cfg
