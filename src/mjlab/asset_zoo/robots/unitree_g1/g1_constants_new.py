"""Unitree G1 mode_15-aligned constants using the isolated g1_new.xml asset."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

G1_NEW_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_g1" / "xmls" / "g1_new.xml"
)
assert G1_NEW_XML.exists()


def get_new_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, G1_NEW_XML.parent / "assets", meshdir)
  return assets


def get_new_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_NEW_XML))
  spec.assets = get_new_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Motor specs (from Unitree).
ROTOR_INERTIAS_5020 = (
  0.139e-4,
  0.017e-4,
  0.169e-4,
)
GEARS_5020 = (
  1,
  1 + (46 / 18),
  1 + (56 / 16),
)
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_5020, GEARS_5020
)

ROTOR_INERTIAS_7520_14 = (
  0.489e-4,
  0.098e-4,
  0.533e-4,
)
GEARS_7520_14 = (
  1,
  4.5,
  1 + (48 / 22),
)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

ROTOR_INERTIAS_7520_22 = (
  0.489e-4,
  0.109e-4,
  0.738e-4,
)
GEARS_7520_22 = (
  1,
  4.5,
  5,
)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_22, GEARS_7520_22
)

ROTOR_INERTIAS_4010 = (
  0.068e-4,
  0.0,
  0.0,
)
GEARS_4010 = (
  1,
  5,
  5,
)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_4010, GEARS_4010
)

ACTUATOR_5020 = ElectricActuator(
  reflected_inertia=ARMATURE_5020,
  velocity_limit=37.0,
  effort_limit=25.0,
)
ACTUATOR_7520_14 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_14,
  velocity_limit=32.0,
  effort_limit=88.0,
)
ACTUATOR_7520_22 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_22,
  velocity_limit=20.0,
  effort_limit=139.0,
)

# mode_15 updates wrist_pitch/yaw limits to 13.4 Nm / 27 rad/s and uses 5010
# wrist hardware. We now have the 5010 wrist meshes locally, but the upstream
# package still does not provide 5010 rotor inertia specs here, so we reuse the
# previous wrist armature as the closest local approximation while matching the
# new URDF effort envelope.
ACTUATOR_WRIST_MODE_15 = ElectricActuator(
  reflected_inertia=ARMATURE_4010,
  velocity_limit=27.0,
  effort_limit=13.4,
)

# Waist pitch/roll and ankles are modeled as 4-bar joints. We keep the doubled
# armature approximation from the original asset, but match the mode_15 joint
# effort envelope of 35 Nm from the latest URDF.
ACTUATOR_FOUR_BAR_MODE_15 = ElectricActuator(
  reflected_inertia=ARMATURE_5020 * 2,
  velocity_limit=30.0,
  effort_limit=35.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_WRIST_MODE_15 = ACTUATOR_WRIST_MODE_15.reflected_inertia * NATURAL_FREQ**2
STIFFNESS_FOUR_BAR_MODE_15 = ACTUATOR_FOUR_BAR_MODE_15.reflected_inertia * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_WRIST_MODE_15 = (
  2.0 * DAMPING_RATIO * ACTUATOR_WRIST_MODE_15.reflected_inertia * NATURAL_FREQ
)
DAMPING_FOUR_BAR_MODE_15 = (
  2.0 * DAMPING_RATIO * ACTUATOR_FOUR_BAR_MODE_15.reflected_inertia * NATURAL_FREQ
)

G1_NEW_ACTUATOR_5020 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
)
G1_NEW_ACTUATOR_7520_14 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint", "waist_yaw_joint"),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)
G1_NEW_ACTUATOR_7520_22 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"),
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)
G1_NEW_ACTUATOR_WRIST_MODE_15 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  stiffness=STIFFNESS_WRIST_MODE_15,
  damping=DAMPING_WRIST_MODE_15,
  effort_limit=ACTUATOR_WRIST_MODE_15.effort_limit,
  armature=ACTUATOR_WRIST_MODE_15.reflected_inertia,
)
G1_NEW_ACTUATOR_FOUR_BAR_MODE_15 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "waist_pitch_joint",
    "waist_roll_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
  ),
  stiffness=STIFFNESS_FOUR_BAR_MODE_15,
  damping=DAMPING_FOUR_BAR_MODE_15,
  effort_limit=ACTUATOR_FOUR_BAR_MODE_15.effort_limit,
  armature=ACTUATOR_FOUR_BAR_MODE_15.reflected_inertia,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.783675),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": 1.28,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.76),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

G1_NEW_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_NEW_ACTUATOR_5020,
    G1_NEW_ACTUATOR_7520_14,
    G1_NEW_ACTUATOR_7520_22,
    G1_NEW_ACTUATOR_WRIST_MODE_15,
    G1_NEW_ACTUATOR_FOUR_BAR_MODE_15,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_new_robot_cfg() -> EntityCfg:
  """Get a fresh mode_15-aligned G1 robot configuration instance."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_new_spec,
    articulation=G1_NEW_ARTICULATION,
  )


G1_NEW_ACTION_SCALE: dict[str, float] = {}
for actuator_cfg in G1_NEW_ARTICULATION.actuators:
  assert isinstance(actuator_cfg, BuiltinPositionActuatorCfg)
  effort_limit = actuator_cfg.effort_limit
  stiffness = actuator_cfg.stiffness
  target_names = actuator_cfg.target_names_expr
  assert effort_limit is not None
  for name in target_names:
    G1_NEW_ACTION_SCALE[name] = 0.25 * effort_limit / stiffness


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_new_robot_cfg())
  viewer.launch(robot.spec.compile())
