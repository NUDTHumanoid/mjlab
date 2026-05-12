"""Nubot Z2 constants and tracking-oriented robot config."""
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

Z2_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "nubot_z2" / "xmls" / "assembly.xml"
)
assert Z2_XML.exists()

Z2_ROOT_BODY_NAME = "Z2_0_Lite_description_0429_1"
Z2_WAIST_YAW_BODY_NAME = "Z2_0_Lite_description_0429_1_waist_yaw_Link"
Z2_WAIST_ROLL_BODY_NAME = "Z2_0_Lite_description_0429_1_waist_roll_Link"

Z2_TRACKING_BODY_NAMES = (
  Z2_ROOT_BODY_NAME,
  "Z2_0_Lite_description_0429_1_L_hip_roll_Link",
  "Z2_0_Lite_description_0429_1_L_knee_Link",
  "Z2_0_Lite_description_0429_1_L_ankle_roll_Link",
  "Z2_0_Lite_description_0429_1_R_hip_roll_Link",
  "Z2_0_Lite_description_0429_1_R_knee_Link",
  "Z2_0_Lite_description_0429_1_R_ankle_roll_Link",
  Z2_WAIST_ROLL_BODY_NAME,
  "Z2_0_Lite_description_0429_1_L_shoulder_roll_Link",
  "Z2_0_Lite_description_0429_1_L_elbow_Link",
  "Z2_0_Lite_description_0429_1_L_wrist_yaw_Link",
  "Z2_0_Lite_description_0429_1_R_shoulder_roll_Link",
  "Z2_0_Lite_description_0429_1_R_elbow_Link",
  "Z2_0_Lite_description_0429_1_R_wrist_yaw_Link",
)

Z2_TRACKING_JOINT_NAMES = (
  "Z2_0_Lite_description_0429_1_waist_yaw_joint",
  "Z2_0_Lite_description_0429_1_waist_pitch_joint",
  "Z2_0_Lite_description_0429_1_waist_roll_joint",
  "Z2_0_Lite_description_0429_1_R_shoulder_pitch_joint",
  "Z2_0_Lite_description_0429_1_R_shoulder_roll_joint",
  "Z2_0_Lite_description_0429_1_R_shoulder_yaw_joint",
  "Z2_0_Lite_description_0429_1_R_elbow_joint",
  "Z2_0_Lite_description_0429_1_R_wrist_roll_joint",
  "Z2_0_Lite_description_0429_1_R_wrist_pitch_joint",
  "Z2_0_Lite_description_0429_1_R_wrist_yaw_joint",
  "Z2_0_Lite_description_0429_1_L_shoulder_pitch_joint",
  "Z2_0_Lite_description_0429_1_L_shoulder_roll_joint",
  "Z2_0_Lite_description_0429_1_L_shoulder_yaw_joint",
  "Z2_0_Lite_description_0429_1_L_elbow_joint",
  "Z2_0_Lite_description_0429_1_L_wrist_roll_joint",
  "Z2_0_Lite_description_0429_1_L_wrist_pitch_joint",
  "Z2_0_Lite_description_0429_1_L_wrist_yaw_joint",
  "Z2_0_Lite_description_0429_1_R_hip_pitch_joint",
  "Z2_0_Lite_description_0429_1_R_hip_roll_joint",
  "Z2_0_Lite_description_0429_1_R_hip_yaw_joint",
  "Z2_0_Lite_description_0429_1_R_knee_joint",
  "Z2_0_Lite_description_0429_1_R_ankle_pitch_joint",
  "Z2_0_Lite_description_0429_1_R_ankle_roll_joint",
  "Z2_0_Lite_description_0429_1_L_hip_pitch_joint",
  "Z2_0_Lite_description_0429_1_L_hip_roll_joint",
  "Z2_0_Lite_description_0429_1_L_hip_yaw_joint",
  "Z2_0_Lite_description_0429_1_L_knee_joint",
  "Z2_0_Lite_description_0429_1_L_ankle_pitch_joint",
  "Z2_0_Lite_description_0429_1_L_ankle_roll_joint",
)

Z2_FLAT_TRACKING_EE_BODY_NAMES = (
  "Z2_0_Lite_description_0429_1_L_ankle_roll_Link",
  "Z2_0_Lite_description_0429_1_R_ankle_roll_Link",
  "Z2_0_Lite_description_0429_1_L_wrist_yaw_Link",
  "Z2_0_Lite_description_0429_1_R_wrist_yaw_Link",
)

Z2_FOOT_GEOM_PATTERN = r"^(left|right)_foot_contact_\d+$"
# Sole footprint fitted to the bottom 5 mm band of the ankle-roll meshes:
# x ~= [-0.034, 0.032], y ~= [-0.019, 0.019], z_min ~= -0.0955.
_Z2_FOOT_CAPSULE_RADIUS = 0.0065
_Z2_FOOT_CAPSULE_Z = -0.0070
_Z2_FOOT_CAPSULES = (
  ((-0.0270, -0.0120, _Z2_FOOT_CAPSULE_Z), (0.0250, -0.0120, _Z2_FOOT_CAPSULE_Z)),
  ((-0.0290, 0.0000, _Z2_FOOT_CAPSULE_Z), (0.0270, 0.0000, _Z2_FOOT_CAPSULE_Z)),
  ((-0.0270, 0.0120, _Z2_FOOT_CAPSULE_Z), (0.0250, 0.0120, _Z2_FOOT_CAPSULE_Z)),
)


def _rename_z2_geoms(spec: mujoco.MjSpec) -> None:
  """Assign stable geom names for DR/contact matching.

  The source MJCF uses unnamed mesh geoms, so we create deterministic names for the
  handful of geoms the tracking configs need to address by regex.
  """
  for body in spec.bodies:
    if not hasattr(body, "geoms"):
      continue

    group3_geoms = [geom for geom in body.geoms if geom.group == 3]
    if not group3_geoms:
      continue

    for index, geom in enumerate(group3_geoms):
      body_name = body.name
      if body_name.endswith("_L_ankle_roll_Link"):
        geom.name = (
          "left_foot_contact" if index == 0 else f"left_foot_contact_aux_{index}"
        )
      elif body_name.endswith("_R_ankle_roll_Link"):
        geom.name = (
          "right_foot_contact" if index == 0 else f"right_foot_contact_aux_{index}"
        )
      else:
        suffix = body_name.split("Z2_0_Lite_description_0429_1_")[-1]
        geom.name = f"{suffix}_collision_{index}"

  for geom_index, geom in enumerate(spec.geoms):
    if not geom.name:
      geom.name = f"z2_geom_{geom_index}"


def _replace_mesh_foot_collisions_with_primitives(spec: mujoco.MjSpec) -> None:
  """Replace ankle mesh collisions with foot capsules similar to G1 feet.

  Z2's source MJCF exposes the entire ankle-roll mesh as the contact geom. That is
  too coarse for replay-time foot height analysis because MuJoCo reports it as a mesh
  geom and our analysis utilities only support primitive shapes. We therefore keep the
  mesh as visual-only and add a small sole made of capsules on each ankle body.
  """
  for side in ("left", "right"):
    body_name = (
      "Z2_0_Lite_description_0429_1_L_ankle_roll_Link"
      if side == "left"
      else "Z2_0_Lite_description_0429_1_R_ankle_roll_Link"
    )
    mesh_contact_name = f"{side}_foot_contact"
    body = spec.body(body_name)

    for geom in body.geoms:
      if geom.name == mesh_contact_name:
        geom.contype = 0
        geom.conaffinity = 0
        geom.condim = 1
        geom.priority = 0

    for index, (start, end) in enumerate(_Z2_FOOT_CAPSULES):
      geom = body.add_geom(
        name=f"{side}_foot_contact_{index}",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=(_Z2_FOOT_CAPSULE_RADIUS, 0.0, 0.0),
        fromto=(*start, *end),
        rgba=(0.2, 0.6, 0.2, 0.3),
        group=3,
      )
      geom.contype = 1
      geom.conaffinity = 1
      geom.condim = 3
      geom.priority = 1
      geom.friction[0] = 0.6
      geom.friction[1] = 0.005
      geom.friction[2] = 0.0001


def _ensure_floating_base(spec: mujoco.MjSpec) -> None:
  """Match the floating-base setup expected by tracking tasks."""
  root_body = spec.bodies[1]
  if not any(joint.type == mujoco.mjtJoint.mjJNT_FREE for joint in root_body.joints):
    root_body.add_freejoint(name="floating_base_joint")


def _ensure_builtin_imu_sensors(spec: mujoco.MjSpec) -> None:
  """Provide the builtin IMU sensor names expected by tracking tasks."""
  root_body = spec.bodies[1]
  site_names = {site.name for site in root_body.sites}
  if "imu_in_base" not in site_names:
    root_body.add_site(name="imu_in_base", pos=(0.0, 0.0, 0.0))

  sensor_names = {sensor.name for sensor in spec.sensors}
  if "imu_ang_vel" not in sensor_names:
    spec.add_sensor(
      name="imu_ang_vel",
      type=mujoco.mjtSensor.mjSENS_GYRO,
      objtype=mujoco.mjtObj.mjOBJ_SITE,
      objname="imu_in_base",
    )
  if "imu_lin_vel" not in sensor_names:
    spec.add_sensor(
      name="imu_lin_vel",
      type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
      objtype=mujoco.mjtObj.mjOBJ_SITE,
      objname="imu_in_base",
    )


def get_z2_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, Z2_XML.parent / "meshes", meshdir, recursive=True)
  return assets


def get_z2_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(Z2_XML))
  spec.assets = get_z2_assets(spec.meshdir)
  _rename_z2_geoms(spec)
  _replace_mesh_foot_collisions_with_primitives(spec)
  for actuator in tuple(spec.actuators):
    spec.delete(actuator)
  _ensure_floating_base(spec)
  _ensure_builtin_imu_sensors(spec)
  return spec


##
# Actuator config.
##

# Reuse the established G1 tracking PD/effort envelopes so the new Z2 tracking
# tasks stay as close as possible to `Mjlab-Tracking-Flat-Unitree-G1-New`.
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
ACTUATOR_WRIST = ElectricActuator(
  reflected_inertia=ARMATURE_4010,
  velocity_limit=27.0,
  effort_limit=13.4,
)
ACTUATOR_FOUR_BAR = ElectricActuator(
  reflected_inertia=ARMATURE_5020 * 2,
  velocity_limit=30.0,
  effort_limit=35.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_WRIST = ACTUATOR_WRIST.reflected_inertia * NATURAL_FREQ**2
STIFFNESS_FOUR_BAR = ACTUATOR_FOUR_BAR.reflected_inertia * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_WRIST = 2.0 * DAMPING_RATIO * ACTUATOR_WRIST.reflected_inertia * NATURAL_FREQ
DAMPING_FOUR_BAR = (
  2.0 * DAMPING_RATIO * ACTUATOR_FOUR_BAR.reflected_inertia * NATURAL_FREQ
)

##
# Keyframe config.
##

Z2_KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.76),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    ".*_L_shoulder_roll_joint": 0.2,
    ".*_L_shoulder_pitch_joint": 0.2,
    ".*_R_shoulder_roll_joint": -0.2,
    ".*_R_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

Z2_FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*",),
  condim={Z2_FOOT_GEOM_PATTERN: 3, ".*": 1},
  priority={Z2_FOOT_GEOM_PATTERN: 1},
  friction={Z2_FOOT_GEOM_PATTERN: (0.6,)},
)

Z2_FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(Z2_FOOT_GEOM_PATTERN,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

Z2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    BuiltinPositionActuatorCfg(
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
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_hip_yaw_joint", ".*_waist_yaw_joint"),
      stiffness=STIFFNESS_7520_14,
      damping=DAMPING_7520_14,
      effort_limit=ACTUATOR_7520_14.effort_limit,
      armature=ACTUATOR_7520_14.reflected_inertia,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"),
      stiffness=STIFFNESS_7520_22,
      damping=DAMPING_7520_22,
      effort_limit=ACTUATOR_7520_22.effort_limit,
      armature=ACTUATOR_7520_22.reflected_inertia,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
      stiffness=STIFFNESS_WRIST,
      damping=DAMPING_WRIST,
      effort_limit=ACTUATOR_WRIST.effort_limit,
      armature=ACTUATOR_WRIST.reflected_inertia,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(
        ".*_waist_pitch_joint",
        ".*_waist_roll_joint",
        ".*_ankle_pitch_joint",
        ".*_ankle_roll_joint",
      ),
      stiffness=STIFFNESS_FOUR_BAR,
      damping=DAMPING_FOUR_BAR,
      effort_limit=ACTUATOR_FOUR_BAR.effort_limit,
      armature=ACTUATOR_FOUR_BAR.reflected_inertia,
    ),
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_z2_robot_cfg() -> EntityCfg:
  """Get a fresh Z2 robot configuration instance."""
  return EntityCfg(
    init_state=Z2_KNEES_BENT_KEYFRAME,
    collisions=(Z2_FEET_ONLY_COLLISION,),
    spec_fn=get_z2_spec,
    articulation=Z2_ARTICULATION,
  )


Z2_ACTION_SCALE: dict[str, float] = {
  ".*_elbow_joint": 0.25 * ACTUATOR_5020.effort_limit / STIFFNESS_5020,
  ".*_shoulder_pitch_joint": 0.25 * ACTUATOR_5020.effort_limit / STIFFNESS_5020,
  ".*_shoulder_roll_joint": 0.25 * ACTUATOR_5020.effort_limit / STIFFNESS_5020,
  ".*_shoulder_yaw_joint": 0.25 * ACTUATOR_5020.effort_limit / STIFFNESS_5020,
  ".*_wrist_roll_joint": 0.25 * ACTUATOR_5020.effort_limit / STIFFNESS_5020,
  ".*_hip_yaw_joint": 0.25 * ACTUATOR_7520_14.effort_limit / STIFFNESS_7520_14,
  ".*_waist_yaw_joint": 0.25 * ACTUATOR_7520_14.effort_limit / STIFFNESS_7520_14,
  ".*_hip_pitch_joint": 0.25 * ACTUATOR_7520_22.effort_limit / STIFFNESS_7520_22,
  ".*_hip_roll_joint": 0.25 * ACTUATOR_7520_22.effort_limit / STIFFNESS_7520_22,
  ".*_knee_joint": 0.25 * ACTUATOR_7520_22.effort_limit / STIFFNESS_7520_22,
  ".*_wrist_pitch_joint": 0.25 * ACTUATOR_WRIST.effort_limit / STIFFNESS_WRIST,
  ".*_wrist_yaw_joint": 0.25 * ACTUATOR_WRIST.effort_limit / STIFFNESS_WRIST,
  ".*_waist_pitch_joint": 0.25 * ACTUATOR_FOUR_BAR.effort_limit / STIFFNESS_FOUR_BAR,
  ".*_waist_roll_joint": 0.25 * ACTUATOR_FOUR_BAR.effort_limit / STIFFNESS_FOUR_BAR,
  ".*_ankle_pitch_joint": 0.25 * ACTUATOR_FOUR_BAR.effort_limit / STIFFNESS_FOUR_BAR,
  ".*_ankle_roll_joint": 0.25 * ACTUATOR_FOUR_BAR.effort_limit / STIFFNESS_FOUR_BAR,
}
