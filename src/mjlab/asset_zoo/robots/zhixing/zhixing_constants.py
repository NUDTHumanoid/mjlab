"""Zhixing humanoid constants and robot configuration."""

from pathlib import Path
import re

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

ZHIXING_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "zhixing"
  / "xmls"
  / "z1_description_v1-5_20dofs.xml"
)
assert ZHIXING_XML.exists()

ZHIXING_FOOT_GEOM_PATTERN = r"^(left|right)_ankle_roll_geom([1-9]|10)$"
_IMU_SITE_NAME = "imu_in_waist"


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, ZHIXING_XML.parent / "assets", meshdir, recursive=True)
  return assets


def _ensure_imu_site_and_sensors(spec: mujoco.MjSpec) -> None:
  site_names = {site.name for site in spec.sites}
  sensor_names = {sensor.name for sensor in spec.sensors}

  if _IMU_SITE_NAME not in site_names:
    spec.body("waist_yaw_link").add_site(
      name=_IMU_SITE_NAME,
      pos=(0.0, 0.0, 0.18),
      size=(0.01, 0.01, 0.01),
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
    )

  if "imu_ang_vel" not in sensor_names:
    spec.add_sensor(
      name="imu_ang_vel",
      type=mujoco.mjtSensor.mjSENS_GYRO,
      objtype=mujoco.mjtObj.mjOBJ_SITE,
      objname=_IMU_SITE_NAME,
    )
  if "imu_lin_vel" not in sensor_names:
    spec.add_sensor(
      name="imu_lin_vel",
      type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
      objtype=mujoco.mjtObj.mjOBJ_SITE,
      objname=_IMU_SITE_NAME,
    )
  if "imu_lin_acc" not in sensor_names:
    spec.add_sensor(
      name="imu_lin_acc",
      type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
      objtype=mujoco.mjtObj.mjOBJ_SITE,
      objname=_IMU_SITE_NAME,
    )


def get_spec() -> mujoco.MjSpec:
  xml_text = ZHIXING_XML.read_text()
  # The source XML ships torque motors; strip them here so the programmatic
  # position actuators below are the only active control path.
  xml_text = re.sub(
    r"<actuator>.*?</actuator>",
    "",
    xml_text,
    count=1,
    flags=re.DOTALL,
  )
  spec = mujoco.MjSpec.from_string(xml_text)
  spec.assets = get_assets(spec.meshdir)
  _ensure_imu_site_and_sensors(spec)
  return spec


##
# Actuator config.
##

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0


def _make_position_actuator(
  *,
  target_names_expr: tuple[str, ...],
  armature: float,
  frictionloss: float,
  effort_limit: float,
) -> BuiltinPositionActuatorCfg:
  stiffness = armature * NATURAL_FREQ**2
  damping = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ
  return BuiltinPositionActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=stiffness,
    damping=damping,
    effort_limit=effort_limit,
    armature=armature,
    frictionloss=frictionloss,
  )


ZHIXING_HIP_PITCH_ACTUATOR = _make_position_actuator(
  target_names_expr=(".*_hip_pitch_joint",),
  armature=0.19094,
  frictionloss=2.625,
  effort_limit=320.0,
)
ZHIXING_HIP_ROLL_ACTUATOR = _make_position_actuator(
  target_names_expr=(".*_hip_roll_joint",),
  armature=0.06102,
  frictionloss=1.175,
  effort_limit=330.0,
)
ZHIXING_HIP_YAW_ACTUATOR = _make_position_actuator(
  target_names_expr=(".*_hip_yaw_joint",),
  armature=0.03138,
  frictionloss=1.029,
  effort_limit=150.0,
)
ZHIXING_KNEE_ACTUATOR = _make_position_actuator(
  target_names_expr=(".*_knee_joint",),
  armature=0.19314,
  frictionloss=2.25,
  effort_limit=400.0,
)
ZHIXING_ANKLE_PITCH_ACTUATOR = _make_position_actuator(
  target_names_expr=(".*_ankle_pitch_joint",),
  armature=0.032,
  frictionloss=0.2585,
  effort_limit=180.0,
)
ZHIXING_SHOULDER_PITCH_ACTUATOR = _make_position_actuator(
  target_names_expr=(".*_shoulder_pitch_joint",),
  armature=0.01,
  frictionloss=0.2,
  effort_limit=90.0,
)
ZHIXING_SHOULDER_ROLL_ACTUATOR = _make_position_actuator(
  target_names_expr=(".*_shoulder_roll_joint",),
  armature=0.01,
  frictionloss=0.2,
  effort_limit=90.0,
)
ZHIXING_SHOULDER_YAW_ACTUATOR = _make_position_actuator(
  target_names_expr=(".*_shoulder_yaw_joint",),
  armature=0.01,
  frictionloss=0.2,
  effort_limit=60.0,
)
ZHIXING_ELBOW_ACTUATOR = _make_position_actuator(
  target_names_expr=(".*_elbow_joint",),
  armature=0.01,
  frictionloss=0.2,
  effort_limit=60.0,
)
ZHIXING_WRIST_ROLL_ACTUATOR = _make_position_actuator(
  target_names_expr=(".*_wrist_roll_joint",),
  armature=0.01,
  frictionloss=0.2,
  effort_limit=36.0,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.02),
  joint_pos={
    ".*_hip_pitch_joint": -0.18,
    ".*_knee_joint": 0.36,
    ".*_ankle_pitch_joint": -0.18,
    ".*_shoulder_pitch_joint": 0.15,
    ".*_elbow_joint": 0.45,
    "left_shoulder_roll_joint": 0.12,
    "right_shoulder_roll_joint": -0.12,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.02),
  joint_pos={
    ".*_hip_pitch_joint": -0.28,
    ".*_knee_joint": 0.62,
    ".*_ankle_pitch_joint": -0.32,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": 0.75,
    "left_shoulder_roll_joint": 0.15,
    "right_shoulder_roll_joint": -0.15,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FOOT_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(ZHIXING_FOOT_GEOM_PATTERN,),
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

ZHIXING_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    ZHIXING_HIP_PITCH_ACTUATOR,
    ZHIXING_HIP_ROLL_ACTUATOR,
    ZHIXING_HIP_YAW_ACTUATOR,
    ZHIXING_KNEE_ACTUATOR,
    ZHIXING_ANKLE_PITCH_ACTUATOR,
    ZHIXING_SHOULDER_PITCH_ACTUATOR,
    ZHIXING_SHOULDER_ROLL_ACTUATOR,
    ZHIXING_SHOULDER_YAW_ACTUATOR,
    ZHIXING_ELBOW_ACTUATOR,
    ZHIXING_WRIST_ROLL_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_zhixing_robot_cfg() -> EntityCfg:
  """Get a fresh Zhixing robot configuration instance."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FOOT_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=ZHIXING_ARTICULATION,
  )


ZHIXING_ACTION_SCALE: dict[str, float] = {}
for actuator_cfg in ZHIXING_ARTICULATION.actuators:
  assert isinstance(actuator_cfg, BuiltinPositionActuatorCfg)
  effort_limit = actuator_cfg.effort_limit
  stiffness = actuator_cfg.stiffness
  target_names = actuator_cfg.target_names_expr
  assert effort_limit is not None
  for name in target_names:
    ZHIXING_ACTION_SCALE[name] = 0.25 * effort_limit / stiffness


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_zhixing_robot_cfg())
  viewer.launch(robot.spec.compile())
