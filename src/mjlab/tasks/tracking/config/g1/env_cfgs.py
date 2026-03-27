"""Unitree G1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from mjlab.terrains import TerrainGeneratorCfg, HfPyramidSlopedTerrainCfg, HfRandomUniformTerrainCfg 
# 自定义地形：随机起伏
CUSTOM_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),                 # 每个子地形 8m x 8m
    border_width=2.0,                # 外边框宽度 2m（足够防止掉落）
    num_rows=5,                      # 5 行难度等级
    num_cols=6,                      # 6 列（3列斜坡+3列随机起伏，比例可调）
    curriculum=True,                 # 课程模式：按列分配地形类型，难度逐行增加
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        "rough": HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.02, 0.15),   # 起伏高度范围
            downsampled_scale=0.2,      # 粗采样间距，使起伏更自然
            border_width=0.5,
            horizontal_scale=0.1,
            vertical_scale=0.005,
        ),
    },
    add_lights=True,
)

def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
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

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg

def unitree_g1_rough_tracking_env_cfg(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 rough terrain tracking configuration."""
    # 先获取平坦配置
    cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation, play)
    # 覆盖地形为粗糙地形生成器
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = CUSTOM_TERRAIN_CFG
    # ===== 粗糙地形特有的参数调整（示例，可根据实际需要修改） =====
    cfg.rewards["motion_global_root_pos"].weight = 2.0   # 原来1.5
    cfg.rewards["motion_body_lin_vel"].weight = 2.0      # 原来1.5
    cfg.rewards["motion_body_pos"].weight = 1.5          # 原来1.0
    cfg.rewards["motion_global_root_ori"].weight = 1.0   # 原来0.5
    cfg.rewards["motion_body_ori"].weight = 1.5          # 原来1.0
    cfg.terminations["ee_body_pos"].params["threshold"] = 0.4
    # ==================================================
    return cfg