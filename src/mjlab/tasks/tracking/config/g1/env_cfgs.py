"""Unitree G1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.asset_zoo.robots.unitree_g1.g1_constants_new import (
    G1_NEW_ACTION_SCALE,
    get_g1_new_robot_cfg,
)#新增

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
    num_cols=3,                      # 3 列
    curriculum=True,                 # 课程模式：按列分配地形类型，难度逐行增加
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        "rough": HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.01, 0.05),   # 起伏高度范围
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

    return cfg


# 新版机器人平坦地形配置函数
def unitree_g1_new_flat_tracking_env_cfg(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 (mode_15) flat terrain tracking configuration."""
    cfg = make_tracking_env_cfg()

    # ===== 核心替换：使用新机器人配置 =====
    cfg.scene.entities = {"robot": get_g1_new_robot_cfg()}

    # ===== 传感器配置（与旧版相同） =====
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

    # ===== 动作缩放替换为新版系数 =====
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_NEW_ACTION_SCALE

    # ===== 运动命令配置（与旧版相同） =====
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

    # ===== 事件配置（与旧版相同） =====
    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = r"^(left|right)_foot[1-7]_collision$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

    # ===== 终止条件配置（与旧版相同） =====
    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )

    # ===== 视角配置（与旧版相同） =====
    cfg.viewer.body_name = "torso_link"

    # ===== 无状态估计时的观测修改（与旧版相同） =====
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

    # ===== 推理/演示模式覆盖（与旧版相同） =====
    if play:
        # 几乎无限的 episode 长度
        cfg.episode_length_s = int(1e9)

        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        # 禁用 RSI 随机化
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}
        motion_cmd.sampling_mode = "start"

    return cfg


# 完整的新版粗糙地形配置函数
def unitree_g1_new_rough_tracking_env_cfg(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 (mode_15) rough terrain tracking configuration."""
    # 先获取新版平坦配置
    cfg = unitree_g1_new_flat_tracking_env_cfg(has_state_estimation, play)
    
    # 覆盖地形为粗糙地形生成器
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = CUSTOM_TERRAIN_CFG

    # 推力强化
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].interval_range_s = (0.5, 1.5)
        cfg.events["push_robot"].params["velocity_range"] = {
            "x": (-1.5, 1.5),
            "y": (-1.5, 1.5),
            "z": (-0.5, 0.5),
            "roll": (-2.0, 2.0),
            "pitch": (-2.0, 2.0),
            "yaw": (-2.0, 2.0),
        }

    # 摩擦力范围扩大
    if "foot_friction" in cfg.events:
        cfg.events["foot_friction"].params["ranges"] = (0.1, 1.5)
    
    return cfg

