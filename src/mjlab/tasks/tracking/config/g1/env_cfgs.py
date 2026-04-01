"""Unitree G1 tracking environment configurations."""

from copy import deepcopy

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
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

_FOOT_GEOM_PATTERN = r"^(left|right)_foot[1-7]_collision$"
_TRACKING_BODY_NAMES = (
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
_FLAT_TRACKING_EE_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)
# JumpRough-specific lower-body tracking list disabled for now.
# _LOWER_BODY_TRACKING_BODY_NAMES = (
#   "left_hip_roll_link",
#   "left_knee_link",
#   "left_ankle_roll_link",
#   "right_hip_roll_link",
#   "right_knee_link",
#   "right_ankle_roll_link",
# )

#Modified by czy:修改rough阶段偏移配置，默认自动捕获rough开始训练时的checkpoint步数，单位为PPO轮数
_TRACKING_ROUGH_STAGE_ITERATION_OFFSET: int | None = None
#Modified by czy:修改rough阶段步数偏移配置，内部按common_step_counter进行阶段判断，None表示自动捕获
_TRACKING_ROUGH_STAGE_STEP_OFFSET = (
  None
  if _TRACKING_ROUGH_STAGE_ITERATION_OFFSET is None
  else _TRACKING_ROUGH_STAGE_ITERATION_OFFSET * 24
)

#Modified by czy:修改rough四阶段地形课程配置，阶段切换改为相对rough开始训练后的0/3k/6k/9k轮
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


def _configure_g1_tracking_cfg(
  cfg: ManagerBasedRlEnvCfg,
  has_state_estimation: bool,
) -> MotionCommandCfg:
  """Apply G1-specific robot, sensor, and observation settings.

  Keeping these shared between flat and rough variants avoids two task definitions
  drifting apart in unrelated details such as body lists, self-collision sensors,
  and observation layout.
  """
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
  motion_cmd.body_names = _TRACKING_BODY_NAMES

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _FOOT_GEOM_PATTERN
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.terminations["ee_body_pos"].params["body_names"] = _FLAT_TRACKING_EE_BODY_NAMES
  cfg.viewer.body_name = "torso_link"

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

  # Disable random start initialization so we can inspect the learned motion from
  # the first frame rather than from perturbed states.
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.sampling_mode = "start"


def _apply_randomized_terrain_play_overrides(
  cfg: ManagerBasedRlEnvCfg,
  motion_cmd: MotionCommandCfg,
  *,
  num_rows: int,
  num_cols: int,
  border_width: float = 10.0,
  within_patch_xy_range: tuple[float, float] | None = None,
) -> None:
  """Enable randomized terrain sampling during play for terrain-based tasks.

  We still want deterministic policy evaluation from the first motion frame, but
  we also want each reset to sample a new terrain patch so the rough-terrain task
  can be visually inspected on multiple randomized layouts. Optional within-patch
  XY offsets move the full reference motion around inside the patch, which makes
  the randomized terrain coverage more obvious during play.
  """
  _apply_tracking_play_overrides(cfg, motion_cmd)
  randomize_params = {}
  if within_patch_xy_range is not None:
    randomize_params["within_patch_xy_range"] = within_patch_xy_range
  cfg.events["randomize_terrain"] = EventTermCfg(
    func=envs_mdp.randomize_terrain,
    mode="reset",
    params=randomize_params,
  )

  if cfg.scene.terrain is not None:
    terrain_generator = cfg.scene.terrain.terrain_generator
    if terrain_generator is not None:
      terrain_generator.curriculum = False
      terrain_generator.num_rows = num_rows
      terrain_generator.num_cols = num_cols
      terrain_generator.border_width = border_width


def _apply_staged_terrain_play_overrides(
  cfg: ManagerBasedRlEnvCfg,
  motion_cmd: MotionCommandCfg,
  *,
  num_rows: int,
  num_cols: int,
  border_width: float = 10.0,
) -> None:
  """Use the same staged terrain sampler during play as during training.

  This keeps terrain visualization aligned with the curriculum stage restored
  from the loaded checkpoint while still using deterministic play-time motion
  settings for qualitative inspection.
  """
  _apply_tracking_play_overrides(cfg, motion_cmd)
  #Modified by czy:增添play阶段地形同步展示逻辑，使play按checkpoint恢复的课程阶段展示rough地形
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


# JumpRough-specific stratified play helper disabled for now.
# def _apply_stratified_terrain_play_overrides(
#   cfg: ManagerBasedRlEnvCfg,
#   motion_cmd: MotionCommandCfg,
#   *,
#   num_envs: int,
#   num_rows: int,
#   num_cols: int,
#   border_width: float = 10.0,
#   patch_margin: float = 0.75,
# ) -> None:
#   """Use a small, clean batch for play-time terrain coverage visualization.
#
#   Play mode is primarily for qualitative inspection, so we intentionally reduce
#   the number of environments and lay them out evenly over the visible terrain
#   patches. This makes the terrain coverage easy to read in the viewer.
#   """
#   _apply_tracking_play_overrides(cfg, motion_cmd)
#   cfg.scene.num_envs = num_envs
#   cfg.events["stratified_terrain_placement"] = EventTermCfg(
#     func=envs_mdp.stratified_terrain_placement,
#     mode="reset",
#     params={
#       "patch_margin": patch_margin,
#     },
#   )
#
#   if cfg.scene.terrain is not None:
#     terrain_generator = cfg.scene.terrain.terrain_generator
#     if terrain_generator is not None:
#       terrain_generator.curriculum = False
#       terrain_generator.num_rows = num_rows
#       terrain_generator.num_cols = num_cols
#       terrain_generator.border_width = border_width


def _make_tracking_rough_terrain_cfg():
  """Create a milder rough-terrain curriculum for motion tracking.

  Tracking jump motions is more fragile than velocity locomotion. The first rough
  version should therefore bias toward gentle terrain and retain some flat patches,
  so the policy can keep the reference motion amplitude while learning terrain
  adaptation instead of immediately collapsing to a conservative gait.
  """
  terrain_cfg = deepcopy(ROUGH_TERRAINS_CFG)
  terrain_cfg.curriculum = True
  terrain_cfg.num_rows = 6
  terrain_cfg.num_cols = 12

  #Modified by czy:删除rough中的台阶与坡面地形，仅保留flat、random_rough、wave_terrain
  terrain_cfg.sub_terrains.pop("pyramid_stairs", None)
  terrain_cfg.sub_terrains.pop("pyramid_stairs_inv", None)
  terrain_cfg.sub_terrains.pop("hf_pyramid_slope", None)
  terrain_cfg.sub_terrains.pop("hf_pyramid_slope_inv", None)

  #Modified by czy:修改rough三类地形的基础列分布，使最终wave主导阶段拥有更多列级空间多样性
  terrain_cfg.sub_terrains["flat"].proportion = 0.25
  terrain_cfg.sub_terrains["random_rough"].proportion = 0.25
  terrain_cfg.sub_terrains["random_rough"].noise_range = (0.01, 0.04)
  terrain_cfg.sub_terrains["wave_terrain"].proportion = 0.50
  terrain_cfg.sub_terrains["wave_terrain"].amplitude_range = (0.0, 0.08)
  return terrain_cfg


# JumpRough-specific terrain curriculum disabled for now; keep the general rough
# curriculum as the only terrain-based tracking variant.
# def _make_tracking_jump_rough_terrain_cfg():
#   """Create a jump-specific rough terrain curriculum.
#
#   This variant is intentionally milder than the general rough task. Jump motions
#   need the take-off and landing patches to remain physically close to the flat
#   reference, so we keep most samples flat and only inject small slopes, shallow
#   roughness, and low-amplitude waves. We explicitly disable stairs here because
#   they tend to break jump-height imitation before they improve robustness.
#   """
#   terrain_cfg = deepcopy(ROUGH_TERRAINS_CFG)
#   terrain_cfg.curriculum = True
#   terrain_cfg.num_rows = 5
#   terrain_cfg.num_cols = 10
#
#   terrain_cfg.sub_terrains["flat"].proportion = 0.5  # modified:jump-rough 0.7→0.5
#   terrain_cfg.sub_terrains["pyramid_stairs"].proportion = 0.0
#   terrain_cfg.sub_terrains["pyramid_stairs_inv"].proportion = 0.0
#   terrain_cfg.sub_terrains["hf_pyramid_slope"].proportion = 0.15  # modified:jump-rough 0.1→0.15
#   terrain_cfg.sub_terrains["hf_pyramid_slope"].slope_range = (0.0, 0.12)
#   terrain_cfg.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.15  # modified:jump-rough 0.1→0.15
#   terrain_cfg.sub_terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.12)
#   terrain_cfg.sub_terrains["random_rough"].proportion = 0.12  # modified:jump-rough 0.07→0.12
#   terrain_cfg.sub_terrains["random_rough"].noise_range = (0.005, 0.02)
#   terrain_cfg.sub_terrains["wave_terrain"].proportion = 0.08  # modified:jump-rough 0.03→0.08
#   terrain_cfg.sub_terrains["wave_terrain"].amplitude_range = (0.0, 0.03)
#   return terrain_cfg


def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()
  motion_cmd = _configure_g1_tracking_cfg(cfg, has_state_estimation)

  if play:
    _apply_tracking_play_overrides(cfg, motion_cmd)

  return cfg


def unitree_g1_rough_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain tracking configuration.

  Design goals for the first rough variant:
  1. Keep the motion recognizable by avoiding world-space z tracking penalties.
  2. Expose the robot to mild terrain variation and mild domain randomization.
  3. Avoid early hard failures from z-only terminations that assume a flat floor.
  """
  cfg = make_tracking_env_cfg()
  motion_cmd = _configure_g1_tracking_cfg(cfg, has_state_estimation)

  #Modified by czy:修改rough为仅保留三类地形的程序化terrain，并从最简单行开始初始化
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=_make_tracking_rough_terrain_cfg(),
    max_init_terrain_level=0,
  )
  cfg.scene.extent = 2.0

  if not play:
    #Modified by czy:增添rough四阶段reset地形采样课程，随着训练推进降低flat占比并提高wave占比
    cfg.events["staged_terrain_sampling"] = EventTermCfg(
      func=mdp.staged_tracking_terrain_sampling,
      mode="reset",
      params={
        "stages": _TRACKING_ROUGH_TERRAIN_STAGES,
        "rough_stage_step_offset": _TRACKING_ROUGH_STAGE_STEP_OFFSET,
        "auto_capture_offset": True,
      },
    )

  # Rough terrain creates more simultaneous contacts than flat ground. Raising
  # these limits reduces the chance of contact truncation becoming the hidden
  # bottleneck when the robot lands on edges or uneven patches.
  cfg.sim.nconmax = 60
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.mujoco.ccd_iterations = 200

  # On rough terrain, absolute world-space z no longer cleanly reflects tracking
  # quality. We therefore keep root position tracking in XY, keep the root-z
  # velocity term for jump timing, and add a soft root-z position term so the
  # policy is still encouraged to reach the reference jump apex.
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
    weight=0.75,  # modified:rough-stage2 add soft root-z tracking
    params={"command_name": "motion", "std": 0.2},
  )

  # Rough-stage2 modification: strengthen continuous orientation supervision so
  # the robot is pushed toward a cleaner landing posture instead of relying on a
  # late compensatory step after touchdown.
  cfg.rewards["motion_global_root_ori"].weight = 1.0  # modified:rough-stage2 0.5→1.0
  cfg.rewards["motion_body_ori"].weight = 1.5  # modified:rough-stage2 1.0→1.5

  # Mild domain randomization helps robustness, but we intentionally keep it
  # narrower than the flat task and remove push disturbances. The policy first
  # needs to preserve the reference jump amplitude while adapting to terrain.
  cfg.events.pop("push_robot", None)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.015, 0.015),
    1: (-0.03, 0.03),
    2: (-0.03, 0.03),
  }
  cfg.events["encoder_bias"].params["bias_range"] = (-0.005, 0.005)
  cfg.events["foot_friction"].params["ranges"] = (0.5, 1.0)

  # These flat-terrain terminations compare world-space z against a flat-motion
  # reference and would incorrectly end rough-terrain episodes just because the
  # robot is standing on a bump or landing in a depression.
  cfg.terminations.pop("anchor_pos", None)
  cfg.terminations.pop("ee_body_pos", None)

  # Rough-stage2 modification: let the policy survive slightly larger torso tilt
  # errors during the landing transition. The goal is to avoid terminating just
  # before touchdown while the stronger orientation rewards pull it back toward a
  # cleaner landing pose.
  cfg.terminations["anchor_ori"].params["threshold"] = 1.2  # modified:rough-stage2 0.8→1.2

  if play:
    #Modified by czy:修改play地形展示逻辑，使其与rough课程阶段同步，而不是独立随机采样地形
    _apply_staged_terrain_play_overrides(
      cfg,
      motion_cmd,
      num_rows=6,
      num_cols=6,
      border_width=10.0,
    )

  return cfg

# JumpRough env config disabled for now; keep the general rough-terrain task only.
# def unitree_g1_jump_rough_tracking_env_cfg(
#   has_state_estimation: bool = True,
#   play: bool = False,
# ) -> ManagerBasedRlEnvCfg:
#   """Create a jump-specific rough terrain tracking configuration.
#
#   Design goals for JumpRough:
#   1. Preserve the reference jump amplitude more aggressively than the general
#      rough task.
#   2. Use only mild terrain variation so take-off and landing stay close enough
#      to the flat reference motion to remain imitable.
#   3. Keep play-time terrain randomization enabled so visual evaluation covers
#      multiple sampled terrain patches instead of a single fixed layout.
#   """
#   cfg = make_tracking_env_cfg()
#   motion_cmd = _configure_g1_tracking_cfg(cfg, has_state_estimation)
#
#   # Jump-specific rough keeps the same terrain-generator pathway as the general
#   # rough task, but the actual terrain mix is much gentler and starts from the
#   # easiest curriculum row. This biases the task toward preserving jump height
#   # instead of solving arbitrary locomotion-like terrain disturbances.
#   cfg.scene.terrain = TerrainEntityCfg(
#     terrain_type="generator",
#     terrain_generator=_make_tracking_jump_rough_terrain_cfg(),
#     max_init_terrain_level=4,  # modified:jump-rough 2→4 for full terrain-row coverage
#   )
#   cfg.scene.extent = 2.0
#
#   # Keep the higher contact budget because even small terrain edges can create
#   # extra contacts during jump take-off and landing.
#   cfg.sim.nconmax = 60
#   cfg.sim.contact_sensor_maxmatch = 128
#   cfg.sim.mujoco.ccd_iterations = 200
#
#   # Reuse the jump-oriented rough rewards: XY root tracking for terrain tolerance,
#   # root-z velocity for timing, and soft root-z position for preserving apex
#   # height. We make the height term slightly stronger here because JumpRough is
#   # explicitly optimized for staying closer to the reference jump.
#   cfg.rewards["motion_global_root_pos"] = RewardTermCfg(
#     func=mdp.motion_global_anchor_xy_position_error_exp,
#     weight=1.0,
#     params={"command_name": "motion", "std": 0.4},
#   )
#   cfg.rewards["motion_global_root_z_vel"] = RewardTermCfg(
#     func=mdp.motion_global_anchor_z_velocity_error_exp,
#     weight=1.0,
#     params={"command_name": "motion", "std": 1.0},
#   )
#   cfg.rewards["motion_global_root_z_pos"] = RewardTermCfg(
#     func=mdp.motion_global_anchor_z_position_error_exp,
#     weight=1.0,  # modified:jump-rough add stronger soft root-z tracking
#     params={"command_name": "motion", "std": 0.18},
#   )
#   cfg.rewards["motion_global_root_ori"].weight = 1.0  # modified:jump-rough 0.5→1.0
#   cfg.rewards["motion_body_ori"].weight = 1.5  # modified:jump-rough 1.0→1.5
#   cfg.rewards["motion_lower_body_pos"] = RewardTermCfg(
#     func=mdp.motion_relative_body_position_error_exp,
#     weight=0.75,  # modified:jump-rough-stage3 add lower-limb pose tracking
#     params={
#       "command_name": "motion",
#       "std": 0.2,
#       "body_names": _LOWER_BODY_TRACKING_BODY_NAMES,
#     },
#   )
#   cfg.rewards["motion_joint_pos"] = RewardTermCfg(
#     func=mdp.motion_joint_position_error_exp,
#     weight=0.5,  # modified:jump-rough-stage3 add joint-space pose tracking
#     params={"command_name": "motion", "std": 0.25},
#   )
#   cfg.rewards["motion_joint_vel"] = RewardTermCfg(
#     func=mdp.motion_joint_velocity_error_exp,
#     weight=0.15,  # modified:jump-rough-stage3 add joint-space velocity tracking
#     params={"command_name": "motion", "std": 3.0},
#   )
#
#   # Keep domain randomization, but narrow it further than the general rough task.
#   # The target here is robustness to light terrain mismatch while preserving jump
#   # height, not full terrain-agnostic behavior.
#   cfg.events.pop("push_robot", None)
#   # JumpRough reset modification: spread the batch evenly over all active terrain
#   # patches and place envs on a local grid within each patch. This gives much more
#   # consistent terrain coverage than IID random sampling, especially with 4096 envs.
#   cfg.events["stratified_terrain_placement"] = EventTermCfg(
#     func=envs_mdp.stratified_terrain_placement,
#     mode="reset",
#     params={
#       "max_terrain_level": 4,  # modified:jump-rough 2→4 to cover every terrain row
#       "patch_margin": 0.75,
#       "local_grid_jitter": 0.15,  # modified:jump-rough 0.0→0.15 for in-patch coverage over time
#       "reshuffle_every_n_resets": 64,  # modified:jump-rough 0→64 for periodic slot refresh
#       "shuffle_patch_order": False,
#     },
#   )
#   cfg.events["base_com"].params["ranges"] = {
#     0: (-0.01, 0.01),
#     1: (-0.02, 0.02),
#     2: (-0.02, 0.02),
#   }
#   cfg.events["encoder_bias"].params["bias_range"] = (-0.003, 0.003)
#   cfg.events["foot_friction"].params["ranges"] = (0.7, 1.0)
#
#   # The flat-specific z-only terminations are still inappropriate on terrain, so
#   # we remove them here as well. We keep the looser torso-orientation threshold
#   # so the policy can survive the landing transition and recover without being
#   # truncated immediately.
#   cfg.terminations.pop("anchor_pos", None)
#   cfg.terminations.pop("ee_body_pos", None)
#   cfg.terminations["anchor_ori"].params["threshold"] = 1.2  # modified:jump-rough 0.8→1.2
#
#   if play:
#     _apply_stratified_terrain_play_overrides(
#       cfg,
#       motion_cmd,
#       num_envs=16,
#       num_rows=4,
#       num_cols=4,
#       border_width=10.0,
#       patch_margin=0.75,
#     )
#
#   return cfg
