from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.actuator import BuiltinPositionActuator, IdealPdActuator, XmlPositionActuator
from mjlab.actuator.delayed_actuator import DelayedActuator
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_yaw,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)
from mjlab.utils.lab_api.string import resolve_matching_names

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


#Modified by czy:增添tracking rough四阶段地形reset采样事件
class TerrainSamplingStage(TypedDict):
  step: int
  max_terrain_level: int
  terrain_type_probs: dict[str, float]


def _resolve_tracking_terrain_stage(
  stages: list[TerrainSamplingStage],
  common_step_counter: int,
) -> TerrainSamplingStage:
  active_stage = stages[0]
  for stage in stages:
    if common_step_counter >= stage["step"]:
      active_stage = stage
  return active_stage


def _terrain_type_columns(terrain_generator_cfg) -> dict[str, list[int]]:
  names = list(terrain_generator_cfg.sub_terrains.keys())
  proportions = [sub_cfg.proportion for sub_cfg in terrain_generator_cfg.sub_terrains.values()]
  total = sum(proportions)
  if total <= 0.0:
    raise ValueError("Terrain proportions must sum to a positive value.")

  cumulative: list[float] = []
  running = 0.0
  for proportion in proportions:
    running += proportion / total
    cumulative.append(running)

  terrain_type_columns = {name: [] for name in names}
  for index in range(terrain_generator_cfg.num_cols):
    threshold = index / terrain_generator_cfg.num_cols + 0.001
    for sub_index, upper in enumerate(cumulative):
      if threshold < upper:
        terrain_type_columns[names[sub_index]].append(index)
        break

  return terrain_type_columns


class staged_tracking_terrain_sampling:
  #Modified by czy:增添带状态的tracking rough阶段采样器，支持自动捕获rough起始offset
  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._captured_stage_step_offset: int | None = None

  #Modified by czy:增添rough阶段采样器状态导出接口，用于checkpoint保存与play恢复
  def get_state(self) -> dict[str, int | None]:
    return {"captured_stage_step_offset": self._captured_stage_step_offset}

  #Modified by czy:增添rough阶段采样器状态恢复接口，用于checkpoint加载后同步课程阶段
  def set_state(self, state: dict[str, int | None] | None) -> None:
    if state is None:
      self._captured_stage_step_offset = None
      return
    self._captured_stage_step_offset = state.get("captured_stage_step_offset")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    stages: list[TerrainSamplingStage],
    rough_stage_step_offset: int | None = None,
    auto_capture_offset: bool = True,
    within_patch_xy_range: tuple[float, float] | None = None,
  ) -> None:
    """Sample tracking terrains by training stage on each reset."""
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
      env_ids = env_ids.to(device=env.device, dtype=torch.long)

    if env_ids.numel() == 0:
      return

    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None or terrain.env_origins is None:
      return

    terrain_generator_cfg = terrain.cfg.terrain_generator
    if terrain_generator_cfg is None:
      return

    #Modified by czy:修改tracking rough阶段判断逻辑，优先使用手动offset，否则自动捕获rough起点
    if rough_stage_step_offset is not None:
      resolved_offset = int(rough_stage_step_offset)
      self._captured_stage_step_offset = resolved_offset
    elif auto_capture_offset:
      if self._captured_stage_step_offset is None:
        self._captured_stage_step_offset = int(env.common_step_counter)
      resolved_offset = self._captured_stage_step_offset
    else:
      resolved_offset = 0

    effective_step = max(env.common_step_counter - resolved_offset, 0)
    active_stage = _resolve_tracking_terrain_stage(stages, effective_step)
    terrain_type_columns = _terrain_type_columns(terrain_generator_cfg)

    valid_type_names = [
      name
      for name, weight in active_stage["terrain_type_probs"].items()
      if weight > 0.0 and len(terrain_type_columns.get(name, [])) > 0
    ]
    if not valid_type_names:
      raise ValueError("No active terrain types available for staged terrain sampling.")

    type_weights = torch.tensor(
      [active_stage["terrain_type_probs"][name] for name in valid_type_names],
      device=env.device,
      dtype=torch.float32,
    )
    type_weights /= torch.sum(type_weights)

    sampled_type_ids = torch.multinomial(
      type_weights,
      num_samples=env_ids.numel(),
      replacement=True,
    )
    sampled_terrain_types = torch.empty_like(env_ids)

    for local_type_id, terrain_type_name in enumerate(valid_type_names):
      assigned_mask = sampled_type_ids == local_type_id
      if not torch.any(assigned_mask):
        continue
      candidate_columns = torch.tensor(
        terrain_type_columns[terrain_type_name],
        device=env.device,
        dtype=torch.long,
      )
      sampled_columns = candidate_columns[
        torch.randint(
          0,
          candidate_columns.numel(),
          (int(torch.sum(assigned_mask).item()),),
          device=env.device,
        )
      ]
      sampled_terrain_types[assigned_mask] = sampled_columns

    max_level = min(active_stage["max_terrain_level"], terrain.terrain_origins.shape[0] - 1)
    sampled_terrain_levels = torch.randint(
      0,
      max_level + 1,
      (env_ids.numel(),),
      device=env.device,
    )

    terrain.terrain_levels[env_ids] = sampled_terrain_levels
    terrain.terrain_types[env_ids] = sampled_terrain_types
    terrain.env_origins[env_ids] = terrain.terrain_origins[
      terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]

    if within_patch_xy_range is not None:
      lower, upper = within_patch_xy_range
      offsets_xy = sample_uniform(lower, upper, (env_ids.numel(), 2), device=env.device)
      terrain.env_origins[env_ids, :2] += offsets_xy


class late_phase_play_disturbance:
  """Inject late-phase DR plus stand-up overshoot/underpowered recovery errors."""

  model_fields = ("actuator_forcerange", "actuator_gainprm", "actuator_biasprm")

  _OVERSHOOT_MODE = 0
  _UNDERPOWERED_MODE = 1

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._asset = env.scene[self._asset_cfg.name]
    self._body_ids = self._asset_cfg.body_ids
    self._num_bodies = (
      len(self._body_ids) if isinstance(self._body_ids, list) else self._asset.num_bodies
    )
    self._stand_up_push_body_cfg: SceneEntityCfg = cfg.params["stand_up_push_body_cfg"]
    self._stand_up_push_body_ids = self._stand_up_push_body_cfg.body_ids
    self._stand_up_push_num_bodies = (
      len(self._stand_up_push_body_ids)
      if isinstance(self._stand_up_push_body_ids, list)
      else self._asset.num_bodies
    )
    self._stand_up_underpowered_actuator_cfg: SceneEntityCfg = cfg.params[
      "stand_up_underpowered_actuator_cfg"
    ]
    self._stand_up_underpowered_asset = env.scene[
      self._stand_up_underpowered_actuator_cfg.name
    ]
    self._stand_up_underpowered_actuators = self._resolve_actuators(
      self._stand_up_underpowered_asset,
      self._stand_up_underpowered_actuator_cfg,
    )
    actuator_name_patterns = self._stand_up_underpowered_actuator_cfg.actuator_names
    if actuator_name_patterns is None:
      self._stand_up_action_target_patterns: list[str] = [".*"]
    elif isinstance(actuator_name_patterns, str):
      self._stand_up_action_target_patterns = [actuator_name_patterns]
    else:
      self._stand_up_action_target_patterns = list(actuator_name_patterns)
    self._joint_pos_action_term = None
    self._stand_up_action_target_ids: torch.Tensor | None = None
    self._stand_up_action_default_scale: torch.Tensor | None = None
    self._sim = env.sim
    self._num_envs = env.num_envs
    self._device = env.device
    self._step_dt = env.step_dt
    body_point_offset = cfg.params.get("stand_up_push_body_point_offset", None)
    self._stand_up_push_body_point_offset: torch.Tensor | None = (
      torch.tensor(body_point_offset, device=self._device, dtype=torch.float32)
      if body_point_offset is not None
      else None
    )

    self._state_cooldown = torch.zeros(self._num_envs, device=self._device)
    self._impulse_cooldown = torch.zeros(self._num_envs, device=self._device)
    self._impulse_time_left = torch.zeros(self._num_envs, device=self._device)
    self._impulse_active = torch.zeros(
      self._num_envs, device=self._device, dtype=torch.bool
    )
    self._stand_up_push_time_left = torch.zeros(self._num_envs, device=self._device)
    self._stand_up_push_active = torch.zeros(
      self._num_envs, device=self._device, dtype=torch.bool
    )
    self._stand_up_underpowered_time_left = torch.zeros(
      self._num_envs, device=self._device
    )
    self._stand_up_underpowered_active = torch.zeros(
      self._num_envs, device=self._device, dtype=torch.bool
    )
    self._stand_up_recovery_triggered = torch.zeros(
      self._num_envs, device=self._device, dtype=torch.bool
    )
    self._stand_up_recovery_mode = torch.full(
      (self._num_envs,), -1, device=self._device, dtype=torch.long
    )
    self._stand_up_overshoot_trigger_frame = torch.full(
      (self._num_envs,), -1, device=self._device, dtype=torch.long
    )
    self._stand_up_underpowered_trigger_frame = torch.full(
      (self._num_envs,), -1, device=self._device, dtype=torch.long
    )
    self._stand_up_velocity_kick_steps_left = torch.zeros(
      self._num_envs, device=self._device, dtype=torch.long
    )
    self._stand_up_velocity_kick_world_delta = torch.zeros(
      (self._num_envs, 3), device=self._device
    )
    self._stand_up_pitch_kick_steps_left = torch.zeros(
      self._num_envs, device=self._device, dtype=torch.long
    )
    self._stand_up_pitch_kick_world_delta = torch.zeros(
      (self._num_envs, 3), device=self._device
    )
    self._previous_motion_time_steps = torch.full(
      (self._num_envs,), -1, device=self._device, dtype=torch.long
    )

  def _resolve_actuators(
    self,
    asset,
    actuator_cfg: SceneEntityCfg,
  ) -> list:
    actuator_ids = actuator_cfg.actuator_ids
    selected_actuators: list = []

    if isinstance(actuator_ids, slice):
      requested_ctrl_ids = set(range(asset.num_actuators)[actuator_ids])
    elif isinstance(actuator_ids, list):
      requested_ctrl_ids = set(int(actuator_id) for actuator_id in actuator_ids)
    else:
      requested_ctrl_ids = {int(actuator_ids)}

    for actuator in asset.actuators:
      actuator_ctrl_ids = actuator.ctrl_ids.detach().cpu().tolist()
      if requested_ctrl_ids.intersection(actuator_ctrl_ids):
        selected_actuators.append(actuator)

    return [
      actuator.base_actuator if isinstance(actuator, DelayedActuator) else actuator
      for actuator in selected_actuators
    ]

  def _ensure_joint_pos_action_term(self, env: ManagerBasedRlEnv):
    if self._joint_pos_action_term is not None:
      return self._joint_pos_action_term

    joint_pos_action = env.action_manager.get_term("joint_pos")
    action_scale = joint_pos_action.scale
    if isinstance(action_scale, (float, int)):
      joint_pos_action._scale = torch.full(
        (self._num_envs, joint_pos_action.action_dim),
        float(action_scale),
        device=self._device,
      )
    else:
      joint_pos_action._scale = action_scale.clone()

    action_target_ids, _ = resolve_matching_names(
      self._stand_up_action_target_patterns,
      joint_pos_action.target_names,
      preserve_order=False,
    )
    self._joint_pos_action_term = joint_pos_action
    self._stand_up_action_target_ids = torch.tensor(
      action_target_ids,
      device=self._device,
      dtype=torch.long,
    )
    self._stand_up_action_default_scale = joint_pos_action._scale.clone()
    return joint_pos_action

  def _eligible_env_ids(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    start_ratio: float,
  ) -> torch.Tensor:
    from mjlab.tasks.tracking.mdp.commands import MotionCommand

    motion_command = cast(MotionCommand, env.command_manager.get_term(command_name))
    start_ratio = min(max(float(start_ratio), 0.0), 1.0)
    start_step = int((motion_command.motion.time_step_total - 1) * start_ratio)
    return torch.nonzero(motion_command.time_steps >= start_step, as_tuple=False).squeeze(-1)

  def _progress_scales(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    late_phase_start_ratio: float,
    late_phase_onset_scale: float,
    late_phase_scale_power: float,
  ) -> torch.Tensor:
    from mjlab.tasks.tracking.mdp.commands import MotionCommand

    motion_command = cast(MotionCommand, env.command_manager.get_term(command_name))
    if env_ids.numel() == 0:
      return torch.zeros(0, device=self._device)

    max_time_step = max(motion_command.motion.time_step_total - 1, 1)
    progress = motion_command.time_steps[env_ids].float() / float(max_time_step)
    late_phase_start_ratio = min(max(float(late_phase_start_ratio), 0.0), 0.999)
    normalized = (progress - late_phase_start_ratio) / max(
      1.0 - late_phase_start_ratio, 1.0e-6
    )
    normalized = normalized.clamp_(0.0, 1.0) ** max(
      float(late_phase_scale_power), 1.0e-6
    )
    onset_scale = min(max(float(late_phase_onset_scale), 0.0), 1.0)
    return onset_scale + (1.0 - onset_scale) * normalized

  def _sample_timer(
    self,
    env_ids: torch.Tensor,
    time_range_s: tuple[float, float],
  ) -> None:
    lower, upper = time_range_s
    self._state_cooldown[env_ids] = (
      torch.rand(len(env_ids), device=self._device) * (upper - lower) + lower
    )

  def _sample_impulse_cooldown(
    self,
    env_ids: torch.Tensor,
    cooldown_s: tuple[float, float],
  ) -> None:
    lower, upper = cooldown_s
    self._impulse_cooldown[env_ids] = (
      torch.rand(len(env_ids), device=self._device) * (upper - lower) + lower
    )

  def _clear_impulses(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    zeros = torch.zeros((len(env_ids), self._num_bodies, 3), device=self._device)
    self._asset.write_external_wrench_to_sim(
      zeros, zeros, env_ids=env_ids, body_ids=self._body_ids
    )
    self._impulse_active[env_ids] = False
    self._impulse_time_left[env_ids] = 0.0

  def _clear_stand_up_pushes(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    zeros = torch.zeros(
      (len(env_ids), self._stand_up_push_num_bodies, 3),
      device=self._device,
    )
    self._asset.write_external_wrench_to_sim(
      zeros,
      zeros,
      env_ids=env_ids,
      body_ids=self._stand_up_push_body_ids,
    )
    self._stand_up_push_active[env_ids] = False
    self._stand_up_push_time_left[env_ids] = 0.0

  def _set_stand_up_action_scales(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    action_scales: torch.Tensor,
  ) -> None:
    if env_ids.numel() == 0:
      return
    joint_pos_action = self._ensure_joint_pos_action_term(env)
    assert self._stand_up_action_target_ids is not None
    assert self._stand_up_action_default_scale is not None
    joint_pos_action._scale[
      env_ids[:, None], self._stand_up_action_target_ids
    ] = (
      self._stand_up_action_default_scale[
        env_ids[:, None], self._stand_up_action_target_ids
      ]
      * action_scales.unsqueeze(1)
    )

  def _clear_stand_up_action_scales(
    self,
    env: ManagerBasedRlEnv | None,
    env_ids: torch.Tensor,
  ) -> None:
    if env_ids.numel() == 0:
      return
    if self._joint_pos_action_term is None:
      if env is None:
        return
      self._ensure_joint_pos_action_term(env)
    assert self._stand_up_action_target_ids is not None
    assert self._stand_up_action_default_scale is not None
    self._joint_pos_action_term._scale[
      env_ids[:, None], self._stand_up_action_target_ids
    ] = self._stand_up_action_default_scale[
      env_ids[:, None], self._stand_up_action_target_ids
    ]

  def _set_stand_up_underpowered_scales(
    self,
    env_ids: torch.Tensor,
    effort_scales: torch.Tensor,
    kp_scales: torch.Tensor,
    kd_scales: torch.Tensor,
  ) -> None:
    if env_ids.numel() == 0 or not self._stand_up_underpowered_actuators:
      return

    for actuator in self._stand_up_underpowered_actuators:
      ctrl_ids = actuator.global_ctrl_ids
      if isinstance(actuator, (BuiltinPositionActuator, XmlPositionActuator)):
        default_forcerange = self._sim.get_default_field("actuator_forcerange")
        self._sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 0] = (
          default_forcerange[ctrl_ids, 0] * effort_scales.unsqueeze(1)
        )
        self._sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 1] = (
          default_forcerange[ctrl_ids, 1] * effort_scales.unsqueeze(1)
        )

        default_gainprm = self._sim.get_default_field("actuator_gainprm")
        default_biasprm = self._sim.get_default_field("actuator_biasprm")
        self._sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = (
          default_gainprm[ctrl_ids, 0] * kp_scales.unsqueeze(1)
        )
        self._sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = (
          default_biasprm[ctrl_ids, 1] * kp_scales.unsqueeze(1)
        )
        self._sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = (
          default_biasprm[ctrl_ids, 2] * kd_scales.unsqueeze(1)
        )
      elif isinstance(actuator, IdealPdActuator):
        assert actuator.default_force_limit is not None
        assert actuator.default_stiffness is not None
        assert actuator.default_damping is not None
        actuator.set_effort_limit(
          env_ids,
          effort_limit=actuator.default_force_limit[env_ids]
          * effort_scales.unsqueeze(1),
        )
        actuator.set_gains(
          env_ids,
          kp=actuator.default_stiffness[env_ids] * kp_scales.unsqueeze(1),
          kd=actuator.default_damping[env_ids] * kd_scales.unsqueeze(1),
        )
      else:
        raise TypeError(
          "stand_up_underpowered only supports BuiltinPositionActuator, "
          "XmlPositionActuator, and IdealPdActuator."
        )

  def _clear_stand_up_underpowered(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    ones = torch.ones(len(env_ids), device=self._device)
    self._set_stand_up_underpowered_scales(env_ids, ones, ones, ones)
    self._stand_up_underpowered_active[env_ids] = False
    self._stand_up_underpowered_time_left[env_ids] = 0.0

  def _clear_stand_up_velocity_kicks(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    self._stand_up_velocity_kick_steps_left[env_ids] = 0
    self._stand_up_velocity_kick_world_delta[env_ids] = 0.0

  def _clear_stand_up_pitch_kicks(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    self._stand_up_pitch_kick_steps_left[env_ids] = 0
    self._stand_up_pitch_kick_world_delta[env_ids] = 0.0

  def _reapply_stand_up_velocity_kicks(self) -> None:
    active_ids = torch.nonzero(
      self._stand_up_velocity_kick_steps_left > 0, as_tuple=False
    ).squeeze(-1)
    if active_ids.numel() == 0:
      return

    root_link_vel = self._asset.data.root_link_vel_w[active_ids].clone()
    root_link_vel[:, 0:3] += self._stand_up_velocity_kick_world_delta[active_ids]
    self._asset.write_root_link_velocity_to_sim(root_link_vel, env_ids=active_ids)
    self._stand_up_velocity_kick_steps_left[active_ids] -= 1

    expired_ids = active_ids[
      self._stand_up_velocity_kick_steps_left[active_ids] <= 0
    ]
    self._clear_stand_up_velocity_kicks(expired_ids)

  def _reapply_stand_up_pitch_kicks(self) -> None:
    active_ids = torch.nonzero(
      self._stand_up_pitch_kick_steps_left > 0, as_tuple=False
    ).squeeze(-1)
    if active_ids.numel() == 0:
      return

    root_link_vel = self._asset.data.root_link_vel_w[active_ids].clone()
    root_link_vel[:, 3:6] += self._stand_up_pitch_kick_world_delta[active_ids]
    self._asset.write_root_link_velocity_to_sim(root_link_vel, env_ids=active_ids)
    self._stand_up_pitch_kick_steps_left[active_ids] -= 1

    expired_ids = active_ids[
      self._stand_up_pitch_kick_steps_left[active_ids] <= 0
    ]
    self._clear_stand_up_pitch_kicks(expired_ids)

  def _reset_motion_cycle_state(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    self._clear_impulses(env_ids)
    self._clear_stand_up_pushes(env_ids)
    self._clear_stand_up_velocity_kicks(env_ids)
    self._clear_stand_up_pitch_kicks(env_ids)
    self._state_cooldown[env_ids] = 0.0
    self._impulse_cooldown[env_ids] = 0.0
    self._stand_up_recovery_triggered[env_ids] = False
    self._stand_up_recovery_mode[env_ids] = -1
    self._stand_up_overshoot_trigger_frame[env_ids] = -1
    self._stand_up_underpowered_trigger_frame[env_ids] = -1

  def _apply_state_perturbation(
    self,
    env_ids: torch.Tensor,
    scales: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    joint_position_range: tuple[float, float],
    joint_velocity_range: tuple[float, float],
  ) -> None:
    if env_ids.numel() == 0:
      return

    root_pos = self._asset.data.root_link_pos_w[env_ids].clone()
    root_quat = self._asset.data.root_link_quat_w[env_ids].clone()
    pose_ranges = torch.tensor(
      [
        pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
      ],
      device=self._device,
      dtype=torch.float32,
    )
    pose_lower = pose_ranges[:, 0].unsqueeze(0) * scales.unsqueeze(1)
    pose_upper = pose_ranges[:, 1].unsqueeze(0) * scales.unsqueeze(1)
    pose_samples = sample_uniform(
      pose_lower,
      pose_upper,
      (len(env_ids), 6),
      device=self._device,
    )
    root_pos += pose_samples[:, :3]
    root_quat = quat_mul(
      quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
      ),
      root_quat,
    )
    self._asset.write_root_link_pose_to_sim(
      torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids
    )

    root_vel = self._asset.data.root_link_vel_w[env_ids].clone()
    velocity_ranges = torch.tensor(
      [
        velocity_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
      ],
      device=self._device,
      dtype=torch.float32,
    )
    velocity_lower = velocity_ranges[:, 0].unsqueeze(0) * scales.unsqueeze(1)
    velocity_upper = velocity_ranges[:, 1].unsqueeze(0) * scales.unsqueeze(1)
    velocity_samples = sample_uniform(
      velocity_lower,
      velocity_upper,
      (len(env_ids), 6),
      device=self._device,
    )
    root_vel += velocity_samples
    self._asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

    joint_pos = self._asset.data.joint_pos[env_ids].clone()
    joint_vel = self._asset.data.joint_vel[env_ids].clone()
    joint_pos += sample_uniform(
      joint_position_range[0],
      joint_position_range[1],
      joint_pos.shape,
      self._device,
    ) * scales.unsqueeze(1)
    joint_vel += sample_uniform(
      joint_velocity_range[0],
      joint_velocity_range[1],
      joint_vel.shape,
      self._device,
    ) * scales.unsqueeze(1)
    soft_joint_pos_limits = self._asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(
      soft_joint_pos_limits[..., 0],
      soft_joint_pos_limits[..., 1],
    )
    self._asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

  def _stand_up_window_env_ids(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    center_frame: int,
    half_window: int,
  ) -> torch.Tensor:
    from mjlab.tasks.tracking.mdp.commands import MotionCommand

    motion_command = cast(MotionCommand, env.command_manager.get_term(command_name))
    window_start = max(int(center_frame) - max(int(half_window), 0), 0)
    window_end = min(
      int(center_frame) + max(int(half_window), 0),
      motion_command.motion.time_step_total - 1,
    )
    in_window = (motion_command.time_steps >= window_start) & (
      motion_command.time_steps <= window_end
    )
    return torch.nonzero(in_window, as_tuple=False).squeeze(-1)

  def _apply_stand_up_push(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    longitudinal_force_range: tuple[float, float],
    lateral_force_range: tuple[float, float],
    duration_s: tuple[float, float],
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if env_ids.numel() == 0:
      return (
        torch.zeros((0, 3), device=self._device),
        torch.zeros((0, 3), device=self._device),
      )

    local_forces = torch.zeros((len(env_ids), 3), device=self._device)
    local_forces[:, 0] = sample_uniform(
      longitudinal_force_range[0],
      longitudinal_force_range[1],
      (len(env_ids),),
      device=self._device,
    )
    local_forces[:, 1] = sample_uniform(
      lateral_force_range[0],
      lateral_force_range[1],
      (len(env_ids),),
      device=self._device,
    )

    push_body_id = (
      self._stand_up_push_body_ids[0]
      if isinstance(self._stand_up_push_body_ids, list)
      else 0
    )
    body_quat = self._asset.data.body_com_quat_w[env_ids][:, push_body_id]
    world_forces = quat_apply_yaw(body_quat, local_forces)
    forces = world_forces.unsqueeze(1).repeat(1, self._stand_up_push_num_bodies, 1)
    torques = torch.zeros_like(forces)
    torque_world = torch.zeros((len(env_ids), 3), device=self._device)
    if self._stand_up_push_body_point_offset is not None:
      offset_world = quat_apply(
        body_quat,
        self._stand_up_push_body_point_offset.expand(len(env_ids), 3),
      )
      torque_world = torch.cross(offset_world, world_forces, dim=-1)
      torques = torque_world.unsqueeze(1).repeat(1, self._stand_up_push_num_bodies, 1)
    self._asset.write_external_wrench_to_sim(
      forces,
      torques,
      env_ids=env_ids,
      body_ids=self._stand_up_push_body_ids,
    )

    duration_low, duration_high = duration_s
    self._stand_up_push_time_left[env_ids] = (
      torch.rand(len(env_ids), device=self._device) * (duration_high - duration_low)
      + duration_low
    )
    self._stand_up_push_active[env_ids] = True
    self._stand_up_recovery_triggered[env_ids] = True
    return world_forces, torque_world

  def _apply_stand_up_actuator_scaling(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    effort_scale_range: tuple[float, float],
    pd_scale_range: tuple[float, float],
    action_scale_range: tuple[float, float],
    duration_s: tuple[float, float],
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if env_ids.numel() == 0:
      empty = torch.zeros(0, device=self._device)
      return empty, empty, empty, empty

    effort_scales = sample_uniform(
      effort_scale_range[0],
      effort_scale_range[1],
      (len(env_ids),),
      device=self._device,
    )
    pd_scales = sample_uniform(
      pd_scale_range[0],
      pd_scale_range[1],
      (len(env_ids),),
      device=self._device,
    )
    action_scales = sample_uniform(
      action_scale_range[0],
      action_scale_range[1],
      (len(env_ids),),
      device=self._device,
    )
    self._set_stand_up_underpowered_scales(
      env_ids, effort_scales, pd_scales, pd_scales
    )
    self._set_stand_up_action_scales(env, env_ids, action_scales)

    duration_low, duration_high = duration_s
    durations = (
      torch.rand(len(env_ids), device=self._device) * (duration_high - duration_low)
      + duration_low
    )
    self._stand_up_underpowered_time_left[env_ids] = durations
    self._stand_up_underpowered_active[env_ids] = True
    self._stand_up_recovery_triggered[env_ids] = True
    return effort_scales, pd_scales, action_scales, durations

  def _apply_stand_up_underpowered(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    effort_scale_range: tuple[float, float],
    pd_scale_range: tuple[float, float],
    action_scale_range: tuple[float, float],
    duration_s: tuple[float, float],
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return self._apply_stand_up_actuator_scaling(
      env,
      env_ids,
      effort_scale_range=effort_scale_range,
      pd_scale_range=pd_scale_range,
      action_scale_range=action_scale_range,
      duration_s=duration_s,
    )

  def _apply_stand_up_pitch_ang_vel_kick(
    self,
    env_ids: torch.Tensor,
    pitch_ang_vel_range: tuple[float, float],
    duration_steps: int,
  ) -> torch.Tensor:
    if env_ids.numel() == 0:
      return torch.zeros((0, 3), device=self._device)

    local_ang_velocity_delta = torch.zeros((len(env_ids), 3), device=self._device)
    local_ang_velocity_delta[:, 1] = sample_uniform(
      pitch_ang_vel_range[0],
      pitch_ang_vel_range[1],
      (len(env_ids),),
      device=self._device,
    )
    root_quat = self._asset.data.root_link_quat_w[env_ids]
    world_ang_velocity_delta = quat_apply(root_quat, local_ang_velocity_delta)
    self._stand_up_pitch_kick_steps_left[env_ids] = max(int(duration_steps), 1)
    self._stand_up_pitch_kick_world_delta[env_ids] = world_ang_velocity_delta
    self._stand_up_recovery_triggered[env_ids] = True
    return world_ang_velocity_delta

  def _apply_stand_up_velocity_kick(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    forward_velocity_range: tuple[float, float],
    backward_velocity_range: tuple[float, float],
    backward_probability: float,
    lateral_velocity_range: tuple[float, float],
    duration_steps: int,
  ) -> torch.Tensor:
    if env_ids.numel() == 0:
      return torch.zeros((0, 3), device=self._device)

    local_velocity_delta = torch.zeros((len(env_ids), 3), device=self._device)
    backward_probability = min(max(float(backward_probability), 0.0), 1.0)
    backward_mask = torch.rand(len(env_ids), device=self._device) < backward_probability
    forward_mask = ~backward_mask
    if forward_mask.any():
      local_velocity_delta[forward_mask, 0] = sample_uniform(
        forward_velocity_range[0],
        forward_velocity_range[1],
        (int(forward_mask.sum().item()),),
        device=self._device,
      )
    if backward_mask.any():
      local_velocity_delta[backward_mask, 0] = sample_uniform(
        backward_velocity_range[0],
        backward_velocity_range[1],
        (int(backward_mask.sum().item()),),
        device=self._device,
    )
    lateral_delta = sample_uniform(
      lateral_velocity_range[0],
      lateral_velocity_range[1],
      (len(env_ids),),
      device=self._device,
    )
    push_body_id = (
      self._stand_up_push_body_ids[0]
      if isinstance(self._stand_up_push_body_ids, list)
      else 0
    )
    body_quat = self._asset.data.body_com_quat_w[env_ids][:, push_body_id]
    local_velocity_delta[:, 1] = lateral_delta
    world_velocity_delta = quat_apply_yaw(body_quat, local_velocity_delta)
    self._stand_up_velocity_kick_steps_left[env_ids] = max(int(duration_steps), 1)
    self._stand_up_velocity_kick_world_delta[env_ids] = world_velocity_delta
    self._stand_up_recovery_triggered[env_ids] = True
    return world_velocity_delta

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    command_name: str,
    late_phase_start_ratio: float,
    late_phase_onset_scale: float,
    late_phase_scale_power: float,
    state_cooldown_s: tuple[float, float],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    joint_position_range: tuple[float, float],
    joint_velocity_range: tuple[float, float],
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    impulse_duration_s: tuple[float, float],
    impulse_cooldown_s: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    stand_up_push_center_frame: int,
    stand_up_push_half_window: int,
    stand_up_push_post_trigger_frame: int,
    stand_up_push_forward_force_range: tuple[float, float],
    stand_up_push_lateral_force_range: tuple[float, float],
    stand_up_push_duration_s: tuple[float, float],
    stand_up_push_body_cfg: SceneEntityCfg,
    stand_up_push_body_point_offset: tuple[float, float, float] | None,
    stand_up_overshoot_trigger_frame: int,
    stand_up_overshoot_half_window: int,
    stand_up_overshoot_effort_scale_range: tuple[float, float],
    stand_up_overshoot_pd_scale_range: tuple[float, float],
    stand_up_overshoot_action_scale_range: tuple[float, float],
    stand_up_overshoot_pitch_ang_vel_range: tuple[float, float],
    stand_up_overshoot_duration_s: tuple[float, float],
    stand_up_underpowered_probability: float,
    stand_up_underpowered_trigger_frame: int,
    stand_up_underpowered_half_window: int,
    stand_up_underpowered_effort_scale_range: tuple[float, float],
    stand_up_underpowered_pd_scale_range: tuple[float, float],
    stand_up_underpowered_action_scale_range: tuple[float, float],
    stand_up_underpowered_pitch_ang_vel_range: tuple[float, float],
    stand_up_underpowered_duration_s: tuple[float, float],
    stand_up_underpowered_actuator_cfg: SceneEntityCfg,
    stand_up_pitch_ang_vel_kick_duration_steps: int,
    stand_up_velocity_kick_forward_range: tuple[float, float],
    stand_up_velocity_kick_backward_range: tuple[float, float],
    stand_up_velocity_kick_backward_probability: float,
    stand_up_velocity_kick_lateral_range: tuple[float, float],
    stand_up_velocity_kick_duration_steps: int,
    log_stand_up_recovery_disturbance: bool,
  ) -> None:
    del env_ids, asset_cfg, stand_up_push_body_cfg, stand_up_push_body_point_offset
    del stand_up_underpowered_actuator_cfg

    from mjlab.tasks.tracking.mdp.commands import MotionCommand

    motion_command = cast(MotionCommand, env.command_manager.get_term(command_name))
    current_time_steps = motion_command.time_steps.clone()
    motion_loop_ids = torch.nonzero(
      (self._previous_motion_time_steps >= 0)
      & (current_time_steps < self._previous_motion_time_steps),
      as_tuple=False,
    ).squeeze(-1)
    self._reset_motion_cycle_state(motion_loop_ids)
    self._clear_stand_up_underpowered(motion_loop_ids)
    self._clear_stand_up_action_scales(env, motion_loop_ids)

    eligible_env_ids = self._eligible_env_ids(
      env, command_name=command_name, start_ratio=late_phase_start_ratio
    )
    eligible_mask = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
    eligible_mask[eligible_env_ids] = True
    inactive_active_ids = (
      self._impulse_active & (~eligible_mask)
    ).nonzero(as_tuple=False).squeeze(-1)
    self._clear_impulses(inactive_active_ids)
    inactive_push_ids = (
      self._stand_up_push_active & (~eligible_mask)
    ).nonzero(as_tuple=False).squeeze(-1)
    self._clear_stand_up_pushes(inactive_push_ids)
    inactive_underpowered_ids = (
      self._stand_up_underpowered_active & (~eligible_mask)
    ).nonzero(as_tuple=False).squeeze(-1)
    self._clear_stand_up_underpowered(inactive_underpowered_ids)
    self._clear_stand_up_action_scales(env, inactive_underpowered_ids)
    inactive_kick_ids = (
      (self._stand_up_velocity_kick_steps_left > 0) & (~eligible_mask)
    ).nonzero(as_tuple=False).squeeze(-1)
    self._clear_stand_up_velocity_kicks(inactive_kick_ids)
    inactive_pitch_kick_ids = (
      (self._stand_up_pitch_kick_steps_left > 0) & (~eligible_mask)
    ).nonzero(as_tuple=False).squeeze(-1)
    self._clear_stand_up_pitch_kicks(inactive_pitch_kick_ids)

    if eligible_env_ids.numel() == 0:
      self._previous_motion_time_steps.copy_(current_time_steps)
      return

    unset_mode_mask = self._stand_up_recovery_mode[eligible_env_ids] < 0
    unset_mode_ids = eligible_env_ids[unset_mode_mask]
    if unset_mode_ids.numel() > 0:
      underpowered_mode = (
        torch.rand(unset_mode_ids.numel(), device=self._device)
        < min(max(float(stand_up_underpowered_probability), 0.0), 1.0)
      )
      sampled_modes = torch.full(
        (unset_mode_ids.numel(),),
        self._OVERSHOOT_MODE,
        device=self._device,
        dtype=torch.long,
      )
      sampled_modes[underpowered_mode] = self._UNDERPOWERED_MODE
      self._stand_up_recovery_mode[unset_mode_ids] = sampled_modes

      underpowered_center_frame = min(
        max(int(stand_up_underpowered_trigger_frame), 0),
        motion_command.motion.time_step_total - 1,
      )
      underpowered_half_window = max(int(stand_up_underpowered_half_window), 0)
      underpowered_lower_frame = max(
        underpowered_center_frame - underpowered_half_window, 0
      )
      underpowered_upper_frame = min(
        underpowered_center_frame + underpowered_half_window,
        motion_command.motion.time_step_total - 1,
      )
      overshoot_center_frame = min(
        max(int(stand_up_overshoot_trigger_frame), 0),
        motion_command.motion.time_step_total - 1,
      )
      overshoot_half_window = max(int(stand_up_overshoot_half_window), 0)
      overshoot_lower_frame = max(overshoot_center_frame - overshoot_half_window, 0)
      overshoot_upper_frame = min(
        overshoot_center_frame + overshoot_half_window,
        motion_command.motion.time_step_total - 1,
      )
      underpowered_ids = unset_mode_ids[underpowered_mode]
      if underpowered_ids.numel() > 0:
        self._stand_up_underpowered_trigger_frame[underpowered_ids] = torch.randint(
          underpowered_lower_frame,
          underpowered_upper_frame + 1,
          (underpowered_ids.numel(),),
          device=self._device,
        )
      overshoot_ids = unset_mode_ids[~underpowered_mode]
      if overshoot_ids.numel() > 0:
        self._stand_up_overshoot_trigger_frame[overshoot_ids] = torch.randint(
          overshoot_lower_frame,
          overshoot_upper_frame + 1,
          (overshoot_ids.numel(),),
          device=self._device,
        )
        self._stand_up_underpowered_trigger_frame[overshoot_ids] = -1
      if underpowered_ids.numel() > 0:
        self._stand_up_overshoot_trigger_frame[underpowered_ids] = -1

    scales = self._progress_scales(
      env,
      eligible_env_ids,
      command_name=command_name,
      late_phase_start_ratio=late_phase_start_ratio,
      late_phase_onset_scale=late_phase_onset_scale,
      late_phase_scale_power=late_phase_scale_power,
    )

    dt = self._step_dt

    self._state_cooldown[eligible_env_ids] -= dt
    self._impulse_cooldown[eligible_env_ids] -= dt
    self._impulse_time_left[self._impulse_active] -= dt
    self._stand_up_push_time_left[self._stand_up_push_active] -= dt
    self._stand_up_underpowered_time_left[self._stand_up_underpowered_active] -= dt

    expired_impulses = self._impulse_active & (self._impulse_time_left <= 0.0)
    if expired_impulses.any():
      expired_ids = expired_impulses.nonzero(as_tuple=False).squeeze(-1)
      self._clear_impulses(expired_ids)
      self._sample_impulse_cooldown(expired_ids, impulse_cooldown_s)

    expired_stand_up_pushes = self._stand_up_push_active & (
      self._stand_up_push_time_left <= 0.0
    )
    if expired_stand_up_pushes.any():
      expired_ids = expired_stand_up_pushes.nonzero(as_tuple=False).squeeze(-1)
      self._clear_stand_up_pushes(expired_ids)

    expired_underpowered = self._stand_up_underpowered_active & (
      self._stand_up_underpowered_time_left <= 0.0
    )
    if expired_underpowered.any():
      expired_ids = expired_underpowered.nonzero(as_tuple=False).squeeze(-1)
      self._clear_stand_up_underpowered(expired_ids)
      self._clear_stand_up_action_scales(env, expired_ids)

    state_trigger_mask = self._state_cooldown[eligible_env_ids] <= 0.0
    state_trigger_ids = eligible_env_ids[state_trigger_mask]
    if state_trigger_ids.numel() > 0:
      self._apply_state_perturbation(
        state_trigger_ids,
        scales=scales[state_trigger_mask],
        pose_range=pose_range,
        velocity_range=velocity_range,
        joint_position_range=joint_position_range,
        joint_velocity_range=joint_velocity_range,
      )
      self._sample_timer(state_trigger_ids, state_cooldown_s)

    impulse_trigger_mask = (~self._impulse_active[eligible_env_ids]) & (
      self._impulse_cooldown[eligible_env_ids] <= 0.0
    )
    impulse_trigger_ids = eligible_env_ids[impulse_trigger_mask]
    if impulse_trigger_ids.numel() > 0:
      size = (len(impulse_trigger_ids), self._num_bodies, 3)
      impulse_scales = scales[impulse_trigger_mask]
      forces = sample_uniform(-1.0, 1.0, size, self._device)
      torques = sample_uniform(-1.0, 1.0, size, self._device)
      force_mag = max(abs(force_range[0]), abs(force_range[1]))
      torque_mag = max(abs(torque_range[0]), abs(torque_range[1]))
      forces = forces * force_mag * impulse_scales[:, None, None]
      torques = torques * torque_mag * impulse_scales[:, None, None]
      self._asset.write_external_wrench_to_sim(
        forces, torques, env_ids=impulse_trigger_ids, body_ids=self._body_ids
      )
      duration_low, duration_high = impulse_duration_s
      self._impulse_time_left[impulse_trigger_ids] = (
        torch.rand(len(impulse_trigger_ids), device=self._device)
        * (duration_high - duration_low)
        + duration_low
      )
      self._impulse_active[impulse_trigger_ids] = True
      self._sample_impulse_cooldown(impulse_trigger_ids, impulse_cooldown_s)

    post_trigger_frame = min(
      int(stand_up_push_post_trigger_frame),
      motion_command.motion.time_step_total - 1,
    )
    post_window_mask = motion_command.time_steps == post_trigger_frame
    stand_up_available_mask = (
      (~self._stand_up_recovery_triggered)
      & (~self._stand_up_push_active)
      & (~self._stand_up_underpowered_active)
    )
    overshoot_window_mask = (
      motion_command.time_steps == self._stand_up_overshoot_trigger_frame
    )
    underpowered_window_mask = (
      motion_command.time_steps == self._stand_up_underpowered_trigger_frame
    )
    push_has_effect = (
      max(
        abs(stand_up_push_forward_force_range[0]),
        abs(stand_up_push_forward_force_range[1]),
      ) > 0.0
      or max(
        abs(stand_up_push_lateral_force_range[0]),
        abs(stand_up_push_lateral_force_range[1]),
      ) > 0.0
    ) and stand_up_push_duration_s[1] > 0.0
    overshoot_execution_has_effect = (
      max(
        abs(stand_up_overshoot_effort_scale_range[0] - 1.0),
        abs(stand_up_overshoot_effort_scale_range[1] - 1.0),
      ) > 1.0e-3
      or max(
        abs(stand_up_overshoot_pd_scale_range[0] - 1.0),
        abs(stand_up_overshoot_pd_scale_range[1] - 1.0),
      ) > 1.0e-3
      or max(
        abs(stand_up_overshoot_action_scale_range[0] - 1.0),
        abs(stand_up_overshoot_action_scale_range[1] - 1.0),
      ) > 1.0e-3
    ) and stand_up_overshoot_duration_s[1] > 0.0
    underpowered_execution_has_effect = (
      min(stand_up_underpowered_effort_scale_range) < 0.999
      or min(stand_up_underpowered_pd_scale_range) < 0.999
      or min(stand_up_underpowered_action_scale_range) < 0.999
    ) and stand_up_underpowered_duration_s[1] > 0.0
    overshoot_pitch_kick_has_effect = (
      max(
        abs(stand_up_overshoot_pitch_ang_vel_range[0]),
        abs(stand_up_overshoot_pitch_ang_vel_range[1]),
      ) > 0.0
    ) and int(stand_up_pitch_ang_vel_kick_duration_steps) > 0
    underpowered_pitch_kick_has_effect = (
      max(
        abs(stand_up_underpowered_pitch_ang_vel_range[0]),
        abs(stand_up_underpowered_pitch_ang_vel_range[1]),
      ) > 0.0
    ) and int(stand_up_pitch_ang_vel_kick_duration_steps) > 0
    kick_has_effect = (
      max(
        abs(stand_up_velocity_kick_forward_range[0]),
        abs(stand_up_velocity_kick_forward_range[1]),
      ) > 0.0
      or max(
        abs(stand_up_velocity_kick_backward_range[0]),
        abs(stand_up_velocity_kick_backward_range[1]),
      ) > 0.0
      or max(
        abs(stand_up_velocity_kick_lateral_range[0]),
        abs(stand_up_velocity_kick_lateral_range[1]),
      ) > 0.0
    )
    if underpowered_execution_has_effect or underpowered_pitch_kick_has_effect:
      underpowered_candidate_ids = torch.nonzero(
        underpowered_window_mask
        & stand_up_available_mask
        & (self._stand_up_recovery_mode == self._UNDERPOWERED_MODE),
        as_tuple=False,
      ).squeeze(-1)
      if underpowered_candidate_ids.numel() > 0:
        effort_scales = torch.ones(len(underpowered_candidate_ids), device=self._device)
        pd_scales = torch.ones(len(underpowered_candidate_ids), device=self._device)
        action_scales = torch.ones(len(underpowered_candidate_ids), device=self._device)
        durations = torch.zeros(len(underpowered_candidate_ids), device=self._device)
        if underpowered_execution_has_effect:
          effort_scales, pd_scales, action_scales, durations = self._apply_stand_up_underpowered(
            env,
            underpowered_candidate_ids,
            effort_scale_range=stand_up_underpowered_effort_scale_range,
            pd_scale_range=stand_up_underpowered_pd_scale_range,
            action_scale_range=stand_up_underpowered_action_scale_range,
            duration_s=stand_up_underpowered_duration_s,
          )
        world_ang_velocity_delta = torch.zeros(
          (len(underpowered_candidate_ids), 3), device=self._device
        )
        if underpowered_pitch_kick_has_effect:
          world_ang_velocity_delta = self._apply_stand_up_pitch_ang_vel_kick(
            underpowered_candidate_ids,
            pitch_ang_vel_range=stand_up_underpowered_pitch_ang_vel_range,
            duration_steps=stand_up_pitch_ang_vel_kick_duration_steps,
          )
        if log_stand_up_recovery_disturbance:
          frames = (
            motion_command.time_steps[underpowered_candidate_ids]
            .detach()
            .cpu()
            .tolist()
          )
          print(
            "[INFO]: stand-up underpowered triggered",
            {
              "frames": frames,
              "effort_scale": effort_scales.detach().cpu().tolist(),
              "pd_scale": pd_scales.detach().cpu().tolist(),
              "action_scale": action_scales.detach().cpu().tolist(),
              "delta_pitch_ang_vel_world_xyz": (
                world_ang_velocity_delta.detach().cpu().tolist()
              ),
              "duration_s": durations.detach().cpu().tolist(),
            },
          )

    stand_up_available_mask = (
      (~self._stand_up_recovery_triggered)
      & (~self._stand_up_push_active)
      & (~self._stand_up_underpowered_active)
    )
    if overshoot_execution_has_effect or overshoot_pitch_kick_has_effect:
      overshoot_candidate_ids = torch.nonzero(
        overshoot_window_mask
        & stand_up_available_mask
        & (self._stand_up_recovery_mode == self._OVERSHOOT_MODE),
        as_tuple=False,
      ).squeeze(-1)
      if overshoot_candidate_ids.numel() > 0:
        effort_scales = torch.ones(len(overshoot_candidate_ids), device=self._device)
        pd_scales = torch.ones(len(overshoot_candidate_ids), device=self._device)
        action_scales = torch.ones(len(overshoot_candidate_ids), device=self._device)
        durations = torch.zeros(len(overshoot_candidate_ids), device=self._device)
        if overshoot_execution_has_effect:
          effort_scales, pd_scales, action_scales, durations = self._apply_stand_up_actuator_scaling(
            env,
            overshoot_candidate_ids,
            effort_scale_range=stand_up_overshoot_effort_scale_range,
            pd_scale_range=stand_up_overshoot_pd_scale_range,
            action_scale_range=stand_up_overshoot_action_scale_range,
            duration_s=stand_up_overshoot_duration_s,
          )
        world_ang_velocity_delta = torch.zeros(
          (len(overshoot_candidate_ids), 3), device=self._device
        )
        if overshoot_pitch_kick_has_effect:
          world_ang_velocity_delta = self._apply_stand_up_pitch_ang_vel_kick(
            overshoot_candidate_ids,
            pitch_ang_vel_range=stand_up_overshoot_pitch_ang_vel_range,
            duration_steps=stand_up_pitch_ang_vel_kick_duration_steps,
          )
        if log_stand_up_recovery_disturbance:
          frames = motion_command.time_steps[overshoot_candidate_ids].detach().cpu().tolist()
          print(
            "[INFO]: stand-up overshoot triggered",
            {
              "frames": frames,
              "effort_scale": effort_scales.detach().cpu().tolist(),
              "pd_scale": pd_scales.detach().cpu().tolist(),
              "action_scale": action_scales.detach().cpu().tolist(),
              "delta_pitch_ang_vel_world_xyz": (
                world_ang_velocity_delta.detach().cpu().tolist()
              ),
              "duration_s": durations.detach().cpu().tolist(),
            },
          )

    stand_up_available_mask = (
      (~self._stand_up_recovery_triggered)
      & (~self._stand_up_push_active)
      & (~self._stand_up_underpowered_active)
    )
    if push_has_effect:
      forward_candidate_ids = torch.nonzero(
        post_window_mask
        & stand_up_available_mask
        & (self._stand_up_recovery_mode == self._OVERSHOOT_MODE),
        as_tuple=False,
      ).squeeze(-1)
      if forward_candidate_ids.numel() > 0:
        world_forces, world_torques = self._apply_stand_up_push(
          env,
          forward_candidate_ids,
          command_name=command_name,
          longitudinal_force_range=stand_up_push_forward_force_range,
          lateral_force_range=stand_up_push_lateral_force_range,
          duration_s=stand_up_push_duration_s,
        )
        if log_stand_up_recovery_disturbance:
          frames = motion_command.time_steps[forward_candidate_ids].detach().cpu().tolist()
          print(
            "[INFO]: stand-up overshoot body push triggered",
            {
              "frames": frames,
              "force_world_xyz": world_forces.detach().cpu().tolist(),
              "torque_world_xyz": world_torques.detach().cpu().tolist(),
            },
          )
    if kick_has_effect:
      stand_up_window_ids = self._stand_up_window_env_ids(
        env,
        command_name=command_name,
        center_frame=stand_up_push_center_frame,
        half_window=stand_up_push_half_window,
      )
      if stand_up_window_ids.numel() > 0:
        stand_up_candidate_mask = (
          ~self._stand_up_recovery_triggered[stand_up_window_ids]
        ) & (~self._stand_up_push_active[stand_up_window_ids]) & (
          ~self._stand_up_underpowered_active[stand_up_window_ids]
        )
        stand_up_candidate_ids = stand_up_window_ids[stand_up_candidate_mask]
        if stand_up_candidate_ids.numel() > 0:
          world_velocity_delta = self._apply_stand_up_velocity_kick(
            env,
            stand_up_candidate_ids,
            command_name=command_name,
            forward_velocity_range=stand_up_velocity_kick_forward_range,
            backward_velocity_range=stand_up_velocity_kick_backward_range,
            backward_probability=stand_up_velocity_kick_backward_probability,
            lateral_velocity_range=stand_up_velocity_kick_lateral_range,
            duration_steps=stand_up_velocity_kick_duration_steps,
          )
          if log_stand_up_recovery_disturbance:
            frames = motion_command.time_steps[stand_up_candidate_ids].detach().cpu().tolist()
            deltas = world_velocity_delta.detach().cpu().tolist()
            print(
              "[INFO]: stand-up velocity kick triggered",
              {
                "frames": frames,
                "delta_v_world_xyz": deltas,
                "duration_steps": max(int(stand_up_velocity_kick_duration_steps), 1),
              },
            )

    self._reapply_stand_up_pitch_kicks()
    self._reapply_stand_up_velocity_kicks()
    self._previous_motion_time_steps.copy_(current_time_steps)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      reset_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.long)
    elif isinstance(env_ids, slice):
      reset_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.long)[
        env_ids
      ]
    else:
      reset_ids = env_ids.to(device=self._device, dtype=torch.long)

    self._clear_impulses(reset_ids)
    self._clear_stand_up_pushes(reset_ids)
    self._clear_stand_up_underpowered(reset_ids)
    self._clear_stand_up_action_scales(None, reset_ids)
    self._clear_stand_up_velocity_kicks(reset_ids)
    self._clear_stand_up_pitch_kicks(reset_ids)
    self._state_cooldown[reset_ids] = 0.0
    self._impulse_cooldown[reset_ids] = 0.0
    self._stand_up_recovery_triggered[reset_ids] = False
    self._stand_up_recovery_mode[reset_ids] = -1
    self._stand_up_overshoot_trigger_frame[reset_ids] = -1
    self._stand_up_underpowered_trigger_frame[reset_ids] = -1
    self._previous_motion_time_steps[reset_ids] = -1
