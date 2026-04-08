from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.g1_new.env_cfgs import (
  unitree_g1_new_flat_tracking_env_cfg,
)
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_apply,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

G1_TRACKING_JOINT_NAMES = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]
G1_FOOT_GEOM_PATTERN = r"^(left|right)_foot[1-7]_collision$"
ALL_COLLISION_GEOM_PATTERN = r".*_collision$"
RobotVariant = Literal["g1", "g1_new"]


@dataclass(frozen=True)
class GroundingStats:
  foot_geom_names: tuple[str, ...]
  frame_min_foot_z: np.ndarray
  global_min_foot_z: float
  recommended_delta_z: float


@dataclass(frozen=True)
class CollisionGroundingStats:
  geom_names: tuple[str, ...]
  frame_min_bottom_z: np.ndarray
  frame_min_geom_indices: np.ndarray
  global_min_bottom_z: float
  recommended_delta_z: float

  @property
  def global_min_geom_name(self) -> str:
    global_idx = int(np.argmin(self.frame_min_bottom_z))
    culprit_idx = int(self.frame_min_geom_indices[global_idx])
    return self.geom_names[culprit_idx]


@dataclass(frozen=True)
class PhaseBlendControlSuggestion:
  control_points: np.ndarray
  source: str
  note: str
  nearby_foot_height: np.ndarray
  grounded_center: float
  airborne_center: float
  airborne_segments: int
  airborne_frame_ratio: float

  @property
  def grounded_height(self) -> float | None:
    zero_weight = self.control_points[self.control_points[:, 1] <= 0.0]
    if zero_weight.size == 0:
      return None
    return float(zero_weight[-1, 0])

  @property
  def airborne_height(self) -> float | None:
    one_weight = self.control_points[self.control_points[:, 1] >= 1.0]
    if one_weight.size == 0:
      return None
    return float(one_weight[0, 0])

  @property
  def blend_points_cli(self) -> str:
    return ",".join(
      f"{float(height):.3f}:{float(weight):.2f}"
      for height, weight in self.control_points
    )


def collect_csv_files(input_dir: Path) -> list[Path]:
  return sorted(p for p in input_dir.rglob("*.csv") if p.is_file())


def is_numeric_motion_csv(path: Path) -> bool:
  try:
    with path.open("r", encoding="utf-8-sig") as f:
      for line in f:
        stripped = line.strip()
        if not stripped:
          continue
        for value in stripped.split(","):
          float(value)
        return True
  except (OSError, UnicodeDecodeError, ValueError):
    return False
  return False


def resolve_batch_output(
  input_root: Path, output_root: Path, input_file: Path
) -> Path:
  relative = input_file.relative_to(input_root)
  return output_root / relative.parent / f"{input_file.stem}.npz"


class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    input_fps: int,
    output_fps: int,
    device: torch.device | str,
    line_range: tuple[int, int] | None = None,
  ):
    self.motion_file = motion_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0
    self.device = device
    self.line_range = line_range
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self):
    """Loads the motion from the csv file."""
    if self.line_range is None:
      motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
    else:
      motion = torch.from_numpy(
        np.loadtxt(
          self.motion_file,
          delimiter=",",
          skiprows=self.line_range[0] - 1,
          max_rows=self.line_range[1] - self.line_range[0] + 1,
        )
      )
    motion = motion.to(torch.float32).to(self.device)
    # motion[:, 2] -= 0.05
    self.motion_base_poss_input = motion[:, :3]
    self.motion_base_rots_input = motion[:, 3:7]
    self.motion_base_rots_input = self.motion_base_rots_input[
      :, [3, 0, 1, 2]
    ]  # convert to wxyz
    self.motion_dof_poss_input = motion[:, 7:]

    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

  def _interpolate_motion(self):
    """Interpolates the motion to the output fps."""
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )
    print(
      f"Motion interpolated, input frames: {self.input_frames}, "
      f"input fps: {self.input_fps}, "
      f"output frames: {self.output_frames}, "
      f"output fps: {self.output_fps}"
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Linear interpolation between two tensors."""
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions."""
    slerped_quats = torch.zeros_like(a)
    for i in range(a.shape[0]):
      slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return slerped_quats

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the frame blend for the motion."""
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _compute_velocities(self):
    """Computes the velocities of the motion."""
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots, self.output_dt
    )

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """Computes the derivative of a sequence of SO3 rotations.

    Args:
      rotations: shape (B, 4).
      dt: time step.
    Returns:
      shape (B, 3).
    """
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
    omega = torch.cat(
      [omega[:1], omega, omega[-1:]], dim=0
    )  # repeat first and last sample
    return omega

  def apply_root_z_offset(
    self, offset_z: float | np.ndarray | torch.Tensor
  ) -> None:
    offset_tensor = torch.as_tensor(
      offset_z, dtype=self.motion_base_poss.dtype, device=self.device
    )
    if offset_tensor.ndim == 0 or offset_tensor.numel() == 1:
      scalar = offset_tensor.reshape(()).clone()
      self.motion_base_poss_input[:, 2] += scalar
      self.motion_base_poss[:, 2] += scalar
    else:
      if offset_tensor.shape != (self.output_frames,):
        raise ValueError(
          "Frame-wise root z offsets must match the interpolated output frame count. "
          f"Expected {(self.output_frames,)}, got {tuple(offset_tensor.shape)}."
        )
      self.motion_base_poss[:, 2] += offset_tensor
    self._compute_velocities()

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    """Gets the next state of the motion."""
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = False
    if self.current_idx >= self.output_frames:
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


def resolve_device(device: str) -> str:
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA is not available. Falling back to CPU. This may be slow.")
    return "cpu"
  return device


def build_tracking_sim(
  output_fps: float,
  device: str,
  render: bool = False,
  robot_variant: RobotVariant = "g1",
) -> tuple[Simulation, Scene, OffscreenRenderer | None]:
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  if robot_variant == "g1_new":
    env_cfg = unitree_g1_new_flat_tracking_env_cfg()
  else:
    env_cfg = unitree_g1_flat_tracking_env_cfg()

  scene = Scene(env_cfg.scene, device=device)
  model = scene.compile()

  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  renderer = None
  if render:
    viewer_cfg = ViewerConfig(
      height=480,
      width=640,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="robot",
      distance=2.0,
      elevation=-5.0,
      azimuth=20,
    )
    renderer = OffscreenRenderer(
      model=sim.mj_model,
      cfg=viewer_cfg,
      scene=scene,
    )
    renderer.initialize()

  return sim, scene, renderer


def get_tracking_robot(scene: Scene) -> tuple[Entity, list[int]]:
  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(
    G1_TRACKING_JOINT_NAMES, preserve_order=True
  )[0]
  return robot, robot_joint_indexes


def _write_motion_frame_to_sim(
  sim: Simulation,
  scene: Scene,
  robot: Entity,
  robot_joint_indexes: list[int],
  motion: MotionLoader,
  frame_idx: int,
) -> None:
  root_states = robot.data.default_root_state.clone()
  root_states[:, 0:3] = motion.motion_base_poss[frame_idx : frame_idx + 1]
  root_states[:, :2] += scene.env_origins[:, :2]
  root_states[:, 3:7] = motion.motion_base_rots[frame_idx : frame_idx + 1]
  root_states[:, 7:10] = motion.motion_base_lin_vels[frame_idx : frame_idx + 1]
  root_states[:, 10:] = motion.motion_base_ang_vels[frame_idx : frame_idx + 1]
  robot.write_root_state_to_sim(root_states)

  joint_pos = robot.data.default_joint_pos.clone()
  joint_vel = robot.data.default_joint_vel.clone()
  joint_pos[:, robot_joint_indexes] = motion.motion_dof_poss[frame_idx : frame_idx + 1]
  joint_vel[:, robot_joint_indexes] = motion.motion_dof_vels[frame_idx : frame_idx + 1]
  robot.write_joint_state_to_sim(joint_pos, joint_vel)

  sim.forward()
  scene.update(sim.mj_model.opt.timestep)


def _compute_geom_bottom_heights(
  geom_pos_w: torch.Tensor,
  geom_quat_w: torch.Tensor,
  geom_sizes: torch.Tensor,
  geom_types: np.ndarray,
) -> torch.Tensor:
  bottom_heights = torch.empty(
    geom_pos_w.shape[0], device=geom_pos_w.device, dtype=geom_pos_w.dtype
  )
  local_z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=geom_pos_w.device)
  box_signs = torch.tensor(
    [
      [-1.0, -1.0, -1.0],
      [-1.0, -1.0, 1.0],
      [-1.0, 1.0, -1.0],
      [-1.0, 1.0, 1.0],
      [1.0, -1.0, -1.0],
      [1.0, -1.0, 1.0],
      [1.0, 1.0, -1.0],
      [1.0, 1.0, 1.0],
    ],
    device=geom_pos_w.device,
  )

  for geom_idx, geom_type in enumerate(geom_types):
    geom_type = int(geom_type)
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
      bottom_heights[geom_idx] = geom_pos_w[geom_idx, 2] - geom_sizes[geom_idx, 0]
    elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
      axis = quat_apply(geom_quat_w[geom_idx : geom_idx + 1], local_z_axis)[0]
      half_length = geom_sizes[geom_idx, 1]
      end_0_z = geom_pos_w[geom_idx, 2] - axis[2] * half_length
      end_1_z = geom_pos_w[geom_idx, 2] + axis[2] * half_length
      bottom_heights[geom_idx] = torch.minimum(end_0_z, end_1_z) - geom_sizes[
        geom_idx, 0
      ]
    elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
      local_corners = box_signs * geom_sizes[geom_idx : geom_idx + 1]
      world_corners = quat_apply(
        geom_quat_w[geom_idx : geom_idx + 1].expand(8, -1), local_corners
      )
      world_corners = world_corners + geom_pos_w[geom_idx : geom_idx + 1]
      bottom_heights[geom_idx] = world_corners[:, 2].min()
    else:
      raise ValueError(
        "Unsupported geom type for ground analysis. "
        f"Geom type id: {geom_type}."
      )

  return bottom_heights


def analyze_collision_grounding(
  sim: Simulation,
  scene: Scene,
  robot: Entity,
  robot_joint_indexes: list[int],
  motion: MotionLoader,
  clearance: float,
  geom_pattern: str,
  show_progress: bool = True,
) -> CollisionGroundingStats:
  geom_ids, geom_names = robot.find_geoms(geom_pattern, preserve_order=True)
  if not geom_ids:
    raise ValueError(f"No geoms matched pattern: {geom_pattern}")

  global_geom_ids = robot.indexing.geom_ids[geom_ids].cpu().numpy()
  geom_sizes = torch.tensor(
    sim.mj_model.geom_size[global_geom_ids],
    dtype=torch.float32,
    device=sim.device,
  )
  geom_types = np.asarray(sim.mj_model.geom_type[global_geom_ids])

  frame_min_bottom_z = np.empty(motion.output_frames, dtype=np.float32)
  frame_min_geom_indices = np.empty(motion.output_frames, dtype=np.int32)
  frame_iterator = range(motion.output_frames)
  if show_progress:
    frame_iterator = tqdm(
      frame_iterator,
      total=motion.output_frames,
      desc="Analyzing collision grounding",
      unit="frame",
      ncols=100,
    )

  scene.reset()
  for frame_idx in frame_iterator:
    _write_motion_frame_to_sim(sim, scene, robot, robot_joint_indexes, motion, frame_idx)
    geom_pose_w = robot.data.geom_pose_w[0, geom_ids]
    bottom_heights = _compute_geom_bottom_heights(
      geom_pos_w=geom_pose_w[:, :3],
      geom_quat_w=geom_pose_w[:, 3:7],
      geom_sizes=geom_sizes,
      geom_types=geom_types,
    )
    culprit_idx = int(bottom_heights.argmin().item())
    frame_min_geom_indices[frame_idx] = culprit_idx
    frame_min_bottom_z[frame_idx] = float(bottom_heights[culprit_idx].item())

  global_min_bottom_z = float(frame_min_bottom_z.min())
  recommended_delta_z = max(0.0, clearance - global_min_bottom_z)
  return CollisionGroundingStats(
    geom_names=tuple(geom_names),
    frame_min_bottom_z=frame_min_bottom_z,
    frame_min_geom_indices=frame_min_geom_indices,
    global_min_bottom_z=global_min_bottom_z,
    recommended_delta_z=recommended_delta_z,
  )


def analyze_foot_penetration(
  sim: Simulation,
  scene: Scene,
  robot: Entity,
  robot_joint_indexes: list[int],
  motion: MotionLoader,
  clearance: float,
  foot_geom_pattern: str = G1_FOOT_GEOM_PATTERN,
  show_progress: bool = True,
) -> GroundingStats:
  collision_stats = analyze_collision_grounding(
    sim=sim,
    scene=scene,
    robot=robot,
    robot_joint_indexes=robot_joint_indexes,
    motion=motion,
    clearance=clearance,
    geom_pattern=foot_geom_pattern,
    show_progress=show_progress,
  )
  return GroundingStats(
    foot_geom_names=collision_stats.geom_names,
    frame_min_foot_z=collision_stats.frame_min_bottom_z,
    global_min_foot_z=collision_stats.global_min_bottom_z,
    recommended_delta_z=collision_stats.recommended_delta_z,
  )


def apply_global_ground_alignment(motion: MotionLoader, stats: GroundingStats) -> None:
  if stats.recommended_delta_z <= 0.0:
    return
  motion.apply_root_z_offset(stats.recommended_delta_z)


def _symmetric_window_max(values: np.ndarray, radius: int) -> np.ndarray:
  if radius <= 0:
    return values.copy()
  out = np.empty_like(values, dtype=np.float64)
  for idx in range(values.shape[0]):
    lo = max(0, idx - radius)
    hi = min(values.shape[0], idx + radius + 1)
    out[idx] = values[lo:hi].max()
  return out


def _forward_window_max(values: np.ndarray, window: int) -> np.ndarray:
  out = np.empty_like(values, dtype=np.float64)
  for idx in range(values.shape[0]):
    hi = min(values.shape[0], idx + window)
    out[idx] = values[idx:hi].max()
  return out


def _edge_smoothed(values: np.ndarray, radius: int) -> np.ndarray:
  if radius <= 0:
    return values.copy()
  kernel = np.ones(2 * radius + 1, dtype=np.float64)
  kernel = kernel / kernel.sum()
  padded = np.pad(values, (radius, radius), mode="edge")
  return np.convolve(padded, kernel, mode="valid")


def _boolean_segments(values: np.ndarray) -> list[tuple[bool, int, int]]:
  if values.size == 0:
    return []
  segments: list[tuple[bool, int, int]] = []
  start = 0
  current = bool(values[0])
  for idx in range(1, values.shape[0]):
    value = bool(values[idx])
    if value == current:
      continue
    segments.append((current, start, idx - 1))
    start = idx
    current = value
  segments.append((current, start, values.shape[0] - 1))
  return segments


def _cleanup_airborne_mask(mask: np.ndarray, min_segment_frames: int) -> np.ndarray:
  if min_segment_frames <= 1 or mask.size == 0:
    return mask.copy()

  cleaned = mask.astype(bool).copy()
  changed = True
  while changed:
    changed = False
    segments = _boolean_segments(cleaned)
    for idx, (value, start, end) in enumerate(segments):
      seg_len = end - start + 1
      if value and seg_len < min_segment_frames:
        cleaned[start : end + 1] = False
        changed = True
        break
      if (
        not value
        and 0 < idx < len(segments) - 1
        and segments[idx - 1][0]
        and segments[idx + 1][0]
        and seg_len < min_segment_frames
      ):
        cleaned[start : end + 1] = True
        changed = True
        break
  return cleaned


def _kmeans_two_centers(values: np.ndarray, max_iters: int = 32) -> tuple[float, float]:
  flat_values = np.asarray(values, dtype=np.float64).reshape(-1)
  if flat_values.size == 0:
    raise ValueError("Cannot infer phase control points from an empty signal.")

  low = float(np.quantile(flat_values, 0.15))
  high = float(np.quantile(flat_values, 0.85))
  if high - low < 1e-6:
    return low, high

  for _ in range(max_iters):
    assign_high = np.abs(flat_values - high) < np.abs(flat_values - low)
    if assign_high.all() or (~assign_high).all():
      break
    new_low = float(flat_values[~assign_high].mean())
    new_high = float(flat_values[assign_high].mean())
    if max(abs(new_low - low), abs(new_high - high)) < 1e-6:
      low, high = new_low, new_high
      break
    low, high = new_low, new_high

  if low > high:
    low, high = high, low
  return low, high


def suggest_phase_blend_control_points(
  *,
  foot_heights: np.ndarray,
  output_dt: float,
  phase_window_s: float,
  phase_grounded_height: float | None = None,
  phase_airborne_height: float | None = None,
  phase_blend_points: str | None = None,
) -> PhaseBlendControlSuggestion:
  phase_window_frames = max(0, int(round(phase_window_s / output_dt)))
  nearby_foot_height = _symmetric_window_max(
    np.asarray(foot_heights, dtype=np.float64),
    phase_window_frames,
  ).astype(np.float64)
  grounded_center, airborne_center = _kmeans_two_centers(nearby_foot_height)

  midpoint = 0.5 * (grounded_center + airborne_center)
  min_segment_frames = max(2, int(round(max(0.08, 0.5 * phase_window_s) / output_dt)))
  airborne_mask = nearby_foot_height >= midpoint
  airborne_mask = _cleanup_airborne_mask(airborne_mask, min_segment_frames)
  airborne_segments = sum(1 for value, _, _ in _boolean_segments(airborne_mask) if value)
  airborne_frame_ratio = float(airborne_mask.mean())

  auto_grounded_height: float
  auto_airborne_height: float
  dynamic_range = max(0.0, airborne_center - grounded_center)
  if (
    dynamic_range < 0.04
    or airborne_segments == 0
    or airborne_frame_ratio < 0.04
    or (~airborne_mask).sum() == 0
  ):
    auto_grounded_height = float(np.quantile(nearby_foot_height, 0.85))
    auto_airborne_height = auto_grounded_height + max(0.04, 0.6 * max(dynamic_range, 0.02))
    auto_note = (
      "No clear airborne plateau was detected, so the suggested ramp stays conservative."
    )
  else:
    grounded_values = nearby_foot_height[~airborne_mask]
    airborne_values = nearby_foot_height[airborne_mask]
    auto_grounded_height = float(np.quantile(grounded_values, 0.90))
    auto_airborne_height = float(np.quantile(airborne_values, 0.10))
    min_gap = max(0.02, 0.20 * dynamic_range)
    if auto_airborne_height <= auto_grounded_height + min_gap:
      midpoint = 0.5 * (auto_grounded_height + auto_airborne_height)
      auto_grounded_height = midpoint - 0.5 * min_gap
      auto_airborne_height = midpoint + 0.5 * min_gap
    auto_note = (
      "Control points were auto-inferred from the motion's nearby foot-height clusters."
    )

  if phase_blend_points is not None and phase_blend_points.strip():
    control_points = _resolve_phase_blend_control_points(
      phase_grounded_height=0.0,
      phase_airborne_height=1.0,
      phase_blend_points=phase_blend_points,
    )
    source = "manual_blend_points"
    note = "Using the explicit `phase_blend_points` override."
  else:
    grounded_height = (
      auto_grounded_height
      if phase_grounded_height is None
      else float(phase_grounded_height)
    )
    airborne_height = (
      auto_airborne_height
      if phase_airborne_height is None
      else float(phase_airborne_height)
    )
    if airborne_height <= grounded_height:
      airborne_height = grounded_height + max(0.02, 0.20 * max(dynamic_range, 0.02))
    control_points = np.asarray(
      [
        [grounded_height, 0.0],
        [airborne_height, 1.0],
      ],
      dtype=np.float64,
    )
    if phase_grounded_height is None and phase_airborne_height is None:
      source = "auto_inferred"
      note = auto_note
    elif phase_grounded_height is None or phase_airborne_height is None:
      source = "mixed_manual_auto"
      note = "One phase height was provided manually; the other was auto-inferred."
    else:
      source = "manual_heights"
      note = "Using the explicit grounded/airborne phase heights."

  return PhaseBlendControlSuggestion(
    control_points=control_points,
    source=source,
    note=note,
    nearby_foot_height=nearby_foot_height.astype(np.float32),
    grounded_center=grounded_center,
    airborne_center=airborne_center,
    airborne_segments=airborne_segments,
    airborne_frame_ratio=airborne_frame_ratio,
  )


def _resolve_phase_blend_control_points(
  *,
  phase_grounded_height: float,
  phase_airborne_height: float,
  phase_blend_points: str | None,
) -> np.ndarray:
  if phase_blend_points is None or not phase_blend_points.strip():
    if phase_airborne_height <= phase_grounded_height:
      raise ValueError(
        "`phase_airborne_height` must be greater than `phase_grounded_height`."
      )
    return np.asarray(
      [
        [phase_grounded_height, 0.0],
        [phase_airborne_height, 1.0],
      ],
      dtype=np.float64,
    )

  parsed_points: list[tuple[float, float]] = []
  for raw_item in phase_blend_points.split(","):
    item = raw_item.strip()
    if not item:
      continue
    if ":" not in item:
      raise ValueError(
        "Each `phase_blend_points` item must use `foot_height:blend_weight` format. "
        f"Got: {item!r}"
      )
    height_str, weight_str = item.split(":", 1)
    height = float(height_str.strip())
    weight = float(weight_str.strip())
    if not 0.0 <= weight <= 1.0:
      raise ValueError(
        "Blend weights in `phase_blend_points` must stay within [0, 1]. "
        f"Got {weight} for item {item!r}."
      )
    parsed_points.append((height, weight))

  if not parsed_points:
    raise ValueError("`phase_blend_points` did not contain any valid control points.")

  parsed_points.sort(key=lambda x: x[0])
  for idx in range(1, len(parsed_points)):
    if parsed_points[idx][0] <= parsed_points[idx - 1][0]:
      raise ValueError(
        "`phase_blend_points` heights must be strictly increasing after sorting."
      )
  return np.asarray(parsed_points, dtype=np.float64)


def compute_phased_ground_alignment_offsets(
  *,
  foot_stats: GroundingStats,
  whole_body_stats: CollisionGroundingStats,
  clearance: float,
  output_dt: float,
  phase_blend_control_points: np.ndarray,
  phase_window_s: float,
  phase_lookahead_s: float,
  phase_smoothing_s: float,
) -> np.ndarray:
  foot_required = np.maximum(0.0, clearance - foot_stats.frame_min_foot_z).astype(
    np.float64
  )
  whole_body_required = np.maximum(
    0.0, clearance - whole_body_stats.frame_min_bottom_z
  ).astype(np.float64)

  phase_window_frames = max(0, int(round(phase_window_s / output_dt)))
  lookahead_frames = max(1, int(round(phase_lookahead_s / output_dt)))
  smoothing_frames = max(0, int(round(phase_smoothing_s / output_dt)))

  nearby_foot_height = _symmetric_window_max(foot_stats.frame_min_foot_z, phase_window_frames)
  if phase_blend_control_points.shape[0] == 1:
    blend_weight = np.full(
      nearby_foot_height.shape,
      phase_blend_control_points[0, 1],
      dtype=np.float64,
    )
  else:
    blend_weight = np.interp(
      nearby_foot_height,
      phase_blend_control_points[:, 0],
      phase_blend_control_points[:, 1],
    )
  upcoming_whole_body_required = _forward_window_max(
    whole_body_required, lookahead_frames
  )
  base_offsets = foot_required + blend_weight * (
    upcoming_whole_body_required - foot_required
  )
  phased_offsets = _edge_smoothed(base_offsets, smoothing_frames)
  phased_offsets = np.maximum(phased_offsets, base_offsets)
  phased_offsets = np.maximum(phased_offsets, foot_required)
  phased_offsets = np.minimum(
    phased_offsets, np.maximum(upcoming_whole_body_required, foot_required)
  )
  return phased_offsets.astype(np.float32)


def run_sim(
  sim: Simulation,
  scene: Scene,
  robot: Entity,
  robot_joint_indexes: list[int],
  input_file,
  input_fps,
  output_fps,
  output_name,
  render,
  line_range,
  ground_align: Literal["none", "global", "phased"],
  clearance: float,
  whole_body_geom_pattern: str,
  phase_control_mode: Literal["manual", "auto"],
  phase_grounded_height: float | None,
  phase_airborne_height: float | None,
  phase_blend_points: str | None,
  phase_window_s: float,
  phase_lookahead_s: float,
  phase_smoothing_s: float,
  renderer: OffscreenRenderer | None = None,
):
  output_path = Path(output_name)
  if output_path.suffix != ".npz":
    output_path = output_path.with_suffix(".npz")
  video_path = output_path.with_suffix(".mp4")

  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    line_range=line_range,
  )

  if ground_align == "global":
    print(
      f"Analyzing foot penetration with clearance target {clearance:.3f} m..."
    )
    grounding_stats = analyze_foot_penetration(
      sim=sim,
      scene=scene,
      robot=robot,
      robot_joint_indexes=robot_joint_indexes,
      motion=motion,
      clearance=clearance,
    )
    print(
      "Grounding analysis complete: "
      f"min foot bottom z = {grounding_stats.global_min_foot_z:.4f} m, "
      f"recommended global z offset = {grounding_stats.recommended_delta_z:.4f} m"
    )
    apply_global_ground_alignment(motion, grounding_stats)
  elif ground_align == "phased":
    print(
      "Analyzing phased ground alignment with "
      f"clearance target {clearance:.3f} m..."
    )
    grounding_stats = analyze_foot_penetration(
      sim=sim,
      scene=scene,
      robot=robot,
      robot_joint_indexes=robot_joint_indexes,
      motion=motion,
      clearance=clearance,
      show_progress=True,
    )
    whole_body_stats = analyze_collision_grounding(
      sim=sim,
      scene=scene,
      robot=robot,
      robot_joint_indexes=robot_joint_indexes,
      motion=motion,
      clearance=clearance,
      geom_pattern=whole_body_geom_pattern,
      show_progress=False,
    )
    phase_control = suggest_phase_blend_control_points(
      foot_heights=grounding_stats.frame_min_foot_z,
      output_dt=motion.output_dt,
      phase_window_s=phase_window_s,
      phase_grounded_height=None if phase_control_mode == "auto" else phase_grounded_height,
      phase_airborne_height=None if phase_control_mode == "auto" else phase_airborne_height,
      phase_blend_points=phase_blend_points,
    )
    phase_blend_control_points = phase_control.control_points
    phased_offsets = compute_phased_ground_alignment_offsets(
      foot_stats=grounding_stats,
      whole_body_stats=whole_body_stats,
      clearance=clearance,
      output_dt=motion.output_dt,
      phase_blend_control_points=phase_blend_control_points,
      phase_window_s=phase_window_s,
      phase_lookahead_s=phase_lookahead_s,
      phase_smoothing_s=phase_smoothing_s,
    )
    print(
      "Phased grounding analysis complete: "
      f"foot min = {grounding_stats.global_min_foot_z:.4f} m, "
      f"whole-body min = {whole_body_stats.global_min_bottom_z:.4f} m "
      f"({whole_body_stats.global_min_geom_name}), "
      f"frame-wise offset range = [{float(phased_offsets.min()):.4f}, "
      f"{float(phased_offsets.max()):.4f}] m"
    )
    print(
      "Phased alignment settings: "
      f"mode={phase_control_mode}, "
      f"source={phase_control.source}, "
      f"blend_points={phase_blend_control_points.tolist()}, "
      f"window={phase_window_s:.3f} s, "
      f"lookahead={phase_lookahead_s:.3f} s, "
      f"smoothing={phase_smoothing_s:.3f} s"
    )
    grounded_height = phase_control.grounded_height
    airborne_height = phase_control.airborne_height
    if grounded_height is not None and airborne_height is not None:
      print(
        "Phased control-point summary: "
        f"grounded_height={grounded_height:.4f} m, "
        f"airborne_height={airborne_height:.4f} m, "
        f"detected_airborne_segments={phase_control.airborne_segments}, "
        f"airborne_frame_ratio={phase_control.airborne_frame_ratio:.3f}"
      )
    print(f"Phased control-point note: {phase_control.note}")
    motion.apply_root_z_offset(phased_offsets)
  else:
    print("Ground alignment disabled; using raw motion root heights.")

  log: dict[str, Any] = {
    "fps": [output_fps],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  file_saved = False

  frames = []
  scene.reset()

  print(f"\nStarting simulation with {motion.output_frames} frames...")
  if render:
    print("Rendering enabled - generating video frames...")

  # Create progress bar
  pbar = tqdm(
    total=motion.output_frames,
    desc="Processing frames",
    unit="frame",
    ncols=100,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
  )

  frame_count = 0
  while not file_saved:
    frame_idx = motion.current_idx
    (
      (
        _motion_base_pos,
        _motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        _motion_dof_pos,
        _motion_dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    _write_motion_frame_to_sim(sim, scene, robot, robot_joint_indexes, motion, frame_idx)
    if render and renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())

    if not file_saved:
      log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
      log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
      log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
      log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
      log["body_lin_vel_w"].append(
        robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
      )
      log["body_ang_vel_w"].append(
        robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
      )

      torch.testing.assert_close(
        robot.data.body_link_lin_vel_w[0, 0], motion_base_lin_vel[0]
      )
      torch.testing.assert_close(
        robot.data.body_link_ang_vel_w[0, 0], motion_base_ang_vel[0]
      )

      frame_count += 1
      pbar.update(1)

      if frame_count % 100 == 0:  # Update every 100 frames to avoid spam
        elapsed_time = frame_count / output_fps
        pbar.set_description(f"Processing frames (t={elapsed_time:.1f}s)")

      if reset_flag and not file_saved:
        file_saved = True
        pbar.close()

        print("\nStacking arrays and saving data...")
        for k in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        ):
          log[k] = np.stack(log[k], axis=0)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {output_path}...")
        np.savez(output_path, **log)

        print(f"[INFO]: Motion saved locally: {output_path}")

        if render:
          import mediapy as media

          print(f"Creating video at {video_path}...")
          media.write_video(str(video_path), frames, fps=output_fps)
          print(f"[INFO]: Video saved locally: {video_path}")


def main(
  input_file: str | None = None,
  output_name: str | None = None,
  inputm: str | None = None,
  outputm: str | None = None,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  ground_align: Literal["none", "global", "phased"] = "none",
  clearance: float = 0.01,
  whole_body_geom_pattern: str = ALL_COLLISION_GEOM_PATTERN,
  phase_control_mode: Literal["manual", "auto"] = "manual",
  phase_grounded_height: float | None = 0.03,
  phase_airborne_height: float | None = 0.10,
  phase_blend_points: str | None = None,
  phase_window_s: float = 0.12,
  phase_lookahead_s: float = 0.24,
  phase_smoothing_s: float = 0.08,
  robot_variant: RobotVariant = "g1",
  line_range: tuple[int, int] | None = None,
):
  """Replay motion from CSV file and output to npz file.

  Args:
    input_file: Path to the input CSV file in single-file mode.
    output_name: Path to the output npz file in single-file mode.
    inputm: Directory containing input CSV files in batch mode.
    outputm: Output directory for batch mode. If omitted, writes next to input CSVs.
    input_fps: Frame rate of the CSV file.
    output_fps: Desired output frame rate.
    device: Device to use.
    render: Whether to render the simulation and save a video.
    ground_align: Ground alignment mode for the motion root.
    clearance: Target minimum foot-bottom clearance above the ground plane.
    whole_body_geom_pattern: Regex used by phased alignment to measure whole-body collisions.
    phase_control_mode: Whether phased alignment uses manual control points or auto-infers them from motion foot heights.
    phase_grounded_height: Default lower control point for phased alignment when no custom blend points are provided.
    phase_airborne_height: Default upper control point for phased alignment when no custom blend points are provided.
    phase_blend_points: Optional custom phased control points in `foot_height:blend_weight` format separated by commas.
    phase_window_s: Symmetric foot-height neighborhood used to classify grounded vs airborne phases.
    phase_lookahead_s: Future lookahead window used to prepare airborne lift before landing.
    phase_smoothing_s: Temporal smoothing applied to phased per-frame root z offsets.
    robot_variant: Tracking robot asset used to replay the motion inside MuJoCo.
    line_range: Range of lines to process from the CSV file.
  """
  using_single = input_file is not None or output_name is not None
  using_batch = inputm is not None or outputm is not None
  if using_single == using_batch:
    raise ValueError(
      "Use exactly one mode: (--input-file --output-name) or (--inputm [--outputm])."
    )
  if using_single and (input_file is None or output_name is None):
    raise ValueError("Single-file mode requires both --input-file and --output-name.")

  device = resolve_device(device)
  sim, scene, renderer = build_tracking_sim(
    output_fps=output_fps,
    device=device,
    render=render,
    robot_variant=robot_variant,
  )
  robot, robot_joint_indexes = get_tracking_robot(scene)
  try:
    if using_single:
      assert input_file is not None
      assert output_name is not None
      run_sim(
        sim=sim,
        scene=scene,
        robot=robot,
        robot_joint_indexes=robot_joint_indexes,
        input_fps=input_fps,
        input_file=input_file,
        output_fps=output_fps,
        output_name=output_name,
        render=render,
        ground_align=ground_align,
        clearance=clearance,
        whole_body_geom_pattern=whole_body_geom_pattern,
        phase_control_mode=phase_control_mode,
        phase_grounded_height=phase_grounded_height,
        phase_airborne_height=phase_airborne_height,
        phase_blend_points=phase_blend_points,
        phase_window_s=phase_window_s,
        phase_lookahead_s=phase_lookahead_s,
        phase_smoothing_s=phase_smoothing_s,
        line_range=line_range,
        renderer=renderer,
      )
      return

    assert inputm is not None
    input_root = Path(inputm)
    if not input_root.exists():
      raise FileNotFoundError(f"Batch input directory not found: {input_root}")
    if not input_root.is_dir():
      raise ValueError(f"Batch input must be a directory: {input_root}")

    output_root = Path(outputm) if outputm is not None else input_root
    files = collect_csv_files(input_root)
    if not files:
      print(f"[INFO]: No CSV files found in: {input_root}")
      return

    numeric_files = [path for path in files if is_numeric_motion_csv(path)]
    numeric_file_set = set(numeric_files)
    skipped_files = [path for path in files if path not in numeric_file_set]
    if not numeric_files:
      print(f"[INFO]: No numeric motion CSV files found in: {input_root}")
      if skipped_files:
        print(
          f"[INFO]: Skipped {len(skipped_files)} non-numeric/header CSV files."
        )
      return

    print(
      f"[INFO]: Found {len(files)} CSV files in batch input: {input_root} "
      f"({len(numeric_files)} numeric motion CSVs, {len(skipped_files)} skipped)"
    )
    if skipped_files:
      print("[INFO]: Skipping non-numeric/header CSV files such as SONIC source CSVs.")

    failed_files: list[tuple[Path, str]] = []
    converted_count = 0
    for src in numeric_files:
      dst = resolve_batch_output(input_root, output_root, src)
      print(f"\n[INFO]: Converting {src} -> {dst}")
      try:
        run_sim(
          sim=sim,
          scene=scene,
          robot=robot,
          robot_joint_indexes=robot_joint_indexes,
          input_fps=input_fps,
          input_file=str(src),
          output_fps=output_fps,
          output_name=str(dst),
          render=render,
          ground_align=ground_align,
          clearance=clearance,
          whole_body_geom_pattern=whole_body_geom_pattern,
          phase_control_mode=phase_control_mode,
          phase_grounded_height=phase_grounded_height,
          phase_airborne_height=phase_airborne_height,
          phase_blend_points=phase_blend_points,
          phase_window_s=phase_window_s,
          phase_lookahead_s=phase_lookahead_s,
          phase_smoothing_s=phase_smoothing_s,
          line_range=line_range,
          renderer=renderer,
        )
        converted_count += 1
      except Exception as exc:
        failed_files.append((src, str(exc)))
        print(f"[ERROR]: Failed to convert {src}: {exc}")

    print(
      f"\n[DONE]: Converted {converted_count}/{len(numeric_files)} numeric motion CSV files."
    )
    if skipped_files:
      print(f"[INFO]: Skipped {len(skipped_files)} non-numeric/header CSV files.")
    if failed_files:
      print(f"[WARN]: {len(failed_files)} files failed during conversion:")
      for src, message in failed_files:
        print(f"  - {src}: {message}")
  finally:
    if renderer is not None:
      renderer.close()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
