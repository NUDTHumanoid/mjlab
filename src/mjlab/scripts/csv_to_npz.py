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


@dataclass(frozen=True)
class GroundingStats:
  foot_geom_names: tuple[str, ...]
  frame_min_foot_z: np.ndarray
  global_min_foot_z: float
  recommended_delta_z: float


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
) -> tuple[Simulation, Scene, OffscreenRenderer | None]:
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
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
  foot_geom_ids, foot_geom_names = robot.find_geoms(
    foot_geom_pattern, preserve_order=True
  )
  if not foot_geom_ids:
    raise ValueError(f"No foot geoms matched pattern: {foot_geom_pattern}")

  global_geom_ids = robot.indexing.geom_ids[foot_geom_ids].cpu().numpy()
  geom_sizes = torch.tensor(
    sim.mj_model.geom_size[global_geom_ids],
    dtype=torch.float32,
    device=sim.device,
  )
  geom_types = np.asarray(sim.mj_model.geom_type[global_geom_ids])

  frame_min_foot_z = np.empty(motion.output_frames, dtype=np.float32)
  frame_iterator = range(motion.output_frames)
  if show_progress:
    frame_iterator = tqdm(
      frame_iterator,
      total=motion.output_frames,
      desc="Analyzing foot grounding",
      unit="frame",
      ncols=100,
    )

  scene.reset()
  for frame_idx in frame_iterator:
    _write_motion_frame_to_sim(sim, scene, robot, robot_joint_indexes, motion, frame_idx)
    geom_pose_w = robot.data.geom_pose_w[0, foot_geom_ids]
    bottom_heights = _compute_geom_bottom_heights(
      geom_pos_w=geom_pose_w[:, :3],
      geom_quat_w=geom_pose_w[:, 3:7],
      geom_sizes=geom_sizes,
      geom_types=geom_types,
    )
    frame_min_foot_z[frame_idx] = float(bottom_heights.min().item())

  global_min_foot_z = float(frame_min_foot_z.min())
  recommended_delta_z = max(0.0, clearance - global_min_foot_z)
  return GroundingStats(
    foot_geom_names=tuple(foot_geom_names),
    frame_min_foot_z=frame_min_foot_z,
    global_min_foot_z=global_min_foot_z,
    recommended_delta_z=recommended_delta_z,
  )


def apply_global_ground_alignment(motion: MotionLoader, stats: GroundingStats) -> None:
  if stats.recommended_delta_z <= 0.0:
    return
  motion.motion_base_poss_input[:, 2] += stats.recommended_delta_z
  motion.motion_base_poss[:, 2] += stats.recommended_delta_z


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
  ground_align: Literal["none", "global"],
  clearance: float,
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
  ground_align: Literal["none", "global"] = "none",
  clearance: float = 0.01,
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
