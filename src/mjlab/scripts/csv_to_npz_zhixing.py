"""Replay Zhixing motion CSV files and export tracking-ready `.npz` files."""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.asset_zoo.robots.zhixing.zhixing_constants import ZHIXING_FOOT_GEOM_PATTERN
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.scripts.csv_to_npz import (
  MotionLoader,
  _write_motion_frame_to_sim,
  analyze_collision_grounding,
  analyze_foot_penetration,
  apply_global_ground_alignment,
  collect_csv_files,
  compute_phased_ground_alignment_offsets,
  is_numeric_motion_csv,
  resolve_batch_output,
  resolve_device,
  suggest_phase_blend_control_points,
)
from mjlab.tasks.tracking.config.zhixing.env_cfgs import zhixing_flat_tracking_env_cfg
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

ZHIXING_TRACKING_JOINT_NAMES = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
]
ZHIXING_WHOLE_BODY_GEOM_PATTERN = r"^(left|right)_ankle_roll_geom([1-9]|10)?$"


def build_tracking_sim(
  output_fps: float,
  device: str,
  render: bool = False,
) -> tuple[Simulation, Scene, OffscreenRenderer | None]:
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  env_cfg = zhixing_flat_tracking_env_cfg()
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
    ZHIXING_TRACKING_JOINT_NAMES,
    preserve_order=True,
  )[0]
  return robot, robot_joint_indexes


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
      foot_geom_pattern=ZHIXING_FOOT_GEOM_PATTERN,
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
      foot_geom_pattern=ZHIXING_FOOT_GEOM_PATTERN,
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

    _write_motion_frame_to_sim(
      sim,
      scene,
      robot,
      robot_joint_indexes,
      motion,
      frame_idx,
    )
    if render and renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())

    if not file_saved:
      log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
      log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
      log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
      log["body_quat_w"].append(
        robot.data.body_link_quat_w[0, :].cpu().numpy().copy()
      )
      log["body_lin_vel_w"].append(
        robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
      )
      log["body_ang_vel_w"].append(
        robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
      )

      torch.testing.assert_close(
        robot.data.body_link_lin_vel_w[0, 0],
        motion_base_lin_vel[0],
      )
      torch.testing.assert_close(
        robot.data.body_link_ang_vel_w[0, 0],
        motion_base_ang_vel[0],
      )

      frame_count += 1
      pbar.update(1)

      if frame_count % 100 == 0:
        elapsed_time = frame_count / output_fps
        pbar.set_description(f"Processing frames (t={elapsed_time:.1f}s)")

      if reset_flag and not file_saved:
        file_saved = True
        pbar.close()

        print("\nStacking arrays and saving data...")
        for key in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        ):
          log[key] = np.stack(log[key], axis=0)

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
  whole_body_geom_pattern: str = ZHIXING_WHOLE_BODY_GEOM_PATTERN,
  phase_control_mode: Literal["manual", "auto"] = "manual",
  phase_grounded_height: float | None = 0.03,
  phase_airborne_height: float | None = 0.10,
  phase_blend_points: str | None = None,
  phase_window_s: float = 0.12,
  phase_lookahead_s: float = 0.24,
  phase_smoothing_s: float = 0.08,
  line_range: tuple[int, int] | None = None,
):
  """Replay Zhixing motion CSV files and output `.npz` files."""
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
        print(f"[INFO]: Skipped {len(skipped_files)} non-numeric/header CSV files.")
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
