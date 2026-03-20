from pathlib import Path

import numpy as np
import tyro

import mjlab
from mjlab.scripts.csv_to_npz import (
  G1_FOOT_GEOM_PATTERN,
  MotionLoader,
  analyze_foot_penetration,
  build_tracking_sim,
  get_tracking_robot,
  resolve_device,
)


def main(
  input_file: str,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  clearance: float = 0.01,
  foot_geom_pattern: str = G1_FOOT_GEOM_PATTERN,
  line_range: tuple[int, int] | None = None,
  worst_frames: int = 10,
):
  """Analyze foot penetration against the ground plane for a motion CSV."""
  device = resolve_device(device)
  sim, scene, _ = build_tracking_sim(output_fps=output_fps, device=device, render=False)
  robot, robot_joint_indexes = get_tracking_robot(scene)

  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=device,
    line_range=line_range,
  )
  stats = analyze_foot_penetration(
    sim=sim,
    scene=scene,
    robot=robot,
    robot_joint_indexes=robot_joint_indexes,
    motion=motion,
    clearance=clearance,
    foot_geom_pattern=foot_geom_pattern,
  )

  worst_k = min(worst_frames, motion.output_frames)
  worst_indices = np.argsort(stats.frame_min_foot_z)[:worst_k]
  input_path = Path(input_file)

  print("\nFoot grounding summary")
  print(f"  input_file: {input_path}")
  print(f"  analyzed_frames: {motion.output_frames}")
  print(f"  foot_geoms: {', '.join(stats.foot_geom_names)}")
  print(f"  target_clearance_m: {clearance:.4f}")
  print(f"  global_min_foot_bottom_z_m: {stats.global_min_foot_z:.4f}")
  print(f"  recommended_global_z_offset_m: {stats.recommended_delta_z:.4f}")

  print("\nWorst frames")
  for frame_idx in worst_indices:
    min_z = float(stats.frame_min_foot_z[frame_idx])
    penetration = max(0.0, clearance - min_z)
    print(
      f"  frame={int(frame_idx):4d}  min_foot_bottom_z={min_z: .4f} m  "
      f"needed_lift={penetration: .4f} m"
    )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
