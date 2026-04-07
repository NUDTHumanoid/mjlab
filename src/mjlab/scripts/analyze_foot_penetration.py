from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.scripts.csv_to_npz import (
  CollisionGroundingStats,
  G1_FOOT_GEOM_PATTERN,
  GroundingStats,
  MotionLoader,
  PhaseBlendControlSuggestion,
  _compute_geom_bottom_heights,
  _write_motion_frame_to_sim,
  build_tracking_sim,
  compute_phased_ground_alignment_offsets,
  get_tracking_robot,
  resolve_device,
  suggest_phase_blend_control_points,
)

ALL_COLLISION_GEOM_PATTERN = r".*_collision$"
_BODY_PART_PATTERNS = (
  ("feet", G1_FOOT_GEOM_PATTERN),
  ("hands", r"^(left|right)_hand_collision$"),
  ("wrists", r"^(left|right)_wrist_collision$"),
  ("head", r"^head_collision$"),
  ("torso", r"^torso_collision$"),
  ("pelvis", r"^pelvis_collision$"),
)


@dataclass(frozen=True)
class CollisionBottomAnalysis:
  geom_names: tuple[str, ...]
  frame_bottom_heights: np.ndarray

  def summarize(
    self,
    *,
    label: str,
    geom_pattern: str,
    clearance: float,
  ) -> "CollisionBottomSummary":
    geom_indices = _match_geom_indices(self.geom_names, geom_pattern)
    if not geom_indices:
      raise ValueError(f"No geoms matched pattern: {geom_pattern}")

    subset_heights = self.frame_bottom_heights[:, geom_indices]
    frame_min_local = np.argmin(subset_heights, axis=1).astype(np.int32)
    frame_min_bottom_z = subset_heights[np.arange(subset_heights.shape[0]), frame_min_local]
    subset_geom_names = tuple(self.geom_names[idx] for idx in geom_indices)
    return CollisionBottomSummary(
      label=label,
      geom_names=subset_geom_names,
      frame_min_bottom_z=frame_min_bottom_z.astype(np.float32),
      frame_min_geom_indices=frame_min_local,
      clearance=clearance,
    )


@dataclass(frozen=True)
class CollisionBottomSummary:
  label: str
  geom_names: tuple[str, ...]
  frame_min_bottom_z: np.ndarray
  frame_min_geom_indices: np.ndarray
  clearance: float

  @property
  def analyzed_frames(self) -> int:
    return int(self.frame_min_bottom_z.shape[0])

  @property
  def global_min_frame(self) -> int:
    return int(np.argmin(self.frame_min_bottom_z))

  @property
  def global_min_bottom_z(self) -> float:
    return float(self.frame_min_bottom_z[self.global_min_frame])

  @property
  def recommended_delta_z(self) -> float:
    return max(0.0, self.clearance - self.global_min_bottom_z)

  @property
  def global_min_geom_name(self) -> str:
    return self.geom_name_for_frame(self.global_min_frame)

  def geom_name_for_frame(self, frame_idx: int) -> str:
    return self.geom_names[int(self.frame_min_geom_indices[frame_idx])]


@dataclass(frozen=True)
class PhaseSuggestion:
  label: str
  start_frame: int
  end_frame: int
  output_dt: float
  nearby_foot_height_min: float
  nearby_foot_height_max: float
  blend_weight_min: float
  blend_weight_max: float
  recommended_lift_min: float
  recommended_lift_mean: float
  recommended_lift_max: float
  max_lift_frame: int
  max_lift_culprit_geom: str

  @property
  def start_time_s(self) -> float:
    return self.start_frame * self.output_dt

  @property
  def end_time_s(self) -> float:
    return self.end_frame * self.output_dt

  @property
  def duration_s(self) -> float:
    return (self.end_frame - self.start_frame + 1) * self.output_dt


def _compute_phase_blend_weight(
  nearby_foot_height: np.ndarray,
  phase_blend_control_points: np.ndarray,
) -> np.ndarray:
  if phase_blend_control_points.shape[0] == 1:
    return np.full(
      nearby_foot_height.shape,
      phase_blend_control_points[0, 1],
      dtype=np.float64,
    )
  return np.interp(
    nearby_foot_height,
    phase_blend_control_points[:, 0],
    phase_blend_control_points[:, 1],
  )


def _build_phase_suggestion(
  *,
  raw_label: str,
  prev_raw_label: str | None,
  next_raw_label: str | None,
  start_frame: int,
  end_frame: int,
  output_dt: float,
  nearby_foot_height: np.ndarray,
  blend_weight: np.ndarray,
  recommended_offsets: np.ndarray,
  whole_body_summary: CollisionBottomSummary,
) -> PhaseSuggestion:
  seg_foot_height = nearby_foot_height[start_frame : end_frame + 1]
  seg_blend_weight = blend_weight[start_frame : end_frame + 1]
  seg_offsets = recommended_offsets[start_frame : end_frame + 1]

  label = raw_label
  if raw_label == "transition":
    if prev_raw_label == "grounded" and next_raw_label == "airborne":
      label = "takeoff"
    elif prev_raw_label == "airborne" and next_raw_label == "grounded":
      label = "landing"
    else:
      foot_height_delta = float(seg_foot_height[-1] - seg_foot_height[0])
      if foot_height_delta > 1e-4:
        label = "takeoff"
      elif foot_height_delta < -1e-4:
        label = "landing"

  local_peak_idx = int(np.argmax(seg_offsets))
  max_lift_frame = start_frame + local_peak_idx
  return PhaseSuggestion(
    label=label,
    start_frame=start_frame,
    end_frame=end_frame,
    output_dt=output_dt,
    nearby_foot_height_min=float(seg_foot_height.min()),
    nearby_foot_height_max=float(seg_foot_height.max()),
    blend_weight_min=float(seg_blend_weight.min()),
    blend_weight_max=float(seg_blend_weight.max()),
    recommended_lift_min=float(seg_offsets.min()),
    recommended_lift_mean=float(seg_offsets.mean()),
    recommended_lift_max=float(seg_offsets.max()),
    max_lift_frame=max_lift_frame,
    max_lift_culprit_geom=whole_body_summary.geom_name_for_frame(max_lift_frame),
  )


def _suggest_phases(
  *,
  foot_summary: CollisionBottomSummary,
  whole_body_summary: CollisionBottomSummary,
  clearance: float,
  output_dt: float,
  phase_grounded_height: float | None,
  phase_airborne_height: float | None,
  phase_blend_points: str | None,
  phase_window_s: float,
  phase_lookahead_s: float,
  phase_smoothing_s: float,
) -> tuple[PhaseBlendControlSuggestion, np.ndarray, list[PhaseSuggestion]]:
  phase_control = suggest_phase_blend_control_points(
    foot_heights=foot_summary.frame_min_bottom_z,
    output_dt=output_dt,
    phase_window_s=phase_window_s,
    phase_grounded_height=phase_grounded_height,
    phase_airborne_height=phase_airborne_height,
    phase_blend_points=phase_blend_points,
  )
  phase_blend_control_points = phase_control.control_points
  foot_stats = GroundingStats(
    foot_geom_names=foot_summary.geom_names,
    frame_min_foot_z=foot_summary.frame_min_bottom_z,
    global_min_foot_z=foot_summary.global_min_bottom_z,
    recommended_delta_z=foot_summary.recommended_delta_z,
  )
  whole_body_stats = CollisionGroundingStats(
    geom_names=whole_body_summary.geom_names,
    frame_min_bottom_z=whole_body_summary.frame_min_bottom_z,
    frame_min_geom_indices=whole_body_summary.frame_min_geom_indices,
    global_min_bottom_z=whole_body_summary.global_min_bottom_z,
    recommended_delta_z=whole_body_summary.recommended_delta_z,
  )
  recommended_offsets = compute_phased_ground_alignment_offsets(
    foot_stats=foot_stats,
    whole_body_stats=whole_body_stats,
    clearance=clearance,
    output_dt=output_dt,
    phase_blend_control_points=phase_blend_control_points,
    phase_window_s=phase_window_s,
    phase_lookahead_s=phase_lookahead_s,
    phase_smoothing_s=phase_smoothing_s,
  ).astype(np.float64)

  nearby_foot_height = phase_control.nearby_foot_height.astype(np.float64)
  blend_weight = _compute_phase_blend_weight(
    nearby_foot_height,
    phase_blend_control_points,
  )

  raw_labels = np.full(blend_weight.shape, "transition", dtype=object)
  raw_labels[blend_weight <= 0.05] = "grounded"
  raw_labels[blend_weight >= 0.95] = "airborne"

  raw_segments: list[tuple[str, int, int]] = []
  start_frame = 0
  current_label = str(raw_labels[0])
  for frame_idx in range(1, raw_labels.shape[0]):
    label = str(raw_labels[frame_idx])
    if label == current_label:
      continue
    raw_segments.append((current_label, start_frame, frame_idx - 1))
    start_frame = frame_idx
    current_label = label
  raw_segments.append((current_label, start_frame, raw_labels.shape[0] - 1))

  phase_suggestions: list[PhaseSuggestion] = []
  for idx, (raw_label, start_frame, end_frame) in enumerate(raw_segments):
    prev_raw_label = raw_segments[idx - 1][0] if idx > 0 else None
    next_raw_label = raw_segments[idx + 1][0] if idx + 1 < len(raw_segments) else None
    phase_suggestions.append(
      _build_phase_suggestion(
        raw_label=raw_label,
        prev_raw_label=prev_raw_label,
        next_raw_label=next_raw_label,
        start_frame=start_frame,
        end_frame=end_frame,
        output_dt=output_dt,
        nearby_foot_height=nearby_foot_height,
        blend_weight=blend_weight,
        recommended_offsets=recommended_offsets,
        whole_body_summary=whole_body_summary,
      )
    )
  return (
    phase_control,
    recommended_offsets,
    phase_suggestions,
  )


def _match_geom_indices(geom_names: tuple[str, ...], geom_pattern: str) -> list[int]:
  compiled = re.compile(geom_pattern)
  return [idx for idx, name in enumerate(geom_names) if compiled.search(name)]


def _analyze_collision_bottom_heights(
  motion: MotionLoader,
  *,
  sim,
  scene,
  robot,
  robot_joint_indexes: list[int],
  geom_pattern: str,
  show_progress: bool = True,
) -> CollisionBottomAnalysis:
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

  frame_bottom_heights = np.empty((motion.output_frames, len(geom_ids)), dtype=np.float32)
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
    frame_bottom_heights[frame_idx] = bottom_heights.detach().cpu().numpy()

  return CollisionBottomAnalysis(
    geom_names=tuple(geom_names),
    frame_bottom_heights=frame_bottom_heights,
  )


def _print_summary(summary: CollisionBottomSummary, input_path: Path) -> None:
  print(f"\n{summary.label} grounding summary")
  print(f"  input_file: {input_path}")
  print(f"  analyzed_frames: {summary.analyzed_frames}")
  print(f"  matched_geoms ({len(summary.geom_names)}): {', '.join(summary.geom_names)}")
  print(f"  target_clearance_m: {summary.clearance:.4f}")
  print(f"  global_min_bottom_z_m: {summary.global_min_bottom_z:.4f}")
  print(f"  global_min_geom: {summary.global_min_geom_name}")
  print(f"  global_min_frame: {summary.global_min_frame}")
  print(f"  recommended_global_z_offset_m: {summary.recommended_delta_z:.4f}")


def _print_worst_frames(summary: CollisionBottomSummary, worst_frames: int) -> None:
  worst_k = min(worst_frames, summary.analyzed_frames)
  worst_indices = np.argsort(summary.frame_min_bottom_z)[:worst_k]

  print(f"\n{summary.label} worst frames")
  for frame_idx in worst_indices:
    frame_idx = int(frame_idx)
    min_z = float(summary.frame_min_bottom_z[frame_idx])
    penetration = max(0.0, summary.clearance - min_z)
    print(
      f"  frame={frame_idx:4d}  min_bottom_z={min_z: .4f} m  "
      f"culprit={summary.geom_name_for_frame(frame_idx)}  "
      f"needed_lift={penetration: .4f} m"
    )


def _print_body_part_minima(
  analysis: CollisionBottomAnalysis,
  *,
  clearance: float,
) -> None:
  print("\nSelected body-part minima")
  for label, geom_pattern in _BODY_PART_PATTERNS:
    try:
      summary = analysis.summarize(
        label=label,
        geom_pattern=geom_pattern,
        clearance=clearance,
      )
    except ValueError:
      continue
    print(
      f"  {label:6s} min_bottom_z={summary.global_min_bottom_z: .4f} m  "
      f"frame={summary.global_min_frame:4d}  "
      f"culprit={summary.global_min_geom_name}  "
      f"needed_lift={summary.recommended_delta_z: .4f} m"
    )


def _format_phase_blend_control_points(phase_blend_control_points: np.ndarray) -> str:
  return ", ".join(
    f"{float(height):.3f}:{float(weight):.2f}"
    for height, weight in phase_blend_control_points
  )


def _print_phase_suggestions(
  *,
  phase_control: PhaseBlendControlSuggestion,
  recommended_offsets: np.ndarray,
  phase_suggestions: list[PhaseSuggestion],
  phase_window_s: float,
  phase_lookahead_s: float,
  phase_smoothing_s: float,
) -> None:
  print("\nAutomatic phased-alignment suggestion")
  print(f"  phase_control_source: {phase_control.source}")
  print(
    "  phase cue control points (foot_height:blend_weight): "
    f"{_format_phase_blend_control_points(phase_control.control_points)}"
  )
  grounded_height = phase_control.grounded_height
  airborne_height = phase_control.airborne_height
  if grounded_height is not None:
    print(f"  suggested_phase_grounded_height: {grounded_height:.4f}")
  if airborne_height is not None:
    print(f"  suggested_phase_airborne_height: {airborne_height:.4f}")
  print(
    "  suggested_phase_blend_points: "
    f"\"{phase_control.blend_points_cli}\""
  )
  print(
    "  csv_to_npz_hint: "
    f"--ground-align phased --phase-blend-points \"{phase_control.blend_points_cli}\""
  )
  print(
    "  detected_foot_height_centers_m: "
    f"grounded={phase_control.grounded_center:.4f}, "
    f"airborne={phase_control.airborne_center:.4f}"
  )
  print(f"  detected_airborne_segments: {phase_control.airborne_segments}")
  print(f"  airborne_frame_ratio: {phase_control.airborne_frame_ratio:.3f}")
  print(f"  phase_window_s: {phase_window_s:.3f}")
  print(f"  phase_lookahead_s: {phase_lookahead_s:.3f}")
  print(f"  phase_smoothing_s: {phase_smoothing_s:.3f}")
  print(
    "  recommended_offset_range_m: "
    f"[{float(recommended_offsets.min()):.4f}, {float(recommended_offsets.max()):.4f}]"
  )
  print(f"  note: {phase_control.note}")

  print("\nSuggested coarse phases")
  for idx, phase in enumerate(phase_suggestions, start=1):
    print(
      f"  phase={idx:2d}  type={phase.label:8s}  "
      f"frames={phase.start_frame:4d}-{phase.end_frame:4d}  "
      f"time=[{phase.start_time_s:.3f}, {phase.end_time_s:.3f}] s  "
      f"duration={phase.duration_s:.3f} s  "
      f"foot_height=[{phase.nearby_foot_height_min:.4f}, {phase.nearby_foot_height_max:.4f}] m  "
      f"blend=[{phase.blend_weight_min:.2f}, {phase.blend_weight_max:.2f}]  "
      f"lift(min/mean/max)=[{phase.recommended_lift_min:.4f}, "
      f"{phase.recommended_lift_mean:.4f}, {phase.recommended_lift_max:.4f}] m  "
      f"peak_frame={phase.max_lift_frame:4d}  "
      f"peak_culprit={phase.max_lift_culprit_geom}"
    )


def main(
  input_file: str,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  clearance: float = 0.01,
  foot_geom_pattern: str = G1_FOOT_GEOM_PATTERN,
  collision_geom_pattern: str = ALL_COLLISION_GEOM_PATTERN,
  line_range: tuple[int, int] | None = None,
  worst_frames: int = 10,
  phase_grounded_height: float | None = None,
  phase_airborne_height: float | None = None,
  phase_blend_points: str | None = None,
  phase_window_s: float = 0.12,
  phase_lookahead_s: float = 0.24,
  phase_smoothing_s: float = 0.08,
):
  """Analyze foot and whole-body collision penetration against the ground plane."""
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
  collision_analysis = _analyze_collision_bottom_heights(
    motion,
    sim=sim,
    scene=scene,
    robot=robot,
    robot_joint_indexes=robot_joint_indexes,
    geom_pattern=collision_geom_pattern,
  )

  whole_body_summary = collision_analysis.summarize(
    label="Whole-body collision",
    geom_pattern=collision_geom_pattern,
    clearance=clearance,
  )
  try:
    foot_summary = collision_analysis.summarize(
      label="Foot",
      geom_pattern=foot_geom_pattern,
      clearance=clearance,
    )
  except ValueError:
    foot_analysis = _analyze_collision_bottom_heights(
      motion,
      sim=sim,
      scene=scene,
      robot=robot,
      robot_joint_indexes=robot_joint_indexes,
      geom_pattern=foot_geom_pattern,
      show_progress=False,
    )
    foot_summary = foot_analysis.summarize(
      label="Foot",
      geom_pattern=foot_geom_pattern,
      clearance=clearance,
    )

  input_path = Path(input_file)
  _print_summary(foot_summary, input_path)
  _print_worst_frames(foot_summary, worst_frames)
  _print_summary(whole_body_summary, input_path)
  _print_worst_frames(whole_body_summary, worst_frames)
  _print_body_part_minima(collision_analysis, clearance=clearance)
  (
    phase_control,
    recommended_offsets,
    phase_suggestions,
  ) = _suggest_phases(
    foot_summary=foot_summary,
    whole_body_summary=whole_body_summary,
    clearance=clearance,
    output_dt=motion.output_dt,
    phase_grounded_height=phase_grounded_height,
    phase_airborne_height=phase_airborne_height,
    phase_blend_points=phase_blend_points,
    phase_window_s=phase_window_s,
    phase_lookahead_s=phase_lookahead_s,
    phase_smoothing_s=phase_smoothing_s,
  )
  _print_phase_suggestions(
    phase_control=phase_control,
    recommended_offsets=recommended_offsets,
    phase_suggestions=phase_suggestions,
    phase_window_s=phase_window_s,
    phase_lookahead_s=phase_lookahead_s,
    phase_smoothing_s=phase_smoothing_s,
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
