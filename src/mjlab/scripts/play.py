"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import torch
import tyro

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  play_window: bool = False
  """Spawn a play-only window-frame obstacle into the MuJoCo scene."""
  play_window_pose_mode: Literal["auto", "manual"] = "auto"
  """Auto aligns the window to the motion path, manual uses explicit center coordinates."""
  play_window_opening_width: float = 0.88
  play_window_opening_height: float = 0.88
  play_window_thickness: float = 0.40
  play_window_sill_height: float = 0.49
  play_window_outer_width: float = 2.0
  play_window_outer_height: float = 2.2
  play_window_center_x: float | None = None
  play_window_center_y: float | None = None
  play_window_center_z: float | None = None
  play_window_offset_x: float = 0.0
  play_window_offset_y: float = 0.0
  play_window_offset_z: float = 0.0
  play_window_align_body_name: str | None = None
  """Body name used for auto placement. Defaults to the tracking anchor body."""
  play_window_frame_index: int | None = None
  """Optional explicit motion frame index to align the window against in auto mode."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def _make_play_window_spec(
  *,
  center: tuple[float, float, float],
  opening_width: float,
  opening_height: float,
  thickness: float,
  sill_height: float,
  outer_width: float,
  outer_height: float,
  rgba: tuple[float, float, float, float] = (0.16, 0.72, 0.92, 0.45),
) -> mujoco.MjSpec:
  if opening_width <= 0.0 or opening_height <= 0.0 or thickness <= 0.0:
    raise ValueError("Window opening width/height and thickness must be positive.")
  if sill_height < 0.0:
    raise ValueError("Window sill height must be non-negative.")
  if outer_width <= opening_width:
    raise ValueError("Window outer width must be larger than the opening width.")
  opening_top = sill_height + opening_height
  if outer_height <= opening_top:
    raise ValueError(
      "Window outer height must be larger than sill height + opening height."
    )

  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="play_window_frame", pos=center)

  depth_half = thickness / 2.0
  bottom_half_height = sill_height / 2.0
  top_height = outer_height - opening_top
  top_half_height = top_height / 2.0
  side_width = (outer_width - opening_width) / 2.0
  side_half_width = side_width / 2.0
  side_center_y = opening_width / 2.0 + side_half_width
  wall_center_z = outer_height / 2.0

  if bottom_half_height > 0.0:
    body.add_geom(
      name="play_window_bottom",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(depth_half, outer_width / 2.0, bottom_half_height),
      pos=(0.0, 0.0, bottom_half_height),
      rgba=rgba,
    )

  body.add_geom(
    name="play_window_top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(depth_half, outer_width / 2.0, top_half_height),
    pos=(0.0, 0.0, opening_top + top_half_height),
    rgba=rgba,
  )
  body.add_geom(
    name="play_window_left",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(depth_half, side_half_width, wall_center_z),
    pos=(0.0, -side_center_y, wall_center_z),
    rgba=rgba,
  )
  body.add_geom(
    name="play_window_right",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(depth_half, side_half_width, wall_center_z),
    pos=(0.0, side_center_y, wall_center_z),
    rgba=rgba,
  )
  return spec


def _resolve_play_window_center(
  *,
  cfg: PlayConfig,
  env_cfg,
  motion_file: str | None,
  default_align_body_name: str | None,
) -> tuple[tuple[float, float, float], str, int | None]:
  opening_center_z = (
    cfg.play_window_center_z
    if cfg.play_window_center_z is not None
    else cfg.play_window_sill_height + 0.5 * cfg.play_window_opening_height
  )

  if cfg.play_window_pose_mode == "manual":
    if cfg.play_window_center_x is None or cfg.play_window_center_y is None:
      raise ValueError(
        "Manual play-window placement requires both `--play-window-center-x` "
        "and `--play-window-center-y`."
      )
    center = (
      cfg.play_window_center_x + cfg.play_window_offset_x,
      cfg.play_window_center_y + cfg.play_window_offset_y,
      opening_center_z + cfg.play_window_offset_z,
    )
    return center, "manual", None

  if motion_file is None:
    raise ValueError(
      "Auto play-window placement requires a resolved motion file. "
      "Provide `--motion-file` or use a tracking run that resolves one."
    )

  align_body_name = cfg.play_window_align_body_name or default_align_body_name
  if align_body_name is None:
    raise ValueError(
      "Auto play-window placement could not determine an alignment body. "
      "Provide `--play-window-align-body-name`."
    )

  robot_cfg = env_cfg.scene.entities["robot"]
  robot_entity = robot_cfg.build()
  try:
    body_index = robot_entity.body_names.index(align_body_name)
  except ValueError as exc:
    raise ValueError(
      f"Body {align_body_name!r} was not found in the robot body list. "
      f"Available bodies include: {robot_entity.body_names}"
    ) from exc

  motion_data = np.load(motion_file)
  body_pos_w = motion_data["body_pos_w"]
  if body_index >= body_pos_w.shape[1]:
    raise ValueError(
      f"Motion file only contains {body_pos_w.shape[1]} bodies, but body index "
      f"{body_index} was requested for {align_body_name!r}."
    )

  if cfg.play_window_frame_index is not None:
    frame_index = int(cfg.play_window_frame_index)
    if not 0 <= frame_index < body_pos_w.shape[0]:
      raise ValueError(
        f"`--play-window-frame-index` must be within [0, {body_pos_w.shape[0] - 1}], "
        f"got {frame_index}."
      )
  else:
    body_heights = body_pos_w[:, body_index, 2]
    frame_index = int(np.argmin(np.abs(body_heights - opening_center_z)))

  center = (
    float(body_pos_w[frame_index, body_index, 0]) + cfg.play_window_offset_x,
    float(body_pos_w[frame_index, body_index, 1]) + cfg.play_window_offset_y,
    float(opening_center_z) + cfg.play_window_offset_z,
  )
  return center, align_body_name, frame_index


def _inject_play_window_entity(
  *,
  env_cfg,
  cfg: PlayConfig,
  motion_file: str | None,
  default_align_body_name: str | None,
) -> None:
  center, align_body_name, frame_index = _resolve_play_window_center(
    cfg=cfg,
    env_cfg=env_cfg,
    motion_file=motion_file,
    default_align_body_name=default_align_body_name,
  )
  opening_center_z = center[2]
  print(
    "[INFO]: Play window enabled: "
    f"opening={cfg.play_window_opening_width:.2f} x {cfg.play_window_opening_height:.2f} m, "
    f"thickness={cfg.play_window_thickness:.2f} m, "
    f"sill={cfg.play_window_sill_height:.2f} m, "
    f"center=({center[0]:.3f}, {center[1]:.3f}, {opening_center_z:.3f})"
  )
  if frame_index is not None:
    print(
      "[INFO]: Play window auto placement: "
      f"aligned to body {align_body_name!r} at motion frame {frame_index}."
    )
  else:
    print("[INFO]: Play window placement: using manual center coordinates.")

  env_cfg.scene.entities = dict(env_cfg.scene.entities)
  env_cfg.scene.entities["play_window"] = EntityCfg(
    spec_fn=lambda: _make_play_window_spec(
      center=center,
      opening_width=cfg.play_window_opening_width,
      opening_height=cfg.play_window_opening_height,
      thickness=cfg.play_window_thickness,
      sill_height=cfg.play_window_sill_height,
      outer_width=cfg.play_window_outer_width,
      outer_height=cfg.play_window_outer_height,
    )
  )


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  resolved_motion_file = None
  default_align_body_name = None
  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    resolved_motion_file = motion_cmd.motion_file or None
    default_align_body_name = motion_cmd.anchor_body_name

  if cfg.play_window:
    _inject_play_window_entity(
      env_cfg=env_cfg,
      cfg=cfg,
      motion_file=resolved_motion_file,
      default_align_body_name=default_align_body_name,
    )

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    requested_num_envs = cfg.num_envs
    terrain_cfg = env_cfg.scene.terrain
    if (
      "stratified_terrain_placement" in env_cfg.events
      and terrain_cfg is not None
      and terrain_cfg.terrain_generator is not None
    ):
      max_visible_envs = (
        terrain_cfg.terrain_generator.num_rows * terrain_cfg.terrain_generator.num_cols
      )
      if requested_num_envs > max_visible_envs:
        print(
          "[WARN]: Stratified terrain play visualization uses at most one env per visible patch. "
          f"Clamping num_envs from {requested_num_envs} to {max_visible_envs}."
        )
        requested_num_envs = max_visible_envs
    env_cfg.scene.num_envs = requested_num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
