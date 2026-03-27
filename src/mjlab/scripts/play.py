"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
# 删除: from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  # 删除: registry_name, wandb_run_path, wandb_checkpoint_name, checkpoint_file
  # 新添: checkpoint 字段，用于指定本地 checkpoint 路径或自动选择
  checkpoint: str | None = None
  """Path to a specific checkpoint file (e.g., 'logs/rsl_rl/exp1/2025-03-13_10-30-00/model_30000.pt').
     If None, the latest checkpoint from the latest run under logs/rsl_rl/{experiment_name} will be used."""
  motion_file: str | None = None
  """Motion file path or name. If a name without path is given (e.g., 'jump_up01_poses'),
     the script will look for 'source/motions/{name}.npz'. Absolute paths are used as is."""
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

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def find_latest_checkpoint(experiment_name: str) -> Path | None:
  """Find the latest checkpoint under logs/rsl_rl/{experiment_name}/.
     Returns the path to the checkpoint with the largest iteration number in the newest run directory.
  """
  log_root = Path("logs") / "rsl_rl" / experiment_name
  if not log_root.exists():
    return None

  # Get all run directories (timestamped) sorted by name (which includes datetime)
  run_dirs = [d for d in log_root.iterdir() if d.is_dir()]
  if not run_dirs:
    return None
  # Sort descending so the newest (largest timestamp) is first
  run_dirs.sort(reverse=True)
  latest_run_dir = run_dirs[0]

  # Find all .pt files in the latest run directory that match "model_*.pt"
  checkpoint_files = list(latest_run_dir.glob("model_*.pt"))
  if not checkpoint_files:
    return None

  # Extract iteration numbers from filenames (e.g., model_30000.pt -> 30000)
  def get_iter(p: Path) -> int:
    try:
      return int(p.stem.split('_')[1])
    except (IndexError, ValueError):
      return 0

  # Select the checkpoint with the largest iteration number
  latest_checkpoint = max(checkpoint_files, key=get_iter)
  return latest_checkpoint


def resolve_motion_file(motion_spec: str | None) -> str | None:
  """Resolve motion file path. If motion_spec is None, return None.
     If motion_spec is an existing file, return its absolute path.
     Otherwise, assume it's a name and look for source/motions/{name}.npz.
  """
  if motion_spec is None:
    return None
  path = Path(motion_spec)
  if path.exists():
    return str(path.absolute())
  # Try default location: source/motions/{motion_spec}.npz
  default_path = Path("source/motions") / f"{motion_spec}.npz"
  if default_path.exists():
    return str(default_path.absolute())
  raise FileNotFoundError(f"Motion file not found: {motion_spec} (tried as absolute and in source/motions/)")


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

    # Resolve motion file (supports name or path)
    if cfg.motion_file is not None:
      resolved_motion = resolve_motion_file(cfg.motion_file)
      print(f"[INFO]: Using motion file: {resolved_motion}")
      motion_cmd.motion_file = resolved_motion
    else:
      # For trained mode, we might need to infer from checkpoint? But usually motion is required.
      # We'll require explicit --motion-file for tracking tasks.
      raise ValueError(
        "Tracking tasks require a motion file. Provide --motion-file <path_or_name>."
      )

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    if cfg.checkpoint is not None:
      # User specified checkpoint path
      resume_path = Path(cfg.checkpoint)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      log_dir = resume_path.parent
      print(f"[INFO]: Loading specified checkpoint: {resume_path.name}")
    else:
      # Auto-detect latest checkpoint
      latest_checkpoint = find_latest_checkpoint(agent_cfg.experiment_name)
      if latest_checkpoint is None:
        raise FileNotFoundError(
          f"No checkpoint found under logs/rsl_rl/{agent_cfg.experiment_name}/. "
          "Please train a policy first or provide --checkpoint explicitly."
        )
      resume_path = latest_checkpoint
      log_dir = resume_path.parent
      print(f"[INFO]: Auto-detected latest checkpoint: {resume_path}")

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
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

  # Parse the rest of the arguments
  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
