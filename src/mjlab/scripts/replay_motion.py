"""Replay tracking motion `.npz` files directly in mjlab viewers.

This reuses the same task-loading flow as ``play``, but instead of stepping a
policy through the environment, it writes each motion frame directly into the
simulation state. The default backend is MuJoCo's native viewer, which makes it
easy to pause, single-step, and inspect the replayed motion.

Example:
  uv run replay-motion Mjlab-Tracking-Flat-Unitree-G1 \
    --motion-file /path/to/motion.npz
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import mujoco
import numpy as np
import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.csv_to_npz import (
  G1_FOOT_GEOM_PATTERN,
  _compute_geom_bottom_heights,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.base import VerbosityLevel, ViewerAction
from mjlab.viewer.native.keys import KEY_D

_REPLAY_TYRO_FLAGS = tuple(
  flag for flag in mjlab.TYRO_FLAGS if flag is not tyro.conf.FlagConversionOff
)


@dataclass(frozen=True)
class ReplayMotionConfig:
  motion_file: str
  num_envs: int = 1
  device: str | None = None
  viewer: Literal["auto", "native", "viser"] = "native"
  loop: bool = True
  start_paused: bool = False
  root_body_name: str | None = None
  foot_geom_pattern: str | None = G1_FOOT_GEOM_PATTERN
  reference_viz: Literal["none", "ghost", "frames"] = "none"
  print_summary: bool = True
  verbosity: Literal["silent", "info", "debug"] = "silent"


class _ReplayZeroPolicy:
  """Dummy policy required by the shared viewer interfaces."""

  def __init__(self, action_shape: tuple[int, ...], device: torch.device | str):
    self._action_shape = action_shape
    self._device = device

  def __call__(self, obs: Any) -> torch.Tensor:
    del obs
    return torch.zeros(self._action_shape, device=self._device)


class MotionReplayEnvAdapter:
  """Wrap a tracking env so viewers can replay motion frames via ``step()``."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    motion_term: MotionCommand,
    *,
    root_body_name: str | None = None,
    foot_geom_pattern: str | None = G1_FOOT_GEOM_PATTERN,
    loop: bool = True,
  ) -> None:
    self._env = env
    self._motion_term = motion_term
    self._motion = motion_term.motion
    self._loop = loop

    self._robot = env.scene[motion_term.cfg.entity_name]
    self._root_body_name = root_body_name or motion_term.cfg.body_names[0]
    self._foot_geom_pattern = foot_geom_pattern
    if self._root_body_name not in motion_term.cfg.body_names:
      raise ValueError(
        f"Root body '{self._root_body_name}' is not present in motion body list "
        f"{motion_term.cfg.body_names}."
      )
    if self._robot.body_names[0] != self._root_body_name:
      raise ValueError(
        "Direct root replay expects the motion root body to match the robot root body. "
        f"Robot root='{self._robot.body_names[0]}', motion root='{self._root_body_name}'."
      )
    self._motion_root_idx = motion_term.cfg.body_names.index(self._root_body_name)
    self._foot_geom_ids: list[int] = []
    self._foot_geom_names: tuple[str, ...] = ()
    self._foot_geom_sizes: torch.Tensor | None = None
    self._foot_geom_types: np.ndarray | None = None
    self._current_min_foot_bottom_z = np.full(self.num_envs, np.nan, dtype=np.float32)
    self._setup_foot_geom_display()
    self._frame_idx = 0

    self._obs, self._extras = self._env.reset()
    self._reward = torch.zeros(self.num_envs, device=self.device)
    self._terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._timeouts = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    if self.total_frames <= 0:
      raise ValueError("Motion file does not contain any replayable frames.")

    self._apply_frame(self._frame_idx)

  @property
  def num_envs(self) -> int:
    return self._env.num_envs

  @property
  def device(self) -> torch.device | str:
    return self._env.device

  @property
  def cfg(self):
    return self._env.cfg

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self._env

  @property
  def total_frames(self) -> int:
    return int(self._motion.time_step_total)

  @property
  def current_frame(self) -> int:
    return int(self._frame_idx)

  @property
  def root_body_name(self) -> str:
    return self._root_body_name

  @property
  def loop_enabled(self) -> bool:
    return self._loop

  @property
  def foot_geom_pattern(self) -> str | None:
    return self._foot_geom_pattern

  @property
  def foot_bottom_display_enabled(self) -> bool:
    return self._foot_geom_sizes is not None and len(self._foot_geom_ids) > 0

  @property
  def foot_geom_count(self) -> int:
    return len(self._foot_geom_ids)

  def get_observations(self) -> Any:
    return self._obs

  def step(self, actions: torch.Tensor) -> tuple[Any, ...]:
    del actions

    if self._loop:
      self._frame_idx = (self._frame_idx + 1) % self.total_frames
    else:
      self._frame_idx = min(self._frame_idx + 1, self.total_frames - 1)

    self._apply_frame(self._frame_idx)
    return (
      self._obs,
      self._reward,
      self._terminated,
      self._timeouts,
      self._extras,
    )

  def reset(
    self,
    *,
    seed: int | None = None,
    env_ids: torch.Tensor | None = None,
    options: dict[str, Any] | None = None,
  ) -> tuple[Any, dict]:
    del env_ids  # The replay uses one shared frame index for all envs.
    self._obs, self._extras = self._env.reset(seed=seed, options=options)
    self._frame_idx = 0
    self._apply_frame(self._frame_idx)
    return self._obs, self._extras

  def close(self) -> None:
    self._env.close()

  def get_min_foot_bottom_z(self, env_idx: int = 0) -> float | None:
    if not self.foot_bottom_display_enabled:
      return None
    env_idx = int(max(0, min(env_idx, self.num_envs - 1)))
    return float(self._current_min_foot_bottom_z[env_idx])

  def describe_current_frame(self, env_idx: int = 0, joint_preview: int = 12) -> str:
    env_idx = int(max(0, min(env_idx, self.num_envs - 1)))
    frame = self.current_frame

    env_origin = self._env.scene.env_origins[env_idx].detach().cpu().numpy()
    root_pos = (
      self._motion.body_pos_w[frame, self._motion_root_idx].detach().cpu().numpy()
      + env_origin
    )
    root_quat = (
      self._motion.body_quat_w[frame, self._motion_root_idx].detach().cpu().numpy()
    )
    root_lin_vel = (
      self._motion.body_lin_vel_w[frame, self._motion_root_idx].detach().cpu().numpy()
    )
    root_ang_vel = (
      self._motion.body_ang_vel_w[frame, self._motion_root_idx].detach().cpu().numpy()
    )
    joint_pos = self._motion.joint_pos[frame].detach().cpu().numpy()
    joint_vel = self._motion.joint_vel[frame].detach().cpu().numpy()
    preview = min(joint_preview, joint_pos.shape[0])
    min_foot_bottom_z = self.get_min_foot_bottom_z(env_idx=env_idx)
    min_foot_bottom_z_str = (
      f"{min_foot_bottom_z:.6f}" if min_foot_bottom_z is not None else "n/a"
    )

    return "\n".join(
      [
        "",
        "[Replay Motion Frame]",
        f"  env: {env_idx}",
        f"  frame: {frame + 1}/{self.total_frames}",
        f"  loop: {'on' if self._loop else 'off'}",
        f"  root_body: {self._root_body_name}",
        f"  min_foot_bottom_z_m: {min_foot_bottom_z_str}",
        f"  env_origin: {np.array2string(env_origin, precision=4, suppress_small=True)}",
        f"  root_pos_w: {np.array2string(root_pos, precision=4, suppress_small=True)}",
        f"  root_quat_w: {np.array2string(root_quat, precision=4, suppress_small=True)}",
        f"  root_lin_vel_w: {np.array2string(root_lin_vel, precision=4, suppress_small=True)}",
        f"  root_ang_vel_w: {np.array2string(root_ang_vel, precision=4, suppress_small=True)}",
        f"  joint_pos[0:{preview}]: {np.array2string(joint_pos[:preview], precision=4, suppress_small=True)}",
        f"  joint_vel[0:{preview}]: {np.array2string(joint_vel[:preview], precision=4, suppress_small=True)}",
      ]
    )

  def _apply_frame(self, frame_idx: int) -> None:
    root_state = self._robot.data.default_root_state.clone()
    root_state[:, 0:3] = (
      self._motion.body_pos_w[frame_idx, self._motion_root_idx].unsqueeze(0)
      + self._env.scene.env_origins
    )
    root_state[:, 3:7] = self._motion.body_quat_w[frame_idx, self._motion_root_idx]
    root_state[:, 7:10] = self._motion.body_lin_vel_w[frame_idx, self._motion_root_idx]
    root_state[:, 10:13] = self._motion.body_ang_vel_w[
      frame_idx, self._motion_root_idx
    ]
    self._robot.write_root_state_to_sim(root_state)

    joint_pos = self._robot.data.default_joint_pos.clone()
    joint_vel = self._robot.data.default_joint_vel.clone()
    joint_pos[:] = self._motion.joint_pos[frame_idx]
    joint_vel[:] = self._motion.joint_vel[frame_idx]
    self._robot.write_joint_state_to_sim(joint_pos, joint_vel)

    self._motion_term.time_steps.fill_(frame_idx)
    self._env.sim.forward()
    self._env.scene.update(self._env.physics_dt)
    self._update_min_foot_bottom_z()

  def _setup_foot_geom_display(self) -> None:
    if self._foot_geom_pattern is None:
      return
    foot_geom_ids, foot_geom_names = self._robot.find_geoms(
      self._foot_geom_pattern, preserve_order=True
    )
    if not foot_geom_ids:
      return
    global_geom_ids = self._robot.indexing.geom_ids[foot_geom_ids].cpu().numpy()
    self._foot_geom_ids = list(foot_geom_ids)
    self._foot_geom_names = tuple(foot_geom_names)
    self._foot_geom_sizes = torch.tensor(
      self._env.sim.mj_model.geom_size[global_geom_ids],
      dtype=torch.float32,
      device=self.device,
    )
    self._foot_geom_types = np.asarray(self._env.sim.mj_model.geom_type[global_geom_ids])

  def _update_min_foot_bottom_z(self) -> None:
    if not self.foot_bottom_display_enabled:
      return
    assert self._foot_geom_sizes is not None
    assert self._foot_geom_types is not None
    foot_geom_ids = self._foot_geom_ids
    for env_idx in range(self.num_envs):
      geom_pose_w = self._robot.data.geom_pose_w[env_idx, foot_geom_ids]
      bottom_heights = _compute_geom_bottom_heights(
        geom_pos_w=geom_pose_w[:, :3],
        geom_quat_w=geom_pose_w[:, 3:7],
        geom_sizes=self._foot_geom_sizes,
        geom_types=self._foot_geom_types,
      )
      self._current_min_foot_bottom_z[env_idx] = float(bottom_heights.min().item())


class ReplayNativeMujocoViewer(NativeMujocoViewer):
  """Native viewer with replay-specific status text and frame dump action."""

  env: MotionReplayEnvAdapter

  def _set_status_overlay(self, viewer) -> None:
    status = self.get_status()
    capped = " [CAPPED]" if status.capped else ""
    loop_suffix = " (loop)" if self.env.loop_enabled else ""
    min_foot_bottom_z = self.env.get_min_foot_bottom_z(env_idx=self.env_idx)
    min_foot_bottom_z_str = (
      f"{min_foot_bottom_z:.4f} m" if min_foot_bottom_z is not None else "n/a"
    )
    text_1 = "Env\nFrame\nMin Foot Z\nReplay Step\nStatus\nSpeed\nTarget RT\nActual RT"
    text_2 = (
      f"{self.env_idx + 1}/{self.env.num_envs}\n"
      f"{self.env.current_frame + 1}/{self.env.total_frames}{loop_suffix}\n"
      f"{min_foot_bottom_z_str}\n"
      f"{status.step_count}\n"
      f"{'PAUSED' if status.paused else 'RUNNING'}{capped}\n"
      f"{status.speed_label}\n"
      f"{status.target_realtime:.2f}x\n"
      f"{status.actual_realtime:.2f}x ({status.smoothed_fps:.0f} FPS)"
    )
    overlay = (
      mujoco.mjtFontScale.mjFONTSCALE_150.value,
      mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
      text_1,
      text_2,
    )
    viewer.set_texts(overlay)

  def _handle_custom_action(self, action: ViewerAction, payload: object | None) -> bool:
    if super()._handle_custom_action(action, payload):
      return True
    if action == ViewerAction.CUSTOM and payload == "dump_frame":
      print(self.env.describe_current_frame(env_idx=self.env_idx))
      return True
    return False


def _resolve_viewer(viewer: Literal["auto", "native", "viser"]) -> Literal["native", "viser"]:
  if viewer != "auto":
    return viewer
  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  return "native" if has_display else "viser"


def _resolve_verbosity(
  verbosity: Literal["silent", "info", "debug"],
) -> VerbosityLevel:
  if verbosity == "debug":
    return VerbosityLevel.DEBUG
  if verbosity == "info":
    return VerbosityLevel.INFO
  return VerbosityLevel.SILENT


def _print_motion_summary(cfg: ReplayMotionConfig, replay_env: MotionReplayEnvAdapter) -> None:
  with np.load(cfg.motion_file) as data:
    print(f"[INFO]: Loaded motion file: {cfg.motion_file}")
    print(f"[INFO]: Replay root body: {replay_env.root_body_name}")
    if replay_env.foot_bottom_display_enabled:
      print(
        "[INFO]: Foot-bottom display: "
        f"enabled via pattern {replay_env.foot_geom_pattern!r} "
        f"({replay_env.foot_geom_count} geoms)"
      )
    else:
      print("[INFO]: Foot-bottom display: unavailable (no matching geoms)")
    print(
      "[INFO]: Replay timing: "
      f"{replay_env.unwrapped.step_dt:.4f}s per frame "
      f"({1.0 / replay_env.unwrapped.step_dt:.1f} FPS)"
    )
    print(f"[INFO]: Total frames: {replay_env.total_frames}")
    print(f"[INFO]: Viewer backend: {cfg.viewer}")
    print("[INFO]: Motion arrays:")
    for key in sorted(data.files):
      value = data[key]
      print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
  print("[INFO]: Native viewer hotkeys: Space pause/resume, Right single-step, Enter reset, D dump current frame.")


def run_replay(task_id: str, cfg: ReplayMotionConfig) -> None:
  configure_torch_backends()
  motion_path = Path(cfg.motion_file)
  if not motion_path.exists():
    raise FileNotFoundError(f"Motion file not found: {motion_path}")
  if cfg.num_envs <= 0:
    raise ValueError("`num_envs` must be >= 1.")

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task_id, play=True)

  motion_cmd_cfg = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd_cfg, MotionCommandCfg):
    raise ValueError(f"Task {task_id} is not a tracking task.")

  motion_cmd_cfg.motion_file = str(motion_path)
  if cfg.reference_viz == "none":
    motion_cmd_cfg.debug_vis = False
  else:
    motion_cmd_cfg.debug_vis = True
    motion_cmd_cfg.viz.mode = cfg.reference_viz

  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.terminations = {}
  env_cfg.viewer.env_idx = min(max(env_cfg.viewer.env_idx, 0), cfg.num_envs - 1)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  motion_term = cast(MotionCommand, env.command_manager.get_term("motion"))
  replay_env = MotionReplayEnvAdapter(
    env,
    motion_term,
    root_body_name=cfg.root_body_name,
    foot_geom_pattern=cfg.foot_geom_pattern,
    loop=cfg.loop,
  )

  if cfg.print_summary:
    _print_motion_summary(cfg, replay_env)

  policy = _ReplayZeroPolicy(env.action_space.shape, env.device)
  resolved_viewer = _resolve_viewer(cfg.viewer)
  verbosity = _resolve_verbosity(cfg.verbosity)

  if resolved_viewer == "native":
    viewer = ReplayNativeMujocoViewer(replay_env, policy, verbosity=verbosity)

    def _on_key(key: int) -> None:
      if key == KEY_D:
        viewer.request_action("dump_frame", "dump_frame")

    viewer.user_key_callback = _on_key
  else:
    viewer = ViserPlayViewer(replay_env, policy, verbosity=verbosity)

  if cfg.start_paused:
    viewer.pause()

  try:
    viewer.run()
  finally:
    replay_env.close()


def main() -> None:
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=_REPLAY_TYRO_FLAGS,
  )

  args = tyro.cli(
    ReplayMotionConfig,
    args=remaining_args,
    default=ReplayMotionConfig(motion_file=""),
    prog=sys.argv[0] + f" {chosen_task}",
    config=_REPLAY_TYRO_FLAGS,
  )
  if not args.motion_file:
    raise ValueError("`--motion-file` is required.")

  run_replay(chosen_task, args)


if __name__ == "__main__":
  main()
