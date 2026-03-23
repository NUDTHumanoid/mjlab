"""Interactive robot debugger for structure and control-chain inspection."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import torch
import tyro
import viser

import mjlab
from mjlab.asset_zoo.robots import get_g1_robot_cfg, get_go1_robot_cfg, get_yam_robot_cfg
from mjlab.entity import Entity, EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import Simulation, SimulationCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer.viser import ViserMujocoScene
from mjlab.viewer.viser.debug_panels import (
  build_control_chain_rows,
  build_actuator_inventory_from_cfg,
  build_joint_inventory,
)

ROBOT_CFG_GETTERS: dict[str, Callable[[], EntityCfg]] = {
  "g1": get_g1_robot_cfg,
  "go1": get_go1_robot_cfg,
  "yam": get_yam_robot_cfg,
}


@dataclass(frozen=True)
class PoseDeltaResult:
  joint_index: int
  joint_name: str
  delta: float
  q: float
  q_default: float
  joint_limit: tuple[float, float]


@dataclass
class RobotDebugSession:
  robot_name: str
  scene: Scene | None
  sim: Simulation
  entity: Entity
  device: str


@dataclass
class TaskDebugSession:
  task_id: str
  env: ManagerBasedRlEnv
  vec_env: RslRlVecEnvWrapper
  entity: Entity
  action_term: Any
  device: str
  obs: Any


@dataclass(frozen=True)
class RobotModeConfig:
  robot: Literal["g1", "go1", "yam"] = "g1"
  mode: Literal["robot", "task"] = "robot"
  device: str | None = None
  task_id: str = "Mjlab-Velocity-Flat-Unitree-G1"
  agent: Literal["zero", "random", "trained"] = "zero"
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  checkpoint_file: str | None = None
  num_envs: int = 1
  task_control_mode: Literal["policy", "hold-reference", "manual-delta"] = "policy"
  manual_delta_limit: float = 0.25


@dataclass
class HoldReferencePolicy:
  action_dim: int
  device: str

  def __call__(self, obs: Any) -> torch.Tensor:
    batch_size = _infer_batch_size(obs)
    return torch.zeros((batch_size, self.action_dim), device=self.device)


@dataclass
class ManualDeltaPolicy:
  action_dim: int
  action_index: int
  raw_delta: float
  device: str

  def __call__(self, obs: Any) -> torch.Tensor:
    batch_size = _infer_batch_size(obs)
    actions = torch.zeros((batch_size, self.action_dim), device=self.device)
    actions[:, self.action_index] = self.raw_delta
    return actions


def create_robot_session(cfg: RobotModeConfig) -> RobotDebugSession:
  """Create a single-robot simulation session for pose browsing."""
  if cfg.robot not in ROBOT_CFG_GETTERS:
    raise ValueError(f"Unsupported robot {cfg.robot!r}.")

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  robot_cfg: EntityCfg = ROBOT_CFG_GETTERS[cfg.robot]()
  scene = Scene(SceneCfg(entities={"robot": robot_cfg}), device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device=device)
  scene.initialize(model, sim.model, sim.data)
  entity = scene["robot"]
  assert isinstance(entity, Entity)
  sim.forward()
  return RobotDebugSession(
    robot_name=cfg.robot,
    scene=scene,
    sim=sim,
    entity=entity,
    device=device,
  )


def _infer_batch_size(obs: Any) -> int:
  if isinstance(obs, torch.Tensor):
    return int(obs.shape[0]) if obs.ndim > 0 else 1
  batch_size = getattr(obs, "batch_size", None)
  if batch_size:
    return int(batch_size[0])
  return 1


def _resolve_action_scale(action_term: Any, action_index: int, env_idx: int = 0) -> float:
  scale = action_term.scale
  if isinstance(scale, (float, int)):
    return float(scale)
  return float(scale[env_idx, action_index].item())


def build_manual_delta_policy(
  action_term: Any,
  selected_joint_index: int,
  joint_delta: float,
  joint_delta_limit: float,
  device: str,
) -> ManualDeltaPolicy:
  """Create a single-joint raw-action policy from a desired joint-space delta."""
  clamped_delta = max(-joint_delta_limit, min(joint_delta, joint_delta_limit))
  scale = _resolve_action_scale(action_term, selected_joint_index)
  if abs(scale) < 1e-12:
    raise ValueError("Action scale is zero; cannot convert joint delta to raw action.")
  return ManualDeltaPolicy(
    action_dim=action_term.action_dim,
    action_index=selected_joint_index,
    raw_delta=clamped_delta / scale,
    device=device,
  )


def build_robot_mode_summary() -> str:
  return (
    "pose browser for learning joint structure. It directly edits joint positions "
    "and refreshes kinematics, so it is not a balance controller."
  )


def build_task_mode_summary(control_mode: str, agent: str) -> str:
  return (
    f"Task mode ({control_mode}, agent={agent}) shows raw actions, position targets, "
    "joint state, and actuator outputs. On real robots, those position targets are "
    "still turned into motor effort by the low-level controller."
  )


def create_task_session(cfg: RobotModeConfig) -> TaskDebugSession:
  """Create a one-env task session for control-chain inspection."""
  configure_torch_backends()
  import mjlab.tasks  # noqa: F401

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(cfg.task_id, play=True)
  agent_cfg = load_rl_cfg(cfg.task_id)
  env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  obs = vec_env.get_observations()
  entity = env.scene["robot"]
  assert isinstance(entity, Entity)
  action_term = env.action_manager.get_term("joint_pos")
  return TaskDebugSession(
    task_id=cfg.task_id,
    env=env,
    vec_env=vec_env,
    entity=entity,
    action_term=action_term,
    device=device,
    obs=obs,
  )


def build_policy_controller(cfg: RobotModeConfig, session: TaskDebugSession):
  """Create the controller used in task-mode `policy` submode."""
  action_dim = session.action_term.action_dim
  if cfg.agent == "zero":
    return HoldReferencePolicy(action_dim=action_dim, device=session.device)
  if cfg.agent == "random":
    return lambda obs: 2 * torch.rand(
      (_infer_batch_size(obs), action_dim), device=session.device
    ) - 1
  if cfg.agent != "trained":
    raise ValueError(f"Unsupported agent {cfg.agent!r}.")

  agent_cfg = load_rl_cfg(cfg.task_id)
  runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(session.vec_env, asdict(agent_cfg), device=session.device)

  if cfg.checkpoint_file is not None:
    resume_path = Path(cfg.checkpoint_file)
    if not resume_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
  else:
    if cfg.wandb_run_path is None:
      raise ValueError(
        "`wandb_run_path` is required for trained task policy mode "
        "when `checkpoint_file` is not provided."
      )
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    resume_path, _ = get_wandb_checkpoint_path(
      log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
    )

  runner.load(
    str(resume_path),
    load_cfg={"actor": True},
    strict=True,
    map_location=session.device,
  )
  return runner.get_inference_policy(device=session.device)


def apply_pose_delta(
  session: RobotDebugSession,
  joint_index: int,
  delta: float,
  clamp: bool = True,
) -> PoseDeltaResult:
  """Apply a delta to one joint relative to the default pose and refresh kinematics."""
  num_joints = session.entity.num_joints
  if joint_index < 0 or joint_index >= num_joints:
    raise IndexError(f"joint_index {joint_index} out of range for {num_joints} joints.")

  q_default = session.entity.data.default_joint_pos.clone()
  q_target = q_default.clone()
  joint_limit = session.entity.data.joint_pos_limits[0, joint_index]
  target_value = float(q_default[0, joint_index].item() + delta)
  if clamp:
    target_value = min(
      max(target_value, float(joint_limit[0].item())),
      float(joint_limit[1].item()),
    )
  q_target[0, joint_index] = target_value

  dq_target = torch.zeros_like(session.entity.data.default_joint_vel)
  session.entity.write_joint_position_to_sim(q_target)
  session.entity.write_joint_velocity_to_sim(dq_target)
  session.sim.forward()

  return PoseDeltaResult(
    joint_index=joint_index,
    joint_name=session.entity.joint_names[joint_index],
    delta=delta,
    q=float(session.entity.data.joint_pos[0, joint_index].item()),
    q_default=float(session.entity.data.default_joint_pos[0, joint_index].item()),
    joint_limit=(float(joint_limit[0].item()), float(joint_limit[1].item())),
  )


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
  html_rows = []
  for row in rows:
    tds = "".join(f"<td>{cell}</td>" for cell in row)
    html_rows.append(f"<tr>{tds}</tr>")
  thead = "".join(f"<th>{header}</th>" for header in headers)
  tbody = "".join(html_rows)
  return f"""
    <style>
      table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.8rem;
      }}
      th, td {{
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding: 0.35rem 0.5rem;
        text-align: left;
        vertical-align: top;
      }}
      th {{
        font-weight: 600;
      }}
      code {{
        font-size: 0.78rem;
      }}
    </style>
    <table>
      <thead><tr>{thead}</tr></thead>
      <tbody>{tbody}</tbody>
    </table>
  """


class RobotDebugApp:
  """Viser app for browsing a robot pose and actuator structure."""

  def __init__(self, cfg: RobotModeConfig):
    self.cfg = cfg
    self.session = create_robot_session(cfg)
    self.server = viser.ViserServer(label="Robot Debugger")
    self.scene = ViserMujocoScene.create(
      self.server, self.session.sim.mj_model, num_envs=1
    )
    self.selected_joint_index = 0
    self.scene.update(self.session.sim.data, env_idx=0)

  def setup(self) -> None:
    tabs = self.server.gui.add_tab_group()

    with tabs.add_tab("Overview", icon=viser.Icon.INFO_CIRCLE):
      self._overview_html = self.server.gui.add_html("")

    with tabs.add_tab("Joints", icon=viser.Icon.LIST_DETAILS):
      with self.server.gui.add_folder("Pose Browser"):
        self._joint_dropdown = self.server.gui.add_dropdown(
          "Joint",
          options=list(self.session.entity.joint_names),
          initial_value=self.session.entity.joint_names[0],
        )

        @self._joint_dropdown.on_update
        def _(event) -> None:
          self.selected_joint_index = self.session.entity.joint_names.index(
            event.target.value
          )
          self._delta_slider.value = 0.0
          self._refresh_panels()

        self._delta_slider = self.server.gui.add_slider(
          "Position Delta",
          min=-1.5,
          max=1.5,
          step=0.01,
          initial_value=0.0,
        )

        @self._delta_slider.on_update
        def _(event) -> None:
          apply_pose_delta(
            self.session,
            joint_index=self.selected_joint_index,
            delta=float(event.target.value),
            clamp=True,
          )
          self.scene.update(self.session.sim.data, env_idx=0)
          self._refresh_panels()

        reset_button = self.server.gui.add_button("Reset Pose")

        @reset_button.on_click
        def _(_) -> None:
          self._delta_slider.value = 0.0
          apply_pose_delta(
            self.session,
            joint_index=self.selected_joint_index,
            delta=0.0,
            clamp=True,
          )
          self.scene.update(self.session.sim.data, env_idx=0)
          self._refresh_panels()

      self._joints_html = self.server.gui.add_html("")

    with tabs.add_tab("Actuators", icon=viser.Icon.SETTINGS):
      self._actuators_html = self.server.gui.add_html("")

    with tabs.add_tab("Scene", icon=viser.Icon.CUBE):
      self.scene.create_visualization_gui(show_debug_viz_control=False)

    self._refresh_panels()

  def _refresh_panels(self) -> None:
    self._overview_html.content = self._get_overview_html()
    self._joints_html.content = self._get_joints_html()
    self._actuators_html.content = self._get_actuators_html()

  def _get_overview_html(self) -> str:
    rows = [
      ["Robot", self.session.robot_name],
      ["Mode", self.cfg.mode],
      ["Joints", str(self.session.entity.num_joints)],
      ["Actuators", str(self.session.entity.num_actuators)],
      ["Selected Joint", self.session.entity.joint_names[self.selected_joint_index]],
      ["Control Type", "pose browser (joint position delta)"],
      ["Summary", build_robot_mode_summary()],
    ]
    return _render_table(["Field", "Value"], rows)

  def _get_joints_html(self) -> str:
    rows = []
    for row in build_joint_inventory(self.session.entity):
      rows.append(
        [
          str(row.joint_index),
          row.joint_name,
          f"{row.q:.3f}",
          f"{row.q_des:.3f}",
          f"{row.dq:.3f}",
          f"{row.q_default:.3f}",
          f"[{row.joint_limit[0]:.2f}, {row.joint_limit[1]:.2f}]",
          row.actuator_group or "—",
        ]
      )
    return _render_table(
      ["idx", "joint", "q", "q_des", "dq", "q_default", "limit", "group"],
      rows,
    )

  def _get_actuators_html(self) -> str:
    rows = []
    for row in build_actuator_inventory_from_cfg(ROBOT_CFG_GETTERS[self.cfg.robot]()):
      rows.append(
        [
          row.group_name,
          "<br/>".join(row.target_expr),
          ", ".join(row.matched_joints),
          row.control_type,
          "—" if row.stiffness is None else f"{row.stiffness:.3f}",
          "—" if row.damping is None else f"{row.damping:.3f}",
          "—" if row.effort_limit is None else f"{row.effort_limit:.3f}",
          f"{row.armature:.4f}",
        ]
      )
    return _render_table(
      [
        "group",
        "target_expr",
        "matched_joints",
        "control_type",
        "stiffness",
        "damping",
        "effort_limit",
        "armature",
      ],
      rows,
    )

  def run(self) -> None:
    self.setup()
    print("Robot debugger running. Press Ctrl+C to exit.")
    try:
      while True:
        if self.scene.needs_update:
          self.scene.refresh_visualization()
        time.sleep(0.1)
    except KeyboardInterrupt:
      self.server.stop()


class TaskDebugApp:
  """Viser app for inspecting control chains in a live task."""

  def __init__(self, cfg: RobotModeConfig):
    self.cfg = cfg
    self.session = create_task_session(cfg)
    self.server = viser.ViserServer(label="Task Debugger")
    self.scene = ViserMujocoScene.create(
      self.server,
      self.session.env.sim.mj_model,
      num_envs=self.session.env.num_envs,
    )
    self.scene.env_idx = 0
    self.scene.update(self.session.env.sim.data, env_idx=0)

    self.selected_joint_index = 0
    self.control_mode = cfg.task_control_mode
    self.policy_controller = build_policy_controller(cfg, self.session)
    self.hold_controller = HoldReferencePolicy(
      action_dim=self.session.action_term.action_dim,
      device=self.session.device,
    )
    self.manual_controller = build_manual_delta_policy(
      self.session.action_term,
      selected_joint_index=0,
      joint_delta=0.0,
      joint_delta_limit=self.cfg.manual_delta_limit,
      device=self.session.device,
    )
    self._is_paused = False

  def setup(self) -> None:
    tabs = self.server.gui.add_tab_group()

    with tabs.add_tab("Overview", icon=viser.Icon.INFO_CIRCLE):
      self._overview_html = self.server.gui.add_html("")

    with tabs.add_tab("Joints", icon=viser.Icon.LIST_DETAILS):
      self._joints_html = self.server.gui.add_html("")

    with tabs.add_tab("Actuators", icon=viser.Icon.SETTINGS):
      self._actuators_html = self.server.gui.add_html("")

    with tabs.add_tab("Control Chain", icon=viser.Icon.ARROWS_EXCHANGE):
      with self.server.gui.add_folder("Task Controls"):
        self._pause_button = self.server.gui.add_button("Pause")

        @self._pause_button.on_click
        def _(_) -> None:
          self._is_paused = not self._is_paused
          self._pause_button.label = "Play" if self._is_paused else "Pause"

        step_button = self.server.gui.add_button("Step")

        @step_button.on_click
        def _(_) -> None:
          self._step_once()

        reset_button = self.server.gui.add_button("Reset Environment")

        @reset_button.on_click
        def _(_) -> None:
          self.session.obs, _ = self.session.vec_env.reset()
          self.scene.update(self.session.env.sim.data, env_idx=0)
          self._refresh_panels()

        self._control_mode_dropdown = self.server.gui.add_dropdown(
          "Control Mode",
          options=["policy", "hold-reference", "manual-delta"],
          initial_value=self.control_mode,
        )

        @self._control_mode_dropdown.on_update
        def _(event) -> None:
          self.control_mode = event.target.value
          self._refresh_panels()

        self._joint_dropdown = self.server.gui.add_dropdown(
          "Manual Joint",
          options=list(self.session.action_term.target_names),
          initial_value=self.session.action_term.target_names[0],
        )

        @self._joint_dropdown.on_update
        def _(event) -> None:
          self.selected_joint_index = self.session.action_term.target_names.index(
            event.target.value
          )
          self._update_manual_controller()
          self._refresh_panels()

        self._delta_slider = self.server.gui.add_slider(
          "Joint Delta",
          min=-self.cfg.manual_delta_limit,
          max=self.cfg.manual_delta_limit,
          step=0.01,
          initial_value=0.0,
        )

        @self._delta_slider.on_update
        def _(_) -> None:
          self._update_manual_controller()
          self._refresh_panels()

      self._control_chain_html = self.server.gui.add_html("")

    with tabs.add_tab("Scene", icon=viser.Icon.CUBE):
      self.scene.create_visualization_gui(show_debug_viz_control=False)

    self._refresh_panels()

  def _update_manual_controller(self) -> None:
    self.manual_controller = build_manual_delta_policy(
      self.session.action_term,
      selected_joint_index=self.selected_joint_index,
      joint_delta=float(self._delta_slider.value),
      joint_delta_limit=self.cfg.manual_delta_limit,
      device=self.session.device,
    )

  def _current_controller(self):
    if self.control_mode == "hold-reference":
      return self.hold_controller
    if self.control_mode == "manual-delta":
      return self.manual_controller
    return self.policy_controller

  def _step_once(self) -> None:
    action = self._current_controller()(self.session.obs)
    self.session.obs, *_ = self.session.vec_env.step(action)
    self.scene.update(self.session.env.sim.data, env_idx=0)
    self._refresh_panels()

  def _refresh_panels(self) -> None:
    self._overview_html.content = self._get_overview_html()
    self._joints_html.content = self._get_joints_html()
    self._actuators_html.content = self._get_actuators_html()
    self._control_chain_html.content = self._get_control_chain_html()

  def _get_overview_html(self) -> str:
    command_rows = self.session.env.command_manager.get_active_iterable_terms(0)
    command_text = "; ".join(
      f"{name}: {[round(v, 3) for v in values]}" for name, values in command_rows
    )
    rows = [
      ["Task", self.cfg.task_id],
      ["Agent", self.cfg.agent],
      ["Control Mode", self.control_mode],
      ["Action Dim", str(self.session.action_term.action_dim)],
      ["Selected Joint", self.session.action_term.target_names[self.selected_joint_index]],
      ["Command", command_text or "—"],
      ["Summary", build_task_mode_summary(self.control_mode, self.cfg.agent)],
    ]
    return _render_table(["Field", "Value"], rows)

  def _get_joints_html(self) -> str:
    rows = []
    for row in build_joint_inventory(self.session.entity):
      rows.append(
        [
          str(row.joint_index),
          row.joint_name,
          f"{row.q:.3f}",
          f"{row.q_des:.3f}",
          f"{row.dq:.3f}",
          f"{row.q_default:.3f}",
          f"[{row.joint_limit[0]:.2f}, {row.joint_limit[1]:.2f}]",
          row.actuator_group or "—",
        ]
      )
    return _render_table(
      ["idx", "joint", "q", "q_des", "dq", "q_default", "limit", "group"],
      rows,
    )

  def _get_actuators_html(self) -> str:
    rows = []
    for row in build_actuator_inventory_from_cfg(self.session.entity.cfg):
      rows.append(
        [
          row.group_name,
          "<br/>".join(row.target_expr),
          ", ".join(row.matched_joints),
          row.control_type,
          "—" if row.stiffness is None else f"{row.stiffness:.3f}",
          "—" if row.damping is None else f"{row.damping:.3f}",
          "—" if row.effort_limit is None else f"{row.effort_limit:.3f}",
          f"{row.armature:.4f}",
        ]
      )
    return _render_table(
      [
        "group",
        "target_expr",
        "matched_joints",
        "control_type",
        "stiffness",
        "damping",
        "effort_limit",
        "armature",
      ],
      rows,
    )

  def _get_control_chain_html(self) -> str:
    rows = []
    for row in build_control_chain_rows(
      self.session.entity, self.session.action_term, env_idx=0
    ):
      rows.append(
        [
          str(row.action_index),
          row.joint_name,
          f"{row.raw_action:.3f}",
          f"{row.processed_action:.3f}",
          f"{row.q_des:.3f}",
          f"{row.q:.3f}",
          f"{row.dq:.3f}",
          "—" if row.actuator_force is None else f"{row.actuator_force:.3f}",
          f"{row.qfrc_actuator:.3f}",
        ]
      )
    return _render_table(
      [
        "a_idx",
        "joint",
        "raw",
        "processed",
        "q_des",
        "q",
        "dq",
        "actuator_force",
        "qfrc_actuator",
      ],
      rows,
    )

  def run(self) -> None:
    self.setup()
    print("Task debugger running. Press Ctrl+C to exit.")
    try:
      while True:
        if not self._is_paused:
          self._step_once()
        elif self.scene.needs_update:
          self.scene.refresh_visualization()
        time.sleep(0.1)
    except KeyboardInterrupt:
      self.session.vec_env.close()
      self.server.stop()


def main() -> None:
  cfg = tyro.cli(RobotModeConfig, config=mjlab.TYRO_FLAGS)
  if cfg.mode == "robot":
    RobotDebugApp(cfg).run()
    return
  TaskDebugApp(cfg).run()


if __name__ == "__main__":
  main()
