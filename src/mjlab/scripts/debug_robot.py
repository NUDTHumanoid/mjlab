"""Interactive robot debugger for structure and control-chain inspection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
import viser

import mjlab
from mjlab.asset_zoo.robots import get_g1_robot_cfg, get_go1_robot_cfg, get_yam_robot_cfg
from mjlab.entity import Entity, EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import Simulation, SimulationCfg
from mjlab.viewer.viser import ViserMujocoScene
from mjlab.viewer.viser.debug_panels import (
  build_actuator_inventory_from_cfg,
  build_joint_inventory,
)

ROBOT_CFG_GETTERS: dict[str, callable] = {
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


@dataclass(frozen=True)
class RobotModeConfig:
  robot: Literal["g1", "go1", "yam"] = "g1"
  mode: Literal["robot", "task"] = "robot"
  device: str | None = None


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


def main() -> None:
  cfg = tyro.cli(RobotModeConfig, config=mjlab.TYRO_FLAGS)
  if cfg.mode != "robot":
    raise NotImplementedError("Task mode is not implemented yet.")
  RobotDebugApp(cfg).run()


if __name__ == "__main__":
  main()
