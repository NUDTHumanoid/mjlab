"""Shared configuration for late-phase disturbance finetuning."""

from __future__ import annotations

from copy import deepcopy

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking import mdp

LATE_PHASE_DR_START_RATIO = 0.42
LATE_PHASE_DR_TERMINATION_THRESHOLDS = {
  "anchor_pos": 0.35,
  "anchor_ori": 1.2,
  "ee_body_pos": 0.35,
}
STAND_UP_RECOVERY_PROBABILITY = 0.4
DEFAULT_STAND_UP_KICK_SCALE = 1.5
_RAW_STAND_UP_PUSH_FORWARD_FORCE_RANGE = (0.0, 0.0)
_RAW_STAND_UP_PUSH_LATERAL_FORCE_RANGE = (0.0, 0.0)
_STAND_UP_PUSH_DURATION_S = (0.0, 0.0)
_RAW_STAND_UP_OVERSHOOT_EFFORT_SCALE_RANGE = (1.10, 1.18)
_RAW_STAND_UP_OVERSHOOT_PD_SCALE_RANGE = (1.04, 1.10)
_RAW_STAND_UP_OVERSHOOT_ACTION_SCALE_RANGE = (1.15, 1.28)
_RAW_STAND_UP_OVERSHOOT_PITCH_ANG_VEL_RANGE = (0.70, 1.10)
_STAND_UP_OVERSHOOT_DURATION_S = (0.05, 0.08)
_STAND_UP_UNDERPOWERED_PROBABILITY = 0.5
_RAW_STAND_UP_UNDERPOWERED_EFFORT_SCALE_RANGE = (0.82, 0.92)
_RAW_STAND_UP_UNDERPOWERED_PD_SCALE_RANGE = (0.84, 0.92)
_RAW_STAND_UP_UNDERPOWERED_ACTION_SCALE_RANGE = (0.72, 0.88)
_RAW_STAND_UP_UNDERPOWERED_PITCH_ANG_VEL_RANGE = (-1.30, -0.90)
_STAND_UP_UNDERPOWERED_DURATION_S = (0.06, 0.10)
_STAND_UP_RECOVERY_ANG_VEL_KICK_DURATION_STEPS = 3
_STAND_UP_UNDERPOWERED_ACTUATOR_PATTERNS = [
  ".*_hip_pitch_joint",
  ".*_hip_roll_joint",
  ".*_knee_joint",
  ".*_ankle_pitch_joint",
  ".*_ankle_roll_joint",
]


def _scale_range(value_range: tuple[float, float], scale: float) -> tuple[float, float]:
  return tuple(round(value * scale, 3) for value in value_range)


def _scale_range_about_one(
  value_range: tuple[float, float],
  scale: float,
) -> tuple[float, float]:
  scaled = [1.0 + (value - 1.0) * scale for value in value_range]
  clamped = [round(min(max(value, 0.1), 2.0), 3) for value in scaled]
  return (min(clamped), max(clamped))


def scale_late_phase_tracking_disturbance_event(
  event_cfg: EventTermCfg,
  *,
  overshoot_scale: float = DEFAULT_STAND_UP_KICK_SCALE,
  underpowered_scale: float = DEFAULT_STAND_UP_KICK_SCALE,
  kick_duration_steps: int | None = None,
) -> None:
  """Apply the shared late-phase recovery scaling used by both train and play."""
  overshoot_scale = max(float(overshoot_scale), 0.0)
  underpowered_scale = max(float(underpowered_scale), 0.0)
  overshoot_relative_scale = overshoot_scale / DEFAULT_STAND_UP_KICK_SCALE
  underpowered_relative_scale = underpowered_scale / DEFAULT_STAND_UP_KICK_SCALE

  if overshoot_relative_scale != 1.0:
    for key in (
      "stand_up_overshoot_pitch_ang_vel_range",
      "stand_up_push_forward_force_range",
      "stand_up_push_lateral_force_range",
    ):
      lower, upper = event_cfg.params[key]
      event_cfg.params[key] = (
        lower * overshoot_relative_scale,
        upper * overshoot_relative_scale,
      )
    for key in (
      "stand_up_overshoot_effort_scale_range",
      "stand_up_overshoot_pd_scale_range",
      "stand_up_overshoot_action_scale_range",
    ):
      event_cfg.params[key] = _scale_range_about_one(
        event_cfg.params[key], overshoot_relative_scale
      )

  if underpowered_relative_scale != 1.0:
    for key in ("stand_up_underpowered_pitch_ang_vel_range",):
      lower, upper = event_cfg.params[key]
      event_cfg.params[key] = (
        lower * underpowered_relative_scale,
        upper * underpowered_relative_scale,
      )
    for key in (
      "stand_up_underpowered_effort_scale_range",
      "stand_up_underpowered_pd_scale_range",
      "stand_up_underpowered_action_scale_range",
    ):
      event_cfg.params[key] = _scale_range_about_one(
        event_cfg.params[key], underpowered_relative_scale
      )

  if kick_duration_steps is not None:
    event_cfg.params["stand_up_pitch_ang_vel_kick_duration_steps"] = max(
      int(kick_duration_steps), 1
    )


_STAND_UP_PUSH_FORWARD_FORCE_RANGE = _scale_range(
  _RAW_STAND_UP_PUSH_FORWARD_FORCE_RANGE, DEFAULT_STAND_UP_KICK_SCALE
)
_STAND_UP_PUSH_LATERAL_FORCE_RANGE = _scale_range(
  _RAW_STAND_UP_PUSH_LATERAL_FORCE_RANGE, DEFAULT_STAND_UP_KICK_SCALE
)
_STAND_UP_OVERSHOOT_EFFORT_SCALE_RANGE = _scale_range_about_one(
  _RAW_STAND_UP_OVERSHOOT_EFFORT_SCALE_RANGE,
  DEFAULT_STAND_UP_KICK_SCALE,
)
_STAND_UP_OVERSHOOT_PD_SCALE_RANGE = _scale_range_about_one(
  _RAW_STAND_UP_OVERSHOOT_PD_SCALE_RANGE,
  DEFAULT_STAND_UP_KICK_SCALE,
)
_STAND_UP_OVERSHOOT_ACTION_SCALE_RANGE = _scale_range_about_one(
  _RAW_STAND_UP_OVERSHOOT_ACTION_SCALE_RANGE,
  DEFAULT_STAND_UP_KICK_SCALE,
)
_STAND_UP_OVERSHOOT_PITCH_ANG_VEL_RANGE = _scale_range(
  _RAW_STAND_UP_OVERSHOOT_PITCH_ANG_VEL_RANGE,
  DEFAULT_STAND_UP_KICK_SCALE,
)
_STAND_UP_UNDERPOWERED_EFFORT_SCALE_RANGE = _scale_range_about_one(
  _RAW_STAND_UP_UNDERPOWERED_EFFORT_SCALE_RANGE,
  DEFAULT_STAND_UP_KICK_SCALE,
)
_STAND_UP_UNDERPOWERED_PD_SCALE_RANGE = _scale_range_about_one(
  _RAW_STAND_UP_UNDERPOWERED_PD_SCALE_RANGE,
  DEFAULT_STAND_UP_KICK_SCALE,
)
_STAND_UP_UNDERPOWERED_ACTION_SCALE_RANGE = _scale_range_about_one(
  _RAW_STAND_UP_UNDERPOWERED_ACTION_SCALE_RANGE,
  DEFAULT_STAND_UP_KICK_SCALE,
)
_STAND_UP_UNDERPOWERED_PITCH_ANG_VEL_RANGE = _scale_range(
  _RAW_STAND_UP_UNDERPOWERED_PITCH_ANG_VEL_RANGE,
  DEFAULT_STAND_UP_KICK_SCALE,
)

_LATE_PHASE_DR_EVENT_PARAMS = {
  "command_name": "motion",
  "late_phase_start_ratio": LATE_PHASE_DR_START_RATIO,
  "late_phase_onset_scale": 0.35,
  "late_phase_scale_power": 1.25,
  "state_cooldown_s": (0.50, 0.95),
  "pose_range": {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (-0.16, 0.16),
    "pitch": (-0.16, 0.16),
    "yaw": (-0.24, 0.24),
  },
  "velocity_range": {
    "x": (-0.35, 0.35),
    "y": (-0.35, 0.35),
    "z": (-0.25, 0.25),
    "roll": (-0.45, 0.45),
    "pitch": (-0.45, 0.45),
    "yaw": (-0.60, 0.60),
  },
  "joint_position_range": (0.0, 0.0),
  "joint_velocity_range": (-1.2, 1.2),
  "force_range": (-60.0, 60.0),
  "torque_range": (-15.0, 15.0),
  "impulse_duration_s": (0.02, 0.05),
  "impulse_cooldown_s": (0.30, 0.65),
  "asset_cfg": SceneEntityCfg(
    "robot",
    body_names=("pelvis",),
  ),
  "stand_up_push_center_frame": 150,
  "stand_up_push_half_window": 4,
  "stand_up_push_post_trigger_frame": 165,
  "stand_up_push_forward_force_range": _STAND_UP_PUSH_FORWARD_FORCE_RANGE,
  "stand_up_push_lateral_force_range": _STAND_UP_PUSH_LATERAL_FORCE_RANGE,
  "stand_up_push_duration_s": _STAND_UP_PUSH_DURATION_S,
  "stand_up_push_body_cfg": SceneEntityCfg(
    "robot",
    body_names=("torso_link",),
  ),
  "stand_up_push_body_point_offset": (0.0, 0.0, 0.18),
  "stand_up_recovery_probability": STAND_UP_RECOVERY_PROBABILITY,
  "stand_up_overshoot_trigger_frame": 165,
  "stand_up_overshoot_half_window": 2,
  "stand_up_overshoot_effort_scale_range": _STAND_UP_OVERSHOOT_EFFORT_SCALE_RANGE,
  "stand_up_overshoot_pd_scale_range": _STAND_UP_OVERSHOOT_PD_SCALE_RANGE,
  "stand_up_overshoot_action_scale_range": _STAND_UP_OVERSHOOT_ACTION_SCALE_RANGE,
  "stand_up_overshoot_pitch_ang_vel_range": _STAND_UP_OVERSHOOT_PITCH_ANG_VEL_RANGE,
  "stand_up_overshoot_duration_s": _STAND_UP_OVERSHOOT_DURATION_S,
  "stand_up_underpowered_probability": _STAND_UP_UNDERPOWERED_PROBABILITY,
  "stand_up_underpowered_trigger_frame": 135,
  "stand_up_underpowered_half_window": 2,
  "stand_up_underpowered_effort_scale_range": _STAND_UP_UNDERPOWERED_EFFORT_SCALE_RANGE,
  "stand_up_underpowered_pd_scale_range": _STAND_UP_UNDERPOWERED_PD_SCALE_RANGE,
  "stand_up_underpowered_action_scale_range": _STAND_UP_UNDERPOWERED_ACTION_SCALE_RANGE,
  "stand_up_underpowered_pitch_ang_vel_range": _STAND_UP_UNDERPOWERED_PITCH_ANG_VEL_RANGE,
  "stand_up_underpowered_duration_s": _STAND_UP_UNDERPOWERED_DURATION_S,
  "stand_up_underpowered_actuator_cfg": SceneEntityCfg(
    "robot",
    actuator_names=_STAND_UP_UNDERPOWERED_ACTUATOR_PATTERNS,
  ),
  "stand_up_pitch_ang_vel_kick_duration_steps": _STAND_UP_RECOVERY_ANG_VEL_KICK_DURATION_STEPS,
  "stand_up_velocity_kick_forward_range": (0.0, 0.0),
  "stand_up_velocity_kick_backward_range": (0.0, 0.0),
  "stand_up_velocity_kick_backward_probability": 0.0,
  "stand_up_velocity_kick_lateral_range": (0.0, 0.0),
  "stand_up_velocity_kick_duration_steps": _STAND_UP_RECOVERY_ANG_VEL_KICK_DURATION_STEPS,
  "log_stand_up_recovery_disturbance": False,
}


def _make_tracking_disturbance_event(params: dict) -> EventTermCfg:
  return EventTermCfg(
    func=mdp.late_phase_play_disturbance,
    mode="step",
    params=deepcopy(params),
  )


def make_late_phase_tracking_disturbance_event(*, name: str = "late_phase_disturbance") -> EventTermCfg:
  """Create the training disturbance event used by Late-Phase-DR-Finetune."""
  del name
  return _make_tracking_disturbance_event(_LATE_PHASE_DR_EVENT_PARAMS)


def make_late_phase_tracking_play_disturbance_event() -> EventTermCfg:
  """Create a cleaner play-time disturbance focused on stand-up recovery errors."""
  params = deepcopy(_LATE_PHASE_DR_EVENT_PARAMS)
  params["pose_range"] = {
    key: (0.0, 0.0) for key in params["pose_range"]
  }
  params["velocity_range"] = {
    key: (0.0, 0.0) for key in params["velocity_range"]
  }
  params["joint_velocity_range"] = (0.0, 0.0)
  params["force_range"] = (0.0, 0.0)
  params["torque_range"] = (0.0, 0.0)
  params["stand_up_recovery_probability"] = 1.0
  params["log_stand_up_recovery_disturbance"] = True
  return _make_tracking_disturbance_event(params)
