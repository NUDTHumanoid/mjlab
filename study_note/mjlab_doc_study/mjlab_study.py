from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene.scene import SceneCfg
from mjlab.sim.sim import SimulationCfg
from mjlab.viewer.viewer_config import ViewerConfig


@dataclass
class MyEnvCfg(ManagerBasedRlEnvCfg):

    # --- Physics ---

    decimation: int = 4
    # Number of physics steps per policy step.
    # Environment step duration = sim.mujoco.timestep * decimation.

    sim: SimulationCfg = field(default_factory=SimulationCfg)
    # Physics parameters: timestep, integrator, solver, contact settings.
    # Default timestep is 0.002 s (500 Hz). Override with MujocoCfg.

    scene: SceneCfg = ...
    # Terrain, entities, and sensors. Also sets num_envs.
    # Required; there is no default.

    # --- Episode ---

    episode_length_s: float = 20.0
    # Episode duration in seconds.
    # Steps = ceil(episode_length_s / (sim.mujoco.timestep * decimation)).

    is_finite_horizon: bool = False
    # False (default): time limit is an artificial cutoff. The agent
    #   receives a truncated signal and bootstraps value beyond the limit.
    # True: time limit defines the task boundary. The agent receives a
    #   terminal done signal with no future value beyond it.

    scale_rewards_by_dt: bool = True
    # When True (default), each reward term is multiplied by step_dt so
    # that cumulative episodic sums are invariant to simulation frequency.
    # Set to False for algorithms that expect unscaled reward signals.

    # --- Managers ---

    observations: dict[str, ObservationGroupCfg] = field(default_factory=dict)
    # Observation groups. Each key is a group name (e.g. "actor", "critic").
    # Groups can differ in noise, history, delay, and concatenation.

    actions: dict[str, ActionTermCfg] = field(default_factory=dict)
    # Action terms. Each term controls one slice of the policy output
    # and routes it to a specific entity's actuators.

    rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
    # Reward terms. The manager computes a weighted sum each step.

    terminations: dict[str, TerminationTermCfg] = field(default_factory=dict)
    # Termination conditions. If empty, episodes never terminate early.
    # Add a time_out term to enforce the episode length limit.

    events: dict[str, EventTermCfg] = field(
        default_factory=lambda: {
            "reset_scene_to_default": EventTermCfg(
                func=reset_scene_to_default,
                mode="reset",
            )
        }
    )
    # Event terms for domain randomization and state resets.
    # The default includes reset_scene_to_default, which resets all
    # entities to their initial pose each episode. Override this dict
    # to replace or extend the default reset behavior.

    commands: dict[str, CommandTermCfg] = field(default_factory=dict)
    # Command generators (e.g. velocity targets for locomotion).
    # Commands are resampled at configurable intervals and on reset.

    curriculum: dict[str, CurriculumTermCfg] = field(default_factory=dict)
    # Curriculum terms that adjust training conditions based on performance.

    metrics: dict[str, MetricsTermCfg] = field(default_factory=dict)
    # Custom metrics logged as episode averages alongside reward terms.

    # --- Misc ---

    seed: int | None = None
    # Random seed for reproducibility. If None, a random seed is chosen
    # and stored back into this field after initialization.

    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    # Camera position, resolution, and tracking target for rendering.