"""RL configuration for Unitree G1 tracking tasks."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def _unitree_g1_tracking_ppo_runner_cfg(
  experiment_name: str,
) -> RslRlOnPolicyRunnerCfg:
  """Create a shared PPO runner configuration for G1 tracking tasks.

  Flat and rough tracking share the same optimizer/model structure so checkpoint
  transfer stays straightforward. We only separate the experiment name to keep
  logs and checkpoints easy to compare.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=100_000,
  )


def unitree_g1_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for the flat G1 tracking task."""
  return _unitree_g1_tracking_ppo_runner_cfg(experiment_name="g1_tracking")


def unitree_g1_tracking_late_phase_dr_finetune_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for late-phase-disturbance flat tracking finetuning."""
  cfg = _unitree_g1_tracking_ppo_runner_cfg(
    experiment_name="g1_tracking_late_phase_dr_finetune"
  )
  cfg.algorithm.learning_rate = 1.0e-4
  cfg.algorithm.entropy_coef = 0.001
  cfg.algorithm.desired_kl = 0.003
  cfg.save_interval = 250
  cfg.max_iterations = 20_000
  return cfg


def unitree_g1_rough_tracking_late_phase_dr_finetune_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for rough late-phase-disturbance finetuning."""
  cfg = unitree_g1_tracking_late_phase_dr_finetune_ppo_runner_cfg()
  cfg.experiment_name = "g1_tracking_rough_late_phase_dr_finetune"
  return cfg


def unitree_g1_rough_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for the rough G1 tracking task."""
  return _unitree_g1_tracking_ppo_runner_cfg(experiment_name="g1_tracking_rough")


# JumpRough runner config disabled for now; keep the general rough-terrain task only.
# def unitree_g1_jump_rough_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
#   """Create RL runner configuration for the jump-specific rough G1 tracking task."""
#   return _unitree_g1_tracking_ppo_runner_cfg(experiment_name="g1_tracking_jump_rough")
