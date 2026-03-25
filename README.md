![Project banner](https://raw.githubusercontent.com/mujocolab/mjlab/main/docs/source/_static/mjlab-banner.jpg)

# mjlab

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/mujocolab/mjlab/ci.yml?branch=main)](https://github.com/mujocolab/mjlab/actions/workflows/ci.yml?query=branch%3Amain)
[![Documentation](https://github.com/mujocolab/mjlab/actions/workflows/docs.yml/badge.svg)](https://mujocolab.github.io/mjlab/)
[![License](https://img.shields.io/github/license/mujocolab/mjlab)](https://github.com/mujocolab/mjlab/blob/main/LICENSE)
[![Nightly Benchmarks](https://img.shields.io/badge/Nightly-Benchmarks-blue)](https://mujocolab.github.io/mjlab/nightly/)
[![PyPI](https://img.shields.io/pypi/v/mjlab)](https://pypi.org/project/mjlab/)

mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s manager-based API with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), a GPU-accelerated version of [MuJoCo](https://github.com/google-deepmind/mujoco).
The framework provides composable building blocks for environment design,
with minimal dependencies and direct access to native MuJoCo data structures.

## Getting Started

mjlab requires an NVIDIA GPU for training. macOS is supported for evaluation only.

**Try it now:**

Run the demo (no installation needed):

```bash
uvx --from mjlab --refresh demo
```

Or try in [Google Colab](https://colab.research.google.com/github/mujocolab/mjlab/blob/main/notebooks/demo.ipynb) (no local setup required).

**Install from source:**

```bash
git clone https://github.com/mujocolab/mjlab.git && cd mjlab
uv run demo
```

For alternative installation methods (PyPI, Docker), see the [Installation Guide](https://mujocolab.github.io/mjlab/main/source/installation.html).

## Training Examples

### 1. Velocity Tracking

Train a Unitree G1 humanoid to follow velocity commands on flat terrain:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096
```

**Multi-GPU Training:** Scale to multiple GPUs using `--gpu-ids`:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --gpu-ids "[0, 1]" \
  --env.scene.num-envs 4096
```

See the [Distributed Training guide](https://mujocolab.github.io/mjlab/main/source/training/distributed_training.html) for details.

Evaluate a policy while training (fetches latest checkpoint from Weights & Biases):

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 2. Motion Imitation

Train a humanoid to mimic reference motions from a local `.npz` motion file:

```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file /path/to/motion.npz \
  --env.scene.num-envs 4096
```

Play a trained policy from a local checkpoint with a local motion file:

```bash
uv run play Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint-file /path/to/model.pt \
  --motion-file /path/to/motion.npz
```

### 3. Sanity-check with Dummy Agents

Use built-in agents to sanity check your MDP before training:

```bash
uv run play Mjlab-Your-Task-Id --agent zero  # Sends zero actions
uv run play Mjlab-Your-Task-Id --agent random  # Sends uniform random actions
```

When running motion-tracking tasks, add `--motion-file /path/to/motion.npz` to the command.


### 4. Local Motion Workflow and Jump Tracking Notes

This repository also supports a fully local motion-tracking workflow without relying on Weights & Biases motion artifacts.

#### Convert CSV motions to local NPZ files

The local `csv_to_npz` workflow was extended with three practical changes:

- It writes directly to `--output-name` instead of assuming a W&B artifact workflow.
- `--ground-align global` applies a global root-height offset so the foot collision geoms clear the floor.
- `--clearance` sets the desired minimum foot clearance during that alignment step.

If you want to inspect foot penetration before converting, run:

```bash
MUJOCO_GL=egl uv run -m mjlab.scripts.analyze_foot_penetration \
  --input-file /home/nubot/workspace/mjlab/datasets/csv/jump_forward02_poses.csv \
  --input-fps 30 \
  --output-fps 50 \
  --clearance 0.01
```

Convert the CSV file into a local `.npz` motion file:

```bash
MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
  --input-file /home/nubot/workspace/mjlab/datasets/csv/jump_forward02_poses.csv \
  --output-name /home/nubot/workspace/mjlab/datasets/npz/jump_forward02_poses.npz \
  --input-fps 30 \
  --output-fps 50 \
  --ground-align global \
  --clearance 0.01 \
  --render True
```

This produces a local `.npz` motion file and, when `--render True` is set, a local `.mp4` preview video as well.

#### Train from a local NPZ motion file

Train a tracking policy directly from a local motion file:

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file /home/nubot/workspace/mjlab/datasets/npz/jump_forward02_poses.npz \
  --env.scene.num-envs 4096 \
  --agent.logger tensorboard \
  --agent.upload-model False
```

Play a local checkpoint against the same local motion file:

```bash
MUJOCO_GL=egl uv run play Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint-file /home/nubot/workspace/mjlab/logs/rsl_rl/g1_tracking/2026-03-12_16-26-30/model_7000.pt \
  --motion-file /home/nubot/workspace/mjlab/datasets/npz/jump_forward02_poses.npz \
  --num-envs 1 \
  --viewer viser
```

Use `--no-terminations True` when you want to inspect the full motion even if the policy would otherwise fall early, and add `--video True --video-length 500` to record a rollout.

#### Jump-up / jump-forward tuning on flat tracking

For jump-like motions such as `jump_up` and `jump_forward`, the flat tracking task uses a slightly less conservative reward shaping than the original defaults:

| Parameter | Default | Jump-tuned | Meaning |
| --- | --- | --- | --- |
| `motion_global_root_pos.weight` | `0.5` | `1.0` | Makes root-position tracking matter more, so the jump height and global translation stay closer to the reference. |
| `motion_global_root_pos.std` | `0.3` | `0.4` | Widens the useful reward basin so early jump attempts still receive learning signal instead of collapsing to near-zero reward. |
| `motion_body_lin_vel.weight` | `1.0` | `1.5` | Emphasizes body linear-velocity tracking, which is important for take-off and landing timing. |
| `motion_body_lin_vel.std` | `1.0` | `1.5` | Makes the velocity reward more forgiving when the policy is still under-shooting the reference jump speed. |
| `action_rate_l2.weight` | `-1e-1` | `-3e-2` | Reduces the smoothness penalty so the policy can produce the sharper, more explosive actions needed for jumps. |

These changes are intentionally small: they keep the task recognizable while making it easier to learn motions with stronger vertical or forward impulse.

#### Rough tracking: `Mjlab-Tracking-Rough-Unitree-G1`

`Mjlab-Tracking-Rough-Unitree-G1` is a terrain-aware tracking variant intended for fine-tuning from a flat tracking checkpoint.

Compared with the flat tracking task, it changes four major pieces:

1. Terrain
   - The scene switches from a flat plane to a generated terrain grid.
   - The terrain mix is deliberately mild rather than locomotion-hard:
     - `flat`: `0.35`
     - `pyramid_stairs`: `0.15` with `step_height_range=(0.0, 0.05)`
     - `pyramid_stairs_inv`: `0.15` with `step_height_range=(0.0, 0.05)`
     - `hf_pyramid_slope`: `0.10` with `slope_range=(0.0, 0.3)`
     - `hf_pyramid_slope_inv`: `0.10` with `slope_range=(0.0, 0.3)`
     - `random_rough`: `0.10` with `noise_range=(0.01, 0.04)`
     - `wave_terrain`: `0.05` with `amplitude_range=(0.0, 0.08)`
   - The initial terrain curriculum is capped to the easier rows with `max_init_terrain_level=2`.

2. Reward shaping
   - `motion_global_root_pos` becomes XY-only root tracking instead of full XYZ tracking.
   - `motion_global_root_z_vel` is added to preserve jump timing through vertical root velocity.
   - `motion_global_root_z_pos` is added as a soft height term so the policy still aims for the reference apex.
   - `motion_global_root_ori.weight` changes from `0.5 -> 1.0`.
   - `motion_body_ori.weight` changes from `1.0 -> 1.5`.

3. Domain randomization and contacts
   - `push_robot` is removed.
   - `base_com`, `encoder_bias`, and `foot_friction` randomization ranges are narrowed.
   - Contact limits are increased with `nconmax=60`, `contact_sensor_maxmatch=128`, and `ccd_iterations=200` to better handle uneven landings.

4. Terminations
   - Flat-ground-specific z-only terminations (`anchor_pos`, `ee_body_pos`) are removed.
   - `anchor_ori.threshold` is relaxed from `0.8 -> 1.2` so the policy can survive the landing transition on uneven terrain.

A typical local rough-training command looks like this:

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Rough-Unitree-G1 \
  --env.commands.motion.motion-file /home/nubot/workspace/mjlab/datasets/npz/jump_forward02_poses.npz \
  --env.scene.num-envs 4096 \
  --checkpoint-file /home/nubot/workspace/mjlab/logs/rsl_rl/g1_tracking/2026-03-12_16-26-30/model_7000.pt
```


## Documentation

Full documentation is available at **[mujocolab.github.io/mjlab](https://mujocolab.github.io/mjlab/)**.

## Development

```bash
make test          # Run all tests
make test-fast     # Skip slow tests
make format        # Format and lint
make docs          # Build docs locally
```

For development setup: `uvx pre-commit install`

## Citation

mjlab is used in published research and open-source robotics projects. See the [Research](https://mujocolab.github.io/mjlab/main/source/research.html) page for publications and projects, or share your own in [Show and Tell](https://github.com/mujocolab/mjlab/discussions/categories/show-and-tell).

If you use mjlab in your research, please consider citing:

```bibtex
@misc{zakka2026mjlablightweightframeworkgpuaccelerated,
  title={mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning},
  author={Kevin Zakka and Qiayuan Liao and Brent Yi and Louis Le Lay and Koushil Sreenath and Pieter Abbeel},
  year={2026},
  eprint={2601.22074},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.22074},
}
```

## License

mjlab is licensed under the [Apache License, Version 2.0](LICENSE).

### Third-Party Code

Some portions of mjlab are forked from external projects:

- **`src/mjlab/utils/lab_api/`** — Utilities forked from [NVIDIA Isaac
  Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause license, see file
  headers)

Forked components retain their original licenses. See file headers for details.

## Acknowledgments

mjlab wouldn't exist without the excellent work of the Isaac Lab team, whose API
design and abstractions mjlab builds upon.

Thanks to the MuJoCo Warp team — especially Erik Frey and Taylor Howell — for
answering our questions, giving helpful feedback, and implementing features
based on our requests countless times.
