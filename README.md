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

Replay a motion `.npz` directly in MuJoCo without loading a policy:

```bash
uv run replay-motion Mjlab-Tracking-Flat-Unitree-G1 \
  --motion-file /path/to/motion.npz
```

This replays each motion frame by writing the root and joint state directly back
into the simulator. By default it uses the MuJoCo native viewer, where `Space`
pauses, `Right Arrow` single-steps, `Enter` resets to frame `0`, and `D` dumps
the current frame's root/joint data to the terminal. The native viewer overlay
also shows the current frame's `min_foot_bottom_z`, computed from the matched
foot collision geoms.

#### Replay Motion Viewer

Activate the project environment first, then run `uv run replay-motion`:

```bash
source /home/nubot/workspace/mjlab/.venv/bin/activate

uv run replay-motion Mjlab-Tracking-Flat-Unitree-G1 \
  --motion-file /path/to/motion.npz \
  --start-paused
```

Example for a local G1 motion:

```bash
source /home/nubot/workspace/mjlab/.venv/bin/activate

uv run replay-motion Mjlab-Tracking-Flat-Unitree-G1 \
  --motion-file /home/nubot/workspace/mjlab/datasets/npz/example_motion.npz \
  --start-paused \
  --reference-viz ghost
```

Main parameters:

- `task_id`: the tracking task to build, for example `Mjlab-Tracking-Flat-Unitree-G1`.
- `--motion-file`: required path to the replayed motion `.npz`.
- `--viewer {native,viser,auto}`: choose the viewer backend. Default is `native`.
- `--num-envs`: number of environments to instantiate. Default is `1`.
- `--start-paused` / `--no-start-paused`: start the replay paused or running.
- `--loop` / `--no-loop`: loop the motion or stop at the last frame.
- `--reference-viz {none,ghost,frames}`: optionally show the tracking reference visualization.
- `--root-body-name`: override which motion body is treated as the replay root.
- `--foot-geom-pattern`: regex used to find the foot collision geoms for `min_foot_bottom_z` display.
- `--print-summary` / `--no-print-summary`: print motion metadata to the terminal at startup.
- `--verbosity {silent,info,debug}`: viewer logging verbosity.

Useful native viewer hotkeys:

- `Space`: pause / resume replay.
- `Right Arrow`: single-step one replay frame while paused.
- `Enter`: reset to frame `0`.
- `D`: dump the current frame's root pose, root velocity, joint values, and `min_foot_bottom_z` to the terminal.

### 3. Sanity-check with Dummy Agents

Use built-in agents to sanity check your MDP before training:

```bash
uv run play Mjlab-Your-Task-Id --agent zero  # Sends zero actions
uv run play Mjlab-Your-Task-Id --agent random  # Sends uniform random actions
```

When running motion-tracking tasks, add `--motion-file /path/to/motion.npz` to the command.


### 4. Local Motion Workflow and Jump Tracking Notes

This repository also supports a fully local motion-tracking workflow without relying on Weights & Biases motion artifacts.

#### Convert SONIC/BONES CSV to mimic-compatible CSV

If your source motion comes from SONIC/BONES-style CSV export, first convert it into
the mimic-compatible numeric CSV expected by the downstream motion tools:

```bash
python src/mjlab/scripts/sonic2mimic.py \
  --inputs /home/nubot/workspace/mjlab/datasets/csv/body_stretch_3_002__A052.csv
```

This writes `/path/to/input_mimic.csv` next to the source file by default. The
script is a data-processing step that:

- Validates that the required SONIC/BONES columns are present.
- Converts `root_translateX/Y/Z` using a fixed position scale of `0.01`.
- Converts `root_rotateX/Y/Z` from Euler degrees to quaternion using a fixed `ZYX` composition order.
- Converts all joint DOF columns from degrees to radians.
- Writes a plain numeric CSV in the format expected by `csv_to_npz`.

Important: `sonic2mimic.py` only converts representation and units. It does not
resample time. If the source SONIC/BONES CSV is `120 fps`, the generated
`*_mimic.csv` is still `120 fps`.

Optional: add `--outputs /path/to/out.csv` to choose the output file name, or
`--z-offset 0.02` to apply a constant vertical offset after scaling.

`sonic2mimic.py` also supports batch conversion. In batch mode it recursively
scans `--inputm` for `.csv` files, skips files that already end with
`_mimic.csv`, and writes matching `_mimic.csv` outputs under `--outputm` while
preserving the relative directory structure. If `--outputm` is omitted, the
converted files are written next to the source CSV files.
```bvh
Frames: 595
Frame Time: 0.008333 #broad_jump_004__A359_M.bvh,0.008333 ≈ 1 / 120fps
```

```bash
python src/mjlab/scripts/sonic2mimic.py \
  --inputm /home/nubot/workspace/mjlab/datasets/csv/jumpforward \
  --outputm /home/nubot/workspace/mjlab/datasets/csv_mimic/jumpforward
```

If your source file is already a mimic-style numeric CSV, you can skip this step.

#### Convert mimic CSV motions to local NPZ files

There are two related scripts in this step:

- `analyze_foot_penetration.py`: diagnostic only. It loads a mimic-style CSV, evaluates the
  lowest foot-bottom height against the ground plane, prints the worst frames, and reports a
  recommended global root-height lift. It does not write an `.npz` file.
- `csv_to_npz.py`: conversion script. It turns a mimic-style CSV into a local `.npz` motion file,
  and optionally renders an `.mp4` preview. When `--ground-align global` is enabled, it runs the
  same foot-ground analysis internally and applies one global root `z` offset so the minimum
  foot-bottom clearance reaches the requested target.

Important for SONIC/BONES data: treat the converted `*_mimic.csv` as `120 fps`
input unless you know the original source was exported at a different frame
rate. In other words, for SONIC-derived mimic CSV files, use `--input-fps 120`
when running `analyze_foot_penetration.py` or `csv_to_npz.py`.

About `--ground-align` and `--clearance`:

- `--ground-align none`: use the raw root heights from the CSV.
- `--ground-align global`: analyze the whole motion and apply one global upward offset if needed.
- `--clearance 0.01`: set the target minimum foot-bottom clearance to `0.01 m` (1 cm).

Ground-alignment workflow notes:

- `analyze_foot_penetration.py` is diagnostic only. It reports the current worst-case foot-bottom height and the recommended global lift, but it does not modify any files.
- `csv_to_npz.py --ground-align global` is the step that actually changes the output motion. It writes a new `.npz` whose root `z` trajectory has been shifted upward by one constant offset.
- `replay-motion`, `play`, and training do not run ground alignment again. They simply consume the `.npz` motion file you produced. In practice, this means the training-time reference motion is the aligned `.npz` if you converted with `--ground-align global`, and the raw motion otherwise.
- The optional `.mp4` preview from `csv_to_npz.py --render True` is rendered from the same aligned motion data, so the robot-ground relationship should match `replay-motion` for the same `.npz`. Small visual differences can still appear because the preview video and `replay-motion` use different default cameras.
- `--clearance` only constrains the minimum height of the matched foot collision geoms. It does not guarantee that other body parts such as the head, hands, or torso will stay above the ground plane during flips, rolls, or hard landings.

Practical interpretation:

- Use `--ground-align global` when you want the exported `.npz` to have a controlled foot-ground offset before replay or training.
- Use `--ground-align none` when you intentionally want to keep the raw source heights.
- If the feet look correct but the head still clips the floor during a landing, that is expected under the current implementation: the alignment target is foot clearance, not full-body clearance.

You do not need to run `analyze_foot_penetration.py` before `csv_to_npz.py`. The standalone
analysis script is optional and is mainly useful when you want to inspect the worst frames or tune
the clearance value before conversion.

If you want to inspect foot penetration before converting, run:

```bash
MUJOCO_GL=egl uv run -m mjlab.scripts.analyze_foot_penetration \
  --input-file /home/nubot/workspace/mjlab/datasets/csv/body_stretch_3_002__A052_mimic.csv \
  --input-fps 120 \
  --output-fps 50 \
  --clearance 0.01
```

Convert the mimic CSV file into a local `.npz` motion file:

```bash
MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
  --input-file /home/nubot/workspace/mjlab/datasets/csv/body_stretch_3_002__A052_mimic.csv \
  --output-name /home/nubot/workspace/mjlab/datasets/npz/body_stretch_3_002__A052.npz \
  --input-fps 120 \
  --output-fps 50 \
  --ground-align global \
  --clearance 0.01 \
  --render True
```

This produces a local `.npz` motion file and, when `--render True` is set, a local `.mp4` preview video as well. If you do not need automatic ground alignment, you can omit `--ground-align global --clearance 0.01`.

`csv_to_npz.py` also supports batch conversion, similar to `sonic2mimic.py`. In
batch mode it recursively scans `--inputm` for `.csv` files and writes matching
`.npz` outputs under `--outputm` while preserving the relative directory
structure. If `--outputm` is omitted, the `.npz` files are written next to the
input CSV files. It automatically skips non-numeric/header CSV files such as the
original SONIC source exports and converts only mimic-style numeric CSV files.

```bash
MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
  --inputm /home/nubot/workspace/mjlab/datasets/csv/jumpforward \
  --outputm /home/nubot/workspace/mjlab/datasets/npz/jumpforward \
  --input-fps 120 \
  --output-fps 50 \
  --render True
```

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
   - The rough task now keeps only three terrain families:
     - `flat`
     - `random_rough` with `noise_range=(0.01, 0.04)`
     - `wave_terrain` with `amplitude_range=(0.0, 0.08)`
   - Stairs and slope terrains are intentionally removed so rough fine-tuning stays closer to the flat reference motion and focuses on mild uneven-ground adaptation.
   - The generator keeps a `6 x 12` curriculum grid. Its base column allocation is biased toward waves so later stages have more smooth-undulating terrain coverage:
     - `flat`: `0.25`
     - `random_rough`: `0.25`
     - `wave_terrain`: `0.50`
   - Training starts from the easiest terrain row with `max_init_terrain_level=0`.

2. Stage curriculum
   - Rough terrain progression is staged relative to the start of the rough fine-tuning phase, not the total lifetime of the policy.
   - The current implementation uses four stages:

| Rough phase progress | `max_terrain_level` | `flat` | `random_rough` | `wave_terrain` |
| --- | --- | --- | --- | --- |
| `>= 0` iterations | `0` | `0.70` | `0.20` | `0.10` |
| `>= 3,000` iterations | `2` | `0.50` | `0.20` | `0.30` |
| `>= 6,000` iterations | `4` | `0.30` | `0.25` | `0.45` |
| `>= 9,000` iterations | `5` | `0.15` | `0.25` | `0.60` |

   - The stage switch points are defined in `src/mjlab/tasks/tracking/config/g1/env_cfgs.py` as `step = iterations * 24`, because tracking PPO uses `num_steps_per_env = 24`.
   - `wave_terrain` is intentionally weighted higher than `random_rough` in later stages so the final rough policy sees more continuous undulating terrain than fully irregular bumps.

3. Reward shaping
   - `motion_global_root_pos` becomes XY-only root tracking instead of full XYZ tracking.
   - `motion_global_root_z_vel` is added to preserve jump timing through vertical root velocity.
   - `motion_global_root_z_pos` is added as a soft height term so the policy still aims for the reference apex.
   - `motion_global_root_ori.weight` changes from `0.5 -> 1.0`.
   - `motion_body_ori.weight` changes from `1.0 -> 1.5`.

4. Domain randomization and contacts
   - `push_robot` is removed.
   - `base_com`, `encoder_bias`, and `foot_friction` randomization ranges are narrowed.
   - Contact limits are increased with `nconmax=60`, `contact_sensor_maxmatch=128`, and `ccd_iterations=200` to better handle uneven landings.

5. Terminations
   - Flat-ground-specific z-only terminations (`anchor_pos`, `ee_body_pos`) are removed.
   - `anchor_ori.threshold` is relaxed from `0.8 -> 1.2` so the policy can survive the landing transition on uneven terrain.

#### How rough-stage offsets work

When you start rough training from a flat checkpoint, the code automatically captures the current total training step counter and treats that point as the beginning of the rough phase. This means a flat checkpoint such as `model_7000.pt` still starts rough curriculum at rough-stage `0`, not at the `7000`-iteration stage.

New rough checkpoints also save this rough-phase offset into the checkpoint metadata. As a result:

- resuming rough training continues from the correct rough curriculum stage
- `play` can restore the same rough stage and visualize the matching terrain mix

Older rough checkpoints created before this metadata was added do not contain the saved offset, so stage-aware `play` cannot perfectly recover their rough progress.

A typical local rough-training command looks like this:

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Rough-Unitree-G1 \
  --env.commands.motion.motion-file /home/nubot/workspace/mjlab/datasets/npz/jump_forward02_poses.npz \
  --env.scene.num-envs 4096 \
  --checkpoint-file /home/nubot/workspace/mjlab/logs/rsl_rl/g1_tracking/2026-03-12_16-26-30/model_7000.pt
```

Play a rough checkpoint against the same local motion file:

```bash
MUJOCO_GL=egl uv run play Mjlab-Tracking-Rough-Unitree-G1 \
  --checkpoint-file /home/nubot/workspace/mjlab/logs/rsl_rl/g1_tracking_rough/2026-03-12_16-26-30/model_1000.pt \
  --motion-file /home/nubot/workspace/mjlab/datasets/npz/jump_forward02_poses.npz \
  --num-envs 16 \
  --viewer viser
```

For rough `play`, the terrain sampler is stage-aware: if the loaded checkpoint corresponds to the second rough stage, the viewer will sample second-stage terrain proportions and difficulty bounds instead of always showing stage `0`.


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
