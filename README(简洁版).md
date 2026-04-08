![Project banner](https://raw.githubusercontent.com/mujocolab/mjlab/main/docs/source/_static/mjlab-banner.jpg)

# mjlab

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/mujocolab/mjlab/ci.yml?branch=main)](https://github.com/mujocolab/mjlab/actions/workflows/ci.yml?query=branch%3Amain)
[![Documentation](https://github.com/mujocolab/mjlab/actions/workflows/docs.yml/badge.svg)](https://mujocolab.github.io/mjlab/)
[![License](https://img.shields.io/github/license/mujocolab/mjlab)](https://github.com/mujocolab/mjlab/blob/main/LICENSE)
[![Nightly Benchmarks](https://img.shields.io/badge/Nightly-Benchmarks-blue)](https://mujocolab.github.io/mjlab/nightly/)
[![PyPI](https://img.shields.io/pypi/v/mjlab)](https://pypi.org/project/mjlab/)

mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s manager-based API with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), a GPU-accelerated version of [MuJoCo](https://github.com/google-deepmind/mujoco). The framework provides composable building blocks for environment design, with minimal dependencies and direct access to native MuJoCo data structures.

---

## Table of Contents

- [mjlab](#mjlab)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
  - [Training Examples](#training-examples)
    - [1. Velocity Tracking](#1-velocity-tracking)
    - [2. Motion Imitation](#2-motion-imitation)
    - [3. Sanity-check with Dummy Agents](#3-sanity-check-with-dummy-agents)
  - [Local Motion Workflow](#local-motion-workflow)
    - [Overview](#overview)
    - [Step 1 — (SONIC/BONES only, Optional) Convert SONIC/BONES CSV to mimic-compatible CSV](#step-1--sonicbones-only-optional-convert-sonicbones-csv-to-mimic-compatible-csv)
    - [Step 2 — (Optional) Analyze Foot Penetration](#step-2--optional-analyze-foot-penetration)
    - [Step 3 —  Convert mimic CSV motions to local NPZ files](#step-3---convert-mimic-csv-motions-to-local-npz-files)
    - [Step 4 — Replay and Inspect](#step-4--replay-and-inspect)
    - [Step 5 — Train](#step-5--train)
    - [Ground Alignment Reference](#ground-alignment-reference)
    - [Phased Alignment Tuning Guide](#phased-alignment-tuning-guide)
  - [Advanced: Rough Terrain Tracking](#advanced-rough-terrain-tracking)
    - [Terrain Setup](#terrain-setup)
    - [Stage Curriculum](#stage-curriculum)
    - [Key Differences vs. Flat Tracking](#key-differences-vs-flat-tracking)
  - [Jump Tracking Reward Tuning](#jump-tracking-reward-tuning)
  - [Documentation](#documentation)
  - [Development](#development)
  - [Citation](#citation)
  - [License](#license)
    - [Third-Party Code](#third-party-code)
  - [Acknowledgments](#acknowledgments)

---

## Getting Started

> **Requirements:** An NVIDIA GPU is required for training. macOS is supported for evaluation only.

**Try it instantly (no installation):**

```bash
uvx --from mjlab --refresh demo
```

Or open the [Google Colab demo](https://colab.research.google.com/github/mujocolab/mjlab/blob/main/notebooks/demo.ipynb).

**Install from source:**

```bash
git clone https://github.com/mujocolab/mjlab.git && cd mjlab
uv run demo
```

For PyPI and Docker installation, see the [Installation Guide](https://mujocolab.github.io/mjlab/main/source/installation.html).

---

## Training Examples

### 1. Velocity Tracking

Train a Unitree G1 humanoid to follow velocity commands on flat terrain:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096
```

**Multi-GPU training:** scale with `--gpu-ids`:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --gpu-ids "[0, 1]" \
  --env.scene.num-envs 4096
```

See the [Distributed Training guide](https://mujocolab.github.io/mjlab/main/source/training/distributed_training.html) for details.

**Evaluate while training** (fetches latest checkpoint from Weights & Biases):

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

---

### 2. Motion Imitation

Train a humanoid to mimic reference motions from a local `.npz` file. See the [Local Motion Workflow](#local-motion-workflow) section below for how to prepare your `.npz`.

**Train:**

```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file /path/to/motion.npz \
  --env.scene.num-envs 4096
```

**Play a local checkpoint:**

```bash
uv run play Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint-file /path/to/model.pt \
  --motion-file /path/to/motion.npz
```

---

### 3. Sanity-check with Dummy Agents

Use built-in agents to validate your MDP before training:

```bash
uv run play Mjlab-Your-Task-Id --agent zero    # All-zero actions
uv run play Mjlab-Your-Task-Id --agent random  # Uniform random actions
```

For motion-tracking tasks, also add `--motion-file /path/to/motion.npz`.

---

## Local Motion Workflow

### Overview

This section covers the full pipeline for turning raw motion data into a training-ready `.npz` file.

```
[SONIC/BONES CSV] ──(Step 1, optional)──▶ mimic CSV ──▶ NPZ ──▶ replay ──▶ train
[mimic-style CSV] ──────────────────────────────────▶ NPZ ──▶ replay ──▶ train
```

**Workflow rule of thumb:**

- If your source is **SONIC/BONES CSV**, first run **Step 1** to convert it into a mimic-compatible numeric CSV, then continue with **Step 2 / Step 3 / Step 4 / Step 5**.
- If your source is **already a mimic-style numeric CSV**, skip **Step 1** and start directly from **Step 2** or **Step 3**.
- **Step 3 is the one required conversion step for all CSV-based workflows**: regardless of where the CSV came from, you must convert the mimic-style CSV into a local `.npz` before replay or training.

**Local Motion Workflow Navigation**

- [Step 1 — (SONIC/BONES only, Optional) Convert SONIC/BONES CSV to mimic-compatible CSV](#step-1--sonicbones-only-optional-convert-sonicbones-csv-to-mimic-compatible-csv)
- [Step 2 — (Optional) Analyze Foot Penetration](#step-2--optional-analyze-foot-penetration)
- [Step 3 — Convert mimic CSV motions to local NPZ files](#step-3--required-convert-mimic-csv-motions-to-local-npz-files)
- [Step 4 — Replay and Inspect](#step-4--replay-and-inspect)
- [Step 5 — Train](#step-5--train)

> **Shortest executable path (Step 2 → Step 3 → Step 5)**  
> If you already have a mimic-style CSV and want the fastest path to start training, this is the minimal practical sequence.  
> Step 4 replay is still strongly recommended, but it is omitted here on purpose to keep the path short.
>
> **Step 2 — Analyze**
> ```bash
> MUJOCO_GL=egl uv run -m mjlab.scripts.analyze_foot_penetration \
>   --input-file /home/nubot/workspace/mjlab/datasets/csv/tiger_jump_to_shoulder_roll_R_001__A415_M_mimic.csv \
>   --input-fps 120 \
>   --output-fps 50 \
>   --clearance 0.01
> ```
>
> **Step 3 — Convert**
> ```bash
> MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
>   --input-file /home/nubot/workspace/mjlab/datasets/csv/tiger_jump_to_shoulder_roll_R_001__A415_M_mimic.csv \
>   --output-name /home/nubot/workspace/mjlab/datasets/npz/tiger_jump_to_shoulder_roll_R_001__A415_M_phased_auto.npz \
>   --input-fps 120 \
>   --output-fps 50 \
>   --ground-align phased \
>   --clearance 0.01 \
>   --phase-blend-points "0.049:0.00,0.593:1.00" \
>   --render True
> ```
>
> **Step 5 — Train**
> ```bash
> MUJOCO_GL=egl uv run train Mjlab-Tracking-Flat-Unitree-G1 \
>   --env.commands.motion.motion-file /home/nubot/workspace/mjlab/datasets/npz/tiger_jump_to_shoulder_roll_R_001__A415_M_phased_auto.npz \
>   --env.scene.num-envs 4096 \
>   --agent.logger tensorboard
> ```

| Step                        | Script                        | Required?                            |
| --------------------------- | ----------------------------- | ------------------------------------ |
| 1. Convert SONIC/BONES CSV  | `sonic2mimic.py`              | Only if source is SONIC/BONES format |
| 2. Analyze foot penetration | `analyze_foot_penetration.py` | Optional diagnostic                  |
| 3. Convert mimic CSV → NPZ  | `csv_to_npz.py`               | **Yes, for all CSV inputs**          |
| 4. Replay and inspect       | `replay-motion`               | Strongly recommended                 |
| 5. Train                    | `train`                       | **Yes**                              |

---

### Step 1 — (SONIC/BONES only, Optional) Convert SONIC/BONES CSV to mimic-compatible CSV

> **Run this step only when your source file is SONIC/BONES-style CSV.**  
> If your source file is already a mimic-style numeric CSV, skip this step entirely and move on to Step 2 or Step 3.

If your source motion comes from a SONIC/BONES-style CSV export, convert it first:

```bash
uv run src/mjlab/scripts/sonic2mimic.py \
  --inputs /path/to/source.csv
```

This writes `*_mimic.csv` next to the source file. What it does:

- Validates required SONIC/BONES columns
- Converts `root_translateX/Y/Z` with a fixed position scale of `0.01`
- Converts `root_rotateX/Y/Z` from Euler degrees to quaternion (ZYX order)
- Converts all joint DOF columns from degrees to radians

> **Note:** `sonic2mimic.py` only converts units and representation — it does **not** resample time. A 120 fps SONIC export stays 120 fps. Use `--input-fps 120` in Steps 2 and 3.

**Optional flags:**

| Flag                                 | Description                                    |
| ------------------------------------ | ---------------------------------------------- |
| `--outputs /path/to/out.csv`         | Specify output file name                       |
| `--z-offset 0.02`                    | Apply a constant vertical offset after scaling |
| `--inputm <dir>` + `--outputm <dir>` | Batch mode: recursively convert a directory    |

**Batch example:**

```bash
uv run src/mjlab/scripts/sonic2mimic.py \
  --inputm /path/to/csv_dir \
  --outputm /path/to/csv_mimic_dir
```

---

### Step 2 — (Optional) Analyze Foot Penetration

Before converting, you can inspect ground-penetration severity and get auto-suggested alignment settings for Step 3.

```bash
MUJOCO_GL=egl uv run -m mjlab.scripts.analyze_foot_penetration \
  --input-file /path/to/motion_mimic.csv \
  --input-fps 120 \
  --output-fps 50 \
  --clearance 0.01
```

> **This script is diagnostic only — it never writes any files.**

**What to look for in the output:**

| Output field                                                 | How to use it                                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `recommended_global_z_offset_m` in `Foot grounding summary`  | Constant lift needed for `--ground-align global`             |
| `recommended_global_z_offset_m` in `Whole-body collision grounding summary` | If much larger than the foot value, prefer `--ground-align phased` |
| `suggested_phase_blend_points`                               | Copy directly into Step 3 as `--phase-blend-points`          |
| `csv_to_npz_hint`                                            | Ready-to-paste CLI fragment for Step 3                       |
| `Suggested coarse phases`                                    | Verify that inferred takeoff / airborne / landing windows look correct |

The analysis auto-infers grounded vs. airborne phases from the motion's own foot-height distribution — no manual stage definition is needed by default.

**For example — inspect a shoulder-roll clip before conversion:**

```bash
MUJOCO_GL=egl uv run -m mjlab.scripts.analyze_foot_penetration \
  --input-file /home/nubot/workspace/mjlab/datasets/csv/tiger_jump_to_shoulder_roll_R_001__A415_M_mimic.csv \
  --input-fps 120 \
  --output-fps 50 \
  --clearance 0.01
```

---

### Step 3 —  Convert mimic CSV motions to local NPZ files

This is the required conversion step for **all CSV-based workflows**.

- If you started from SONIC/BONES data, Step 1 should have produced a mimic-compatible CSV for you, and **this step converts that mimic CSV into `.npz`**.
- If you already started from a mimic-style numeric CSV, **this is your first mandatory step**.
- Replay, `play`, and training all consume the exported `.npz`, not the raw CSV.

**Simple motions (walking, stretching)** — one global lift:

```bash
MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
  --input-file /path/to/motion_mimic.csv \
  --output-name /path/to/output.npz \
  --input-fps 120 \
  --output-fps 50 \
  --ground-align global \
  --clearance 0.01 \
  --render True
```

**Aerial motions (flips, shoulder rolls, jumps)** — frame-varying lift:

```bash
# Option A: paste phase-blend-points from the Step 2 analysis output
MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
  --input-file /path/to/motion_mimic.csv \
  --output-name /path/to/output.npz \
  --input-fps 120 \
  --output-fps 50 \
  --ground-align phased \
  --clearance 0.01 \
  --phase-blend-points "0.049:0.00,0.593:1.00" \
  --render True

# Option B: let csv_to_npz auto-infer phase control points directly (easier starting point)
MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
  --input-file /path/to/motion_mimic.csv \
  --output-name /path/to/output.npz \
  --input-fps 120 \
  --output-fps 50 \
  --ground-align phased \
  --phase-control-mode auto \
  --clearance 0.01 \
  --render True
```

When `--render True` is set, an `.mp4` preview is also written alongside the `.npz`.

**For example — convert a broad jump with one global lift:**

```bash
MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
  --input-file /home/nubot/workspace/mjlab/datasets/csv/jumpforward/broad_jump_002__A362_M_mimic.csv \
  --output-name /home/nubot/workspace/mjlab/datasets/npz/jumpforward/broad_jump_002__A362_M_mimic_grounded.npz \
  --input-fps 120 \
  --output-fps 50 \
  --ground-align global \
  --clearance 0.01
```

**For example — convert a shoulder roll with the blend points suggested by Step 2:**

```bash
MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
  --input-file /home/nubot/workspace/mjlab/datasets/csv/tiger_jump_to_shoulder_roll_R_001__A415_M_mimic.csv \
  --output-name /home/nubot/workspace/mjlab/datasets/npz/tiger_jump_to_shoulder_roll_R_001__A415_M_phased_auto.npz \
  --input-fps 120 \
  --output-fps 50 \
  --ground-align phased \
  --clearance 0.01 \
  --phase-blend-points "0.049:0.00,0.593:1.00" \
  --render True
```

**Batch conversion:**

```bash
MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
  --inputm /path/to/csv_dir \
  --outputm /path/to/npz_dir \
  --input-fps 120 \
  --output-fps 50 \
  --render True
```

Non-numeric / header CSV files (e.g., original SONIC exports) are automatically skipped.

---

### Step 4 — Replay and Inspect

Strongly recommended: replay the converted `.npz` before committing to training:

```bash
uv run replay-motion Mjlab-Tracking-Flat-Unitree-G1 \
  --motion-file /path/to/output.npz \
  --start-paused \
  --reference-viz ghost
```

**What to check:**

- Step through landing frames with `→` and watch for head / torso / hand clipping.
- Read the overlay `min_foot_bottom_z` to confirm foot clearance is as expected.
- If landing still dips too low, return to Step 3 and tune phased parameters (see [Phased Alignment Tuning Guide](#phased-alignment-tuning-guide)).

**Replay parameters:**

| Parameter                              | Description                                                  |
| -------------------------------------- | ------------------------------------------------------------ |
| `--motion-file`                        | Path to the `.npz` file                                      |
| `--viewer {native,viser,auto}`         | Viewer backend (default: `native`)                           |
| `--num-envs`                           | Number of environments (default: `1`)                        |
| `--start-paused` / `--no-start-paused` | Start paused or running                                      |
| `--loop` / `--no-loop`                 | Loop or stop at last frame                                   |
| `--reference-viz {none,ghost,frames}`  | Show tracking reference visualization                        |
| `--root-body-name`                     | Override which body is treated as the replay root            |
| `--foot-geom-pattern`                  | Regex for foot collision geoms (used for `min_foot_bottom_z`) |
| `--print-summary`                      | Print motion metadata at startup                             |
| `--verbosity {silent,info,debug}`      | Viewer logging verbosity                                     |

**Native viewer hotkeys:**

| Key     | Action                                                       |
| ------- | ------------------------------------------------------------ |
| `Space` | Pause / resume                                               |
| `→`     | Single-step one frame (while paused)                         |
| `Enter` | Reset to frame 0                                             |
| `D`     | Dump current frame's root pose, velocity, joint values, and `min_foot_bottom_z` to terminal |

**For example — replay the converted shoulder-roll motion and inspect the landing:**

```bash
uv run replay-motion Mjlab-Tracking-Flat-Unitree-G1 \
  --motion-file /home/nubot/workspace/mjlab/datasets/npz/tiger_jump_to_shoulder_roll_R_001__A415_M_phased_auto.npz \
  --start-paused \
  --reference-viz ghost
```

---

### Step 5 — Train

Use the **same `.npz` file** you verified in Step 4:

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file /path/to/output.npz \
  --env.scene.num-envs 4096 \
  --agent.logger tensorboard \
  --agent.upload-model False
```

Play a checkpoint after training:

```bash
MUJOCO_GL=egl uv run play Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint-file /path/to/model.pt \
  --motion-file /path/to/output.npz \
  --num-envs 1 \
  --viewer viser
```

Add `--no-terminations True` to inspect the full motion even if the policy falls early, and `--video True --video-length 500` to record a rollout.

> **Best practice:** The `.npz` you replay and the `.npz` you train on should ideally be the same file. This keeps your inspection result, training reference motion, and later evaluation consistent.

**For example — train a grounded broad-jump motion:**

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file /home/nubot/workspace/mjlab/datasets/npz/jumpforward/broad_jump_002__A362_M_mimic_grounded.npz \
  --env.scene.num-envs 4096 \
  --agent.logger tensorboard
```

**For example — if jump-forward training is unstable, start from a conservative preset:**

This disables push perturbations entirely and relaxes the termination thresholds, giving the policy more room to explore explosive take-off actions without being terminated too early.

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file /home/nubot/workspace/mjlab/datasets/npz/jumpforward/broad_jump_002__A362_M_mimic_grounded.npz \
  --env.scene.num-envs 4096 \
  --env.terminations.anchor-pos.params.threshold 0.35 \
  --env.terminations.ee-body-pos.params.threshold 0.35 \
  --env.events.push-robot.params.velocity-range.x '(0.0, 0.0)' \
  --env.events.push-robot.params.velocity-range.y '(0.0, 0.0)' \
  --env.events.push-robot.params.velocity-range.z '(0.0, 0.0)' \
  --env.events.push-robot.params.velocity-range.roll '(0.0, 0.0)' \
  --env.events.push-robot.params.velocity-range.pitch '(0.0, 0.0)' \
  --env.events.push-robot.params.velocity-range.yaw '(0.0, 0.0)' \
  --agent.logger tensorboard
```

**For example — play a trained checkpoint on the same shoulder-roll motion:**

```bash
MUJOCO_GL=egl uv run play Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint-file /home/nubot/workspace/mjlab/logs/rsl_rl/g1_tracking/2026-04-08_01-55-09/model_12000.pt \
  --motion-file /home/nubot/workspace/mjlab/datasets/npz/tiger_jump_to_shoulder_roll_R_001__A415_M_phased_auto.npz \
  --num-envs 1
```

---

### Ground Alignment Reference

| Mode                    | When to use                                                  |
| ----------------------- | ------------------------------------------------------------ |
| `--ground-align none`   | Keep raw source heights unchanged                            |
| `--ground-align global` | Apply one constant upward offset — good for simple grounded motions |
| `--ground-align phased` | Apply a frame-varying offset — better for flips / rolls where a single global lift is too conservative |

> **When to choose `phased` over `global`:** If the whole-body `recommended_global_z_offset_m` from Step 2 is much larger than the foot-based value, a single global lift will make the whole motion look too high. Use `phased` instead.
>
> **Heads-up:** `--clearance` only constrains foot collision geoms. It does **not** guarantee that the head, hands, or torso stay above ground during flips or hard landings. If the feet look fine but the head still clips, switch from `global` to `phased`.

---

### Phased Alignment Tuning Guide

**Core parameters:**

| Parameter                            | Effect                                                       |
| ------------------------------------ | ------------------------------------------------------------ |
| `--clearance`                        | Target minimum foot-bottom clearance in meters (start with `0.01`) |
| `--phase-control-mode {auto,manual}` | `auto` infers control points from the motion; `manual` uses explicit flags below |
| `--phase-grounded-height`            | Foot height below which alignment stays foot-based (raise to activate airborne logic earlier) |
| `--phase-airborne-height`            | Foot height above which full whole-body lift is allowed (lower to protect landings sooner) |
| `--phase-window-s`                   | Foot-height context window width; larger = steadier, less reactive |
| `--phase-lookahead-s`                | How far ahead to prepare for an upcoming landing; larger = lift starts earlier in flight |
| `--phase-smoothing-s`                | Temporal smoothing on frame-wise offsets; larger = softer transitions |
| `--whole-body-geom-pattern`          | Regex for geoms allowed to request extra airborne lift (default `.*_collision$`) |

**Iterative tuning workflow:**

1. Start from `--ground-align phased --phase-control-mode auto --clearance 0.01`.
2. **Landing still clips ground?** → increase `--phase-lookahead-s`, then decrease `--phase-airborne-height`.
3. **Jump floats too early or too high?** → decrease `--phase-lookahead-s` or increase `--phase-airborne-height`.
4. **Transitions look abrupt?** → increase `--phase-smoothing-s`. Too much of the motion lifted? Decrease it.
5. **Near-ground frames getting airborne-style lift?** → lower `--phase-grounded-height`.

**Custom multi-stage schedules with `--phase-blend-points`:**

Provide control points as `foot_height:blend_weight` pairs (comma-separated). `0.0` = fully foot-based lift; `1.0` = full whole-body lift:

```bash
--phase-blend-points "0.00:0.0,0.03:0.0,0.06:0.35,0.10:1.0"
# 0.00–0.03 m : stay foot-based
#    ~0.06 m  : begin mixing in airborne lift
#   ≥ 0.10 m  : full whole-body lift
```

> When `--phase-blend-points` is provided it overrides `--phase-grounded-height` and `--phase-airborne-height`.

---

## Advanced: Rough Terrain Tracking

`Mjlab-Tracking-Rough-Unitree-G1` is a terrain-aware fine-tuning variant. Start from a flat tracking checkpoint:

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Rough-Unitree-G1 \
  --env.commands.motion.motion-file /path/to/motion.npz \
  --env.scene.num-envs 4096 \
  --checkpoint-file /path/to/flat_checkpoint.pt
```

Play a rough checkpoint:

```bash
MUJOCO_GL=egl uv run play Mjlab-Tracking-Rough-Unitree-G1 \
  --checkpoint-file /path/to/rough_checkpoint.pt \
  --motion-file /path/to/motion.npz \
  --num-envs 16 \
  --viewer viser
```

The terrain sampler is **stage-aware**: the viewer automatically samples terrain proportions and difficulty matching the loaded checkpoint's curriculum stage.

**For example — fine-tune a grounded broad jump on rough terrain from an existing flat checkpoint:**

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Rough-Unitree-G1 \
  --env.commands.motion.motion-file /home/nubot/workspace/mjlab/datasets/npz/jumpforward/broad_jump_002__A362_M_mimic_grounded.npz \
  --env.scene.num-envs 4096 \
  --checkpoint-file /home/nubot/workspace/mjlab/logs/rsl_rl/g1_tracking/2026-04-08_01-55-09/model_13500.pt
```

**For example — play an existing rough-terrain checkpoint:**

```bash
MUJOCO_GL=egl uv run play Mjlab-Tracking-Rough-Unitree-G1 \
  --checkpoint-file /home/nubot/workspace/mjlab/logs/rsl_rl/g1_tracking_rough/2026-04-01_16-02-57/model_30500.pt \
  --motion-file /home/nubot/workspace/mjlab/datasets/npz/jumpforward/broad_jump_002__A362_M_mimic_grounded.npz \
  --num-envs 16 \
  --viewer viser
```

### Terrain Setup

Three terrain families on a `6 × 12` curriculum grid. Stairs and slopes are intentionally excluded to keep fine-tuning close to the flat reference:

| Terrain        | Parameters                    | Base column weight |
| -------------- | ----------------------------- | ------------------ |
| `flat`         | —                             | 0.25               |
| `random_rough` | `noise_range=(0.01, 0.04)`    | 0.25               |
| `wave_terrain` | `amplitude_range=(0.0, 0.08)` | 0.50               |

### Stage Curriculum

Stages are measured from the start of rough fine-tuning (not total training lifetime). A flat checkpoint like `model_7000.pt` still begins rough curriculum at stage 0:

| Rough phase iterations | `max_terrain_level` | `flat` | `random_rough` | `wave_terrain` |
| ---------------------- | ------------------- | ------ | -------------- | -------------- |
| ≥ 0                    | 0                   | 0.70   | 0.20           | 0.10           |
| ≥ 3,000                | 2                   | 0.50   | 0.20           | 0.30           |
| ≥ 6,000                | 4                   | 0.30   | 0.25           | 0.45           |
| ≥ 9,000                | 5                   | 0.15   | 0.25           | 0.60           |

### Key Differences vs. Flat Tracking

| Aspect                          | Flat     | Rough                         |
| ------------------------------- | -------- | ----------------------------- |
| Root position tracking          | Full XYZ | XY-only                       |
| `motion_global_root_z_vel`      | —        | Added (preserves jump timing) |
| `motion_global_root_z_pos`      | —        | Added (soft height term)      |
| `motion_global_root_ori.weight` | 0.5      | 1.0                           |
| `motion_body_ori.weight`        | 1.0      | 1.5                           |
| `push_robot`                    | Enabled  | Disabled                      |
| `anchor_ori.threshold`          | 0.8      | 1.2                           |
| Contact limits (`nconmax`)      | Default  | 60                            |

> **Note:** New rough checkpoints save their rough-phase offset into metadata so training can be resumed correctly and `play` can restore the matching curriculum stage. Older rough checkpoints created before this feature do not contain this metadata.

---

## Jump Tracking Reward Tuning

For jump-like motions (`jump_up`, `jump_forward`), the flat tracking task already applies these jump-friendly values by default. They are intentionally tuned relative to the original conservative defaults to allow more explosive behavior without losing stability:

| Parameter                       | Default | Jump-tuned | Purpose                                                 |
| ------------------------------- | ------- | ---------- | ------------------------------------------------------- |
| `motion_global_root_pos.weight` | 0.5     | 1.0        | More emphasis on jump height and global translation     |
| `motion_global_root_pos.std`    | 0.3     | 0.4        | Wider reward basin so early attempts still get signal   |
| `motion_body_lin_vel.weight`    | 1.0     | 1.5        | Better take-off and landing timing                      |
| `motion_body_lin_vel.std`       | 1.0     | 1.5        | More forgiving when policy under-shoots reference speed |
| `action_rate_l2.weight`         | −1e-1   | −3e-2      | Allow sharper, more explosive actions                   |

If you need to further adjust these values for a specific motion, you can override them directly from the command line:

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file /path/to/motion.npz \
  --env.scene.num-envs 4096 \
  --env.rewards.motion-global-root-pos.weight 1.2 \
  --env.rewards.motion-global-root-pos.std 0.5 \
  --env.rewards.motion-body-lin-vel.weight 2.0 \
  --env.rewards.action-rate-l2.weight -0.01 \
  --agent.logger tensorboard
```

---

## Documentation

Full documentation: **[mujocolab.github.io/mjlab](https://mujocolab.github.io/mjlab/)**

---

## Development

```bash
make test          # Run all tests
make test-fast     # Skip slow tests
make format        # Format and lint
make docs          # Build docs locally
```

For development setup: `uvx pre-commit install`

---

## Citation

mjlab is used in published research and open-source robotics projects. See the [Research page](https://mujocolab.github.io/mjlab/main/source/research.html) or share your own in [Show and Tell](https://github.com/mujocolab/mjlab/discussions/categories/show-and-tell).

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

---

## License

mjlab is licensed under the [Apache License, Version 2.0](LICENSE).

### Third-Party Code

- **`src/mjlab/utils/lab_api/`** — Utilities forked from [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause license, see file headers)

Forked components retain their original licenses.

---

## Acknowledgments

mjlab wouldn't exist without the excellent work of the Isaac Lab team, whose API design and abstractions mjlab builds upon.

Thanks to the MuJoCo Warp team — especially Erik Frey and Taylor Howell — for answering our questions, giving helpful feedback, and implementing features based on our requests.
