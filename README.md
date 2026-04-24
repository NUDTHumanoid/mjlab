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

## Common Workflows

### 1. Motion Data Pipeline: Raw Human Data -> Keypoints -> Retarget -> mjlab Motion

For motion-tracking tasks, the final artifact used by mjlab is a local
`motion.npz` file. The repository includes a full pipeline for converting raw
AMASS-style `.npz` files into that final tracking asset.

If you keep ProtoMotions at `./.external/ProtoMotions`, `report-and-preview-motion`
will find it automatically. Otherwise set `PROTOMOTIONS_ROOT=/abs/path/to/ProtoMotions`.
If PyRoki lives in a separate Python environment, set
`PYROKI_PYTHON=/abs/path/to/python`.

#### Recommended: one command for report + build + optional preview

```bash
uv run report-and-preview-motion \
  --input-file /abs/path/raw_motion.npz \
  --motion-name handstand1 \
  --build true \
  --preview true
```

By default this writes to:

```text
artifacts/motion_reports/handstand1/
  report.json
  pipeline/
    manifest.json
    raw/
    keypoints/
    retarget/
    mjlab/
```

The important outputs are:

- `artifacts/motion_reports/handstand1/report.json`
  Summary of the raw file and the requested actions.
- `artifacts/motion_reports/handstand1/pipeline/keypoints/<raw_stem>_keypoints.npy`
  ProtoMotions keypoints extracted from the raw human motion.
- `artifacts/motion_reports/handstand1/pipeline/retarget/handstand1_g1_retargeted.npz`
  Retargeted Unitree G1 trajectory in PyRoki-style `.npz` format.
- `artifacts/motion_reports/handstand1/pipeline/mjlab/handstand1.csv`
  CSV in the mjlab tracking layout.
- `artifacts/motion_reports/handstand1/pipeline/mjlab/motion.npz`
  Final mjlab tracking asset to pass to training, play, or evaluation.
- `artifacts/motion_reports/handstand1/pipeline/manifest.json`
  End-to-end manifest with all generated artifact paths.

#### Manual pipeline, step by step

1. Extract ProtoMotions keypoints from raw human data:

```bash
uv run raw-human-npz-to-keypoints \
  --input-file /abs/path/raw_motion.npz \
  --output-dir artifacts/keypoints/handstand1 \
  --protomotions-root /abs/path/to/ProtoMotions
```

This writes:

- `artifacts/keypoints/handstand1/<raw_stem>_keypoints.npy`
- `artifacts/keypoints/handstand1/<raw_stem>_manifest.json`
- `artifacts/keypoints/handstand1/<raw_stem>.motion`

2. Retarget SMPL keypoints to Unitree G1:

```bash
uv run smpl-keypoints-to-g1-npz \
  --input-file artifacts/keypoints/handstand1/<raw_stem>_keypoints.npy \
  --output-file artifacts/retarget/handstand1_g1_retargeted.npz \
  --protomotions-root /abs/path/to/ProtoMotions \
  --pyroki-python /abs/path/to/python
```

This writes:

- `artifacts/retarget/handstand1_g1_retargeted.npz`
- `artifacts/retarget/handstand1_g1_retargeted_manifest.json`

3. Convert the retargeted `.npz` into mjlab CSV:

```bash
uv run pyroki-npz-to-csv \
  --input-file artifacts/retarget/handstand1_g1_retargeted.npz \
  --output-file artifacts/mjlab_motion/handstand1.csv
```

4. Convert the CSV into the final mjlab `motion.npz`:

```bash
uv run -m mjlab.scripts.csv_to_npz \
  --input-file artifacts/mjlab_motion/handstand1.csv \
  --output-name handstand1 \
  --input-fps 30 \
  --output-fps 50 \
  --output-file artifacts/mjlab_motion/motion.npz
```

If you want the full pipeline but do not need the extra report/preview wrapper,
use `build-tracking-motion` directly:

```bash
uv run build-tracking-motion \
  --input-file /abs/path/raw_motion.npz \
  --motion-name handstand1 \
  --work-dir artifacts/motion_builds/handstand1 \
  --protomotions-root /abs/path/to/ProtoMotions
```

This creates:

```text
artifacts/motion_builds/handstand1/
  manifest.json
  raw/
  keypoints/
  retarget/
  mjlab/
```

### 2. Training a Policy

All training entry points go through the same command:

```bash
uv run train <TASK_ID> [--backend rsl_rl|flashsac|reference_ppo]
```

Training logs are written under:

- `logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/...`
- `logs/flashsac/<experiment_name>/<timestamp>_<run_name>/...`
- `logs/reference_ppo/<experiment_name>/<timestamp>_<run_name>/...`

Typical contents include:

- `params/env.yaml`
- `params/agent.yaml`
- `params/runtime.yaml`
- `summary/metrics.json`
- checkpoints such as `model_*.pt`, `step_*`, or `agent_state.pt`
- `videos/train/` when training-time video recording is enabled

#### Velocity tracking

Default backend (`rsl_rl`):

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096
```

FlashSAC:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --backend flashsac \
  --env.scene.num-envs 1024
```

Reference PPO:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --backend reference_ppo \
  --env.scene.num-envs 4096
```

`reference_ppo` currently ships as a training/debug backend. Its runs are
saved under `logs/reference_ppo/...`; inspect `summary/metrics.json`,
`params/runtime.yaml`, and saved checkpoints there after training.

Multi-GPU training is available on the default `rsl_rl` path:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --gpu-ids "[0, 1]" \
  --env.scene.num-envs 4096
```

See the [Distributed Training guide](https://mujocolab.github.io/mjlab/main/source/training/distributed_training.html) for details.

#### Motion imitation / tracking

Tracking tasks require a motion input. You must provide either:

- `--registry-name your-org/motions/motion-name`
- `--env.commands.motion.motion-file /abs/path/to/motion.npz`

Examples:

Default backend (`rsl_rl`):

```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file /abs/path/to/motion.npz \
  --env.scene.num-envs 4096
```

FlashSAC:

```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 \
  --backend flashsac \
  --env.commands.motion.motion-file /abs/path/to/motion.npz \
  --env.scene.num-envs 1024
```

Reference PPO:

```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 \
  --backend reference_ppo \
  --env.commands.motion.motion-file /abs/path/to/motion.npz \
  --env.scene.num-envs 4096
```

### 3. Playing Back a Policy

`play` currently supports the default `rsl_rl` path and the `flashsac` path.

Play the latest checkpoint from a Weights & Biases run:

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 \
  --wandb-run-path your-org/mjlab/run-id
```

Play a local checkpoint file:

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint-file /abs/path/to/model_4000.pt
```

Play a FlashSAC checkpoint:

```bash
uv run play Mjlab-Tracking-Flat-Unitree-G1 \
  --backend flashsac \
  --checkpoint-file /abs/path/to/step_1000 \
  --motion-file /abs/path/to/motion.npz \
  --viewer viser
```

When `--video true` is used, playback videos are written under:

- `logs/rsl_rl/.../videos/play/`
- `logs/flashsac/.../videos/play/`

### 4. Evaluating a Tracking Policy

Use `evaluate-tracking` for tracking metrics such as `success_rate`, `mpkpe`,
`r_mpkpe`, `joint_vel_error`, `ee_pos_error`, and `ee_ori_error`.

Default `rsl_rl` path:

```bash
uv run evaluate-tracking Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint-file /abs/path/to/model_31500.pt \
  --motion-file /abs/path/to/motion.npz \
  --output-file artifacts/eval/rsl_rl_eval.json
```

FlashSAC path:

```bash
uv run evaluate-tracking Mjlab-Tracking-Flat-Unitree-G1 \
  --backend flashsac \
  --checkpoint-file /abs/path/to/step_1000 \
  --motion-file /abs/path/to/motion.npz \
  --output-file artifacts/eval/flashsac_eval.json
```

`evaluate-tracking` is documented here for the default `rsl_rl` path and the
explicit `flashsac` path. The `reference_ppo` backend currently exposes its
training outputs under `logs/reference_ppo/...`, but does not yet have a
parallel README-documented `play`/`evaluate-tracking` workflow.

### 5. Sanity-check with Dummy Agents

Use built-in agents to sanity check your MDP before training:

```bash
uv run play Mjlab-Your-Task-Id --agent zero
uv run play Mjlab-Your-Task-Id --agent random
```

For motion-tracking tasks, also pass a local motion file or registry artifact:

```bash
uv run play Mjlab-Tracking-Flat-Unitree-G1 \
  --agent zero \
  --motion-file /abs/path/to/motion.npz
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
