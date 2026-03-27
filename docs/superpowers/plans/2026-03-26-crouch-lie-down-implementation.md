# Crouch-To-LieDown Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retarget `data/crouch/crouch to lie down1.npz` toward Unitree G1, convert it into an `mjlab`-compatible motion file, and add a dedicated `mjlab` tracking task variant for training this motion.

**Architecture:** Keep retargeting and training loosely coupled. The repository owns two thin bridges: a script that converts PyRoki-style G1 retarget output into `mjlab` CSV, and a dedicated `mjlab` tracking task variant tuned for this non-cyclic crouch-to-lie-down motion. External retargeting remains in a separate environment, but the repo tracks the workflow and required inputs.

**Tech Stack:** Python, pytest, MuJoCo/mjlab, PyRoki/ProtoMotions-compatible data bridge

---

## Chunk 1: Plan Tracking

### Task 1: Land and Maintain the Plan File

**Files:**
- Create: `docs/superpowers/plans/2026-03-26-crouch-lie-down-implementation.md`

- [x] **Step 1: Write the implementation plan into the repository**

- [x] **Step 2: Update checklist status as implementation progresses**

## Chunk 2: Repository-Side Data Bridge

### Task 2: Add a PyRoki-to-mjlab CSV Converter

**Files:**
- Create: `src/mjlab/scripts/pyroki_npz_to_csv.py`
- Create: `tests/test_pyroki_npz_to_csv.py`

- [x] **Step 1: Write failing tests for the converter**
  Test the converter on a synthetic retargeted G1 `.npz` containing:
  `base_frame_pos`, `base_frame_wxyz`, and `joint_angles`.
  Verify:
  - output CSV has 36 columns (`3 + 4 + 29`)
  - quaternion is written in `xyzw`
  - rows preserve frame count
  - joint order is unchanged

- [x] **Step 2: Run the converter tests and confirm they fail for the expected missing module/function reason**

- [x] **Step 3: Implement the minimal converter**
  Add a small CLI and reusable function that:
  - loads the PyRoki-style `.npz`
  - validates required keys and shapes
  - converts root quaternion from `wxyz` to `xyzw`
  - writes a comma-separated motion CSV compatible with `mjlab.scripts.csv_to_npz`

- [x] **Step 4: Re-run the converter tests and confirm they pass**

## Chunk 3: Dedicated mjlab Training Task

### Task 3: Add `Mjlab-Tracking-Flat-Unitree-G1-CrouchToLieDown`

**Files:**
- Modify: `src/mjlab/tasks/tracking/config/g1/env_cfgs.py`
- Modify: `src/mjlab/tasks/tracking/config/g1/rl_cfg.py`
- Modify: `src/mjlab/tasks/tracking/config/g1/__init__.py`
- Modify: `tests/test_tracking_task.py`

- [x] **Step 1: Write failing tests for the new tracking task variant**
  Cover:
  - task is registered
  - motion command exists and is `MotionCommandCfg`
  - training and play configs both use `sampling_mode="start"`
  - training config disables RSI perturbations (`pose_range`, `velocity_range`, `joint_position_range`)
  - episode length is set for this one-shot motion

- [x] **Step 2: Run the targeted tracking-task tests and confirm they fail for the expected missing task/config behavior**

- [x] **Step 3: Implement the dedicated env/rl config**
  Add a G1 task variant tuned for crouch-to-lie-down:
  - fixed start-frame sampling
  - no RSI perturbations
  - longer rollout horizon and shorter total training run than the baseline
  - slightly relaxed tracking termination thresholds for terminal lie-down frames
  Keep the default tracking tasks unchanged.

- [x] **Step 4: Re-run the targeted tracking-task tests and confirm they pass**

## Chunk 4: External Retarget Workflow Wrapper

### Task 4: Add a Repository-Tracked Retarget Entry Point

**Files:**
- Create: `src/mjlab/scripts/raw_human_npz_to_smpl_keypoints.py`
- Create: `tests/test_raw_human_npz_to_smpl_keypoints.py`

- [x] **Step 1: Write failing tests for input parsing and CLI validation**
  Cover:
  - raw AMASS-style `.npz` validation (`trans`, `poses`, `betas`, `mocap_framerate`)
  - clear error when required body-model assets are missing
  - metadata is preserved for downstream retargeting

- [x] **Step 2: Run the targeted tests and confirm they fail for missing module/function behavior**

- [x] **Step 3: Implement the minimal bridge**
  Do not embed PyRoki or SMPL assets in this repo. Add a script that:
  - validates raw human motion input
  - requires explicit model/asset paths from CLI
  - emits a focused, PyRoki-oriented keypoint export for the single-motion workflow
  - fails with actionable messages if optional body-model dependencies are absent

- [x] **Step 4: Re-run the targeted tests and confirm they pass**

## Chunk 5: End-to-End Verification

### Task 5: Verify the Local Repository Integration

**Files:**
- Update: `docs/superpowers/plans/2026-03-26-crouch-lie-down-implementation.md`

- [x] **Step 1: Generate a synthetic or real PyRoki-style `.npz` and convert it to CSV**

- [x] **Step 2: Run `mjlab.scripts.csv_to_npz` on the produced CSV and confirm it creates a training motion artifact**

- [x] **Step 3: Run repository tests**
  Run at least:
  - `tests/test_pyroki_npz_to_csv.py`
  - `tests/test_raw_human_npz_to_smpl_keypoints.py`
  - `tests/test_tracking_task.py`

- [x] **Step 4: Update this plan file with the final verification status and any remaining external blockers**

## Verification Status

- `tests/test_pyroki_npz_to_csv.py`: passing
- `tests/test_raw_human_npz_to_smpl_keypoints.py`: passing
- `tests/test_tracking_task.py`: passing
- Combined targeted verification: `29 passed`
- Synthetic bridge verification completed:
  - synthetic PyRoki-style `.npz` -> `pyroki_npz_to_csv.py` -> CSV succeeded
  - `csv_to_npz.py` succeeded with a local `wandb` stub and writable temporary `HOME`
  - generated local training artifact: `/tmp/motion.npz`
- Real source bridge verification completed:
  - real `data/crouch/crouch to lie down1.npz` -> ProtoMotions `.motion` succeeded
  - real `.motion` -> keypoints `.npy` succeeded
  - generated artifacts in `/tmp/crouch_lie_down_real/`
- Real GPU retarget verification completed:
  - real keypoints -> G1 retargeted `.npz` succeeded on GPU
  - generated artifact: `/tmp/crouch_lie_down_retargeted_gpu/crouch_to_lie_down1_keypoints_retargeted.npz`
- Real training-data verification completed:
  - retargeted G1 `.npz` -> `pyroki_npz_to_csv.py` -> CSV succeeded
  - real retargeted CSV -> `csv_to_npz.py` succeeded
  - generated real training artifact: `/tmp/motion.npz`
- GPU smoke training completed:
  - `Mjlab-Tracking-Flat-Unitree-G1-CrouchToLieDown` ran for `1` iteration on `cuda:0`
  - contact-capacity follow-up fix (`cfg.sim.nconmax = 55`) removed the previously observed broadphase overflow warnings in the rerun smoke test

## Remaining External Blockers

- No remaining preprocessing blocker for launching a real training run from `data/crouch/crouch to lie down1.npz`.
- The next step is a longer GPU training run using the generated `/tmp/motion.npz`.
