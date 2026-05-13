# Z2 Tracking Integration Notes

This document records the main changes made when integrating the Nubot Z2 model
into the tracking pipeline, along with the current recommended training and
evaluation workflow.

## Scope

The Z2 integration adds two task IDs:

- `Mjlab-Tracking-Flat-Z2`
- `Mjlab-Tracking-Rough-Z2`

These tasks are intended to mirror the existing G1/G1-New tracking workflow as
closely as practical while adapting the robot asset, joint naming, contact
model, and preprocessing pipeline to Z2.

## Main changes

### 1. New Z2 task registration

Added:

- `src/mjlab/tasks/tracking/config/z2/__init__.py`
- `src/mjlab/tasks/tracking/config/z2/env_cfgs.py`
- `src/mjlab/tasks/tracking/config/z2/rl_cfg.py`

Registered tasks:

- `Mjlab-Tracking-Flat-Z2`
- `Mjlab-Tracking-Rough-Z2`

RL experiment names:

- flat: `z2_tracking`
- rough: `z2_tracking_rough`

### 2. Z2 robot config and asset wrapper

Added:

- `src/mjlab/asset_zoo/robots/nubot_z2/z2_constants.py`

This file is the Z2-specific robot wrapper used by the new tasks. It handles:

- loading `src/mjlab/asset_zoo/robots/nubot_z2/xmls/assembly.xml`
- loading mesh assets recursively
- ensuring the model uses a floating base (`freejoint`)
- adding built-in IMU sensors expected by tracking observations:
  - `robot/imu_ang_vel`
  - `robot/imu_lin_vel`
- defining tracking body names and flat end-effector body names
- defining the Z2 action scale map

### 3. Tracking actuator strategy

Current strategy:

- Use the **G1-style built-in position actuator / PD parameterization**
- Control only the **29 body joints used by tracking**
- Do **not** include finger joints in the action space

Current controlled joint families:

- shoulder pitch / roll / yaw
- elbow
- wrist roll
- wrist pitch / yaw
- waist yaw / pitch / roll
- hip yaw / pitch / roll
- knee
- ankle pitch / roll

The `joint_pos` action space remains 29-dimensional, while the robot still keeps
the full 51-joint state needed by the motion command / reward pipeline.

### 4. Z2 CSV -> NPZ preprocessing

Modified:

- `src/mjlab/scripts/csv_to_npz_z2.py`

This script is now Z2-specific rather than a G1 copy.

Current assumptions:

- input CSV has **58 columns**
- layout is:
  - 7 root columns
  - 51 joint columns
- the 51-DoF order matches the real Z2 model `joint_names`

The script:

- validates the Z2 column count
- uses the Z2 flat tracking env for replay/conversion
- uses Z2 foot geom patterns for grounding analysis
- outputs `.npz` with:
  - `joint_pos` shape `(T, 51)`
  - `joint_vel` shape `(T, 51)`

### 5. Foot collision model replacement

The original Z2 MJCF used the entire ankle-roll mesh as the foot contact geom.
That caused two problems:

- replay-time foot-bottom analysis could not handle mesh geoms cleanly
- training/replay could start with the robot appearing to "pop out of the ground"

Current strategy:

- keep the ankle mesh as visual-only
- disable the original mesh foot collision
- add **3 capsule sole geoms per foot**

Current foot contact regex:

- `^(left|right)_foot_contact_\\d+$`

Current foot sole capsules are defined in:

- `src/mjlab/asset_zoo/robots/nubot_z2/z2_constants.py`

These were first fitted against the Z2 ankle-roll mesh footprint and later
manually re-tuned after direct MuJoCo viewer inspection using
`src/mjlab/asset_zoo/robots/nubot_z2/xmls/scene_z2.xml`.

Latest manually synced capsule parameters:

- radius: `0.0065`
- z height: `-0.035`
- side rows:
  - start `(-0.03, -0.03, -0.035)` end `(0.11, -0.03, -0.035)`
- center row:
  - start `(-0.06, 0.0, -0.035)` end `(0.14, 0.0, -0.035)`
- side rows:
  - start `(-0.03, 0.03, -0.035)` end `(0.11, 0.03, -0.035)`

If `scene_z2.xml` is manually edited again, these values should be synchronized
back into `_Z2_FOOT_CAPSULE_RADIUS`, `_Z2_FOOT_CAPSULE_Z`, and
`_Z2_FOOT_CAPSULES` in `z2_constants.py`.

### 6. Collision simplification for performance

Earlier versions enabled broad mesh collision on many Z2 bodies, which caused:

- high contact counts
- `contact_sensor_maxmatch` overflow
- `ccd_iterations` warnings
- very slow iterations (several seconds per PPO iteration)

Current strategy:

- use **feet-only collision** during tracking
- active collision geoms are just the 6 foot sole capsules

This is defined by:

- `Z2_FEET_ONLY_COLLISION`

### 7. Replay support fixes

Modified:

- `src/mjlab/scripts/replay_motion.py`

Changes:

- auto-detect Z2 task IDs and use Z2 foot geom regex
- avoid crashing if foot-bottom display encounters unsupported geom types

## Important current behavior

### Action space vs. state space

These are intentionally different:

- action space: 29
- full joint state in motion / reward / observations: 51

This is expected and required because the Z2 source motions contain finger
joints, but the tracking controller only actuates the body joints.

### Current foot initialization check

The current Z2 flat play-time initialization was checked numerically after the
latest foot-capsule adjustment.

Representative result:

- `play-mode min foot z ≈ 0.0027`

This is close to the G1 initialization behavior and avoids obvious buried-foot
starts.

## Training

### Flat tracking

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Flat-Z2 \
  --env.commands.motion.motion-file /home/dp/czy/mjlab/datasets/npz/jump_forward02_poses_z2lite.npz \
  --env.scene.num-envs 4096
```

Logs/checkpoints go to:

```text
logs/rsl_rl/z2_tracking/<timestamp>/
```

### Rough tracking

Start rough from a flat checkpoint:

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Rough-Z2 \
  --env.commands.motion.motion-file /home/dp/czy/mjlab/datasets/npz/jump_forward02_poses_z2lite.npz \
  --checkpoint-file /home/dp/czy/mjlab/logs/rsl_rl/z2_tracking/<flat_run>/model_<iter>.pt
```

Rough logs/checkpoints go to:

```text
logs/rsl_rl/z2_tracking_rough/<timestamp>/
```

## Motion inspection

### Replay the NPZ directly

```bash
uv run replay-motion Mjlab-Tracking-Flat-Z2 \
  --motion-file /home/dp/czy/mjlab/datasets/npz/jump_forward02_poses_z2lite.npz
```

### Replay in rough terrain

```bash
uv run replay-motion Mjlab-Tracking-Rough-Z2 \
  --motion-file /home/dp/czy/mjlab/datasets/npz/jump_forward02_poses_z2lite.npz
```

## Policy evaluation

### Visualize a trained flat policy

```bash
MUJOCO_GL=egl uv run play Mjlab-Tracking-Flat-Z2 \
  --checkpoint-file /home/dp/czy/mjlab/logs/rsl_rl/z2_tracking/<run>/model_<iter>.pt \
  --motion-file /home/dp/czy/mjlab/datasets/npz/jump_forward02_poses_z2lite.npz
```

### Visualize a trained rough policy

```bash
MUJOCO_GL=egl uv run play Mjlab-Tracking-Rough-Z2 \
  --checkpoint-file /home/dp/czy/mjlab/logs/rsl_rl/z2_tracking_rough/<run>/model_<iter>.pt \
  --motion-file /home/dp/czy/mjlab/datasets/npz/jump_forward02_poses_z2lite.npz
```

### Sanity-check the environment with zero actions

```bash
uv run play Mjlab-Tracking-Flat-Z2 \
  --agent zero \
  --motion-file /home/dp/czy/mjlab/datasets/npz/jump_forward02_poses_z2lite.npz
```

## Practical debugging checklist

If training behaves oddly, check in this order:

1. Replay the `.npz` directly with `replay-motion`
2. Check the first-frame foot height against the ground
3. Run `play --agent zero` to see raw environment behavior
4. Compare flat-policy playback against direct motion replay
5. Inspect whether reward is dominated by:
   - `joint_limit`
   - `self_collisions`
   - weak body/orientation tracking

## Current caveats

- The Z2 body-joint controller currently uses the G1-style built-in PD/actuator
  parameterization rather than the full original XML actuator stack.
- Finger joints remain in state/motion tensors but are not actuated.
- The current foot sole capsule geometry is a simplified approximation of the
  real Z2 sole, not a full per-link collision reconstruction.

## Files touched during Z2 integration

- `src/mjlab/asset_zoo/robots/__init__.py`
- `src/mjlab/asset_zoo/robots/nubot_z2/z2_constants.py`
- `src/mjlab/tasks/tracking/config/z2/__init__.py`
- `src/mjlab/tasks/tracking/config/z2/env_cfgs.py`
- `src/mjlab/tasks/tracking/config/z2/rl_cfg.py`
- `src/mjlab/scripts/csv_to_npz_z2.py`
- `src/mjlab/scripts/replay_motion.py`
