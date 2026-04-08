# G1-New: mode\_15-Aligned MJCF Asset — Technical Reference

> **Asset identity:** `g1_29dof_mode_15_aligned`
> **Scope:** Training-oriented MuJoCo asset pair (`g1_new.xml` + `g1_constants_new.py`)
> **Alignment target:** `g1_29dof_mode_15.urdf` (torso inertials, wrist inertials, wrist meshes, actuator envelopes)
> **Design contract:** Preserves the original MuJoCo training structure; is **not** a byte-for-byte URDF mirror.

---

## Table of Contents

- [G1-New: mode\_15-Aligned MJCF Asset — Technical Reference](#g1-new-mode_15-aligned-mjcf-asset--technical-reference)
  - [Table of Contents](#table-of-contents)
  - [1. Overview](#1-overview)
  - [2. File Layout and Runtime Dependencies](#2-file-layout-and-runtime-dependencies)
    - [Canonical Asset Paths](#canonical-asset-paths)
  - [3. Quick Migration Guide](#3-quick-migration-guide)
    - [3.1 CLI Commands](#31-cli-commands)
    - [3.2 Python Import Paths](#32-python-import-paths)
    - [3.3 Motion Preprocessing — `csv_to_npz.py`](#33-motion-preprocessing--csv_to_npzpy)
  - [4. MJCF Changes — `g1_new.xml`](#4-mjcf-changes--g1_newxml)
    - [4.1 Structural Overview](#41-structural-overview)
    - [4.2 Wrist Mesh Substitution](#42-wrist-mesh-substitution)
    - [4.3 Inertial Parameter Updates](#43-inertial-parameter-updates)
      - [Torso Link](#torso-link)
      - [Left Wrist Pitch Link](#left-wrist-pitch-link)
      - [Left Wrist Yaw Link](#left-wrist-yaw-link)
      - [Right Wrist Pitch Link](#right-wrist-pitch-link)
      - [Right Wrist Yaw Link](#right-wrist-yaw-link)
    - [4.4 Elements Intentionally Preserved](#44-elements-intentionally-preserved)
  - [5. Constants Changes — `g1_constants_new.py`](#5-constants-changes--g1_constants_newpy)
    - [5.1 Actuator Group Assignments](#51-actuator-group-assignments)
    - [5.2 Joint Effort / Velocity Envelope Comparison](#52-joint-effort--velocity-envelope-comparison)
    - [5.3 Unchanged Constants](#53-unchanged-constants)
  - [6. Alignment Coverage Assessment](#6-alignment-coverage-assessment)
    - [6.1 Fully Aligned Items](#61-fully-aligned-items)
    - [6.2 Known Approximations](#62-known-approximations)
    - [6.3 Out-of-Scope Items](#63-out-of-scope-items)
  - [7. Asset Selection Decision Guide](#7-asset-selection-decision-guide)

---

## 1. Overview

This document describes the rationale, scope, and technical details of the
`g1_new.xml` / `g1_constants_new.py` asset pair, introduced to track the
upstream `g1_29dof_mode_15.urdf` revision without disturbing the original
`g1.xml` / `g1_constants.py` training asset.

The primary motivation for a **separate, isolated asset** (rather than an
in-place update) is backward compatibility: existing checkpoints trained against
the original asset must remain reproducible without modification.

The new asset achieves the following alignment objectives relative to `mode_15`:

- Updated torso link inertial parameters (CoM position, orientation, mass, diagonal inertia tensor)
- Updated wrist pitch / yaw link inertial parameters (both left and right)
- Substituted `_5010`-series wrist visual meshes for all six wrist links
- Updated wrist-yaw joint origin offsets
- Updated actuator effort / velocity envelopes for `hip_pitch`, `wrist_pitch`, and `wrist_yaw`

The following elements are **intentionally not mirrored** from the URDF, as they
are MuJoCo-specific training constructs with no direct URDF equivalent:

- Foot and torso simplified collision geometry
- IMU site definitions used by the reward and observation pipeline
- Contact exclusion pairs tuned for stable MuJoCo simulation

---

## 2. File Layout and Runtime Dependencies

### Canonical Asset Paths

| Role | Path |
|---|---|
| Original MJCF | `src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml` |
| New MJCF | `src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1_new.xml` |
| Original constants | `src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py` |
| New constants | `src/mjlab/asset_zoo/robots/unitree_g1/g1_constants_new.py` |
| `_5010` wrist meshes | `src/mjlab/asset_zoo/robots/unitree_g1/xmls/assets/` |

At runtime, the project depends only on `g1_new.xml`, `g1_constants_new.py`, and
the six `_5010` mesh files listed in [Section 4.2](#42-wrist-mesh-substitution).
The temporary URDF import folder used during the extraction of `mode_15` values
is not a runtime dependency and may be removed.

---

## 3. Quick Migration Guide

### 3.1 CLI Commands

For all registered task commands — `train`, `play`, `replay-motion`, and
`csv_to_npz` — migration requires only a task ID substitution:

```
# Before
*-Unitree-G1

# After
*-Unitree-G1-New
```

No other CLI arguments require modification for these commands.

### 3.2 Python Import Paths

Direct module imports must be updated when a script bypasses the registered task
ID system and references config modules or the constants file explicitly:

| Original import target | Replacement |
|---|---|
| `mjlab.tasks.tracking.config.g1*` | `mjlab.tasks.tracking.config.g1_new*` |
| `mjlab.tasks.velocity.config.g1*` | `mjlab.tasks.velocity.config.g1_new*` |
| `g1_constants.py` (direct import) | `g1_constants_new.py` |

Scripts that access the robot only through registered task IDs do **not** require
code changes.

### 3.3 Motion Preprocessing — `csv_to_npz.py`

The 29-DoF joint tree, joint names, and joint order are identical between
`g1.xml` and `g1_new.xml`. Consequently, for lower-body motion conversion
workflows, the original `G1` config would introduce only a minor consistency gap,
not a structural incompatibility.

To maintain full consistency with the new asset, pass the `--robot-variant` flag:

```bash
# Add this flag; keep all other arguments unchanged
--robot-variant g1_new
```

`csv_to_npz.py` supports both variants through this switch.

---

## 4. MJCF Changes — `g1_new.xml`

### 4.1 Structural Overview

| Element | `g1.xml` | `g1_new.xml` | Rationale |
|---|---|---|---|
| Model name | `g1_29dof_rev_1_0` | `g1_29dof_mode_15_aligned` | Explicit asset isolation |
| Joint tree / moving DoF | 29-DoF | 29-DoF (unchanged) | No robot topology change in `mode_15` |
| Joint names | Original | Unchanged | Preserves compatibility with motion files, actions, and rewards |
| Foot collision capsules | Present | Unchanged | MuJoCo training simplification; intentionally retained |
| Torso / head collision geoms | Simplified | Unchanged | MuJoCo training simplification; intentionally retained |
| Wrist visual meshes | Original series | `_5010` series | Aligns `mode_15` wrist geometry |
| Torso inertial parameters | Original values | Updated | Matches `mode_15` torso inertial |
| Wrist pitch inertial parameters | Original values | Updated | Matches `mode_15` wrist inertial |
| Wrist yaw origin offset + inertials | Original values | Updated | Matches `mode_15` wrist-yaw geometry and inertial |

### 4.2 Wrist Mesh Substitution

All six wrist visual meshes have been replaced with the `_5010`-series variants.
The replacement files are present at
`src/mjlab/asset_zoo/robots/unitree_g1/xmls/assets/`.

| Link | Original Mesh | Replacement Mesh |
|---|---|---|
| `left_wrist_roll_link` | `left_wrist_roll_link.STL` | `left_wrist_roll_link_5010.STL` |
| `left_wrist_pitch_link` | `left_wrist_pitch_link.STL` | `left_wrist_pitch_link_5010.STL` |
| `left_wrist_yaw_link` | `left_wrist_yaw_link.STL` | `left_wrist_yaw_link_5010.STL` |
| `right_wrist_roll_link` | `right_wrist_roll_link.STL` | `right_wrist_roll_link_5010.STL` |
| `right_wrist_pitch_link` | `right_wrist_pitch_link.STL` | `right_wrist_pitch_link_5010.STL` |
| `right_wrist_yaw_link` | `right_wrist_yaw_link.STL` | `right_wrist_yaw_link_5010.STL` |

### 4.3 Inertial Parameter Updates

The tables below list every numerical field that was modified. All changes derive
directly from the `g1_29dof_mode_15.urdf` reference.

#### Torso Link

| Field | `g1.xml` | `g1_new.xml` |
|---|---|---|
| `inertial.pos` | `0.00203158 0.000339683 0.184568` | `0.000931 0.000346 0.15082` |
| `inertial.quat` | `0.999803 -6.03319e-05 0.0198256 0.00131986` | `0.72554 0.000699461 -0.688178 0.00132538` |
| `inertial.mass` | `7.818` | `6.78` |
| `inertial.diaginertia` | `0.121847 0.109825 0.0273735` | `0.0255583 0.0470139 0.0591438` |

#### Left Wrist Pitch Link

| Field | `g1.xml` | `g1_new.xml` |
|---|---|---|
| `inertial.pos` | `0.0229999 -0.00111685 -0.00111658` | `0.0254915 -0.000540425 -0.000541439` |
| `inertial.quat` | `0.249998 0.661363 0.293036 0.643608` | `0.724353 -0.689303 -0.0131972 0.000353948` |
| `inertial.mass` | `0.48405` | `0.684` |
| `inertial.diaginertia` | `0.000430353 0.000429873 0.000164648` | `0.000255669 0.000716079 0.000716495` |

#### Left Wrist Yaw Link

| Field | `g1.xml` | `g1_new.xml` |
|---|---|---|
| `joint.pos` (origin offset) | `0.046 0 0` | `0.051 0 0` |
| `inertial.pos` | `0.0708244 0.000191745 0.00161742` | `0.0220038 0.000494851 0.000538611` |
| `inertial.quat` | `0.510571 0.526295 0.468078 0.493188` | `0.414631 -0.387242 -0.586226 -0.578329` |
| `inertial.mass` | `0.254576` | `0.0845765` |
| `inertial.diaginertia` | `0.000646113 0.000559993 0.000147566` | `3.75684e-05 5.09807e-05 5.97564e-05` |

#### Right Wrist Pitch Link

| Field | `g1.xml` | `g1_new.xml` |
|---|---|---|
| `inertial.pos` | `0.0229999 0.00111685 -0.00111658` | `0.0254915 0.000540425 -0.000541439` |
| `inertial.quat` | `0.643608 0.293036 0.661363 0.249998` | `0.724353 -0.689303 0.0131972 -0.000353948` |
| `inertial.mass` | `0.48405` | `0.684` |
| `inertial.diaginertia` | `0.000430353 0.000429873 0.000164648` | `0.000255669 0.000716079 0.000716495` |

#### Right Wrist Yaw Link

| Field | `g1.xml` | `g1_new.xml` |
|---|---|---|
| `joint.pos` (origin offset) | `0.046 0 0` | `0.051 0 0` |
| `inertial.pos` | `0.0708244 -0.000191745 0.00161742` | `0.0220038 -0.000494851 0.000538611` |
| `inertial.quat` | `0.493188 0.468078 0.526295 0.510571` | `0.578329 0.586226 0.387242 -0.414631` |
| `inertial.mass` | `0.254576` | `0.0845765` |
| `inertial.diaginertia` | `0.000646113 0.000559993 0.000147566` | `3.75684e-05 5.09807e-05 5.97564e-05` |

### 4.4 Elements Intentionally Preserved

The following MuJoCo-specific constructs are **not** derived from the URDF and
were therefore not modified:

| Element | Reason for Preservation |
|---|---|
| Foot collision capsule layout | Simulation-stable simplification integrated with the contact reward |
| Torso / head simplified collision geoms | Simulation-stable simplification; no equivalent in the URDF collision mesh |
| IMU site definitions | Referenced directly by observation and reward modules |
| Contact exclusion pairs | Tuned for MuJoCo solver stability during training |

`g1_new.xml` is therefore not a raw URDF-to-MJCF conversion. It is a
`mode_15`-aligned MJCF that retains the full original training-specific
MuJoCo structure.

---

## 5. Constants Changes — `g1_constants_new.py`

### 5.1 Actuator Group Assignments

| Parameter | `g1_constants.py` | `g1_constants_new.py` | Notes |
|---|---|---|---|
| Loaded XML | `g1.xml` | `g1_new.xml` | Full asset isolation |
| `hip_pitch` actuator group | `7520_14` | `7520_22` | Corrects to `139 Nm / 20 rad/s` envelope |
| `hip_roll` actuator group | `7520_22` | `7520_22` | No change required |
| `hip_yaw` actuator group | `7520_14` | `7520_14` | No change required |
| `waist_yaw` actuator group | `7520_14` | `7520_14` | No change required |
| `wrist_pitch` / `wrist_yaw` group | `4010` | Dedicated `mode_15` wrist group | Corrects to `13.4 Nm / 27 rad/s` |
| `waist` / `ankle` envelope | `5020 × 2`, 50 Nm nominal | Doubled-armature approx., 35 Nm | Effort limit aligned; armature remains approximate |

### 5.2 Joint Effort / Velocity Envelope Comparison

| Joint / Group | `g1_constants.py` | `g1_constants_new.py` | `mode_15` Target | Alignment Status |
|---|---|---|---|---|
| `hip_pitch` | 88 Nm / 32 rad·s⁻¹ | 139 Nm / 20 rad·s⁻¹ | 139 Nm / 20 rad·s⁻¹ | ✅ Aligned |
| `hip_roll` | 139 Nm / 20 rad·s⁻¹ | 139 Nm / 20 rad·s⁻¹ | 139 Nm / 20 rad·s⁻¹ | ✅ Previously aligned |
| `hip_yaw` | 88 Nm / 32 rad·s⁻¹ | 88 Nm / 32 rad·s⁻¹ | 88 Nm / 32 rad·s⁻¹ | ✅ Previously aligned |
| `waist_yaw` | 88 Nm / 32 rad·s⁻¹ | 88 Nm / 32 rad·s⁻¹ | 88 Nm / 32 rad·s⁻¹ | ✅ Previously aligned |
| `wrist_roll` | 25 Nm / 37 rad·s⁻¹ | 25 Nm / 37 rad·s⁻¹ | 25 Nm / 37 rad·s⁻¹ | ✅ Previously aligned |
| `wrist_pitch` | 5 Nm / 22 rad·s⁻¹ | 13.4 Nm / 27 rad·s⁻¹ | 13.4 Nm / 27 rad·s⁻¹ | ✅ Aligned |
| `wrist_yaw` | 5 Nm / 22 rad·s⁻¹ | 13.4 Nm / 27 rad·s⁻¹ | 13.4 Nm / 27 rad·s⁻¹ | ✅ Aligned |
| `waist_pitch` | 50 Nm / 37 rad·s⁻¹ (nominal) | 35 Nm / 30 rad·s⁻¹ | 35 Nm / 30 rad·s⁻¹ | ⚠️ Effort limit aligned; armature approximate |
| `waist_roll` | 50 Nm / 37 rad·s⁻¹ (nominal) | 35 Nm / 30 rad·s⁻¹ | 35 Nm / 30 rad·s⁻¹ | ⚠️ Effort limit aligned; armature approximate |
| `ankle_pitch` | 50 Nm / 37 rad·s⁻¹ (nominal) | 35 Nm / 30 rad·s⁻¹ | 35 Nm / 30 rad·s⁻¹ | ⚠️ Effort limit aligned; armature approximate |
| `ankle_roll` | 50 Nm / 37 rad·s⁻¹ (nominal) | 35 Nm / 30 rad·s⁻¹ | 35 Nm / 30 rad·s⁻¹ | ⚠️ Effort limit aligned; armature approximate |

### 5.3 Unchanged Constants

The following parameters are carried forward unchanged from `g1_constants.py`,
as they represent local simulation design choices rather than values derivable
from the URDF:

- Rotor inertia assumptions for actuator groups `5020`, `7520_14`, `7520_22`, `4010`
- Stiffness and damping generation (10 Hz natural frequency, nominal damping ratio)
- Keyframe presets
- Collision configuration presets

---

## 6. Alignment Coverage Assessment

### 6.1 Fully Aligned Items

`g1_constants_new.py` achieves full numerical alignment with `mode_15` on all
joint groups that govern control authority and policy action scaling:

- All hip-group effort / velocity envelopes
- All waist / ankle effort / velocity envelopes
- All wrist effort / velocity envelopes

`g1_new.xml` achieves full alignment on:

- All wrist visual meshes (`_5010` series)
- Torso link inertial parameters
- Left and right wrist pitch / yaw inertial parameters
- Left and right wrist-yaw joint origin offsets

### 6.2 Known Approximations

The following items remain approximate and are documented here for traceability:

| Item | Current State | Root Cause |
|---|---|---|
| `5010` wrist rotor (reflected) inertia | Old armature value reused | `5010` hardware inertia not published by Unitree |
| Waist / ankle 4-bar mechanism armature | Doubled-`5020` nominal approximation | True 4-bar linkage geometry unavailable |
| Keyframe and collision presets | Inherited from original asset | Not URDF-derivable; local simulation design choices |

`g1_constants_new.py` is therefore correctly characterized as
**control-envelope aligned** but **not fully hardware-parameter exact**.

### 6.3 Out-of-Scope Items

The following `mode_15` URDF elements are outside the current alignment scope
and have not been incorporated:

- Camera sensor fixed frames (`d435`, `mid360`)
- Per-auxiliary-link sensor frame mirrors on the MuJoCo side
- Raw-mesh collision geometry conversion for all visual / collision links
- Exact `5010` reflected inertia values (pending Unitree disclosure)

These items are candidates for a future update if the deployment or perception
stack requires them.

---

## 7. Asset Selection Decision Guide

| Use Case | Recommended Asset |
|---|---|
| Train or replay with `mode_15`-aligned geometry and envelopes | `g1_new.xml` + `g1_constants_new.py` + `*-G1-New` task IDs |
| Reproduce results from checkpoints trained on the original asset | `g1.xml` + `g1_constants.py` + `*-G1` task IDs |
| Prevent any behavior drift relative to prior experiments | Original asset (no migration) |

For the majority of CLI workflows, the complete migration is a single token
substitution: replace every occurrence of `Unitree-G1` with `Unitree-G1-New`.
For direct Python imports, additionally update the module paths and constants
file reference as described in [Section 3.2](#32-python-import-paths).

