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
uv run src/mjlab/scripts/sonic2mimic.py \
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
uv run src/mjlab/scripts/sonic2mimic.py \
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
- `--ground-align phased`: use foot height to distinguish near-ground vs airborne phases, keep grounded frames close to the foot-based lift, and allow larger temporary lift during airborne windows to protect whole-body collisions near landing.
- `--clearance 0.01`: set the target minimum foot-bottom clearance to `0.01 m` (1 cm).

Ground-alignment workflow notes:

- `analyze_foot_penetration.py` is diagnostic only. It reports the current worst-case foot-bottom height and the recommended global lift, but it does not modify any files.
- `analyze_foot_penetration.py` can also auto-suggest a coarse phased schedule. By default it first infers grounded-vs-airborne control points from the motion's own nearby foot-height distribution, then roughly segments the motion into grounded / takeoff / airborne / landing windows and prints a reference lift range for each segment, but it still does not modify any files.
- `csv_to_npz.py --ground-align global` is the step that actually changes the output motion. It writes a new `.npz` whose root `z` trajectory has been shifted upward by one constant offset.
- `csv_to_npz.py --ground-align phased` also changes the output motion, but with a frame-varying root `z` offset. It uses foot height as the phase cue: grounded windows stay closer to the foot-based lift, while airborne windows can rise more to keep head/torso/hand collisions from dipping below the plane near landing.
- `replay-motion`, `play`, and training do not run ground alignment again. They simply consume the `.npz` motion file you produced. In practice, this means the training-time reference motion is the aligned `.npz` if you converted with `--ground-align global`, and the raw motion otherwise.
- The optional `.mp4` preview from `csv_to_npz.py --render True` is rendered from the same aligned motion data, so the robot-ground relationship should match `replay-motion` for the same `.npz`. Small visual differences can still appear because the preview video and `replay-motion` use different default cameras.
- `--clearance` only constrains the minimum height of the matched foot collision geoms. It does not guarantee that other body parts such as the head, hands, or torso will stay above the ground plane during flips, rolls, or hard landings.

Practical interpretation:

- Use `--ground-align global` when you want the exported `.npz` to have a controlled foot-ground offset before replay or training.
- Use `--ground-align phased` when a single global lift is too conservative, especially for flips or rolls where the feet are clear of the floor during flight but the head or torso threatens to clip the ground near landing.
- Use `--ground-align none` when you intentionally want to keep the raw source heights.
- If the feet look correct but the head still clips the floor during a landing, switch from `global` to `phased`: `global` only enforces foot clearance, while `phased` also uses whole-body collision minima during airborne windows.

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

This diagnostic command does two things:

- It reports the worst foot and whole-body collision frames, so you can see which geom is actually requesting the lift.
- It also auto-suggests coarse phases and a reference lift range for each phase, using the same logic as `--ground-align phased`.

Automatic vs manual phase control:

- By default, the analysis step is automatic. You do not need to hand-split the motion.
- The analysis script first estimates a grounded foot-height band and an airborne foot-height band from the motion itself, so the suggested phases are not tied to one specific flip or roll clip.
- Near-ground windows stay close to foot-based lift; clearly airborne windows are allowed to follow whole-body-driven lift more strongly.
- Manual editing is only needed if you want to override that default behavior with your own `--phase-blend-points`, or if you already know the exact control points you want to test.
- The analysis script can be used as a first pass for future aerial motions: let it auto-segment the motion, inspect the suggested phases and lift ranges, then decide whether the default automatic schedule is already good enough.

If you want the analysis script to auto-infer the grounded / airborne split and print a reference `suggested_phase_blend_points`, you can keep the command minimal and omit all phased tuning overrides:

```bash
MUJOCO_GL=egl uv run -m mjlab.scripts.analyze_foot_penetration \
  --input-file /home/nubot/workspace/mjlab/datasets/csv/tiger_jump_to_shoulder_roll_R_001__A415_M_mimic.csv \
  --input-fps 120 \
  --output-fps 50 \
  --clearance 0.01
```

In this default form, the script will infer the phase split from the motion itself. It will automatically print:

- `suggested_phase_grounded_height`
- `suggested_phase_airborne_height`
- `suggested_phase_blend_points`
- `csv_to_npz_hint`
- `Suggested coarse phases`, including each phase's `lift(min/mean/max)` range

So the command itself is not manually splitting the motion into stages. The stage suggestion and the recommended phase offsets come from the analysis output.

How to use the analysis output:

- `recommended_global_z_offset_m` under `Foot grounding summary`: this is the constant upward shift you would need if you used `--ground-align global`. If the foot value is already acceptable and the whole body is also safe, `global` may be enough.
- `recommended_global_z_offset_m` under `Whole-body collision grounding summary`: this is the constant upward shift needed to make the worst whole-body collision safe. If this value is much larger than the foot-based value, a single global lift will usually make the whole motion look too high, so `phased` is usually the better next step.
- `suggested_phase_grounded_height` and `suggested_phase_airborne_height`: these are auto-inferred reference heights for the phase cue. They tell you where the script believes the motion is still near-ground and where it is clearly airborne.
- `suggested_phase_blend_points`: this is the most practical field for the next step. Copy it into `csv_to_npz.py` as `--phase-blend-points`.
- `csv_to_npz_hint`: this is the same idea in ready-to-paste CLI form.
- `Suggested coarse phases`: this is the script's rough segmentation of the motion. Use it to sanity-check whether the inferred grounded / takeoff / airborne / landing windows match your intuition.
- `lift(min/mean/max)` inside each suggested phase: this is the recommended frame-wise root `z` lift range for that phase. A large `max` during `airborne` means the landing needs protection from head / torso / hand penetration. Near-zero lift in the final `grounded` phase means the motion can return to a normal stance height after landing.

Recommended next step after running the analysis:

1. Start from the printed `csv_to_npz_hint`, keep the same `--clearance`, and convert the CSV into a new `.npz`.
2. Replay that `.npz` with `replay-motion` and inspect the landing.
3. If the landing still dips too low, increase `--phase-lookahead-s` a little, or make the blend enter airborne mode earlier by lowering the first height in `--phase-blend-points`.
4. If the robot starts floating too early during takeoff or after landing, decrease `--phase-lookahead-s`, or make airborne mode harder to enter by raising the heights in `--phase-blend-points`.
5. Once the replay looks reasonable, use that converted `.npz` for training.

Worked example using the exact output shown above:

```text
Automatic phased-alignment suggestion
  suggested_phase_blend_points: "0.049:0.00,0.593:1.00"
  csv_to_npz_hint: --ground-align phased --phase-blend-points "0.049:0.00,0.593:1.00"

Suggested coarse phases
  phase= 1  type=grounded  ...
  phase= 2  type=takeoff   ...
  phase= 3  type=airborne  ...
  phase= 4  type=landing   ...
  phase= 5  type=grounded  ...
```

Example A: replay the exact inferred control points from the analysis output:

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

That command uses the auto-inferred phase control points from the analysis output. It is the best choice when you want your first conversion to match the analysis result exactly.

Example B: skip the manual copy step and let `csv_to_npz.py` auto-infer phase control points directly:

```bash
MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
  --input-file /home/nubot/workspace/mjlab/datasets/csv/tiger_jump_to_shoulder_roll_R_001__A415_M_mimic.csv \
  --output-name /home/nubot/workspace/mjlab/datasets/npz/tiger_jump_to_shoulder_roll_R_001__A415_M_phased_auto_direct.npz \
  --input-fps 120 \
  --output-fps 50 \
  --ground-align phased \
  --phase-control-mode auto \
  --clearance 0.01 \
  --render True
```

This new `--phase-control-mode auto` mode uses the same motion-driven phase-control inference directly inside `csv_to_npz.py`, so it is easier to migrate to future aerial motions. You can still use `replay-motion` afterward to check the result and then fine-tune if needed.

Shortest end-to-end path for a new aerial motion:

1. Analyze the CSV and inspect the suggested phases:

```bash
MUJOCO_GL=egl uv run -m mjlab.scripts.analyze_foot_penetration \
  --input-file /home/nubot/workspace/mjlab/datasets/csv/tiger_jump_to_shoulder_roll_R_001__A415_M_mimic.csv \
  --input-fps 120 \
  --output-fps 50 \
  --clearance 0.01
```

What to look for:

- If `Whole-body collision recommended_global_z_offset_m` is much larger than the foot-based one, prefer `--ground-align phased` instead of `global`.
- Check whether `Suggested coarse phases` looks reasonable for the motion, especially the `takeoff`, `airborne`, and `landing` windows.

2. Convert the CSV directly into an aligned training `.npz` using auto phase control:

```bash
MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
  --input-file /home/nubot/workspace/mjlab/datasets/csv/tiger_jump_to_shoulder_roll_R_001__A415_M_mimic.csv \
  --output-name /home/nubot/workspace/mjlab/datasets/npz/tiger_jump_to_shoulder_roll_R_001__A415_M_train_auto.npz \
  --input-fps 120 \
  --output-fps 50 \
  --ground-align phased \
  --phase-control-mode auto \
  --clearance 0.01 \
  --render True
```

This creates both:

- `/home/nubot/workspace/mjlab/datasets/npz/tiger_jump_to_shoulder_roll_R_001__A415_M_train_auto.npz`
- `/home/nubot/workspace/mjlab/datasets/npz/tiger_jump_to_shoulder_roll_R_001__A415_M_train_auto.mp4`

3. Replay the converted `.npz` in MuJoCo before training:

```bash
uv run replay-motion Mjlab-Tracking-Flat-Unitree-G1 \
  --motion-file /home/nubot/workspace/mjlab/datasets/npz/tiger_jump_to_shoulder_roll_R_001__A415_M_train_auto.npz \
  --start-paused \
  --reference-viz ghost
```

What to inspect in `replay-motion`:

- Step through the `landing` frames with `Right Arrow`.
- Watch whether the head, torso, and hands still dip into the ground.
- Watch the overlay `min_foot_bottom_z` to confirm the post-conversion foot clearance is behaving as expected.
- If the motion still lands too low, go back to step 2 and tune the phased parameters, or run the analysis script again and use the printed `suggested_phase_blend_points` explicitly.

4. Train with the same `.npz` after the replay looks acceptable:

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file /home/nubot/workspace/mjlab/datasets/npz/tiger_jump_to_shoulder_roll_R_001__A415_M_train_auto.npz \
  --env.scene.num-envs 4096 \
  --agent.logger tensorboard \
  --agent.upload-model False
```

The important rule is: the `.npz` you replay and the `.npz` you train on should be the same file. That keeps your inspection result, training reference motion, and later evaluation aligned.

If you already know the exact phased settings you want to test, you can still override the auto-inferred behavior manually:

```bash
MUJOCO_GL=egl uv run -m mjlab.scripts.analyze_foot_penetration \
  --input-file /home/nubot/workspace/mjlab/datasets/csv/tiger_jump_to_shoulder_roll_R_001__A415_M_mimic.csv \
  --input-fps 120 \
  --output-fps 50 \
  --clearance 0.01 \
  --phase-grounded-height 0.03 \
  --phase-airborne-height 0.10 \
  --phase-window-s 0.12 \
  --phase-lookahead-s 0.24 \
  --phase-smoothing-s 0.08
```

These extra analysis parameters still do not change any files. They only change how the diagnostic script groups the motion into coarse phases and how it computes the suggested frame-wise lift range. Treat them as optional advanced tuning knobs, not as required stage definitions.

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

For aerial motions such as flips and shoulder rolls, `phased` alignment can be a better starting point than one global lift:

```bash
MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
  --input-file /home/nubot/workspace/mjlab/datasets/csv/tiger_jump_to_shoulder_roll_R_001__A415_M_mimic.csv \
  --output-name /home/nubot/workspace/mjlab/datasets/npz/tiger_jump_to_shoulder_roll_R_001__A415_M_phased.npz \
  --input-fps 120 \
  --output-fps 50 \
  --ground-align phased \
  --phase-control-mode auto \
  --clearance 0.01 \
  --phase-window-s 0.12 \
  --phase-lookahead-s 0.24 \
  --phase-smoothing-s 0.08 \
  --render True
```

The phased mode still keeps the final training motion self-consistent: after the frame-wise root `z` offsets are applied, the converter recomputes the root linear velocity before exporting the `.npz`.

Detailed phased-parameter guide:

- `--clearance`: target minimum foot-bottom clearance after alignment. Start with `0.01`. Increase it if the feet still scrape the plane; decrease it if the whole motion looks unnecessarily lifted near takeoff or stance.
- `--whole-body-geom-pattern`: regex for the collision geoms that are allowed to request extra airborne lift. The default `.*_collision$` includes head, torso, hands, feet, and other collision bodies. Narrow it if you only want a subset of bodies to matter.
- `--phase-control-mode`: `manual` keeps the previous behavior and uses `--phase-grounded-height` / `--phase-airborne-height` unless `--phase-blend-points` is provided. `auto` infers the phase control points from the motion's nearby foot-height distribution, which is the easier starting point for new aerial motions.
- `--phase-grounded-height`: lower default control point for foot height. When the local foot-bottom height is at or below this value, phased alignment behaves like foot-based alignment and avoids large extra lift. Raise it if you want the airborne logic to activate earlier; lower it if you want more frames to stay close to the original grounded height.
- `--phase-airborne-height`: upper default control point for foot height. When the local foot-bottom height is at or above this value, phased alignment allows full whole-body-driven lift. Lower it if flips should enter airborne protection sooner; raise it if only clearly airborne windows should get the larger lift.
- `--phase-window-s`: symmetric foot-height context window used to classify whether a frame belongs to a grounded or airborne neighborhood. Larger values make phase decisions steadier but less responsive; smaller values make them react faster but can create jitter around takeoff and landing.
- `--phase-lookahead-s`: how far ahead phased alignment looks when preparing for an upcoming landing. Larger values start lifting earlier in flight; smaller values delay the lift and keep the takeoff closer to the original reference.
- `--phase-smoothing-s`: temporal smoothing applied to the frame-wise offsets. Larger values create softer transitions but can spread lift farther across the motion; smaller values make the alignment more local and more literal.

Recommended phased-tuning workflow:

- Step 1: run `analyze_foot_penetration.py` and note the worst foot frame, the worst whole-body frame, and the suggested coarse phases with their recommended lift ranges.
- Step 2: start from either `--ground-align phased --phase-control-mode auto --clearance 0.01`, or from the exact `--phase-blend-points` printed by the analysis output.
- Step 3: if landing bodies still dip below the plane, first increase `--phase-lookahead-s`, then decrease `--phase-airborne-height`.
- Step 4: if the whole jump or roll starts to look too high too early, decrease `--phase-lookahead-s` or increase `--phase-airborne-height`.
- Step 5: if transitions look abrupt, increase `--phase-smoothing-s` a little. If too much of the motion is being lifted, decrease it.
- Step 6: if near-ground frames are still getting too much airborne-style lift, lower `--phase-grounded-height`.

Custom stages and how to change them:

- The default phased behavior is effectively a 2-stage schedule:
  foot near the ground -> mostly foot-based lift
  foot clearly airborne -> allow whole-body-driven lift
- You are not locked to those two stages. Use `--phase-blend-points` to provide your own control points in `foot_height:blend_weight` format, separated by commas. Example:

```bash
--phase-blend-points "0.00:0.0,0.03:0.0,0.06:0.35,0.10:1.0"
```

- In that example, the staged meaning is:
  `0.00` to `0.03` m: stay fully foot-based
  around `0.06` m: begin mixing in extra airborne lift
  at `0.10` m and above: allow full whole-body-driven lift
- More control points mean more stages. You can use 1 point, 2 points, or many points:
  1 point means a constant blend weight everywhere
  2 points behaves like the default grounded/airborne ramp
  3 or more points gives you custom multi-stage behavior
- No, all stages do not have to exist. If `--phase-blend-points` is omitted, the converter falls back to the default 2-point schedule controlled by `--phase-grounded-height` and `--phase-airborne-height`.
- If `--phase-blend-points` is provided, it overrides `--phase-grounded-height` and `--phase-airborne-height`.

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
