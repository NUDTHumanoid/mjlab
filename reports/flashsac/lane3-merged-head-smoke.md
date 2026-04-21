# Lane3 merged-head FlashSAC smoke run

- Generated at: 2026-04-15T11:33:20.277734+00:00
- Head SHA: `606d84a` (606d84a1e748cb188510a23a0368eb0b39e6d8a5)
- Baseline SHA: `2fc0c19dfc4b87187d6372bf97965b3d40bda6d0`

## Source-of-truth inputs

- `spec_path` → `/home/ssy/ssy_files/mjlab/.omx/specs/deep-interview-flashsac-rootcause-parallel-refactor.md`
- `prd_path` → `/home/ssy/ssy_files/mjlab/.omx/plans/prd-flashsac-tracking-rootcause-team.md`
- `test_spec_path` → `/home/ssy/ssy_files/mjlab/.omx/plans/test-spec-flashsac-tracking-rootcause-team.md`

## Executed smoke commands

```bash
cd "/home/ssy/ssy_files/mjlab/.omx/team/source-of-truth-omx-specs-deep/worktrees/worker-3" && uv run train Mjlab-Tracking-Flat-Unitree-G1 --backend flashsac --env.commands.motion.motion-file /home/ssy/ssy_files/mjlab/artifacts/motion_runs/handstand1_high/pipeline/mjlab/motion.npz --agent.seed 42 --agent.run-name lane3-merged-head-smoke-seed-42 --agent.num-env-steps 250000 --agent.logger tensorboard
cd "/home/ssy/ssy_files/mjlab/.omx/team/source-of-truth-omx-specs-deep/worktrees/worker-3" && uv run evaluate-tracking Mjlab-Tracking-Flat-Unitree-G1 --backend flashsac --checkpoint-file /home/ssy/ssy_files/mjlab/.omx/team/source-of-truth-omx-specs-deep/worktrees/worker-3/logs/flashsac/tracking_flat_unitree_g1_flashsac/2026-04-15_19-27-32_lane3-merged-head-smoke-seed-42/step_62 --motion-file /home/ssy/ssy_files/mjlab/artifacts/motion_runs/handstand1_high/pipeline/mjlab/motion.npz --num-envs 1024 --device cuda:0 --output-file /home/ssy/ssy_files/mjlab/.omx/artifacts/flashsac_tracking/lane3-merged-head/seed-42/eval/eval.json
cd "/home/ssy/ssy_files/mjlab/.omx/team/source-of-truth-omx-specs-deep/worktrees/worker-3" && uv run python /home/ssy/ssy_files/mjlab/.omx/artifacts/flashsac_tracking/lane3-merged-head/seed-42/commands/play-headless-recorder.py
```

## Artifact bundle

- Artifact root → `/home/ssy/ssy_files/mjlab/.omx/artifacts/flashsac_tracking/lane3-merged-head/seed-42`
- Log dir → `/home/ssy/ssy_files/mjlab/.omx/team/source-of-truth-omx-specs-deep/worktrees/worker-3/logs/flashsac/tracking_flat_unitree_g1_flashsac/2026-04-15_19-27-32_lane3-merged-head-smoke-seed-42`
- Final checkpoint → `/home/ssy/ssy_files/mjlab/.omx/team/source-of-truth-omx-specs-deep/worktrees/worker-3/logs/flashsac/tracking_flat_unitree_g1_flashsac/2026-04-15_19-27-32_lane3-merged-head-smoke-seed-42/step_62`
- Eval JSON → `/home/ssy/ssy_files/mjlab/.omx/artifacts/flashsac_tracking/lane3-merged-head/seed-42/eval/eval.json`
- Video → `/home/ssy/ssy_files/mjlab/.omx/artifacts/flashsac_tracking/lane3-merged-head/seed-42/play/lane3-merged-head-step-0.mp4`
- Hashes → `/home/ssy/ssy_files/mjlab/.omx/artifacts/flashsac_tracking/lane3-merged-head/seed-42/hashes.json`

## Runtime summary

- Target env steps → `250000`
- Final env steps → `253952`
- Final interaction steps → `62`
- Actual update steps → `72`
- Replay size → `245760`
- Checkpoint count → `6`

## Smoke evaluation metrics

| Metric | Value |
| --- | ---: |
| success_rate | 0.0 |
| mpkpe | 0.13610216975212097 |
| r_mpkpe | 0.06861449778079987 |
| joint_vel_error | 9.927271842956543 |
| ee_pos_error | 0.1659158319234848 |
| ee_ori_error | 0.45917844772338867 |

## Training scalar tail

| Tag | Step | Value |
| --- | ---: | ---: |
| actor/loss | 253952 | 0.9592125415802002 |
| actor/entropy | 253952 | -46.77478790283203 |
| actor/mean_action | 253952 | -0.03713666647672653 |
| temperature/value | 253952 | 0.010089605115354061 |
| temperature/loss | 253952 | -0.3320239186286926 |
| critic/loss | 253952 | 4.359889984130859 |
| critic/max_entropy_bonus | 253952 | 1.297623634338379 |
| Episode_Reward/total | 253952 | -4.663280487060547 |
| Metrics/motion/error_body_pos | 253952 | 0.6487683653831482 |
| Metrics/motion/error_joint_vel | 253952 | 16.448335647583008 |
| Perf/effective_updates_per_interaction_step | 253952 | 1.1612902879714966 |
| Perf/replay_fill_ratio | 253952 | 2.4576001167297363 |

## Verdict

- Status: **alive_but_not_learning_at_250k_gate**
- Not blocked on worker-1 or any missing shared-path fix: the current merged-head code produced real checkpoints, a canonical eval JSON, and a headless play video.
- The 250k gate still failed qualitatively and quantitatively: success_rate stayed at `0.0`, and all 1024 evaluation episodes terminated instead of truncating to success.
- Semantics split remains visible in the captured run: training and headless play used FlashSAC tracking overrides without `ee_body_pos`, while canonical evaluation kept `ee_body_pos` enabled.

## Recommended next steps

1. Compare this bundle against the authoritative PPO floor and any shared-path diff findings before spending more GPU on a longer FlashSAC run.
2. If the team extends FlashSAC, resume from the generated checkpoint bundle rather than starting a fresh run without a new hypothesis.
