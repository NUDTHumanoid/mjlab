# Worker 4 lane3 merged-head auditable artifact capture

- Captured at (UTC): 2026-04-15T11:32:39.561469+00:00
- Lane: `lane3-merged-head`
- Seed: `42`
- Run dir: `/home/ssy/ssy_files/mjlab/.omx/team/source-of-truth-omx-specs-deep/worktrees/worker-3/logs/flashsac/tracking_flat_unitree_g1_flashsac/2026-04-15_19-27-32_lane3-merged-head-smoke-seed-42`

## Runtime metadata paths
- `runtime.yaml` → `/home/ssy/ssy_files/mjlab/.omx/team/source-of-truth-omx-specs-deep/worktrees/worker-3/logs/flashsac/tracking_flat_unitree_g1_flashsac/2026-04-15_19-27-32_lane3-merged-head-smoke-seed-42/params/runtime.yaml`
- `agent.yaml` → `/home/ssy/ssy_files/mjlab/.omx/team/source-of-truth-omx-specs-deep/worktrees/worker-3/logs/flashsac/tracking_flat_unitree_g1_flashsac/2026-04-15_19-27-32_lane3-merged-head-smoke-seed-42/params/agent.yaml`
- `env.yaml` → `/home/ssy/ssy_files/mjlab/.omx/team/source-of-truth-omx-specs-deep/worktrees/worker-3/logs/flashsac/tracking_flat_unitree_g1_flashsac/2026-04-15_19-27-32_lane3-merged-head-smoke-seed-42/params/env.yaml`

## Hashes
- `config_hash` → `c77d6b86e0c3afe917109b085e7e62d138f312f98cbe0133a6860543d57deb49`
- `runtime_yaml_sha256` → `70987f7658ce34e0860c14233a353d7c120135ec583484d59a1c6799db43ab06`
- `eval_json_sha256` → `3f3171eec36c24268d5967ef0e962a9ac8c3b2564cc8769e91a62739c9eda7cf`
- `play_mp4_sha256` → `43d7213785846831c2c400dc10adcd88f4614a57639c052b2f721d061062d3fe`

## Runtime snapshot
- device → `cuda:0`
- buffer_device_type → `cuda:0`
- use_amp → `True`
- num_envs → `4096`
- num_env_steps → `250000`
- num_interaction_steps → `62`
- target_update_budget → `124.0`
- actual_update_steps → `72`
- actual_updates_per_interaction_step → `1.1612903225806452`
- final_replay_size → `245760`
- checkpoint_count → `6`
- final_checkpoint_dir → `logs/flashsac/tracking_flat_unitree_g1_flashsac/2026-04-15_19-27-32_lane3-merged-head-smoke-seed-42/step_62`

## Checkpoints
- `/home/ssy/ssy_files/mjlab/.omx/team/source-of-truth-omx-specs-deep/worktrees/worker-3/logs/flashsac/tracking_flat_unitree_g1_flashsac/2026-04-15_19-27-32_lane3-merged-head-smoke-seed-42/step_28`
- `/home/ssy/ssy_files/mjlab/.omx/team/source-of-truth-omx-specs-deep/worktrees/worker-3/logs/flashsac/tracking_flat_unitree_g1_flashsac/2026-04-15_19-27-32_lane3-merged-head-smoke-seed-42/step_35`
- `/home/ssy/ssy_files/mjlab/.omx/team/source-of-truth-omx-specs-deep/worktrees/worker-3/logs/flashsac/tracking_flat_unitree_g1_flashsac/2026-04-15_19-27-32_lane3-merged-head-smoke-seed-42/step_42`
- `/home/ssy/ssy_files/mjlab/.omx/team/source-of-truth-omx-specs-deep/worktrees/worker-3/logs/flashsac/tracking_flat_unitree_g1_flashsac/2026-04-15_19-27-32_lane3-merged-head-smoke-seed-42/step_49`
- `/home/ssy/ssy_files/mjlab/.omx/team/source-of-truth-omx-specs-deep/worktrees/worker-3/logs/flashsac/tracking_flat_unitree_g1_flashsac/2026-04-15_19-27-32_lane3-merged-head-smoke-seed-42/step_56`
- `/home/ssy/ssy_files/mjlab/.omx/team/source-of-truth-omx-specs-deep/worktrees/worker-3/logs/flashsac/tracking_flat_unitree_g1_flashsac/2026-04-15_19-27-32_lane3-merged-head-smoke-seed-42/step_62`

## Eval artifact
- path → `/home/ssy/ssy_files/mjlab/.omx/artifacts/flashsac_tracking/lane3-merged-head/seed-42/eval/eval.json`
- success_rate → `0.0`
- mpkpe → `0.13610216975212097`
- r_mpkpe → `0.06861449778079987`
- joint_vel_error → `9.927271842956543`
- ee_pos_error → `0.1659158319234848`
- ee_ori_error → `0.45917844772338867`

## Play artifact
- mp4 → `/home/ssy/ssy_files/mjlab/.omx/artifacts/flashsac_tracking/lane3-merged-head/seed-42/play/lane3-merged-head-step-0.mp4`
- note → `/home/ssy/ssy_files/mjlab/.omx/artifacts/flashsac_tracking/lane3-merged-head/seed-42/play/video-or-note.txt`

## Metadata gaps
- worker-3 live run predates worker-4 audit patch, so no summary/metrics.json was emitted next to the training run
- worker-3 live run predates worker-4 audit patch, so no summary/checkpoints.json or summary/log-history.json was emitted next to the training run
- runtime.yaml records checkpoint_count/final_checkpoint_dir but not per-checkpoint absolute paths or final replay fill ratio
- no play-side contact sheet or frame manifest is present yet; only mp4 + video-or-note.txt exist
