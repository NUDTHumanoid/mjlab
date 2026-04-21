# LaneV cross-lane acceptance precheck

- Generated at: 2026-04-15T11:11:30.859348+00:00
- Leader HEAD: `407660e` (407660e91ad7468826c68990f3ea9ec017575f2c)
- Baseline SHA: `2fc0c19dfc4b87187d6372bf97965b3d40bda6d0`
- Lane0 observed head: `3406849fabb795c1438e842844318bc7bdf6dc18`
- Mailbox follow-up: `ca5c31ca-2f3b-4f00-afcf-40a5d60daf5a`

## Source-of-truth inputs

- `spec_path` → `/home/ssy/ssy_files/mjlab/.omx/specs/deep-interview-flashsac-rootcause-parallel-refactor.md`
- `prd_path` → `/home/ssy/ssy_files/mjlab/.omx/plans/prd-flashsac-tracking-rootcause-team.md`
- `test_spec_path` → `/home/ssy/ssy_files/mjlab/.omx/plans/test-spec-flashsac-tracking-rootcause-team.md`
- `lane0_manifest_path` → `/home/ssy/ssy_files/mjlab/.omx/specs/flashsac_lane0/shared-manifest.json`
- `lane0_baseline_provenance_path` → `/home/ssy/ssy_files/mjlab/.omx/specs/flashsac_lane0/baseline-provenance.json`
- `lane0_ppo_control_manifest_path` → `/home/ssy/ssy_files/mjlab/.omx/specs/flashsac_lane0/ppo-control-manifest.json`
- `lane0_comparison_template_path` → `/home/ssy/ssy_files/mjlab/.omx/specs/flashsac_lane0/comparison-template.csv`
- `task_10` → `/home/ssy/ssy_files/mjlab/.omx/state/team/use-omx-specs-deep-interview-f/tasks/task-10.json`
- `task_11` → `/home/ssy/ssy_files/mjlab/.omx/state/team/use-omx-specs-deep-interview-f/tasks/task-11.json`

## Completed-lane snapshot

| Lane | Commit | Present on leader HEAD | Promotability | Summary |
| --- | --- | --- | --- | --- |
| lane0-manifest | `ad9a6cc58053f02a9b165e6a5aaa176017dce078` | True | required-input | Published the shared manifest, PPO control manifest, baseline provenance, and comparison template. |
| laneA-parity | `cf96fba` | True | mainline-candidate-awaiting-tracking-smoke | Fixed FlashSAC LR warmup/decay scheduling to use interaction-step-derived update budgets after the real num_envs is observed. |
| laneB-bridge | `b5c8189` | True | mainline-candidate-with-post-manifest-validation | Canonical evaluation overrides and zero-safe metric aggregation are on leader HEAD, and post-manifest validation produced a PASS-with-caution bundle. |
| laneC-trainer-systems | `e3082b3` | True | mainline-candidate-awaiting-tracking-smoke | Normalized runtime device metadata and recorded runtime.yaml device/buffer/amp settings during training; follow-up metadata integration also landed cleanly. |
| laneD-env-contract-control | `eb4cc72` | True | quarantined-control-only | Added opt-in planar/height env-contract probes for counterexample measurement without changing default tracking config wiring. |

## PPO gold baseline snapshot

- Status: **authoritative native PPO success is now the verifier floor**
- Authoritative run dir → `/home/ssy/ssy_files/mjlab/logs/rsl_rl/g1_tracking_handstand1/2026-04-14_12-19-21_handstand1_acrobatics_ft_40000`
- Preferred checkpoint → `/home/ssy/ssy_files/mjlab/logs/rsl_rl/g1_tracking_handstand1/2026-04-14_12-19-21_handstand1_acrobatics_ft_40000/model_31500.pt`
- Motion file → `/home/ssy/ssy_files/mjlab/artifacts/motion_runs/handstand1_high/pipeline/mjlab/motion.npz`
- Checkpoint sha256 → `29bf464dfb8bf28bde707054a58aff6e363af9e97f23eafd90a58bc13ae2940c`
- Motion sha256 → `89d469b7ac8b1c0cb75425ad3ef43f2e6336fecd809d5981b41794e3c9204818`
- LaneV artifact bundle root → `/home/ssy/ssy_files/mjlab/.omx/artifacts/flashsac_tracking/laneV-ppo-control/seed-42`
- LaneV summary → `/home/ssy/ssy_files/mjlab/.omx/artifacts/flashsac_tracking/laneV-ppo-control/seed-42/summary/lanev-smoke-summary.json`
- LaneV comparison row → `/home/ssy/ssy_files/mjlab/.omx/artifacts/flashsac_tracking/laneV-ppo-control/seed-42/comparison-row.json`
- Local reproduced metrics (debug-signal-only) → success_rate=0.25, mpkpe=0.10915069282054901, ee_pos_error=0.1373407244682312
- Guidance: Separate native PPO capability from local reproduced eval/play behavior in verifier notes and future FlashSAC comparisons.

## Pending gaps

- **lane0-manifest-stale-on-current-head** — Shared manifest observed head 3406849fabb795c1438e842844318bc7bdf6dc18 but leader head is now 407660e91ad7468826c68990f3ea9ec017575f2c; refresh before using comparison hashes as current truth.
- **comparison-table-still-empty-despite-available-bundles** — /home/ssy/ssy_files/mjlab/.omx/specs/flashsac_lane0/comparison-template.csv still contains only the header row even though LaneB post-manifest validation and LaneV PPO baseline bundles already exist.
- **laneA-and-laneC-merged-head-tracking-smoke-still-missing** — Lane A and Lane C still lack merged-head tracking smoke/eval/play artifacts, so there is no mainline-candidate row to compare against the PPO floor.
- **laneV-local-reproduction-divergence-still-open** — Task 11 remains in progress: the local checkpoint+motion evaluate/play path still diverges from the authoritative native PPO success and must remain debug-only telemetry.
- **baseline-training-override-risk-still-open** — FlashSAC training overrides in src/mjlab/flashsac/config.py still widen anchor thresholds and drop ee_body_pos during training; this predates the follow-up lanes and remains a genericity risk to review during acceptance.

## Cosmetic env-relaxation leak check

- Status: **no_new_laneD_leak_into_canonical_eval_path_detected; acceptance_video_semantics_still_split; baseline_flashsac_train_override_risk_remains**
- Task 8 post-manifest validation confirmed evaluate.py uses apply_tracking_evaluation_overrides() for both PPO and FlashSAC evaluation paths on leader HEAD.
- The same validation also confirmed play.py still uses apply_flashsac_tracking_inference_overrides(), so acceptance-video semantics remain distinct from canonical evaluation semantics.
- LaneD helper names still appear only in src/mjlab/tasks/tracking/mdp/{rewards,metrics,terminations}.py and their dedicated test; they are not referenced by default tracking config wiring.
- The threshold widening and ee_body_pos removal in src/mjlab/flashsac/config.py already existed at baseline 2fc0c19, so they are not a new leak from LaneD or the evaluation refactor.

## Contract-hash drift vs pinned baseline manifest

| Contract | Baseline hash | Current leader-head hash | Changed on leader HEAD |
| --- | --- | --- | --- |
| flashsac_train | `ea7a6140cd9e760c152fea6f2f341bcc59b12a111a5c49c31d6746e83f6585ee` | `74f308a133e730cadaccf2421889a8f49c47811a5109d21345b7ac86f0537efa` | True |
| flashsac_eval | `42b4542c65a195db01dc99018552bfa8d1821a9e87ee2627484c51da745b4173` | `e2b3f0b043f43319487282c55cdc482a5fba303bba34a186c52639221d59c49c` | True |
| flashsac_play | `7a9a22d3b508b41c57dd3a6ba06b86517cf5b8220ee18b80e5d5f134050dc1e1` | `75542c8030056251da12ba49e55561715b271bdbb63d835ee6671a9ea00e0e63` | True |
| ppo_train | `e980459f1f318602b162cbe8080366c8917256f3f276d8048835210ee2c85a38` | `e980459f1f318602b162cbe8080366c8917256f3f276d8048835210ee2c85a38` | False |
| ppo_eval | `2ce4fd9ff1055df33f79c692e9dbbf4c085ea7490525e77587c787a97f2a6b25` | `7b99d8b6d5eac738214548f815700f73e59c187b0969934030cfe5690be5c546` | True |

## Exact next commands

1. Refresh shared manifest:
   ```bash
   cd "$OMX_TEAM_LEADER_CWD" && uv run flashsac-lane0-manifest --leader-cwd /home/ssy/ssy_files/mjlab --output-dir /home/ssy/ssy_files/mjlab/.omx/specs/flashsac_lane0
   ```
2. Lane A upstream parity:
   ```bash
   cd "$OMX_TEAM_LEADER_CWD" && uv run train Mjlab-Tracking-Flat-Unitree-G1 --backend flashsac --registry-name <motion-artifact> --agent.seed 42 --agent.run-name laneA-parity-seed-42
   cd "$OMX_TEAM_LEADER_CWD" && uv run evaluate-tracking Mjlab-Tracking-Flat-Unitree-G1 --backend flashsac --checkpoint-file <laneA-parity-seed-42-checkpoint-file> --motion-file <absolute-motion-npz> --num-envs 1024 --output-file $OMX_TEAM_LEADER_CWD/.omx/artifacts/flashsac_tracking/laneA-parity/seed-42/eval/eval.json
   cd "$OMX_TEAM_LEADER_CWD" && uv run play Mjlab-Tracking-Flat-Unitree-G1 --backend flashsac --checkpoint-file <laneA-parity-seed-42-checkpoint-file> --motion-file <absolute-motion-npz> --num-envs 1 --viewer viser --video
   ```
3. Lane C trainer/systems:
   ```bash
   cd "$OMX_TEAM_LEADER_CWD" && uv run train Mjlab-Tracking-Flat-Unitree-G1 --backend flashsac --registry-name <motion-artifact> --agent.seed 42 --agent.run-name laneC-trainer-systems-seed-42
   cd "$OMX_TEAM_LEADER_CWD" && uv run evaluate-tracking Mjlab-Tracking-Flat-Unitree-G1 --backend flashsac --checkpoint-file <laneC-trainer-systems-seed-42-checkpoint-file> --motion-file <absolute-motion-npz> --num-envs 1024 --output-file $OMX_TEAM_LEADER_CWD/.omx/artifacts/flashsac_tracking/laneC-trainer-systems/seed-42/eval/eval.json
   cd "$OMX_TEAM_LEADER_CWD" && uv run play Mjlab-Tracking-Flat-Unitree-G1 --backend flashsac --checkpoint-file <laneC-trainer-systems-seed-42-checkpoint-file> --motion-file <absolute-motion-npz> --num-envs 1 --viewer viser --video
   ```
4. Lane V authoritative PPO checkpoint local reproduction (debug-signal-only):
   ```bash
   # no train step; existing authoritative artifact
   local reproduced path: cd "$OMX_TEAM_LEADER_CWD" && uv run evaluate-tracking Mjlab-Tracking-Flat-Unitree-G1 --checkpoint-file /home/ssy/ssy_files/mjlab/logs/rsl_rl/g1_tracking_handstand1/2026-04-14_12-19-21_handstand1_acrobatics_ft_40000/model_31500.pt --motion-file /home/ssy/ssy_files/mjlab/artifacts/motion_runs/handstand1_high/pipeline/mjlab/motion.npz --num-envs 4 --device cpu --output-file /home/ssy/ssy_files/mjlab/.omx/logs/lanev-ppo-authoritative-baseline-eval.json
   local reproduced path: headless-equivalent authoritative PPO playback: cd "$OMX_TEAM_LEADER_CWD" && uv run python <inline-rslrl-video-recorder> --checkpoint-file /home/ssy/ssy_files/mjlab/logs/rsl_rl/g1_tracking_handstand1/2026-04-14_12-19-21_handstand1_acrobatics_ft_40000/model_31500.pt --motion-file /home/ssy/ssy_files/mjlab/artifacts/motion_runs/handstand1_high/pipeline/mjlab/motion.npz --num-envs 1 --device cpu --video-length 200 --output-video /home/ssy/ssy_files/mjlab/.omx/artifacts/flashsac_tracking/laneV-ppo-control/seed-42/play/lanev-ppo-control-step-0.mp4
   ```

## Acceptance precheck verdict

- Overall status: **authoritative_ppo_floor_available_but_acceptance_still_blocked_on_manifest_refresh_and_shared_path_diagnosis**
- Refresh the Lane0 manifest on current leader HEAD before recording new comparison rows.
- Keep the user-provided native PPO run/checkpoint as the PPO gold baseline; do **not** schedule a replacement PPO smoke to redefine the floor.
- Use the existing laneV local reproduced eval/play bundle only as shared-path debugging evidence until task 11 resolves.
- Run merged-head tracking smoke/eval/play for LaneA and LaneC, then backfill comparison rows under the refreshed manifest.
