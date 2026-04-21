# FlashSAC parity overnight launch

- Date: 2026-04-15
- Branch: `feature/flashsac-integration`
- Goal: verify checkpoint-declared FlashSAC env parity restoration across resume/evaluate/play, then run an overnight parity-tagged training job on the new code.

## Code lane

- Restore FlashSAC checkpoint env parity from `params/env.yaml`.
- Keep precedence explicit: `CLI > checkpoint > default`.
- Do not auto-override `num_envs` in `play/evaluate`; audit only by default.
- Treat `ee_body_pos` conservatively: restore only when unambiguous, otherwise audit-only.

## Verification before launch

- `./.venv/bin/pytest tests/test_flashsac_checkpoint_parity.py -q`
- `./.venv/bin/pytest tests/test_rslrl_checkpoint_parity.py tests/test_flashsac_bridge_contract.py -q`
- `./.venv/bin/pytest tests/test_flashsac_backend.py -k "load_flashsac_runner_cfg_reads_yaml_and_normalizes_for_inference or flashsac_tracking_inference_overrides_remove_randomization or apply_flashsac_tracking_train_overrides_stationarizes_tracking_env or resolve_flashsac_checkpoint_dir_accepts_dir_and_file" -q`
- Real smoke:
  `./.venv/bin/python -m mjlab.tasks.tracking.scripts.evaluate Mjlab-Tracking-Flat-Unitree-G1 --backend flashsac --checkpoint-file logs/flashsac/tracking_flat_unitree_g1_flashsac/2026-04-15_20-14-10_flashsac-1024env-100m-seed42/step_29298 --num-envs 1 --device cpu --output-file /tmp/flashsac-parity-smoke-eval.json`

## Overnight run intent

- New run directory only; do not append to an old run.
- Resume source checkpoint:
  `logs/flashsac/tracking_flat_unitree_g1_flashsac/2026-04-15_20-14-10_flashsac-1024env-100m-seed42/step_29298`
- Resume mode: expected `weights-only resume` unless `replay_buffer.pt` exists next to that checkpoint.
- Launch should preserve the same 1024-env lane and only add parity/runtime consistency changes.

## Morning readout checklist

- TensorBoard is still writing.
- Parity audit appears in launch logs.
- New run directory contains `params/env.yaml`, `params/agent.yaml`, `params/runtime.yaml`.
- Resume mode is visible from logs/runtime metadata.
- Early failure mode changed or stayed stable for an attributable reason.
