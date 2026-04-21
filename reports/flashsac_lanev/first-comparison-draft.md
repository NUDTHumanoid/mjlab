# LaneV comparison and acceptance notes

- Baseline SHA: `2fc0c19dfc4b87187d6372bf97965b3d40bda6d0`
- Comparator policy: authoritative native PPO floor > debug-only local reproduced PPO telemetry > FlashSAC candidate artifacts.

## Comparison table

| Row | Role | Backend | success_rate | mpkpe | r_mpkpe | joint_vel_error | ee_pos_error | ee_ori_error | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| authoritative_native_ppo | authoritative_floor | ppo | n/a | n/a | n/a | n/a | n/a | n/a | Canonical PPO floor from the user-supplied successful native run. Do not downgrade PPO capability based on local reproduced eval/play telemetry. |
| local_reproduced_ppo | debug_signal_only | ppo | 0.2500 | 0.0982 | 0.0527 | 4.2959 | 0.1228 | 0.2172 | Checkpoint+motion replay under shared local runtime/eval/inference plumbing. Useful for reproduction debugging, not for redefining PPO capability. |
| flashsac_reference_smoke | candidate_reference | flashsac | 0.0000 | 0.1361 | 0.0986 | 6.6340 | 0.1659 | 0.5170 | Current FlashSAC smoke/reference artifact. Compare against the authoritative PPO floor only after shared-path reproduction issues are separated. |

## Local reproduced PPO minus FlashSAC reference delta

| Metric | Delta |
| --- | --- |
| success_rate | 0.2500 |
| mpkpe | -0.0379 |
| r_mpkpe | -0.0459 |
| joint_vel_error | -2.3381 |
| ee_pos_error | -0.0431 |
| ee_ori_error | -0.2998 |

## Acceptance notes

- Authoritative native PPO success remains the only PPO capability floor. Local reproduced PPO eval/play must stay debug-only until parity is restored.
- Current local reproduced PPO metrics are debugging telemetry for the shared path, not a replacement for the authoritative PPO baseline.
- Current best shared-path root cause: evaluate.py and play.py load env/agent config from the current task_id instead of the saved run params in the checkpoint directory. Replaying the authoritative PPO checkpoint under the default task semantics reproduces failure; replaying under the acrobatics-equivalent task semantics restores success.
- FlashSAC comparison implication: FlashSAC comparisons can be invalidated by the same config-loading/task-id mismatch before any algorithm-specific conclusion is warranted.

## Evidence gaps blocking an apples-to-apples judgment

- The authoritative PPO floor is a native successful run, while the local reproduced PPO path is still debug-only telemetry because replay currently depends on current task_id config instead of the saved run params.
- The current FlashSAC evidence is only a smoke artifact (~253952 env steps, success_rate=0.0); it is not yet a matched-seed, matched-window, acceptance-grade run.
- No apples-to-apples same-protocol FlashSAC evaluation bundle with >=2 seeds, >=3 accepted windows, and stable play/video evidence exists yet.
