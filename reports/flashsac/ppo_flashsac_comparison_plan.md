# PPO vs FlashSAC Comparison Plan

## Goal

在 `main` 分支上建立一套可重复、可审计、对 `reference_ppo` 与 `flashsac` 都公平的对比流程。

本计划只回答两个问题：

1. 在相同任务与共享评估口径下，哪个算法更高效。
2. 在相同训练预算下，哪个算法最终结果更好、更稳定。

## Comparison Rules

- 所有对比都固定在同一个 git SHA 上进行，不允许边跑边改训练逻辑。
- 所有对比都使用同一任务 ID、同一 motion 资源、同一评估脚本、同一评估设备配置。
- 训练期允许算法保留各自已经验证过的后端实现与优化器形态，不强行把两者调成“伪相同”。
- 主表以 `env steps` 对齐，样本效率作为一等指标。
- 补充表以 `wall clock` 对齐，用来说明工程吞吐差异。
- 每个结论都必须同时给出数值结果和至少一段可复现视频或回放证据。
- 如果某个 run 中途改过 reward、termination、observation contract，就作废，不进入主表。

## Tasks

### Track A: Velocity

- Task ID: `Mjlab-Velocity-Flat-Unitree-G1`
- 目的：先确认两个后端在非 motion-tracking 任务上的基础竞争力。
- 主要指标：
  - `total_reward`
  - `episode_length_seconds`
  - `fell_over`
- 次要指标：
  - reward curve 斜率
  - 达到稳定不摔倒所需 env steps

### Track B: Tracking

- Task ID: `Mjlab-Tracking-Flat-Unitree-G1`
- Motion asset: 固定一个 `motion.npz` 或等价 registry artifact，并记录绝对路径与 SHA256。
- 目的：比较两个算法在完整 tracking 闭环上的真实能力。
- 主要指标：
  - `success_rate`
  - `mpkpe`
  - `r_mpkpe`
  - `joint_vel_error`
- 次要指标：
  - `ee_pos_error`
  - `ee_ori_error`
  - 视频中的跟踪连续性、是否出现明显卡顿或摔倒

## Seeds

统一使用 5 个种子：

- `7`
- `42`
- `1337`
- `2026`
- `3407`

如果 GPU 预算不足，先跑 `42` 做 smoke，再补满 5 个种子进入正式表。

## Training Budgets

### Shared Smoke Gate

所有正式实验前都先过同一套 smoke：

- 每个算法、每个任务先跑 `250k env steps`
- 产出：
  - train command
  - eval command
  - config hash
  - 一次 eval JSON
  - 一段短视频或 headless 回放

只有 smoke 通过，才能进入正式长跑。

### Velocity Main Budget

- 主预算：`50M env steps`
- 记录里程碑：
  - `1M`
  - `5M`
  - `10M`
  - `25M`
  - `50M`

### Tracking Main Budget

- 主预算：`100M env steps`
- 记录里程碑：
  - `1M`
  - `5M`
  - `10M`
  - `25M`
  - `50M`
  - `100M`

### Wall Clock Supplement

在同一张卡、同一机器上额外记录：

- `30 min`
- `2 h`
- `6 h`
- `12 h`

这张表不替代 env-step 主表，只作为工程吞吐补充。

## Checkpoint Policy

- 每个里程碑都保留 checkpoint。
- 每个 run 至少汇报两类 checkpoint：
  - `final checkpoint`
  - `best checkpoint by primary metric`
- Velocity 以 `total_reward` 选 best。
- Tracking 以 `success_rate` 为主排序，`mpkpe` 为并列时的 tie-breaker。
- 不允许只挑“最好看的视频”而不对应实际汇报 checkpoint。

## Evaluation Contract

### Velocity

- 使用固定 eval seed 与固定 episode 数。
- 所有评估都在同一设备类型上跑完。
- 输出统一 JSON，至少包含：
  - `env_steps`
  - `total_reward`
  - `episode_length_seconds`
  - `fell_over`

### Tracking

- 统一使用 `uv run evaluate-tracking`
- 统一指定同一个 `--motion-file`
- 统一 `--num-envs 1024`
- 输出统一 JSON，至少包含：
  - `success_rate`
  - `mpkpe`
  - `r_mpkpe`
  - `joint_vel_error`
  - `ee_pos_error`
  - `ee_ori_error`

## Command Templates

### Velocity Train

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --backend flashsac --agent.seed <seed>
uv run train Mjlab-Velocity-Flat-Unitree-G1 --backend reference_ppo --agent.seed <seed>
```

### Tracking Train

```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 --backend flashsac --env.commands.motion.motion-file <absolute-motion.npz> --agent.seed <seed>
uv run train Mjlab-Tracking-Flat-Unitree-G1 --backend reference_ppo --env.commands.motion.motion-file <absolute-motion.npz> --agent.seed <seed>
```

### Tracking Eval

```bash
uv run evaluate-tracking Mjlab-Tracking-Flat-Unitree-G1 --backend flashsac --checkpoint-file <checkpoint> --motion-file <absolute-motion.npz> --num-envs 1024 --output-file <eval.json>
uv run evaluate-tracking Mjlab-Tracking-Flat-Unitree-G1 --backend reference_ppo --checkpoint-file <checkpoint> --motion-file <absolute-motion.npz> --num-envs 1024 --output-file <eval.json>
```

## Required Artifacts Per Run

- `params/env.yaml`
- `params/agent.yaml`
- exact train command
- exact eval command
- primary metrics JSON
- checkpoint path
- at least one video or headless playback artifact
- hardware note:
  - GPU model
  - CUDA visibility
  - elapsed wall clock

## Result Tables

每次正式更新都维护两张表。

### Table 1: Env-Step Matched

字段：

- task
- backend
- seed
- env_steps
- wall_clock_minutes
- primary metric
- secondary metrics
- checkpoint path
- eval json path
- notes

### Table 2: Aggregated Summary

字段：

- task
- backend
- seeds_completed
- median primary metric
- mean primary metric
- std primary metric
- final checkpoint summary
- best checkpoint summary
- verdict

## Verdict Rules

- `sample_efficiency_win`:
  同任务、同 env steps 下，算法在 5 个种子的中位数明显更优。
- `final_quality_win`:
  在主预算结束时，算法的 final checkpoint 中位数更优。
- `stability_win`:
  在主预算结束时，算法方差更小，且没有明显更差的最终质量。
- `inconclusive`:
  中位数接近、方差过大、或有效种子数不足 5。
- `invalid`:
  训练或评估合同不一致，或缺少必要 artifact。

## Execution Order

1. 先做 velocity smoke。
2. 再做 tracking smoke。
3. 通过 smoke 后并行启动 velocity 正式 5-seed。
4. tracking 正式实验在 velocity 第一批结果稳定后启动，避免同时烧掉全部 GPU。
5. 每次里程碑先写表，再下结论，不允许只凭曲线截图口头判断。

## Immediate Next Steps

1. 固定 tracking 的 motion asset 路径与 SHA256。
2. 先在 `main` 上跑 `seed=42` 的 velocity smoke，分别覆盖 `flashsac` 和 `reference_ppo`。
3. 再在 `main` 上跑 `seed=42` 的 tracking smoke，确认 train/eval/play artifact 全链路齐全。
4. smoke 全通过后，建立正式 comparison 表并发起 5-seed 长跑。
