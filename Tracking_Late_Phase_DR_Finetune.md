# Tracking Late-Phase DR Finetune

## 背景

这次 `Late-Phase-DR-Finetune` 的目标不是重新学习整段空翻，而是在一个已经能较好跟踪完整动作的 checkpoint 基础上，专门强化后半段的恢复能力，尤其是：

- 翻滚后站起时的前冲过大
- 翻滚后站起时的冲量不足

对应任务：

- `Mjlab-Tracking-Flat-Unitree-G1-Late-Phase-DR-Finetune`
- `Mjlab-Tracking-Flat-Unitree-G1-New-Late-Phase-DR-Finetune`
- `Mjlab-Tracking-Rough-Unitree-G1-Late-Phase-DR-Finetune`
- `Mjlab-Tracking-Rough-Unitree-G1-New-Late-Phase-DR-Finetune`

## 与原框架的兼容性

结论先说：

- 用这套改动训练出来的策略，通常可以直接放回“原来没有这些 late-phase 扰动逻辑”的 clean 框架里跑。
- 但旧框架只能用它做正常 `play/train`，不能使用本文新增的 late-phase 独立 scale 参数和扰动评测能力。

原因是这次改动没有改变策略接口本身：

- 没改 observation 维度
- 没改 action 维度
- 没改机器人 asset / action scale 配置
- 没改 `Mjlab-Tracking-Flat-Unitree-G1-New` 的 clean inference 入口

本次新增的内容主要是：

- 训练任务里多了一个 `late_phase_dr_disturbance`
- `play` 里可以显式打开 late-phase 扰动评测
- `train/play` 新增了独立的 overshoot / underpowered scale 参数

所以兼容性可以理解成：

- checkpoint 权重本身通常兼容原 clean 框架
- 新的扰动逻辑和新 CLI 参数不兼容旧代码，因为旧代码里没有这些实现

建议的实际使用方式：

- 如果只是部署或 clean 回放 checkpoint，可以直接放回原框架
- 如果还要复现本文的 late-phase 扰动训练或评测，请使用当前这版代码

对 rough 版也是一样：

- checkpoint 本身仍然不依赖新增 observation / action 接口
- rough late-phase 只是把“rough tracking 的环境逻辑”和“late-phase 恢复扰动逻辑”叠加到一起
- 不带这些新任务和新 CLI 的旧代码，无法直接复现本文的 rough late-phase 训练 / 评测

## 当前设计目标

这版 late-phase finetune 的设计原则是：

1. 保留完整动作的正常采样，不把 reset 强行压到后半段。
2. 只把更强的恢复类扰动集中到动作后半段。
3. 不再用“持续推着机器人走”的外力近似恢复失败。
4. 把“过冲”和“冲量不足”建模成更接近执行误差的站起失败模式。
5. `train` 和 `play` 使用同一套缩放逻辑，保证参数大小一一对应。

## 核心思路

### 1. 保留 full-motion 训练

`Late-Phase-DR-Finetune` 不会只从后半段采样，也不会混入旧版那种破坏整段动作分布的 recovery-mixed reset。

训练时仍然：

- 使用完整 motion
- `sampling_mode="adaptive"`
- 从整段动作分布中采样

这样做是为了避免前半段空翻能力被明显遗忘。

## Rough 版本如何叠加

rough late-phase 不是另一套完全不同的恢复任务，而是：

- 保留原 `Rough` 地形生成与四阶段 terrain curriculum
- 保留 rough play 的 staged terrain 展示逻辑
- 保留 rough 专用 reward / termination 设计
- 在此基础上再叠加同一套 `late_phase_dr_disturbance`
- 继续使用同一套 conservative PPO finetune 配置

也就是说，rough 版额外保留的是：

- `motion_global_root_pos` 改为 `XY-only`
- 保留 `motion_global_root_z_vel`
- 保留 `motion_global_root_z_pos`
- 保留 rough 中更宽松的 `anchor_ori.threshold = 1.2`
- 继续移除 rough 下不合适的 `anchor_pos` / `ee_body_pos`
- 继续关闭 rough 里的 `push_robot`

因此 rough late-phase 任务更适合解决这种组合问题：

- 地形有轻微不平
- 空翻 / 翻滚后的姿态恢复本身又容易因为过冲或冲量不足失败

如果你的目标是实机后半段恢复，而部署地面并不完全理想，这个版本会比 flat late-phase 更接近真实问题。

### 2. late-phase 渐进式扰动

训练任务会在后半段开启一个 step event：

- `late_phase_dr_disturbance`

其强度不是在某一帧突然跳满，而是从后半段开始后逐步爬升。

当前关键参数位于：

- [late_phase_dr.py](/home/dp/czy/mjlab/src/mjlab/tasks/tracking/config/late_phase_dr.py)

基础 late-phase 调度参数：

- `late_phase_start_ratio = 0.42`
- `late_phase_onset_scale = 0.35`
- `late_phase_scale_power = 1.25`
- `stand_up_recovery_probability = 0.4`

这里的 `stand_up_recovery_probability` 很重要：

- 训练时不是每个 episode 的后半段都一定被打
- 默认只有 40% 的 episode 会触发一次恢复型扰动
- 剩下 60% 的 episode 仍然保持 clean late-phase

这样可以显著降低“策略默认预判后半段一定会出事”的过拟合趋势。

### 3. 两类恢复失败建模

当前不再把恢复失败主要建模成持续 body push，而是建模成两类“站起执行误差”：

#### overshoot

含义：

- 站起时前冲过大
- 身体前栽过头
- 下肢发力和控制幅度偏强

当前实现：

- 在 `165 +/- 2` 帧附近触发
- 对 root / pelvis 注入一次短时前栽方向的 `pitch angular-velocity kick`
- 同时对下肢施加轻度偏强的：
  - `effort scale`
  - `PD scale`
  - `joint_pos action scale`

默认基线参数：

- `stand_up_overshoot_effort_scale_range = (1.15, 1.27)`
- `stand_up_overshoot_pd_scale_range = (1.06, 1.15)`
- `stand_up_overshoot_action_scale_range = (1.225, 1.42)`
- `stand_up_overshoot_pitch_ang_vel_range = (1.05, 1.65)`
- `stand_up_pitch_ang_vel_kick_duration_steps = 3`

#### underpowered

含义：

- 站起时冲量不足
- 身体抬不起来或起立不够
- 下肢发力和控制幅度偏弱

当前实现：

- 在 `135 +/- 2` 帧附近触发
- 对 root / pelvis 注入一次短时后仰方向的 `pitch angular-velocity kick`
- 同时对下肢施加轻度偏弱的：
  - `effort scale`
  - `PD scale`
  - `joint_pos action scale`

默认基线参数：

- `stand_up_underpowered_effort_scale_range = (0.73, 0.88)`
- `stand_up_underpowered_pd_scale_range = (0.76, 0.88)`
- `stand_up_underpowered_action_scale_range = (0.58, 0.82)`
- `stand_up_underpowered_pitch_ang_vel_range = (-1.95, -1.35)`
- `stand_up_pitch_ang_vel_kick_duration_steps = 3`

### 4. 为什么不用位置错位和持续推力

这版明确避免两种不太符合问题本身的扰动：

- 强行 root 位置偏移
- 长时间持续把身体往前推

原因是它们更像“外部拖拽”或“瞬移错位”，而不是实机里常见的：

- 起身时身体角动量不对
- 发力过大
- 发力不足
- 关节控制输出略强或略弱

因此当前设计更强调：

- root/pelvis 的短时 pitch 角速度误差
- 下肢执行强弱偏差

### 5. 抑制单腿应激补偿

为了减少“clean 轨迹里也出现左脚蹬腿、单腿补偿”这类现象，当前 `Late-Phase-DR-Finetune` 任务额外加入了轻量 joint-space tracking reward：

- `motion_joint_pos`
  - `weight = 0.25`
  - `std = 0.5`
- `motion_joint_vel`
  - `weight = 0.1`
  - `std = 2.5`

它们不是主导项，但会提高“某一条腿单独发神经”这种策略的代价。

### 6. 更保守的 PPO finetune

为了避免强扰动二次训练把原始 nominal 轨迹写坏，当前 late-phase finetune 的 PPO 也做了进一步收紧：

- `learning_rate = 1e-4`
- `entropy_coef = 0.001`
- `desired_kl = 0.003`

相比之前，这一版更强调“在原 checkpoint 附近做小步微调”，而不是快速重写策略。

## Train / Play 参数一一对应

这版已经把 `train` 和 `play` 接到了同一个缩放 helper：

- [late_phase_dr.py](/home/dp/czy/mjlab/src/mjlab/tasks/tracking/config/late_phase_dr.py#L55)

因此：

- `train --late-phase-train-overshoot-scale 7.0`
- `play --late-phase-play-overshoot-scale 7.0`

会映射到同一套 overshoot 扰动参数。

同理：

- `train --late-phase-train-underpowered-scale 3.0`
- `play --late-phase-play-underpowered-scale 3.0`

也会映射到同一套 underpowered 扰动参数。

这一点已经有测试保证：

- [test_tracking_task.py](/home/dp/czy/mjlab/tests/test_tracking_task.py#L323)

## 当前可用 CLI

### 训练端

当前新增的训练参数位于：

- [train.py](/home/dp/czy/mjlab/src/mjlab/scripts/train.py#L28)

可直接使用：

- `--late-phase-train-overshoot-scale`
- `--late-phase-train-underpowered-scale`

### 评测端

当前新增的评测参数位于：

- [play.py](/home/dp/czy/mjlab/src/mjlab/scripts/play.py#L35)

可直接使用：

- `--simulate-late-phase-aggressive-dr True`
- `--late-phase-play-overshoot-scale`
- `--late-phase-play-underpowered-scale`
- `--late-phase-play-kick-duration-steps`

## 参考训练命令

### 1. Flat 默认强度训练

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Flat-Unitree-G1-New-Late-Phase-DR-Finetune \
  --checkpoint-file /home/dp/czy/mjlab/logs/rsl_rl/g1_tracking/2026-04-10_11-12-04/model_53000.pt \
  --env.commands.motion.motion-file /home/dp/czy/mjlab/datasets/npz/tiger_jump_g1_new.npz \
  --env.scene.num-envs 4096 \
  --agent.run-name tiger_jump_late_phase_dr_ft
```

### 2. Flat 推荐的稳健训练起点

当前更推荐把训练强度设得明显低于最终 stress test 评测强度，例如：

- `overshoot scale = 3.0`
- `underpowered scale = 1.5`

对应训练命令：

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Flat-Unitree-G1-New-Late-Phase-DR-Finetune \
  --checkpoint-file /home/dp/czy/mjlab/logs/rsl_rl/g1_tracking/2026-04-10_11-12-04/model_53000.pt \
  --env.commands.motion.motion-file /home/dp/czy/mjlab/datasets/npz/tiger_jump_g1_new.npz \
  --late-phase-train-overshoot-scale 3.0 \
  --late-phase-train-underpowered-scale 1.5 \
  --env.scene.num-envs 4096 \
  --agent.run-name tiger_jump_late_phase_dr_os3_up1p5_ft
```

### 3. Flat 按较强恢复分布训练

如果你当前观察下来觉得下面这组比较符合：

- `overshoot scale = 7.0`
- `underpowered scale = 3.0`

那么对应训练命令为：

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Flat-Unitree-G1-New-Late-Phase-DR-Finetune \
  --checkpoint-file /home/dp/czy/mjlab/logs/rsl_rl/g1_tracking/2026-04-10_11-12-04/model_53000.pt \
  --env.commands.motion.motion-file /home/dp/czy/mjlab/datasets/npz/tiger_jump_g1_new.npz \
  --late-phase-train-overshoot-scale 7.0 \
  --late-phase-train-underpowered-scale 3.0 \
  --env.scene.num-envs 4096 \
  --agent.run-name tiger_jump_late_phase_dr_os7_up3_ft
```

### 4. Rough 默认强度训练

如果你已经有 rough clean checkpoint，更推荐从 rough checkpoint 继续做 second-stage late-phase finetune：

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Rough-Unitree-G1-New-Late-Phase-DR-Finetune \
  --checkpoint-file /path/to/rough_checkpoint.pt \
  --env.commands.motion.motion-file /home/dp/czy/mjlab/datasets/npz/tiger_jump_g1_new.npz \
  --env.scene.num-envs 4096 \
  --agent.run-name tiger_jump_rough_late_phase_dr_ft
```

如果你当前还没有 rough checkpoint，也可以直接从 flat checkpoint 起步，只是同时学习 rough 地形和 late-phase 恢复，训练难度会更高：

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Rough-Unitree-G1-New-Late-Phase-DR-Finetune \
  --checkpoint-file /home/dp/czy/mjlab/logs/rsl_rl/g1_tracking/2026-04-10_11-12-04/model_53000.pt \
  --env.commands.motion.motion-file /home/dp/czy/mjlab/datasets/npz/tiger_jump_g1_new.npz \
  --env.scene.num-envs 4096 \
  --agent.run-name tiger_jump_rough_late_phase_dr_from_flat_ft
```

### 5. Rough 按较强恢复分布训练

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Rough-Unitree-G1-New-Late-Phase-DR-Finetune \
  --checkpoint-file /path/to/rough_checkpoint.pt \
  --env.commands.motion.motion-file /home/dp/czy/mjlab/datasets/npz/tiger_jump_g1_new.npz \
  --late-phase-train-overshoot-scale 7.0 \
  --late-phase-train-underpowered-scale 3.0 \
  --env.scene.num-envs 4096 \
  --agent.run-name tiger_jump_rough_late_phase_dr_os7_up3_ft
```

## 参考评测命令

### 1. Flat clean 完整动作

```bash
MUJOCO_GL=egl uv run play Mjlab-Tracking-Flat-Unitree-G1-New \
  --checkpoint-file /path/to/model.pt \
  --motion-file /home/dp/czy/mjlab/datasets/npz/tiger_jump_g1_new.npz \
  --num-envs 1
```

### 2. Flat late-phase 扰动评测

```bash
MUJOCO_GL=egl uv run play Mjlab-Tracking-Flat-Unitree-G1-New \
  --checkpoint-file /path/to/model.pt \
  --motion-file /home/dp/czy/mjlab/datasets/npz/tiger_jump_g1_new.npz \
  --simulate-late-phase-aggressive-dr True \
  --late-phase-play-overshoot-scale 7.0 \
  --late-phase-play-underpowered-scale 3.0 \
  --no-terminations True \
  --num-envs 1
```

注意：

- 训练端默认 `stand_up_recovery_probability = 0.4`
- `play` 端在打开 `--simulate-late-phase-aggressive-dr True` 后，默认会把 `stand_up_recovery_probability` 固定成 `1.0`

也就是说，训练时是“部分 episode 触发”，评测时是“打开后就稳定触发”，这样更方便做压力测试。

如果只想看 clean 成功率，不想看被打断后的恢复过程，可以去掉：

- `--simulate-late-phase-aggressive-dr True`

如果想按 termination 判定是否真的失败，可以去掉：

- `--no-terminations True`

### 3. Rough clean 完整动作

```bash
MUJOCO_GL=egl uv run play Mjlab-Tracking-Rough-Unitree-G1-New \
  --checkpoint-file /path/to/model.pt \
  --motion-file /home/dp/czy/mjlab/datasets/npz/tiger_jump_g1_new.npz \
  --num-envs 1
```

### 4. Rough late-phase 扰动评测

```bash
MUJOCO_GL=egl uv run play Mjlab-Tracking-Rough-Unitree-G1-New \
  --checkpoint-file /path/to/model.pt \
  --motion-file /home/dp/czy/mjlab/datasets/npz/tiger_jump_g1_new.npz \
  --simulate-late-phase-aggressive-dr True \
  --late-phase-play-overshoot-scale 7.0 \
  --late-phase-play-underpowered-scale 3.0 \
  --no-terminations True \
  --num-envs 1
```

rough play 和 flat play 的差别主要在于：

- rough 会继续按 checkpoint 对应课程阶段展示地形
- 即使打开 late-phase 扰动评测，也不会丢掉 rough 地形逻辑
- 因此这个命令更适合直接看“地形误差 + 恢复误差”共同存在时的表现

## 推荐评测对比

建议至少比较四组：

1. 基础 checkpoint，clean play
2. finetune checkpoint，clean play
3. finetune checkpoint，late-phase disturbed play
4. 基础 checkpoint，late-phase disturbed play

重点看：

- 前半段空翻是否仍稳定
- 后半段翻滚后是否更容易重新站稳
- overshoot 和 underpowered 两类失败是否都比基础 checkpoint 更稳
- clean 轨迹是否没有被明显破坏

如果你在做 rough late-phase，建议再额外比较两组：

1. rough clean checkpoint，rough clean play
2. rough late-phase finetune checkpoint，rough disturbed play

## 当前实现位置

和本任务直接相关的代码位置：

- 共享 late-phase 参数和缩放逻辑：
  - [late_phase_dr.py](/home/dp/czy/mjlab/src/mjlab/tasks/tracking/config/late_phase_dr.py)
- 站起过冲 / 冲量不足事件实现：
  - [events.py](/home/dp/czy/mjlab/src/mjlab/tasks/tracking/mdp/events.py)
- `play` 侧独立 overshoot / underpowered scale：
  - [play.py](/home/dp/czy/mjlab/src/mjlab/scripts/play.py)
- `train` 侧独立 overshoot / underpowered scale：
  - [train.py](/home/dp/czy/mjlab/src/mjlab/scripts/train.py)
- flat / rough late-phase 任务注册：
  - [g1/__init__.py](/home/dp/czy/mjlab/src/mjlab/tasks/tracking/config/g1/__init__.py)
  - [g1_new/__init__.py](/home/dp/czy/mjlab/src/mjlab/tasks/tracking/config/g1_new/__init__.py)
- flat / rough late-phase 环境配置：
  - [env_cfgs.py](/home/dp/czy/mjlab/src/mjlab/tasks/tracking/config/g1/env_cfgs.py)
  - [env_cfgs.py](/home/dp/czy/mjlab/src/mjlab/tasks/tracking/config/g1_new/env_cfgs.py)
- 一致性测试：
  - [test_tracking_task.py](/home/dp/czy/mjlab/tests/test_tracking_task.py)

## 总结

这版 `Late-Phase-DR-Finetune` 的核心，不是“只训练后半段”，而是：

- 保留整段动作分布
- 只在后半段增强恢复压力
- 用更接近执行误差的方式建模“过冲”和“冲量不足”
- 让 `train` 和 `play` 使用完全一致的缩放规则

因此它更适合作为一个第二阶段课程学习任务，用来解决“前半段空翻没问题，但翻滚后起不来”的真实部署问题。

而 rough 版则是在这个基础上进一步覆盖：

- 地形不平带来的姿态误差
- 后半段站起时的过冲 / 冲量不足
- 二者叠加后的恢复鲁棒性
