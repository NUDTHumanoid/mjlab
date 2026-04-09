# Beyondmimic 实物部署指南

> `origin: mzy` · `edited: czy`
>
> **适用平台：** Unitree G1 · **策略文件：** `policy.onnx` · **运行环境：** Docker + ROS 2 Humble

---

## 目录

- [Beyondmimic 实物部署指南](#beyondmimic-实物部署指南)
  - [目录](#目录)
  - [1. 快速参考](#1-快速参考)
  - [2. 前置准备（首次使用）](#2-前置准备首次使用)
    - [2.1 Docker 镜像导入](#21-docker-镜像导入)
    - [2.2 tmux 安装与配置](#22-tmux-安装与配置)
  - [3. 标准部署流程](#3-标准部署流程)
    - [Step 1 · 开机](#step-1--开机)
    - [Step 2 · 连接机载电脑](#step-2--连接机载电脑)
    - [Step 3 · 进入调试模式](#step-3--进入调试模式)
    - [Step 4 · 启动容器并加载策略](#step-4--启动容器并加载策略)
      - [4a · 启动容器](#4a--启动容器)
      - [4b · 拷贝策略文件](#4b--拷贝策略文件)
      - [4c · 进入容器并激活环境](#4c--进入容器并激活环境)
      - [4d · 启动控制节点](#4d--启动控制节点)
    - [Step 5 · 启动策略](#step-5--启动策略)
    - [Step 6 · 结束与关机](#step-6--结束与关机)
  - [4. 手柄按键速查](#4-手柄按键速查)
  - [5. tmux 使用手册](#5-tmux-使用手册)
    - [会话管理](#会话管理)
    - [窗格操作快捷键](#窗格操作快捷键)
    - [为什么用 tmux？](#为什么用-tmux)

---

## 1. 快速参考

| 项目 | 值 |
|---|---|
| 机载电脑 IP | `192.168.123.164` |
| 用户名 | `unitree` |
| 密码 | `123` |
| 容器名 | `beyondmimic_ctrl` |
| 镜像名 | `beyongdmimic_armkongzhidiceng:v0` |
| 策略文件路径（容器内） | `/root/colcon_ws/policy.onnx` |
| ROS 网络接口 | `eth0` |

---

## 2. 前置准备（首次使用）

本节内容**仅需执行一次**。环境已就绪时可直接跳至 [第 3 节](#3-标准部署流程)。

### 2.1 Docker 镜像导入

将镜像文件 `qyk_beyondmimic_diceng.tar` 拷贝至机载电脑，然后执行：

```bash
# 导入镜像（耗时较长，请耐心等待）
docker load -i qyk_beyondmimic_diceng.tar

# 确认镜像已导入
docker images
```

创建部署容器：

```bash
docker run -itd \
    -e DISPLAY=$DISPLAY \
    --name beyondmimic_ctrl \
    --net=host \
    --privileged \
    beyongdmimic_armkongzhidiceng:v0
```

> 容器创建成功后，后续每次部署直接 `docker start` 即可，**无需重复 `docker run`**。

### 2.2 tmux 安装与配置

tmux 可以保证 SSH 断连后远端进程继续运行，**强烈推荐**在首次连接后立即配置。

```bash
# 安装 tmux
sudo apt install tmux

# 创建配置文件；若尚未配置鼠标支持，再追加此行
#（点击切换分屏，拖拽调整大小，滚轮翻页）
touch ~/.tmux.conf
grep -qxF 'set -g mouse on' ~/.tmux.conf || echo 'set -g mouse on' >> ~/.tmux.conf

# 若 tmux 已在运行，立即使配置生效
tmux source-file ~/.tmux.conf
```

---

## 3. 标准部署流程

### Step 1 · 开机

1. **机器人：** 按住机器人**左腰处**电源键（点按 + 长按），等待启动完成。
2. **手柄：** 按住手柄**下侧**电源键（点按 + 长按），手柄自动与机器人配对。

> ⚠️ 请确保机器人处于安全站立或支撑姿态后再开机。

---

### Step 2 · 连接机载电脑

标准部署流程**全程通过 SSH 连接机载电脑**，不需要拓展坞。开始部署前，以及演示结束后准备停程序时，都需要先将网线接回机载电脑。

```bash
ssh unitree@192.168.123.164
# 密码：123
tmux new -s deploy
```

VSCode 用户安装 *Remote - SSH* 插件后可获得完整的图形化文件管理与终端体验。

若已存在会话，可直接恢复：

```bash
tmux ls
tmux attach -t deploy
```

> tmux 详细用法见 [第 5 节](#5-tmux-使用手册)。

参考宇树官方文档：
[宇树科技文档中心 · G1 开发者](https://support.unitree.com/home/zh/G1_developer/about_G1)

---

### Step 3 · 进入调试模式

机器人开机后默认处于**零力矩模式**，需依次执行以下操作：

| 操作 | 按键 | 效果 |
|---|---|---|
| 进入调试模式 | `L2 + R2` | 从零力矩切换至可调试状态 |
| 诊断动作（可选） | `L2 + A` | G1 抬起手臂，确认关节响应正常 |
| 进入阻尼状态 | `L2 + B` | G1 放下手臂，关节进入阻尼保持 |

> 建议在加载策略前先执行一次诊断动作，确认机器人运动正常。

---

### Step 4 · 启动容器并加载策略

#### 4a · 启动容器

```bash
# 查看容器状态
docker ps -a

# 启动容器（容器已存在时使用此命令）
docker start beyondmimic_ctrl
```

#### 4b · 拷贝策略文件

将策略文件复制进容器。**此命令在宿主机（机载电脑）终端执行，不要在容器内执行：**

```bash
docker cp /home/unitree/policy.onnx \
    beyondmimic_ctrl:/root/colcon_ws/policy.onnx
```

> 也可以直接将 `.onnx` 文件拖入 VSCode 的容器文件管理器至 `/root/colcon_ws/` 目录。

#### 4c · 进入容器并激活环境

```bash
# 进入容器
docker exec -it beyondmimic_ctrl bash

# 在容器终端内依次执行：
cd /root/colcon_ws/
source /opt/ros/humble/setup.bash   # 激活系统 ROS 2 环境
source install/setup.bash           # 激活工作空间覆盖环境
```

#### 4d · 启动控制节点

```bash
ros2 launch motion_tracking_controller real.launch.py \
    network_interface:=eth0 \
    policy_path:=policy.onnx
```

> 指令运行成功后，终端会持续输出日志，G1 进入**待机状态**。若演示阶段需要脱离有线网络，可在此时拔掉网线。

---

### Step 5 · 启动策略

| 操作 | 按键 | 说明 |
|----|----|----|
| 启动策略 | `R1 + A` | G1 开始执行策略动作，有可能机器人初始姿态不能保证站立，注意安排人员在后端拉住机器人保持稳定姿态 |
| 暂停 / 回待机 | `L1 + A` | 停止策略，回到待机状态 |
| **紧急停止** | `B` | 立即停止所有动作，优先级最高 |

> ⚠️ 演示期间操作人员应全程手持手柄，保持紧急停止按键可随时触发。

---

### Step 6 · 结束与关机

1. 若策略仍在运行，先按 `L1 + A` 使 G1 回到待机状态。
2. 将网线重新接回机载电脑，SSH 登录并恢复 `tmux` 会话：

```bash
ssh unitree@192.168.123.164
tmux attach -t deploy
```

3. 在容器终端按 `Ctrl+C` 停止 ROS 节点，并确认程序已退出。
4. 按 `L2 + R2` 重新进入调试模式。
5. 按 `L2 + B` 使 G1 进入阻尼模式，机器人放下手臂。
6. 确认 G1 姿态稳定后，**短按 + 长按**电源键安全关机。

---

## 4. 手柄按键速查

| 按键组合 | 时机 | 效果 |
|---|---|---|
| `L2 + R2` | 开机后 / 结束后 | 进入 / 重新进入调试模式 |
| `L2 + A` | 调试模式中 | 诊断动作（抬臂） |
| `L2 + B` | 调试模式中 | 进入阻尼状态（放臂） |
| `R1 + A` | 待机状态中 | 启动策略 |
| `L1 + A` | 策略运行中 | 停止策略，回待机 |
| `B` | 任意时刻 | **紧急停止** |

---

## 5. tmux 使用手册

### 会话管理

```bash
tmux new -s deploy     # 新建名为 deploy 的会话
tmux ls                # 列出所有会话
tmux attach -t deploy  # 重新连接会话（断线恢复用）
exit  或  Ctrl+D       # 关闭当前会话
```

### 窗格操作快捷键

所有快捷键均先按 `Ctrl+B`，松手后再按对应键：

| 功能 | 按键序列 |
|---|---|
| 左右分屏 | `Ctrl+B` → `%` |
| 上下分屏 | `Ctrl+B` → `"` |
| 切换窗格 | `Ctrl+B` → 方向键 |
| 关闭当前窗格 | `Ctrl+D` 或输入 `exit` |
| 进入历史翻阅模式 | `Ctrl+B` → `[` |
| 退出历史翻阅模式 | `q` |

> **鼠标滚轮无法直接翻历史**：需先按 `Ctrl+B → [` 进入复制模式，再用滚轮或 PageUp/PageDown 翻看。

### 为什么用 tmux？

SSH 直连时，一旦网络断开，远端正在运行的进程会立即终止。tmux 会话运行在**服务器端**，本地断连不影响其运行状态。重新 SSH 登录后执行 `tmux attach -t deploy` 即可恢复现场。
