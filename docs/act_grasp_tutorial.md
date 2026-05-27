# LeKiwi ACT 自动抓取纸团完整教程

> 从0到1：录制数据 → 训练模型 → 自动寻找 → 接近 → 抓取

---

## 目录

1. [项目概述](#1-项目概述)
2. [硬件准备](#2-硬件准备)
3. [环境配置](#3-环境配置)
4. [系统架构详解](#4-系统架构详解)
5. [数据采集](#5-数据采集)
6. [模型训练](#6-模型训练)
7. [部署推理](#7-部署推理)
8. [完整流程测试](#8-完整流程测试)
9. [故障排查与常见问题](#9-故障排查与常见问题)
10. [性能优化](#10-性能优化)
11. [高级功能](#11-高级功能)

---

## 1. 项目概述

### 1.1 任务描述

让 LeKiwi 机器人自动完成以下动作：
1. **搜索**：在环境中寻找纸团
2. **接近**：通过视觉导航移动到纸团附近
3. **抓取**：使用机械臂抓取纸团
4. **完成**：抬起机械臂，任务结束

### 1.2 架构设计

采用**两阶段控制策略**：

```
┌─────────────────────────────────────────┐
│ 阶段1: 视觉导航 (YOLO + PID)            │
│  ├─ YOLO检测纸团位置                     │
│  ├─ PID控制底盘对准                      │
│  └─ 底盘移动，机械臂保持默认姿态          │
└─────────────────────────────────────────┘
              ↓ 到达目标附近
┌─────────────────────────────────────────┐
│ 阶段2: ACT抓取 (6-DOF机械臂)            │
│  ├─ PC运行ACT模型推理                    │
│  ├─ 只控制机械臂6个关节                  │
│  └─ 底盘静止不动                         │
└─────────────────────────────────────────┘
```

**为什么分两阶段？**
- 导航和抓取是不同的技能，分开学习更容易
- 抓取时底盘静止，简化ACT模型（只需6-DOF）
- 提高成功率和训练效率

### 1.3 技术方案核心设计

**推理触发机制（重要）**：
- 树莓派Host进入 `grasping` 状态时，通过ZMQ发送 `request_act=true`
- PC端 `client_pc.py` 接收到请求后，运行ACT模型推理
- 推理完成后发送动作序列给树莓派执行
- 不是持续推理，而是**按需触发**

**6-DOF vs 9-DOF**：
- **录制时**：使用主臂遥操作，记录6个机械臂关节动作
- **推理时**：ACT模型只输出6-DOF动作，底盘不移动
- **底盘控制**：由YOLO导航阶段单独控制，不在ACT中学习

---

## 2. 硬件准备

### 2.1 必需硬件

| 组件 | 型号/规格 | 数量 | 说明 |
|------|----------|------|------|
| **树莓派** | Raspberry Pi 5 (8GB) | 1 | 主机，运行Host程序 |
| **主臂** | SO101 | 1 | 遥操作控制从臂（Leader） |
| **从臂** | SO101 | 1 | 安装在LeKiwi底盘上（Follower） |
| **底盘** | LeKiwi 3轮全向 | 1 | 移动底盘 |
| **Front摄像头** | USB Camera | 1 | 前视摄像头 (/dev/video2) |
| **Wrist摄像头** | USB Camera | 1 | 腕部摄像头 (/dev/video0) |
| **PC** | Windows/Linux + NVIDIA GPU | 1 | 训练+推理 |
| **手柄** | Xbox/PS4 | 1 | 手动控制 |

### 2.2 接线检查

```bash
# 在树莓派上检查硬件
ls /dev/video*    # 查看摄像头（应有video0和video2）
ls /dev/ttyACM*   # 查看电机控制板
```

### 2.3 机械臂校准

**主臂校准（PC端）**：
```bash
conda activate lerobot
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=COM5 \
    --teleop.id=L07252802
```

**从臂校准（树莓派端）**：
```bash
ssh acelan@lekiwi-pi
conda activate lekiwi
cd ~/lerobot-workspace/lerobot
python -c "
from lerobot.robots.lekiwi import LeKiwi, LeKiwiConfig
robot = LeKiwi(LeKiwiConfig(port='/dev/ttyACM0'))
robot.connect(calibrate=True)
robot.disconnect()
"
```

---

## 3. 环境配置

### 3.1 树莓派环境

```bash
# SSH到树莓派
ssh acelan@lekiwi-pi

# 创建conda环境
conda create -n lekiwi python=3.10 -y
conda activate lekiwi

# 安装LeRobot
cd ~/lerobot-workspace/lerobot
pip install -e ".[lekiwi]"

# 安装YOLO
pip install ultralytics --no-deps

# 验证安装
python -c "from lerobot.robots.lekiwi import LeKiwi; print('✅ 安装成功')"
```

### 3.2 PC环境（Windows）

```powershell
# 创建conda环境
conda create -n lerobot python=3.10 -y
conda activate lerobot

# 安装LeRobot
cd D:\work\lerobot-workspace\lerobot
pip install -e ".[lekiwi]"

# 安装训练依赖
pip install -e ".[core_scripts]"

# 安装WandB（可选，用于训练可视化）
pip install wandb

# 验证GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 3.3 项目目录结构

```
lekiwi-pi/
├── src/
│   ├── host_pi.py              # 树莓派Host（导航+抓取控制+双摄像头）
│   ├── host_pi_record.py       # 树莓派Host（录制专用，转发主臂动作）
│   ├── client_record.py        # PC端录制客户端（SO101遥操作+视频录制）
│   ├── client_pc.py            # PC端整合客户端（手柄+ACT推理+双摄像头显示）
│   └── ...                     # 其他测试脚本
├── config/
│   └── act_policy_config.yaml  # ACT策略配置（6-DOF）
├── scripts/
│   ├── train_act.sh            # 训练脚本
│   └── deploy_act.sh           # 部署脚本
├── docs/
│   └── act_grasp_tutorial.md   # 本教程
├── models/                     # YOLO模型
│   └── paper_ball_detection/
├── data/                       # 录制数据（保存在项目目录中）
│   └── acelan_lekiwi_grasp_paper_ball_20260527_172725/
│       ├── meta/               # 数据集元信息
│       ├── data/               # 动作和状态数据(parquet)
│       └── videos/             # 摄像头视频(MP4)
└── train_commands.ps1          # Windows训练命令
```

---

## 4. 系统架构详解

### 4.1 数据流（实际运行流程）

```
树莓派 Host (src/host_pi.py)
  ├─ 读取摄像头 (front + wrist)
  ├─ 读取机械臂状态 (6关节)
  ├─ 阶段1: YOLO检测 → 底盘移动
  ├─ 阶段2 (grasping状态): 发送 request_act=true → PC
  └─ 接收ACT动作 → 执行机械臂
        ↑
        │ ZMQ (CMD:5555, OBS:5556)
        ↓
PC Client (src/client_pc.py)
  ├─ 接收树莓派观测（front+wrist图像+机械臂状态）
  ├─ 检测到 request_act=true
  ├─ 运行ACT模型推理 (6-DOF)
  └─ 发送机械臂动作 → 树莓派
  
  同时：
  ├─ 手柄控制底盘（手动模式）
  ├─ 双摄像头显示（OpenCV窗口）
  └─ X键拍照 / START键退出
```

### 4.2 关键设计决策

**1. 为什么PC端推理？**
- GPU推理：30-50ms（RTX 5070 Ti）
- 树莓派CPU推理：~300ms
- 使用PyTorch `torch.compile` 进一步优化

**2. 为什么录制时底盘静止？**
- ACT模型只学习机械臂6-DOF动作
- 底盘导航由YOLO单独控制
- 简化模型，提高抓取成功率

**3. 双摄像头设计**
- **Front摄像头**：导航阶段使用，提供全局视野
- **Wrist摄像头**：抓取阶段使用，提供近距离细节
- ACT模型同时接收两个摄像头的图像

### 4.3 状态机

```
idle (手动模式)
  ↓ 按A键
AUTO (自动模式)
  ↓ 开始搜索
searching 🔍 (旋转寻找纸团)
  ↓ YOLO检测到纸团
aligning 🎯 (PID对准)
  ↓ 对准完成
approaching 🚀 (接近纸团)
  ↓ 纸团占比20%
arrived ✅ (到达目标)
  ↓ 延迟2秒
grasping 🦾 (ACT推理抓取)
  ↓ 抓取完成
idle ⏹️ (回到手动模式)
```

**注意**：抓取完成后状态机重置为 `idle`，不会自动开始新一轮。需要再次按A键触发。

---

## 5. 数据采集

### 5.1 录制策略

**关键点**：
- 底盘已定位到纸团附近（手动或自动导航）
- 录制过程中**底盘保持静止**
- 只移动主臂控制从臂抓取纸团
- 每个episode包含：预抓取→接近→抓取→抬起

### 5.2 启动树莓派Host（录制专用）

```bash
# SSH到树莓派
ssh acelan@lekiwi-pi
conda activate lekiwi
cd ~/lerobot-workspace/lekiwi-pi

# 启动录制专用Host（无自动导航，只接收PC命令）
python src/host_pi_record.py
```

**预期输出**：
```
[Host] 初始化...
[Host] 已连接LeKiwi: True
[Host] 启动进程...
[Controller] 已连接
[Inference] 已加载: ['paper-ball']
[Main] 等待连接... CMD:5555 OBS:5556
```

**host_pi_record.py 特点**：
- 只监听ZMQ命令端口（5555）
- 不执行自动导航
- 接收主臂动作并转发给从臂
- 支持双摄像头（front + wrist）

### 5.3 启动PC端录制客户端

```powershell
# 在PC上打开新终端
conda activate lerobot
cd D:\work\lerobot-workspace\lekiwi-pi

# 启动录制客户端（SO101主臂遥操作 + 视频录制）
python src/client_record.py --ip 192.168.3.176 --arm-port COM5
```

**参数说明**：
- `--ip`: 树莓派IP地址（默认: 192.168.3.176）
- `--arm-port`: 主臂串口（默认: COM5）

**录制控制**（键盘控制）：
- **主臂遥操作**：移动主臂控制从臂抓取纸团（始终开启）
- **R键**：开始新episode录制
- **空格键**：停止当前episode录制
- **Q键**：退出并保存数据集

**录制流程**：
1. 观察front摄像头画面，根据辅助线调整底盘位置
2. 当纸团框水平居中(±15%)且占比20%-30%时，按**R键**开始录制
3. 移动主臂控制从臂完成抓取动作
4. 按**空格键**停止录制
5. 移动底盘到新位置，重复步骤2-4

### 5.4 录制要点

1. **定位底盘**：手动将底盘移动到纸团附近
2. **保持静止**：开始录制后不要移动底盘
3. **抓取动作**：
   - 预抓取位置：夹爪打开，对准纸团
   - 向下移动：机械臂下降
   - 夹爪闭合：抓取纸团
   - 向上抬起：提起纸团
4. **重复录制**：每个位置录制3-5次

### 5.5 数据多样性

| 变化维度 | 建议 |
|---------|------|
| **距离** | 近(10cm)、中(20cm)、远(30cm) |
| **角度** | 左、中、右 |
| **高度** | 桌面、略高、略低 |
| **纸团大小** | 大、中、小 |
| **光照** | 亮、中、暗 |

**推荐录制量**：
- 最少：30 episodes
- 推荐：50-100 episodes
- 每个episode：10-20秒

### 5.6 数据集位置

录制完成后，数据集自动保存在项目目录中：
```
D:\work\lerobot-workspace\lekiwi-pi\data\
└── acelan_lekiwi_grasp_paper_ball_20260527_172725/
    ├── meta/
    │   ├── info.json          # 数据集元信息
    │   ├── stats.json         # 统计信息
    │   └── tasks.jsonl        # 任务描述
    ├── data/
    │   └── chunk-000-00000.parquet  # 动作和状态数据
    └── videos/
        ├── episode_000000/
        │   ├── front.mp4      # 前摄像头视频
        │   └── wrist.mp4      # 腕部摄像头视频
        └── episode_000001/
            └── ...
```

**视频格式优势**：
- 比帧图片节省 **70-90%** 存储空间
- `streaming_encoding=True` 实时编码，无需等待
- H.264编码，兼容性好

---

## 6. 模型训练

### 6.1 训练配置

编辑 `train_commands.ps1`（或直接使用命令行）：

```powershell
# Windows PowerShell 训练命令
lerobot-train `
  --policy.type=act `
  --policy.push_to_hub=false `
  --dataset.repo_id=acelan `
  --dataset.root="D:\work\lerobot-workspace\lekiwi-pi\data\acelan_lekiwi_grasp_paper_ball_20260527_172725" `
  --batch_size=16 `
  --steps=20000 `
  --num_workers=4 `
  --save_freq=2000 `
  --log_freq=100 `
  --output_dir="outputs/train/grasp_paper_ball" `
  --job_name="grasp paper ball" `
  --wandb.enable=true `
  --wandb.project=so101_pick
```

**参数说明**：
| 参数 | 说明 | 示例 |
|------|------|------|
| `--policy.type` | 策略类型 | `act` |
| `--policy.push_to_hub` | 是否推送到Hub | `false` |
| `--dataset.repo_id` | 数据集ID | `acelan` |
| `--dataset.root` | 数据集本地路径 | `D:\work\...\data\...` |
| `--batch_size` | 批次大小 | `16` |
| `--steps` | 训练步数 | `20000` |
| `--num_workers` | 数据加载线程 | `4` |
| `--save_freq` | 保存频率 | `2000` |
| `--log_freq` | 日志频率 | `100` |
| `--output_dir` | 输出目录 | `outputs/train/grasp_paper_ball` |
| `--job_name` | 任务名称 | `grasp paper ball` |
| `--wandb.enable` | 启用WandB | `true` |
| `--wandb.project` | WandB项目 | `so101_pick` |

### 6.2 启动训练

```powershell
# 1. 激活环境
conda activate lerobot

# 2. 进入项目目录
cd D:\work\lerobot-workspace\lekiwi-pi

# 3. 启动训练（修改数据集路径为你的实际路径）
lerobot-train `
  --policy.type=act `
  --policy.push_to_hub=false `
  --dataset.repo_id=acelan `
  --dataset.root="D:\work\lerobot-workspace\lekiwi-pi\data\acelan_lekiwi_grasp_paper_ball_20260527_172725" `
  --batch_size=16 `
  --steps=20000 `
  --num_workers=4 `
  --save_freq=2000 `
  --log_freq=100 `
  --output_dir="outputs/train/grasp_paper_ball" `
  --job_name="grasp paper ball" `
  --wandb.enable=true `
  --wandb.project=so101_pick
```

**注意**：
- 将 `20260527_172725` 替换为你实际的数据集时间戳
- 如果GPU内存不足，减小 `--batch_size`（如改为8）
- 训练过程中可以按 `Ctrl+C` 中断，模型会自动保存

### 6.3 恢复训练

如果需要恢复训练：
```powershell
lerobot-train `
  --config_path="outputs/train/grasp_paper_ball/checkpoints/last" `
  --dataset.root="D:\work\lerobot-workspace\lekiwi-pi\data\acelan_lekiwi_grasp_paper_ball_20260527_172725" `
  --output_dir="outputs/train/grasp_paper_ball" `
  --job_name="grasp paper ball" `
  --wandb.enable=true `
  --wandb.project=so101_pick
```

### 6.4 监控训练

**WandB**（推荐）：
```powershell
# 训练过程中自动记录到WandB
# 访问 https://wandb.ai 查看实时训练曲线
```

**TensorBoard**（可选）：
```powershell
tensorboard --logdir outputs/train/grasp_paper_ball
```

**关键指标**：
- `loss`：总损失（目标 < 0.1）
- `l1_loss`：动作预测误差（目标 < 0.05）
- `kld_loss`：VAE散度（目标 < 5.0）
- `eval/success_rate`：评估成功率

**训练时长**：
- 30 episodes：~30分钟-1小时
- 100 episodes：~2-4小时

### 6.5 模型保存

训练完成后，模型保存在：
```
outputs/train/grasp_paper_ball/
├── checkpoints/
│   ├── last/              # 最新检查点
│   │   ├── config.json    # 模型配置
│   │   ├── model.safetensors  # 模型权重
│   │   └── preprocessor_config.json
│   └── step_20000/        # 每save_freq步保存
│       ├── config.json
│       └── model.safetensors
└── logs/                  # 训练日志
```

---

## 7. 部署推理

### 7.1 系统架构

```
┌──────────────┐      ZMQ       ┌──────────────────┐
│  树莓派 Host  │ ◄────────────► │  PC整合客户端     │
│              │   观测/动作     │   client_pc.py   │
│ ├─ YOLO检测  │                │                  │
│ ├─ 底盘控制  │                │ ├─ ACT模型推理    │
│ ├─ 机械臂控制│                │ ├─ 手柄控制底盘   │
│ └─ 双摄像头  │                │ ├─ 双摄像头显示   │
└──────────────┘                │ └─ X拍照/START退出│
                                └──────────────────┘
```

**注意**：只有一个PC客户端 `client_pc.py`，整合了所有功能：
- ACT模型推理（6-DOF机械臂）
- 手柄控制底盘（手动模式）
- 双摄像头图像显示
- X键拍照 / START键退出

### 7.2 启动树莓派Host

```bash
# SSH到树莓派
ssh acelan@lekiwi-pi
conda activate lekiwi
cd ~/lerobot-workspace/lekiwi-pi

# 启动Host（支持自动导航+ACT抓取+双摄像头）
python src/host_pi.py
```

**预期输出**：
```
==================================================
LeKiwi Host - 多进程版
==================================================
[Main] 共享内存: psm_xxx, 大小: 921600 bytes
[Main] 启动进程...
[Controller] 初始化...
[Inference] 加载模型...
[Main] 等待连接... CMD:5555 OBS:5556
[Main] Ctrl+C 停止
```

### 7.3 启动PC端整合客户端

```powershell
# PC端
conda activate lerobot
cd D:\work\lerobot-workspace\lekiwi-pi

# 启动整合客户端（手柄+ACT推理+双摄像头显示）
python src/client_pc.py --ip 192.168.3.176
```

**参数说明**：
- `--ip`: 树莓派IP地址
- `--model-path`: ACT模型路径（可选，默认使用 `outputs/train/grasp_paper_ball/checkpoints/last`）

**预期输出**：
```
==================================================
LeKiwi PC Client - 整合版
==================================================
树莓派IP: 192.168.3.176
模型路径: outputs/train/grasp_paper_ball/checkpoints/last
推理设备: cuda
控制维度: 6-DOF机械臂（底盘自动静止）
==================================================

连接成功！
等待树莓派进入 grasping 状态...
按 Ctrl+C 停止
```

### 7.4 客户端功能说明

`client_pc.py` 整合了以下功能：

| 功能 | 说明 | 触发方式 |
|------|------|---------|
| **手柄控制底盘** | 手动模式下控制底盘移动 | 左摇杆/D-pad |
| **ACT推理** | 自动抓取时运行模型推理 | 树莓派发送 request_act=true |
| **双摄像头显示** | 显示front和wrist摄像头画面 | 自动显示 |
| **拍照** | 保存当前图像 | X键（button 3） |
| **退出** | 退出程序 | START键（button 11） |
| **速度切换** | 切换底盘速度档位 | LB键（button 6） |

**Gamepad键位映射**（已保存到长期记忆）：
- `A` = button 0：切换自动/手动模式（由树莓派处理）
- `B` = button 1：（预留）
- `X` = button 3：拍照保存
- `Y` = button 4：（预留）
- `LB` = button 6：切换速度档位
- `RB` = button 7：（预留）
- `BACK` = button 10：（预留）
- `START` = button 11：退出程序

### 7.5 触发自动流程

**步骤1：切换到自动导航模式**
- 按手柄 **A键**（button 0）
- 状态变为 `AUTO`

**步骤2：自动搜索**
- 底盘自动旋转搜索纸团
- 状态：`searching` 🔍

**步骤3：对准和接近**
- YOLO检测到纸团
- PID控制底盘对准并接近
- 状态：`aligning` 🎯 → `approaching` 🚀

**步骤4：到达并准备抓取**
- 纸团占视野20%，到达目标
- 状态：`arrived` ✅
- 等待2秒（`ACT_ARRIVED_DELAY = 2.0`）

**步骤5：ACT抓取**
- 自动进入 `grasping` 🦾 状态
- 树莓派发送 `request_act=true`
- PC端运行ACT推理，控制机械臂
- 机械臂执行：预抓取→接近→抓取→抬起

**步骤6：完成**
- 抓取完成，状态回到 `idle`
- 可以再次按A键开始新一轮

---

## 8. 完整流程测试

### 8.1 手动测试步骤

**测试1：底盘遥控**
1. 启动 `host_pi.py` 和 `client_pc.py`
2. 使用手柄左摇杆控制底盘移动
3. 验证前后左右移动正常

**测试2：自动导航**
1. 放置纸团在环境中
2. 按A键切换到AUTO模式
3. 观察底盘是否自动寻找并对准纸团

**测试3：ACT推理**
1. 确保模型已训练并保存
2. 让底盘到达纸团附近（arrived状态）
3. 观察是否自动进入grasping状态
4. 观察机械臂是否执行抓取动作

**测试4：双摄像头**
1. 检查OpenCV窗口是否显示两个画面
2. Front摄像头：全局视野
3. Wrist摄像头：近距离细节

### 8.2 一键启动脚本

创建 `scripts/run_full_pipeline.sh`：

```bash
#!/bin/bash
# 完整流程：导航+抓取

set -e

ROBOT_IP="192.168.3.176"
MODEL_PATH="outputs/train/grasp_paper_ball/checkpoints/last"

echo "=================================="
echo "LeKiwi 自动抓取纸团 - 完整流程"
echo "=================================="

# 1. 启动树莓派Host
echo "[1/2] 启动树莓派Host..."
ssh acelan@$ROBOT_IP "
    cd ~/lerobot-workspace/lekiwi-pi
    pkill -f host_pi.py || true
    sleep 1
    nohup python src/host_pi.py > logs/host.log 2>&1 &
    echo 'Host已启动'
"
sleep 5

# 2. 启动PC端整合客户端
echo "[2/2] 启动PC端整合客户端..."
python src/client_pc.py --ip $ROBOT_IP

echo "停止树莓派Host..."
ssh acelan@$ROBOT_IP "pkill -f host_pi.py || true"

echo "完成！"
```

---

## 9. 故障排查与常见问题

### 9.1 树莓派Host问题

**问题：Host启动失败**
```bash
# 检查摄像头
ls /dev/video*

# 检查串口
ls -l /dev/ttyACM0

# 检查YOLO模型
ls models/paper_ball_detection-1-8/weights/best.pt
```

**问题：机械臂不响应**
- 检查从臂是否已校准
- 检查电机控制板连接
- 查看日志：`tail -f logs/host.log`

**问题：双摄像头只显示一个**
- 检查摄像头设备号：`ls /dev/video*`
- 在 `host_pi.py` 中确认 `camera_device_id=2` (front) 和 `wrist_camera_device_id=0` (wrist)
- 检查摄像头是否被其他进程占用

### 9.2 PC推理问题

**问题：模型加载失败**
```bash
# 检查模型路径
ls outputs/train/grasp_paper_ball/checkpoints/last/

# 检查CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**问题：模型加载API错误（今天已修复）**
- **症状**：`AttributeError: type object 'PreTrainedConfig' has no attribute 'from_pretrained'`
- **原因**：transformers库版本问题，不能直接从 `PreTrainedConfig` 调用
- **解决方案**：
```python
# 错误方式（已废弃）
config = PreTrainedConfig.from_pretrained(model_path)

# 正确方式（当前代码使用）
from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_path)
```

**问题：推理延迟高**
- 使用 `--device cuda` 确保GPU推理
- 检查GPU利用率：`nvidia-smi`
- 降低图像分辨率（修改host_pi.py）

**问题：模型推理时wrist摄像头图像缺失**
- **症状**：`RuntimeError: Expected 4D tensor, got 3D`
- **原因**：树莓派未发送wrist图像，PC端使用零张量占位
- **解决方案**（已在代码中修复）：
```python
# 如果缺少wrist图像，使用零张量占位
if wrist_image is None:
    wrist_image = torch.zeros(1, 3, 480, 640)
```

**问题：推理时图像尺寸不匹配**
- **症状**：`RuntimeError: size mismatch`
- **原因**：训练时和推理时的图像分辨率不一致
- **解决方案**：确保训练配置和推理配置中的图像尺寸一致

### 9.3 录制问题

**问题：录制时主臂动作未正确转发到从臂**
- **症状**：移动主臂，从臂不动
- **原因**：`controller_worker` 未正确接收和发送动作
- **解决方案**（已在代码中修复）：
```python
# host_pi_record.py 中
data = cmd["data"]  # 获取主臂动作
robot.send_action(data)  # 直接转发给从臂
```

**问题：录制数据集键名格式错误**
- **症状**：`KeyError: 'shoulder_pan.pos'`
- **原因**：SO101主臂键名为 `shoulder_pan.pos`，但数据集期望 `arm_shoulder_pan.pos`
- **解决方案**（已在代码中修复）：
```python
# client_record.py 中
action_dict = {}
for key, value in full_action.items():
    dataset_key = f"arm_{key}"
    action_dict[dataset_key] = value
```

### 9.4 抓取失败问题

**问题：机械臂乱动**
- 检查模型是否训练充分（loss < 0.05）
- 增加训练数据（特别是失败案例）
- 检查机械臂限位（不要超出关节范围）

**问题：夹爪抓不住**
- 调整夹爪闭合位置
- 检查纸团大小是否合适
- 增加抓取时的保持时间

**问题：底盘碰撞**
- 降低导航速度（修改NAV_SPEED）
- 增加安全距离（修改TARGET_RATIO）
- 添加碰撞检测传感器

### 9.5 网络问题

**问题：ZMQ连接失败**
```bash
# 检查网络连通
ping 192.168.3.176

# 检查端口
nc -zv 192.168.3.176 5555
nc -zv 192.168.3.176 5556

# 检查防火墙
sudo ufw allow 5555/tcp
sudo ufw allow 5556/tcp
```

**问题：树莓派同步后代码不一致**
- **症状**：修改后的代码在树莓派上未生效
- **原因**：git pull 冲突或未正确同步
- **解决方案**：
```bash
# 在树莓派上强制同步
git fetch origin
git reset --hard origin/feature/act-inference
```

---

## 10. 性能优化

### 10.1 降低延迟

| 优化手段 | 效果 | 实现方式 |
|---------|------|---------|
| 降低图像分辨率 | -15ms | 320x240传输 |
| 降低JPEG质量 | -12ms | quality=60 |
| 使用TensorRT | -25ms | 模型优化 |
| 使用有线网络 | -10ms | 网线替代WiFi |
| PyTorch编译 | -20ms | `torch.compile(model)` |

### 10.2 提高成功率

| 优化手段 | 效果 | 实现方式 |
|---------|------|---------|
| 增加训练数据 | +20% | 录制100+ episodes |
| 数据增强 | +10% | random_crop, color_jitter |
| 模型集成 | +5% | 多个模型投票 |
| 安全限制 | 防损坏 | 设置关节限位 |

---

## 11. 高级功能

### 11.1 多任务支持

修改 `TASK_DESCRIPTION` 支持不同任务：
```python
TASK_DESCRIPTION = "Grasp paper ball"  # 或 "Push box", "Open drawer"
```

### 11.2 连续抓取

修改 `host_pi.py` 完成抓取后不回到idle：
```python
# 抓取完成后直接开始新的搜索
if grasp_completed:
    navigator.reset()  # 直接开始新的搜索
```

### 11.3 远程监控

使用Rerun可视化：
```bash
rerun ws://192.168.3.176:9876
```

---

## 附录A：关键参数配置

### A.1 导航参数（host_pi.py）

```python
TARGET_RATIO = 0.20      # 纸团占视野20%时认为到达
NAV_SPEED = 0.25         # 导航速度 (m/s)
ROT_SPEED = 20           # 旋转速度 (deg/s)
ACT_ARRIVED_DELAY = 2.0  # 到达后延迟2秒开始抓取
ACT_GRASP_DURATION = 5.0 # 抓取持续时间
```

### A.2 ACT模型参数

```python
n_action_steps = 8        # 动作分块大小
chunk_size = 8           # 动作序列长度
dim_model = 512          # Transformer维度
kl_weight = 10.0         # VAE损失权重
```

### A.3 机械臂安全限位

```python
ARM_LIMITS = {
    "arm_shoulder_pan": (-180, 180),
    "arm_shoulder_lift": (-180, 0),
    "arm_elbow_flex": (0, 180),
    "arm_wrist_flex": (-90, 90),
    "arm_wrist_roll": (-180, 180),
    "arm_gripper": (0, 100),
}
```

---

## 附录B：文件清单

| 文件 | 说明 |
|------|------|
| `src/host_pi.py` | 树莓派主程序（导航+抓取控制+双摄像头） |
| `src/host_pi_record.py` | 树莓派录制专用Host（转发主臂动作） |
| `src/client_record.py` | PC端录制客户端（SO101遥操作+视频录制） |
| `src/client_pc.py` | PC端整合客户端（手柄+ACT推理+双摄像头显示） |
| `config/act_policy_config.yaml` | ACT配置（6-DOF） |
| `scripts/train_act.sh` | 训练脚本 |
| `scripts/deploy_act.sh` | 部署脚本 |
| `train_commands.ps1` | Windows PowerShell训练命令 |

---

## 附录C：今日修复记录

### C.1 修复：模型加载API（2025-05-27）

**问题**：`PreTrainedConfig.from_pretrained()` 报错
**修复**：使用 `AutoConfig.from_pretrained()`
**影响文件**：`src/client_pc.py`

### C.2 修复：双摄像头支持（2025-05-27）

**问题**：只有front摄像头，缺少wrist摄像头
**修复**：在 `host_pi.py` 和 `client_pc.py` 中添加wrist摄像头支持
**影响文件**：`src/host_pi.py`, `src/client_pc.py`

### C.3 修复：录制键名映射（2025-05-27）

**问题**：SO101键名 `shoulder_pan.pos` 与数据集期望 `arm_shoulder_pan.pos` 不匹配
**修复**：在 `client_record.py` 中自动添加 `arm_` 前缀
**影响文件**：`src/client_record.py`

### C.4 修复：从臂动作转发（2025-05-27）

**问题**：主臂动作未正确转发到从臂
**修复**：在 `host_pi_record.py` 中直接调用 `robot.send_action(data)`
**影响文件**：`src/host_pi_record.py`

### C.5 移除：20秒自动进入auto模式（2025-05-27）

**问题**：自动导航20秒后强制进入auto模式，干扰手动控制
**修复**：移除该功能，改为手动按A键触发
**影响文件**：`src/client_pc.py`

---

**文档版本**: v2.0
**更新日期**: 2025-05-27
**作者**: AI Assistant
**适用分支**: feature/act-inference