# LeKiwi ACT 自动抓取纸团完整教程

> 从0到1：录制数据 → 训练模型 → 自动寻找 → 接近 → 抓取

---

## 目录

1. [项目概述](#1-项目概述)
2. [硬件准备](#2-硬件准备)
3. [环境配置](#3-环境配置)
4. [数据采集](#4-数据采集)
5. [模型训练](#5-模型训练)
6. [部署推理](#6-部署推理)
7. [完整流程测试](#7-完整流程测试)
8. [故障排查](#8-故障排查)

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

### 1.3 数据流

```
树莓派 Host (src/host_pi.py)
  ├─ 读取摄像头 (front)
  ├─ 读取机械臂状态 (6关节)
  ├─ 阶段1: YOLO检测 → 底盘移动
  ├─ 阶段2: 发送观测 (图像+状态) → PC
  └─ 接收ACT动作 → 执行机械臂

PC Client (src/act_grasp_client.py)
  ├─ 接收树莓派观测
  ├─ 运行ACT模型推理 (6-DOF)
  └─ 发送机械臂动作 → 树莓派
```

---

## 2. 硬件准备

### 2.1 必需硬件

| 组件 | 型号/规格 | 数量 | 说明 |
|------|----------|------|------|
| **树莓派** | Raspberry Pi 5 (8GB) | 1 | 主机，运行Host程序 |
| **主臂** | SO100/SO101 | 1 | 遥操作控制从臂 |
| **从臂** | SO100/SO101 | 1 | 安装在LeKiwi底盘上 |
| **底盘** | LeKiwi 3轮全向 | 1 | 移动底盘 |
| **摄像头** | USB Camera | 1 | front摄像头 (/dev/video2) |
| **PC** | Windows/Linux | 1 | 训练+推理 |
| **手柄** | Xbox/PS4 | 1 | 手动控制（可选） |

### 2.2 接线检查

```bash
# 在树莓派上检查硬件
ls /dev/video*    # 查看摄像头
ls /dev/ttyACM*   # 查看电机控制板
```

### 2.3 机械臂校准

**主臂校准（PC端）**：
```bash
conda activate lerobot
lerobot-calibrate \
    --teleop.type=so100_leader \
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

### 3.2 PC环境

```bash
# 创建conda环境
conda create -n lerobot python=3.10 -y
conda activate lerobot

# 安装LeRobot
cd ~/lerobot-workspace/lerobot
pip install -e ".[lekiwi]"

# 安装训练依赖
pip install -e ".[core_scripts]"

# 验证GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 3.3 项目目录结构

```
lekiwi-pi/
├── src/
│   ├── host_pi.py              # 树莓派Host（导航+抓取控制）
│   ├── act_grasp_client.py     # PC端ACT推理客户端
│   ├── client_pc.py            # PC端手柄控制+显示
│   └── host_pi_act.py          # 树莓派Host（纯ACT推理模式）
├── config/
│   └── act_policy_config.yaml  # ACT策略配置（6-DOF）
├── scripts/
│   ├── train_act.sh            # 训练脚本
│   └── deploy_act.sh           # 部署脚本
├── docs/
│   └── act_grasp_tutorial.md   # 本教程
├── models/                     # YOLO模型
│   └── paper_ball_detection/
└── data/                       # 录制数据
```

---

## 4. 数据采集

### 4.1 录制策略

**关键点**：
- 底盘已定位到纸团附近（手动或自动导航）
- 录制过程中**底盘保持静止**
- 只移动主臂控制从臂抓取纸团
- 每个episode包含：预抓取→接近→抓取→抬起

### 4.2 启动树莓派Host

```bash
# SSH到树莓派
ssh acelan@lekiwi-pi
conda activate lekiwi
cd ~/lerobot-workspace/lekiwi-pi

# 启动Host
python src/host_pi.py
```

### 4.3 启动PC端录制

```bash
# 在PC上打开新终端
conda activate lerobot
cd ~/lerobot-workspace/lekiwi-pi

# 录制数据
python record_grasp.py
```

**录制控制**：
- **主臂**：控制从臂抓取纸团
- **键盘W/A/S/D**：底盘移动（定位时使用）
- **空格键**：提前结束当前episode
- **R键**：重新录制
- **Q键**：退出

### 4.4 录制要点

1. **定位底盘**：手动将底盘移动到纸团附近
2. **保持静止**：开始录制后不要移动底盘
3. **抓取动作**：
   - 预抓取位置：夹爪打开，对准纸团
   - 向下移动：机械臂下降
   - 夹爪闭合：抓取纸团
   - 向上抬起：提起纸团
4. **重复录制**：每个位置录制3-5次

### 4.5 数据多样性

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

### 4.6 数据集上传

```bash
# 录制完成后自动上传到Hugging Face
# 数据集路径: ~/.cache/huggingface/lerobot/your_username/lekiwi_grasp_paper_ball

# 或手动上传
huggingface-cli upload your_username/lekiwi_grasp_paper_ball ./data
```

---

## 5. 模型训练

### 5.1 训练配置

编辑 `config/act_policy_config.yaml`：

```yaml
policy:
  type: act
  
  # 6-DOF机械臂配置
  input_features:
    observation.images.front:
      type: image
      shape: [3, 480, 640]
    observation.state:
      type: state
      shape: [6]  # 只包含机械臂6关节
  
  output_features:
    action:
      type: state
      shape: [6]  # 只输出机械臂6关节
  
  # ACT超参数
  n_action_steps: 8        # 动作分块大小
  chunk_size: 8
  dim_model: 512
  n_heads: 8
  use_vae: true
  kl_weight: 10.0
```

### 5.2 启动训练

```bash
# 使用脚本训练
bash scripts/train_act.sh your_username/lekiwi_grasp_paper_ball

# 或手动训练
conda activate lerobot
cd ~/lerobot-workspace/lerobot

lerobot-train \
    --policy.type=act \
    --dataset.repo_id=your_username/lekiwi_grasp_paper_ball \
    --output_dir=outputs/lekiwi_grasp_act_arm_only \
    --device=cuda \
    --policy.input_features.state.shape=[6] \
    --policy.output_features.action.shape=[6] \
    --policy.n_action_steps=8 \
    --policy.chunk_size=8 \
    --training.num_epochs=3000
```

### 5.3 监控训练

**TensorBoard**：
```bash
tensorboard --logdir outputs/lekiwi_grasp_act_arm_only
```

**关键指标**：
- `l1_loss`：动作预测误差（目标 < 0.05）
- `kld_loss`：VAE散度（目标 < 5.0）
- `eval/success_rate`：评估成功率

**训练时长**：
- 30 episodes：~1-2小时
- 100 episodes：~4-6小时

### 5.4 模型保存

训练完成后，模型保存在：
```
outputs/lekiwi_grasp_act_arm_only/checkpoints/last/
├── config.json           # 模型配置
├── model.safetensors     # 模型权重
└── preprocessor_config.json
```

---

## 6. 部署推理

### 6.1 系统架构

```
┌──────────────┐      ZMQ       ┌──────────────────┐
│  树莓派 Host  │ ◄────────────► │  PC推理客户端     │
│              │   观测/动作     │                  │
│ ├─ YOLO检测  │                │ ├─ ACT模型推理    │
│ ├─ 底盘控制  │                │ └─ 6-DOF机械臂    │
│ └─ 机械臂控制│                │                  │
└──────────────┘                └──────────────────┘
       │                                │
       └──── 手柄控制 (client_pc.py) ────┘
```

### 6.2 启动树莓派Host

```bash
# SSH到树莓派
ssh acelan@lekiwi-pi
conda activate lekiwi
cd ~/lerobot-workspace/lekiwi-pi

# 启动Host（支持自动导航+ACT抓取）
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

### 6.3 启动PC端ACT推理

```bash
# PC端（窗口1）
conda activate lerobot
cd ~/lerobot-workspace/lekiwi-pi

# 启动ACT推理客户端
python src/act_grasp_client.py \
    --model_path outputs/lekiwi_grasp_act_arm_only/checkpoints/last \
    --robot_ip 192.168.3.176
```

**预期输出**：
```
==================================================
LeKiwi ACT 6-DOF机械臂推理客户端
==================================================
树莓派IP: 192.168.3.176
模型路径: outputs/lekiwi_grasp_act_arm_only/checkpoints/last
推理设备: cuda
控制维度: 6-DOF机械臂（底盘自动静止）
==================================================

等待树莓派进入 grasping 状态...
按 Ctrl+C 停止
```

### 6.4 启动PC端显示（可选）

```bash
# PC端（窗口2）
conda activate lerobot
cd ~/lerobot-workspace/lekiwi-pi

# 启动显示客户端
python src/client_pc.py --ip 192.168.3.176
```

### 6.5 触发自动流程

**步骤1：切换到自动导航模式**
- 按手柄 **A键**（或键盘对应键）
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
- 等待2秒

**步骤5：ACT抓取**
- 自动进入 `grasping` 🦾 状态
- 状态：`grasping`（显示进度条）
- PC运行ACT推理，控制机械臂
- 机械臂执行：预抓取→接近→抓取→抬起

**步骤6：完成**
- 抓取完成，状态回到 `idle`
- 可以再次按A键开始新一轮

---

## 7. 完整流程测试

### 7.1 一键启动脚本

创建 `scripts/run_full_pipeline.sh`：

```bash
#!/bin/bash
# 完整流程：导航+抓取

set -e

ROBOT_IP="192.168.3.176"
MODEL_PATH="outputs/lekiwi_grasp_act_arm_only/checkpoints/last"

echo "=================================="
echo "LeKiwi 自动抓取纸团 - 完整流程"
echo "=================================="

# 1. 启动树莓派Host
echo "[1/3] 启动树莓派Host..."
ssh acelan@$ROBOT_IP "
    cd ~/lerobot-workspace/lekiwi-pi
    pkill -f host_pi.py || true
    sleep 1
    nohup python src/host_pi.py > logs/host.log 2>&1 &
    echo 'Host已启动'
"
sleep 5

# 2. 启动ACT推理
echo "[2/3] 启动ACT推理客户端..."
python src/act_grasp_client.py \
    --model_path $MODEL_PATH \
    --robot_ip $ROBOT_IP &
ACT_PID=$!

# 3. 启动显示
echo "[3/3] 启动显示客户端..."
python src/client_pc.py --ip $ROBOT_IP

# 清理
echo "停止推理客户端..."
kill $ACT_PID || true

echo "停止树莓派Host..."
ssh acelan@$ROBOT_IP "pkill -f host_pi.py || true"

echo "完成！"
```

### 7.2 使用手柄触发

```
按键功能：
├─ A键: 切换自动/手动模式
├─ X键: 拍照保存
├─ LB键: 切换速度档位
├─ START键: 退出程序
└─ D-pad: 手动控制底盘（手动模式下）
```

### 7.3 预期流程演示

```
时间轴:
T+0s   按A键 → AUTO模式 → searching 🔍
T+3s   发现纸团 → aligning 🎯
T+8s   对准完成 → approaching 🚀
T+15s  到达纸团 → arrived ✅
T+17s  延迟2秒 → grasping 🦾
T+17s  PC推理: 预抓取 (夹爪打开)
T+18s  PC推理: 向下接近
T+19s  PC推理: 夹爪闭合 (抓取)
T+20s  PC推理: 向上抬起
T+22s  抓取完成 → idle ⏹️
T+25s  可以再次按A键开始新一轮
```

---

## 8. 故障排查

### 8.1 树莓派Host问题

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

### 8.2 PC推理问题

**问题：模型加载失败**
```bash
# 检查模型路径
ls outputs/lekiwi_grasp_act_arm_only/checkpoints/last/

# 检查CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**问题：推理延迟高**
- 使用 `--device cuda` 确保GPU推理
- 检查GPU利用率：`nvidia-smi`
- 降低图像分辨率（修改host_pi.py）

### 8.3 抓取失败问题

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

### 8.4 网络问题

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

---

## 9. 性能优化

### 9.1 降低延迟

| 优化手段 | 效果 | 实现方式 |
|---------|------|---------|
| 降低图像分辨率 | -15ms | 320x240传输 |
| 降低JPEG质量 | -12ms | quality=60 |
| 使用TensorRT | -25ms | 模型优化 |
| 使用有线网络 | -10ms | 网线替代WiFi |

### 9.2 提高成功率

| 优化手段 | 效果 | 实现方式 |
|---------|------|---------|
| 增加训练数据 | +20% | 录制100+ episodes |
| 数据增强 | +10% | random_crop, color_jitter |
| 模型集成 | +5% | 多个模型投票 |
| 安全限制 | 防损坏 | 设置关节限位 |

---

## 10. 高级功能

### 10.1 多任务支持

修改 `TASK_DESCRIPTION` 支持不同任务：
```python
TASK_DESCRIPTION = "Grasp paper ball"  # 或 "Push box", "Open drawer"
```

### 10.2 连续抓取

修改 `host_pi.py` 完成抓取后不回到idle：
```python
# 抓取完成后直接开始新的搜索
if grasp_completed:
    navigator.reset()  # 直接开始新的搜索
```

### 10.3 远程监控

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
| `src/host_pi.py` | 树莓派主程序（导航+抓取控制） |
| `src/act_grasp_client.py` | PC端ACT推理（6-DOF） |
| `src/client_pc.py` | PC端显示+手柄控制 |
| `record_grasp.py` | 数据采集脚本 |
| `config/act_policy_config.yaml` | ACT配置（6-DOF） |
| `scripts/train_act.sh` | 训练脚本 |
| `scripts/deploy_act.sh` | 部署脚本 |

---

**文档版本**: v1.0
**更新日期**: 2025-05-27
**作者**: AI Assistant
**适用分支**: feature/act-inference
