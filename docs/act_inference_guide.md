# LeKiwi ACT 策略推理部署指南

## 概述

本文档说明如何在 LeKiwi 上部署 ACT (Action Chunking with Transformers) 策略模型，实现自动抓取纸团。

**架构设计**:
```
┌─────────────────────────────────────────────────────────────┐
│                         树莓派 (Host)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ 摄像头读取    │  │ 机械臂状态    │  │ 底盘控制         │  │
│  │ (front)      │  │ (6自由度)     │  │ (3轮全向)        │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                  │                    │            │
│  ┌──────▼──────────────────▼────────────────────▼─────────┐│
│  │              Host Main Process                          ││
│  │  - 读取观测 (图像 + 机械臂状态 + 底盘速度)              ││
│  │  - ZMQ发送观测 ────────────────────────→ PC端          ││
│  │  - ZMQ接收动作 ←──────────────────────── PC端          ││
│  │  - 执行动作 (机械臂位置 + 底盘速度)                     ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              │ ZMQ (tcp://192.168.3.176:5555/5556)
                              │
┌─────────────────────────────────────────────────────────────┐
│                      PC (推理端)                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              ACT Policy Model                          │  │
│  │  - 输入: 图像(front) + 机械臂状态 + 底盘速度           │  │
│  │  - 输出: 未来N步动作 (机械臂位置 + 底盘速度)           │  │
│  │  - 推理延迟: ~30-50ms (RTX 5070 Ti)                    │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**为什么选择在PC端推理？**
1. **性能**: 树莓派5 CPU推理ACT模型延迟~200-500ms，PC端GPU推理仅30-50ms
2. **灵活性**: 可随时切换策略模型，无需重新部署到树莓派
3. **稳定性**: 避免树莓派CPU过载导致控制不稳定

---

## 一、数据采集

### 1.1 硬件准备

- [ ] LeKiwi 底盘 + 从臂已组装校准
- [ ] 主臂 (SO100/SO101) 已连接PC并校准
- [ ] 前视摄像头 (/dev/video2) 可用
- [ ] 腕部摄像头 (/dev/video0) 可用（可选）
- [ ] 纸团若干（建议黄色/橙色）

### 1.2 录制脚本

使用 `record_grasp.py` 录制数据：

```bash
# PC端运行
conda activate lerobot
cd ~/lerobot-workspace/lekiwi-pi
python record_grasp.py
```

**录制要点**:
1. **多样性**: 纸团放在不同位置、距离、高度
2. **成功率**: 确保 >80% 的抓取成功
3. **流畅性**: 动作平滑，避免急停急转
4. **数量**: 最少 30 个 episodes，推荐 50+

### 1.3 数据集结构

录制完成后数据格式：
```
dataset/
├── videos/
│   ├── front_episode_000000.mp4
│   └── wrist_episode_000000.mp4
├── data/
│   └── episode_000000.parquet
└── meta/
    └── info.jsonl
```

**关键字段**:
- `observation.images.front`: 前视图像 (640×480)
- `observation.images.wrist`: 腕部图像 (480×640, 可选)
- `observation.state`: [arm_shoulder_pan, arm_shoulder_lift, arm_elbow_flex, arm_wrist_flex, arm_wrist_roll, arm_gripper, x.vel, y.vel, theta.vel]
- `action`: 同上（ACT使用绝对位置控制）

---

## 二、模型训练

### 2.1 ACT 策略配置

ACT (Action Chunking with Transformers) 特点：
- **动作分块**: 一次预测未来 N 个时间步的动作
- **Transformer架构**: CVAE编码器 + Transformer解码器
- **时序集成**: 使用Temporal Ensembling平滑动作

**关键超参数**:
```yaml
# ACT 配置
policy:
  type: act
  n_action_steps: 8          # 动作分块大小（预测未来8步）
  chunk_size: 8              # 同上
  n_obs_steps: 1             # 观测历史步数
  
  # 图像编码器
  vision_backbone: resnet18
  pretrained_backbone: true
  
  # Transformer
  dim_model: 512
  n_heads: 8
  n_encoder_layers: 4
  n_decoder_layers: 7
  
  # CVAE
  latent_dim: 32
  use_vae: true
  kl_weight: 10.0
```

### 2.2 训练命令

```bash
# 基础训练
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=your_username/lekiwi_grasp_paper_ball \
    --output_dir=outputs/lekiwi_grasp_act \
    --job_name=lekiwi_grasp \
    --device=cuda \
    --wandb.enable=true \
    --policy.n_action_steps=8 \
    --policy.chunk_size=8

# 高级配置（推荐）
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=your_username/lekiwi_grasp_paper_ball \
    --output_dir=outputs/lekiwi_grasp_act_v2 \
    --job_name=lekiwi_grasp_v2 \
    --device=cuda \
    --batch_size=16 \
    --num_workers=4 \
    --policy.n_action_steps=8 \
    --policy.chunk_size=8 \
    --policy.dim_model=512 \
    --policy.n_heads=8 \
    --policy.kl_weight=10.0 \
    --training.lr=1e-5 \
    --training.num_epochs=3000 \
    --training.save_freq=1000
```

### 2.3 训练监控

- **训练时长**: ~2-4小时（取决于数据量和GPU）
- **收敛指标**: L1 loss < 0.05，KLD loss < 5.0
- **评估**: 每1000步在验证集上评估成功率

---

## 三、推理部署

### 3.1 部署方案选择

**方案A：PC端推理（推荐）**
- **优点**: 低延迟(30-50ms)、可切换模型、不占用树莓派CPU
- **缺点**: 需要PC开机运行
- **适用**: 有PC在旁边，追求实时性

**方案B：树莓派推理**
- **优点**: 独立运行，无需PC
- **缺点**: 高延迟(200-500ms)、需要模型优化（NCNN/ONNX）
- **适用**: 无PC场景，接受较低实时性

### 3.2 方案A：PC端推理

#### 步骤1：启动树莓派 Host

```bash
# SSH到树莓派
ssh acelan@lekiwi-pi
conda activate lekiwi
cd ~/lerobot-workspace/lekiwi-pi
python src/host_pi_act.py --mode=inference
```

**host_pi_act.py 功能**:
- 读取前视摄像头图像
- 读取从臂6个关节位置
- 读取底盘3个轮速
- ZMQ发送观测（图像 + 状态）
- ZMQ接收动作（机械臂位置 + 底盘速度）
- 执行动作

#### 步骤2：PC端运行推理

```bash
# PC端
conda activate lerobot
cd ~/lerobot-workspace/lekiwi-pi
python src/act_inference_client.py \
    --model_path=outputs/lekiwi_grasp_act/checkpoints/last/pretrained_model \
    --robot_ip=192.168.3.176 \
    --fps=30
```

**推理流程**:
1. 连接树莓派ZMQ
2. 加载ACT模型到GPU
3. 循环:
   - 接收观测（图像 + 状态）
   - 预处理（归一化、调整维度）
   - 策略推理 → 动作分块（8步）
   - 发送第一步动作到树莓派
   - 缓存剩余7步动作
   - 时序集成平滑
4. 退出时保存统计数据

### 3.3 方案B：树莓派推理（实验性）

#### 模型优化

```bash
# 1. 将PyTorch模型导出为ONNX
python -c "
from lerobot import ACTPolicy
import torch

policy = ACTPolicy.from_pretrained('outputs/lekiwi_grasp_act/checkpoints/last')
policy.eval()

# 创建示例输入
dummy_input = {
    'observation.images.front': torch.randn(1, 3, 480, 640),
    'observation.state': torch.randn(1, 9),
}

# 导出ONNX
torch.onnx.export(
    policy,
    dummy_input,
    'lekiwi_act.onnx',
    opset_version=14,
    input_names=['front', 'state'],
    output_names=['action'],
)
"

# 2. 使用ONNX Runtime在树莓派推理
pip install onnxruntime
```

#### 树莓派推理脚本

```python
# host_pi_onnx.py
import onnxruntime as ort
import numpy as np

# 加载ONNX模型
session = ort.InferenceSession("lekiwi_act.onnx")

while True:
    # 读取观测
    obs = robot.get_observation()
    
    # 预处理
    front = obs["front"].transpose(2, 0, 1) / 255.0  # HWC → CHW
    state = np.array([obs[k] for k in state_keys])
    
    # 推理
    outputs = session.run(
        None,
        {"front": front[np.newaxis], "state": state[np.newaxis]}
    )
    action = outputs[0]
    
    # 执行动作
    robot.send_action(action_dict)
```

---

## 四、实现原理

### 4.1 ACT 策略原理

**核心思想**:
1. **动作分块(Action Chunking)**: 一次预测未来N个时间步的动作序列，而不是只预测下一步
2. **条件变分自编码器(CVAE)**: 编码器学习动作分布，解码器生成动作序列
3. **Transformer架构**: 使用注意力机制建模观测与动作序列的关系

**模型结构**:
```
输入: 图像 + 机器人状态
  ↓
[图像编码器] (ResNet18)
  ↓
[拼接] 图像特征 + 状态向量
  ↓
[Transformer编码器] 处理观测
  ↓
[CVAE编码器] 学习动作隐变量 (训练时)
  ↓
[Transformer解码器] 生成动作序列
  ↓
输出: [a_t, a_t+1, ..., a_t+N-1] (N步动作)
```

**推理时序**:
```
t=0: 观测 → 模型 → [a0, a1, a2, ..., a7] → 执行a0, 缓存[a1-a7]
t=1: 观测 → 模型 → [a1', a2', ..., a8'] → 执行a1(集成), 缓存[a2'-a8']
t=2: 观测 → 模型 → ...
```

**时序集成(Temporal Ensembling)**:
- 每个时间步可能有多个预测（来自不同chunk）
- 使用指数加权平均平滑动作
- 权重: w_i = exp(-temporal_ensemble_coeff * i)

### 4.2 观测到动作的映射

**观测空间**:
```python
obs = {
    "observation.images.front": (480, 640, 3),  # BGR图像
    "observation.images.wrist": (640, 480, 3),  # BGR图像（可选）
    "observation.state": [
        arm_shoulder_pan.pos,   # 肩关节旋转 (度)
        arm_shoulder_lift.pos,  # 肩关节抬升 (度)
        arm_elbow_flex.pos,     # 肘关节弯曲 (度)
        arm_wrist_flex.pos,     # 腕关节弯曲 (度)
        arm_wrist_roll.pos,     # 腕关节旋转 (度)
        arm_gripper.pos,        # 夹爪开合 (0-100%)
        x.vel,                   # 底盘X速度 (m/s)
        y.vel,                   # 底盘Y速度 (m/s)
        theta.vel               # 底盘旋转速度 (度/s)
    ]
}
```

**动作空间**:
```python
action = [
    arm_shoulder_pan.pos_target,   # 目标位置
    arm_shoulder_lift.pos_target,
    arm_elbow_flex.pos_target,
    arm_wrist_flex.pos_target,
    arm_wrist_roll.pos_target,
    arm_gripper.pos_target,
    x.vel_target,                   # 目标速度
    y.vel_target,
    theta.vel_target
]
```

### 4.3 数据流

**训练时**:
```
Episode 数据: [(o_0, a_0), (o_1, a_1), ..., (o_T, a_T)]
  ↓
构建 batch:
  - 观测: o_t
  - 动作序列: [a_t, a_t+1, ..., a_t+N-1]
  ↓
模型输入:
  - 图像编码: ResNet18提取特征
  - 状态编码: MLP投影
  - 拼接后传入Transformer
  ↓
模型输出: 动作序列预测
  ↓
损失函数: L1(pred, target) + KL(VAE)
```

**推理时**:
```
实时循环 (30Hz):
  1. 获取观测 o_t
  2. 预处理:
     - 图像: BGR→RGB, HWC→CHW, /255, resize
     - 状态: normalize
  3. 传入策略模型
  4. 获取动作分块 [a_t, ..., a_t+N-1]
  5. 执行 a_t（或时序集成）
  6. 缓存剩余动作
```

### 4.4 LeKiwi 特殊处理

**机械臂控制**:
- ACT输出绝对关节位置（度数）
- LeKiwi使用Feetech总线，支持位置控制模式
- 需要设置 `max_relative_target` 限制单步最大变化量（安全）

**底盘控制**:
- ACT输出底盘速度 (x, y, theta)
- LeKiwi使用3轮全向底盘
- 需要逆运动学转换: (x,y,theta) → (wheel1, wheel2, wheel3)

**坐标系**:
- 图像坐标: (u, v), 原点在左上角
- 机器人坐标: x向前, y向左, theta逆时针
- 动作单位: 机械臂(度), 底盘(m/s, deg/s)

---

## 五、代码结构

### 5.1 文件说明

```
lekiwi-pi/
├── src/
│   ├── host_pi_act.py          # 树莓派Host（支持推理模式）
│   ├── act_inference_client.py # PC端ACT推理脚本
│   ├── host_pi.py              # 原版Host（底盘控制+YOLO）
│   └── client_pc.py            # 原版Client（手柄控制）
├── scripts/
│   ├── train_act.sh            # ACT训练脚本
│   └── deploy_act.sh           # 部署脚本
├── docs/
│   └── act_inference_guide.md  # 本文档
└── config/
    └── act_policy_config.yaml  # ACT策略配置
```

### 5.2 host_pi_act.py 关键逻辑

```python
class ACTHost:
    def __init__(self, mode="inference"):
        self.mode = mode  # "inference" 或 "teleop"
        self.robot = LeKiwi(config)
        
    def run(self):
        while True:
            # 1. 获取观测
            obs = self.get_observation()
            
            if self.mode == "inference":
                # 2a. 推理模式: 发送观测到PC，接收动作
                action = self.get_action_from_pc(obs)
            else:
                # 2b. 遥操作模式: 接收手柄命令
                action = self.get_action_from_gamepad()
            
            # 3. 执行动作
            self.robot.send_action(action)
    
    def get_observation(self):
        # 读取摄像头
        front_frame = self.camera.read()
        
        # 读取机械臂状态
        arm_pos = self.robot.bus.sync_read("Present_Position", arm_motors)
        
        # 读取底盘速度
        base_vel = self.robot.get_base_velocity()
        
        return {
            "images": {"front": front_frame},
            "state": {**arm_pos, **base_vel}
        }
```

### 5.3 act_inference_client.py 关键逻辑

```python
class ACTInferenceClient:
    def __init__(self, model_path):
        # 加载ACT模型
        self.policy = ACTPolicy.from_pretrained(model_path)
        self.policy.eval()
        
        # 连接到树莓派
        self.zmq_cmd = zmq.Context().socket(zmq.PUSH)
        self.zmq_obs = zmq.Context().socket(zmq.PULL)
        
    def run(self):
        while True:
            # 1. 接收观测
            obs = self.receive_observation()
            
            # 2. 预处理
            obs_tensor = self.preprocess_observation(obs)
            
            # 3. 推理
            with torch.inference_mode():
                action = self.policy.select_action(obs_tensor)
            
            # 4. 发送动作
            self.send_action(action)
    
    def preprocess_observation(self, obs):
        # 图像预处理
        front = obs["front"]  # BGR, HWC
        front = cv2.cvtColor(front, cv2.COLOR_BGR2RGB)
        front = front.transpose(2, 0, 1) / 255.0  # CHW, [0,1]
        front = torch.from_numpy(front).unsqueeze(0).cuda()
        
        # 状态预处理
        state = np.array([obs[k] for k in state_keys], dtype=np.float32)
        state = torch.from_numpy(state).unsqueeze(0).cuda()
        
        return {
            "observation.images.front": front,
            "observation.state": state,
        }
```

---

## 六、性能优化

### 6.1 推理延迟优化

| 优化手段 | 延迟改善 | 说明 |
|---------|---------|------|
| GPU推理 | 30-50ms | 使用CUDA加速 |
| 半精度(fp16) | -20% | `torch.cuda.amp` |
| 批处理 | -10% | 单次处理多帧 |
| 模型量化 | -50% | INT8量化 |
| TensorRT | -30% | NVIDIA优化 |

### 6.2 通信优化

```python
# ZMQ优化
socket.setsockopt(zmq.CONFLATE, 1)  # 只保留最新消息
socket.setsockopt(zmq.RCVHWM, 1)    # 接收队列大小为1
socket.setsockopt(zmq.SNDHWM, 1)    # 发送队列大小为1
```

### 6.3 控制频率

- **目标频率**: 30Hz
- **实际频率**: 
  - 树莓派Host: 30Hz（受限于摄像头帧率）
  - PC推理: ~20Hz（受限于模型推理+通信）
  - 端到端延迟: ~80-120ms

---

## 七、故障排查

### 7.1 常见问题

**Q1: 推理时机械臂抖动**
- **原因**: 动作变化过大或频率不稳定
- **解决**: 
  - 设置 `max_relative_target` 限制单步变化量
  - 使用时序集成平滑动作
  - 降低推理频率到20Hz

**Q2: 模型输出动作不执行**
- **原因**: 动作格式不匹配或超出关节限制
- **解决**:
  - 检查动作维度是否与训练时一致
  - 使用 `ensure_safe_goal_position` 裁剪动作
  - 检查机械臂是否已使能

**Q3: 推理延迟过高**
- **原因**: 模型太大或通信延迟
- **解决**:
  - 使用更小的backbone（MobileNet替代ResNet18）
  - 减少action chunk大小（8→4）
  - 使用有线网络替代WiFi

**Q4: 抓取成功率低**
- **原因**: 训练数据不足或分布不匹配
- **解决**:
  - 增加训练数据（特别是失败案例）
  - 数据增强（随机裁剪、颜色抖动）
  - 微调模型（fine-tuning）

---

## 八、扩展功能

### 8.1 多摄像头输入

```python
# 同时使用front和wrist摄像头
obs = {
    "observation.images.front": front_frame,
    "observation.images.wrist": wrist_frame,
    "observation.state": state,
}
```

### 8.2 多任务支持

```python
# 不同任务使用不同模型
task = "grasp_paper_ball"  # 或 "push_box", "open_door"
obs["task"] = task
```

### 8.3 安全保护

```python
# 紧急停止
if emergency_stop_button:
    robot.stop_base()
    robot.bus.disable_torque(arm_motors)
    
# 碰撞检测
if force_sensor > threshold:
    robot.stop_base()
```

---

## 九、参考资源

- [ACT论文](https://arxiv.org/abs/2304.13705)
- [LeRobot文档](https://github.com/huggingface/lerobot)
- [LeKiwi组装指南](https://github.com/SIGRobotics-UIUC/LeKiwi)
- [SO101机械臂文档](https://github.com/huggingface/lerobot/blob/main/docs/source/so101.mdx)

---

**文档版本**: v1.0
**更新日期**: 2025-05-27
**作者**: AI Assistant
