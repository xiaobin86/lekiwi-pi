# LeKiwi 抓取纸团遥操作录制指南

## 概述

本指南说明如何使用 LeKiwi 的从臂，通过主臂遥操作录制抓取纸团的数据集，用于后续训练策略模型。

**硬件配置**:
- **主臂** (Leader): SO100/SO101 机械臂，连接 PC
- **从臂** (Follower): 安装在 LeKiwi 移动底盘上的机械臂
- **移动底盘**: LeKiwi 差分驱动底盘
- **摄像头**: Front (前视) + Wrist (腕部)

**数据流**:
```
主臂 (PC) → 从臂 (LeKiwi) → 录制数据集
   ↓              ↓
键盘控制  →  底盘移动
```

---

## 前置准备

### 1. 硬件检查清单

- [ ] LeKiwi 底盘已组装并上电
- [ ] 从臂已安装在底盘上并校准
- [ ] 主臂已连接 PC 并校准
- [ ] 树莓派已启动，网络正常
- [ ] 前视摄像头 (/dev/video0) 和腕部摄像头 (/dev/video2) 可用
- [ ] 纸团已准备

### 2. 确认端口

**在 PC 上查找主臂端口:**

Windows:
```powershell
# 查看设备管理器中的 COM 端口
# 或运行
python -c "from lerobot.common.utils import find_ports; print(find_ports())"
```

Linux/Mac:
```bash
lerobot-find-port
```

**根据你的配置:**
- 主臂 ID: `L07252802`
- 主臂端口: `COM5` (Windows) 或 `/dev/ttyACM0` (Linux)
- 树莓派 IP: `192.168.3.176`

### 3. 环境配置

确保已安装 LeRobot:
```bash
# PC 端
cd ~/lerobot-workspace/lerobot
conda activate lerobot
pip install -e ".[lekiwi]"

# 树莓派端
ssh acelan@lekiwi-pi
conda activate lekiwi
pip install -e ".[lekiwi]"
```

---

## 录制流程

### 第一步：启动树莓派 Host

在树莓派上启动 host 程序:

```bash
ssh acelan@lekiwi-pi
conda activate lekiwi
cd ~/lerobot-workspace/lekiwi-pi
python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_lekiwi
```

**预期输出:**
```
[INFO] LeKiwi Host started
[INFO] Waiting for connection on port 5555 (cmd) and 5556 (obs)
```

### 第二步：配置录制脚本

编辑 `record_grasp.py` 文件，修改以下配置:

```python
# 录制参数
NUM_EPISODES = 10          # 录制 episode 数量
EPISODE_TIME_SEC = 30      # 每个 episode 时长（秒）
TASK_DESCRIPTION = "Grasp paper ball with follower arm"

# 硬件配置
RASPBERRY_PI_IP = "192.168.3.176"  # 树莓派 IP
LEADER_ARM_PORT = "COM5"            # 主臂串口
LEADER_ARM_ID = "L07252802"         # 主臂 ID

# 数据集
HF_REPO_ID = "your_username/lekiwi_grasp_paper_ball"
```

### 第三步：开始录制

在 PC 上运行录制脚本:

```bash
conda activate lerobot
cd ~/lerobot-workspace/lekiwi-pi
python record_grasp.py
```

### 第四步：遥操作录制

**控制方式:**

| 设备 | 功能 | 说明 |
|------|------|------|
| **主臂** | 控制从臂 | 直接映射主臂姿态到从臂 |
| **键盘 W** | 前进 | 底盘向前移动 |
| **键盘 S** | 后退 | 底盘向后移动 |
| **键盘 A** | 左移 | 底盘向左平移 |
| **键盘 D** | 右移 | 底盘向右平移 |
| **键盘 Z** | 左转 | 底盘原地左转 |
| **键盘 X** | 右转 | 底盘原地右转 |
| **键盘 R** | 加速 | 增加底盘移动速度 |
| **键盘 F** | 减速 | 降低底盘移动速度 |
| **空格键** | 结束 | 提前结束当前 episode |
| **键盘 Q** | 退出 | 停止录制 |

**录制策略:**

1. **Episode 开始**: 将纸团放在从臂可抓取范围内
2. **底盘定位**: 使用键盘控制底盘移动到合适位置
3. **机械臂抓取**: 使用主臂控制从臂抓取纸团
4. **Episode 结束**: 按空格键提前结束或等待超时

**建议录制数量:**
- 最少 10 个 episodes
- 建议 30-50 个 episodes
- 多样性：不同位置、不同角度、不同距离

---

## 数据集结构

录制完成后，数据集包含:

```
lekiwi_grasp_paper_ball/
├── videos/
│   ├── front_episode_000000.mp4      # 前视摄像头视频
│   ├── wrist_episode_000000.mp4      # 腕部摄像头视频
│   └── ...
├── data/
│   ├── episode_000000.parquet        # 状态和动作数据
│   └── ...
├── meta/
│   ├── info.jsonl                    # 元数据
│   └── tasks.jsonl                   # 任务描述
└── README.md                         # 数据集说明
```

**数据字段:**
- `observation.images.front`: 前视图像
- `observation.images.wrist`: 腕部图像
- `observation.state`: 机器人状态（关节位置等）
- `action`: 动作命令（包括底盘速度和关节位置）

---

## 训练模型

录制完成后，使用数据集训练策略模型:

```bash
# 训练 ACT 策略
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=your_username/lekiwi_grasp_paper_ball \
    --output_dir=outputs/lekiwi_grasp_act \
    --job_name=lekiwi_grasp \
    --device=cuda \
    --wandb.enable=true

# 训练 Diffusion 策略
lerobot-train \
    --policy.type=diffusion \
    --dataset.repo_id=your_username/lekiwi_grasp_paper_ball \
    --output_dir=outputs/lekiwi_grasp_diffusion \
    --job_name=lekiwi_grasp \
    --device=cuda
```

---

## 故障排查

### 问题 1: 无法连接到树莓派

**症状:** `Connection refused` 或超时

**排查:**
1. 检查树莓派是否已启动 host 程序
2. 检查 IP 地址是否正确: `ping 192.168.3.176`
3. 检查端口是否开放: `telnet 192.168.3.176 5555`
4. 检查防火墙设置

### 问题 2: 主臂无法连接

**症状:** `Failed to connect to leader arm`

**排查:**
1. 检查 USB 线是否连接
2. 检查端口是否正确: `lerobot-find-port`
3. 检查主臂是否上电（LED 灯亮）
4. 尝试重新插拔 USB

### 问题 3: 从臂不跟随主臂

**症状:** 移动主臂，从臂不动

**排查:**
1. 检查从臂是否已校准
2. 检查树莓派 host 程序是否正常运行
3. 检查网络延迟是否过高

### 问题 4: 数据集上传失败

**症状:** `Failed to push to hub`

**排查:**
1. 检查 Hugging Face token: `hf auth login`
2. 检查 repo_id 格式是否正确
3. 检查网络连接

---

## 最佳实践

### 1. 录制质量

- **光照**: 确保环境光照充足且均匀
- **背景**: 保持背景简洁，避免干扰
- **纸团**: 使用明显颜色（如黄色/橙色）
- **多样性**: 
  - 不同距离（近/中/远）
  - 不同角度（左/中/右）
  - 不同高度（高/中/低）

### 2. 动作设计

- **平滑性**: 避免突然的快速动作
- **成功率**: 确保大多数抓取成功（>80%）
- **稳定性**: 抓取后保持 2-3 秒稳定

### 3. 数据量

- **最小量**: 10 episodes
- **推荐量**: 30-50 episodes
- **最优量**: 100+ episodes（复杂任务）

---

## 参考资料

- [LeRobot 文档](https://github.com/huggingface/lerobot)
- [LeKiwi 组装指南](https://github.com/SIGRobotics-UIUC/LeKiwi)
- [SO100/SO101 机械臂文档](https://github.com/huggingface/lerobot/blob/main/docs/source/so101.mdx)

---

**文档版本**: v1.0
**更新日期**: 2025-05-27
**作者**: AI Assistant
