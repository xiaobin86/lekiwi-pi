# LeKiwi 项目分支说明文档

本文档说明 LeKiwi 项目的各个 Git 分支及其功能。

---

## 分支列表

| 分支名称 | 功能描述 | 状态 | 主要文件 |
|---------|---------|------|---------|
| `master` | 基础底盘遥控 + 图像传输 | 稳定 | `test/host_pi.py`, `test/client_pc.py` |
| `feature/yolo-detection` | YOLO纸团检测 + 结果可视化 | 稳定 | `test/host_pi.py`, `test/client_pc.py`, `docs/yolo_detection_guide.md` |
| `feature/auto-navigation` | 自动导航（基于YOLO检测） | 开发中 | `test/client_pc.py` |

---

## 各分支详细说明

### 1. `master` 分支

**功能**：基础底盘遥控 + 实时图像传输

**实现内容**：
- 树莓派端 (`host_pi.py`)：
  - 初始化 LeKiwi 机器人
  - 连接 front 摄像头 (`/dev/video2`)
  - 通过 ZMQ 接收底盘控制命令
  - 编码图像并通过 ZMQ 发送给电脑端
  - 看门狗机制（超时自动停止底盘）

- 电脑端 (`client_pc.py`)：
  - Xbox 手柄控制底盘移动
  - 显示摄像头实时画面
  - D-pad 控制平移，RB+左右旋转
  - LB 切换速度档
  - START 退出

**按键映射**：
| 按键 | 功能 |
|------|------|
| D-pad 上/下 | 前进/后退 |
| D-pad 左/右 | 左平移/右平移 |
| RB + 左/右 | 原地旋转 |
| LB | 切换速度档（慢/中/快）|
| START | 退出程序 |

**使用方式**：
```bash
# 树莓派端
python test/host_pi.py

# 电脑端
python test/client_pc.py
```

---

### 2. `feature/yolo-detection` 分支

**功能**：在基础功能上集成 YOLOv8 纸团检测

**基于分支**：`master`

**新增功能**：
- 树莓派端 (`host_pi.py`)：
  - 加载 YOLOv8 模型 (`models/paper_ball_detection-1-8/weights/best.pt`)
  - 每3帧进行一次纸团检测（降低负载）
  - 推理分辨率 320x320（提升速度）
  - 缓存检测结果（保持显示连贯）
  - 将检测结果通过 ZMQ 发送给电脑端

- 电脑端 (`client_pc.py`)：
  - 接收并解析检测结果
  - 在图像上绘制检测框（绿色）和中心点（红色）
  - 显示类别和置信度标签
  - 手柄 X 键拍照保存到 `data/` 目录
  - 控制台打印检测信息

**按键映射（新增）**：
| 按键 | 功能 |
|------|------|
| X | 拍照保存到 data 目录 |

**检测结果格式**：
```json
{
  "detections": [
    {
      "class": "paper_ball",
      "confidence": 0.925,
      "bbox": [120.5, 80.3, 340.2, 280.7],
      "center": [230.4, 180.5],
      "size": [219.7, 200.4]
    }
  ]
}
```

**性能优化**：
- 检测频率：每3帧一次（10fps）
- 推理分辨率：320x320（计算量减少75%）
- 图像压缩：JPEG 质量85%
- 缓存机制：非检测帧复用上次结果

**依赖安装**（关键）：
```bash
# 必须按顺序执行，避免 OpenCV 冲突
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
pip uninstall -y ultralytics numpy
cd ~/lerobot-workspace/lerobot
pip install -e ".[feetech]"
pip install ultralytics --no-deps
```

**使用方式**：
```bash
# 树莓派端
python test/host_pi.py

# 电脑端
python test/client_pc.py
```

---

### 3. `feature/auto-navigation` 分支

**功能**：基于 YOLO 检测结果的自动导航功能

**基于分支**：`feature/yolo-detection`

**架构变更**：
- **自动导航逻辑固化到树莓派端**（减少网络延迟，响应更快）
- 电脑端只负责发送切换命令和显示状态

**新增功能**：
- 树莓派端 (`host_pi.py`)：
  - 接收 `toggle_auto` 命令切换模式
  - 根据 YOLO 检测结果实时计算底盘速度
  - 自动导航状态机（搜索/对准/接近/到达）
  - 每3秒打印一次导航状态日志

- 电脑端 (`client_pc.py`)：
  - **A 键发送切换命令**给树莓派
  - 接收并显示当前导航状态
  - 图像上叠加模式指示（MANUAL / AUTO: xxx）

**自动导航算法**（树莓派端执行）：
```
1. 无检测 → 原地旋转寻找（搜索模式）
2. 检测到纸团 → 计算偏差
   a. 水平偏差 > 15% → 旋转对准（对准模式）
   b. 水平偏差 <= 15% → 前进靠近（接近模式）
3. 纸团面积 >= 20% → 停止等待（到达模式）
4. 持续检测，纸团移动则重新跟踪
```

**配置参数**：
```python
TARGET_AREA_RATIO = 0.20    # 目标占视野20%时到达
CENTER_THRESHOLD = 0.15     # 中心偏差阈值（图像宽度的15%）
NAV_SPEED = 0.3            # 自动导航速度
ROT_SPEED = 45             # 自动旋转速度
```

**按键映射（新增）**：
| 按键 | 功能 | 说明 |
|------|------|------|
| A | 发送自动导航切换命令 | 树莓派执行实际导航逻辑 |
| X | 拍照保存到 data 目录 | |

**图像显示指示**：
- **MANUAL**（白色）：手动控制模式
- **AUTO NAV**（黄色）：自动导航模式
- **Progress: xx%**：接近目标的进度

**使用方式**：
```bash
# 树莓派端
python test/host_pi.py

# 电脑端
python test/client_pc.py

# 操作步骤：
# 1. 启动后默认手动模式（树莓派端控制）
# 2. 按 A 键发送自动导航命令（电脑→树莓派）
# 3. 树莓派自动寻找并靠近纸团
# 4. 到达后保持自动导航，持续搜索新目标
# 5. 如果纸团被移动，树莓派自动跟踪新位置
# 6. 按 A 键发送命令切回手动控制
```

**导航状态说明**：
| 状态 | 说明 | 底盘动作 |
|------|------|---------|
| searching | 搜索中 | 原地旋转 |
| aligning | 对准中 | 旋转调整 |
| approaching | 接近中 | 前进 |
| arrived | 已到达 | 停止等待 |
| arrived_waiting | 到达等待 | 持续搜索新目标 |

**持续搜索功能**：
- 到达目标后不会退出自动导航模式
- 保持原地等待，持续检测新目标
- 如果纸团被移动：
  - 消失 → 进入搜索模式（原地旋转寻找）
  - 移动 → 重新计算路径并跟踪
- 用户需要手动按 A 键才能切回手动模式

---

## 分支演进关系

```
master (基础遥控)
  │
  ├── feature/yolo-detection (YOLO检测)
  │     │
  │     └── feature/auto-navigation (自动导航)
  │
  └── feature/gamepad-teleop (手柄遥操作)
```

---

## 切换分支方法

```bash
# 查看所有分支
git branch -a

# 切换到指定分支
git checkout master
git checkout feature/yolo-detection
git checkout feature/auto-navigation

# 拉取最新代码
git pull origin <branch-name>
```

---

## 各分支代码差异总结

### `master` vs `feature/yolo-detection`

**host_pi.py 差异**：
- 新增 YOLO 模型加载
- 新增检测推理逻辑
- 新增检测结果序列化发送
- 性能优化（降频、降分辨率）

**client_pc.py 差异**：
- 新增检测结果解析
- 新增检测框绘制
- 新增 X 键拍照功能
- 新增检测信息打印

### `feature/yolo-detection` vs `feature/auto-navigation`

**host_pi.py 差异**：
- 新增 AutoNavigator 类（自动导航控制器）
- 新增自动导航配置参数（目标面积、速度、阈值）
- 新增自动导航状态机（搜索/对准/接近/到达）
- 新增 `toggle_auto` 命令处理
- 自动导航模式下，根据YOLO检测结果直接计算并发送底盘速度
- 发送导航状态给电脑端显示

**client_pc.py 差异**：
- **移除** AutoNavigator 类（逻辑移至树莓派）
- **移除** 自动导航计算逻辑
- A 键只发送 `toggle_auto` 命令
- 从观测数据读取并显示 `auto_mode` 和 `nav_state`
- 图像上显示当前导航状态

---

## 开发计划

| 阶段 | 分支 | 功能 | 状态 |
|------|------|------|------|
| 阶段1 | master | 基础遥控 | ✅ 完成 |
| 阶段2 | feature/yolo-detection | 目标检测 | ✅ 完成 |
| 阶段3 | feature/auto-navigation | 自动导航 | 🚧 开发中 |
| 阶段4 | - | 机械臂抓取 | 📋 计划中 |

---

## 注意事项

1. **分支依赖**：
   - `feature/yolo-detection` 基于 `master`
   - `feature/auto-navigation` 基于 `feature/yolo-detection`
   - 切换分支时需确保依赖已正确安装

2. **模型文件**：
   - YOLO 检测和自动导航都需要模型文件
   - 确保 `models/paper_ball_detection-1-8/weights/best.pt` 存在

3. **性能考虑**：
   - 树莓派5建议：320x320 推理 + 每3帧检测
   - 如需更流畅，可降低检测频率或分辨率

4. **安全提醒**：
   - 自动导航时请注意周围环境
   - 确保有紧急停止方案（拔电源或强制退出）
   - 初次测试建议降低速度（修改 NAV_SPEED）

---

**文档版本**：v1.0
**更新日期**：2025-05-26
**作者**：AI Assistant
