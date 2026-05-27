# LeKiwi YOLO 纸团检测入门文档

## 项目概述

本项目在 LeKiwi 机器人平台上集成 YOLOv8 目标检测，实现纸团实时识别。系统分为两部分：
- **host_pi.py**：运行在树莓派上，负责图像采集、YOLO推理、底盘控制
- **client_pc.py**：运行在电脑上，负责手柄控制、图像显示、检测结果可视化

## 环境准备

### 1. 基础环境（已配置好 LeRobot）

确保已安装：
- Python 3.10+
- OpenCV (通过 LeRobot 环境安装)
- Pygame
- PyZMQ
- NumPy

### 2. 安装依赖（关键步骤，必须按顺序执行）

**⚠️ 警告**：安装顺序非常重要，否则会导致 OpenCV 冲突！

#### 步骤 1：卸载可能冲突的包

```bash
# 卸载所有 opencv 相关包（避免版本冲突）
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python

# 卸载 ultralytics 和 numpy（后续重新安装）
pip uninstall -y ultralytics numpy
```

#### 步骤 2：安装 LeRobot 依赖（包含正确的 OpenCV）

```bash
# 进入 LeRobot 目录
cd ~/lerobot-workspace/lerobot

# 安装 LeRobot 及其依赖（会自动安装兼容的 numpy 和 opencv）
pip install -e ".[feetech]"
```

#### 步骤 3：安装 Ultralytics（不安装依赖）

```bash
# 使用 --no-deps 安装 ultralytics（避免覆盖 OpenCV）
pip install ultralytics --no-deps
```

#### 完整的安装流程

```bash
# 1. 清理冲突包
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
pip uninstall -y ultralytics numpy

# 2. 安装 LeRobot（包含正确版本的 numpy 和 opencv）
cd ~/lerobot-workspace/lerobot
pip install -e ".[feetech]"

# 3. 安装 ultralytics（不安装依赖）
pip install ultralytics --no-deps
```

**为什么必须这样安装？**

| 步骤 | 目的 | 说明 |
|------|------|------|
| 卸载 opencv | 清理冲突 | ultralytics 会安装 `opencv-python`，与 LeRobot 的 `opencv-python-headless` 冲突 |
| 卸载 ultralytics | 重新安装 | 确保使用 `--no-deps` 参数 |
| 卸载 numpy | 版本对齐 | LeRobot 和 ultralytics 可能对 numpy 版本要求不同 |
| 安装 LeRobot | 安装基础依赖 | 安装 LeRobot 所需的正确版本的 numpy、opencv 等 |
| 安装 ultralytics | 安装 YOLO | `--no-deps` 避免覆盖已安装的 opencv 和 numpy |

**验证安装是否成功：**

```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "from ultralytics import YOLO; print('Ultralytics: OK')"
python -c "from lerobot.robots.lekiwi import LeKiwi; print('LeRobot: OK')"
```

如果以上命令都成功执行，说明环境配置正确。

### 3. 模型文件放置

将训练好的 YOLO 模型放到项目目录：

```
lekiwi-pi/
├── models/
│   └── paper_ball_detection-1-8/
│       └── weights/
│           └── best.pt      # 模型文件
├── test/
│   ├── host_pi.py          # 树莓派端代码
│   └── client_pc.py        # 电脑端代码
└── data/                    # 拍照保存目录（自动创建）
```

**模型路径配置**（在 `host_pi.py` 中）：
```python
YOLO_MODEL_PATH = Path.home() / "lerobot-workspace/lekiwi-pi/models/paper_ball_detection-1-8/weights/best.pt"
```

## host_pi.py 详细实现

### 文件位置
`test/host_pi.py`

### 完整代码实现

```python
#!/usr/bin/env python
"""
树莓派端 Host - 简化版
功能：
- 创建 LeKiwi Host
- 只使用 front 相机 (/dev/video0)
- robot id = lekiwi
- 不控制从臂，只处理底盘命令
- YOLO 实时检测纸团
- 等待电脑端连接
"""

import sys
import time
import logging
import json
import base64
from pathlib import Path

# 先配置日志（在导入其他库之前）
# 注意：ultralytics 导入时会覆盖 logging 配置，所以必须先设置好
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True  # 强制覆盖其他库的配置
)

# 添加 LeRobot 路径
sys.path.insert(0, str(Path.home() / "lerobot-workspace/lerobot/src"))

import cv2
import zmq
from ultralytics import YOLO
from lerobot.robots.lekiwi import LeKiwi
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig, LeKiwiHostConfig
from lerobot.cameras import Cv2Rotation
from lerobot.cameras.opencv import OpenCVCameraConfig


# ==================== 配置区域 ====================

ROBOT_ID = "lekiwi"
FRONT_CAMERA = "/dev/video2"      # 前置摄像头设备
SERIAL_PORT = "/dev/ttyACM0"      # 底盘串口
ZMQ_CMD_PORT = 5555               # 命令接收端口
ZMQ_OBS_PORT = 5556               # 图像发送端口
WATCHDOG_MS = 2000                # 看门狗超时（毫秒）
FPS = 30                          # 帧率

# YOLO 模型配置
YOLO_MODEL_PATH = Path.home() / "lerobot-workspace/lekiwi-pi/models/paper_ball_detection-1-8/weights/best.pt"
YOLO_CONFIDENCE = 0.5             # 置信度阈值（0-1）
YOLO_INFER_SIZE = 320             # 推理分辨率（320x320，降低以提升性能）
YOLO_SKIP_FRAMES = 2              # 每3帧检测一次（跳过2帧）

# ==================== 函数定义 ====================

def get_logger():
    """获取logger实例"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger


def create_robot_config():
    """创建机器人配置（只使用 front 相机）"""
    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=FRONT_CAMERA,
            fps=FPS,
            width=640,
            height=480,
            rotation=Cv2Rotation.ROTATE_180,  # 旋转180度（根据实际安装调整）
            warmup_s=3.0,                     # 预热3秒
        ),
    }
    
    return LeKiwiConfig(
        port=SERIAL_PORT,
        id=ROBOT_ID,
        cameras=cameras,
    )


def create_host_config():
    """创建 ZMQ Host 配置"""
    return LeKiwiHostConfig(
        port_zmq_cmd=ZMQ_CMD_PORT,
        port_zmq_observations=ZMQ_OBS_PORT,
        connection_time_s=10000,  # 10000秒运行时间
        watchdog_timeout_ms=WATCHDOG_MS,
        max_loop_freq_hz=FPS,
    )


# ==================== 主函数 ====================

def main():
    logger = get_logger()
    
    logger.info("=" * 60)
    logger.info(f"LeKiwi Host - Robot ID: {ROBOT_ID}")
    logger.info("=" * 60)
    
    # 加载 YOLO 模型
    logger.info("加载 YOLO 模型...")
    logger.info(f"  模型路径: {YOLO_MODEL_PATH}")
    
    # 检查模型文件是否存在
    if not YOLO_MODEL_PATH.exists():
        logger.error(f"模型文件不存在: {YOLO_MODEL_PATH}")
        logger.info("  请检查路径是否正确")
        yolo_model = None
    else:
        logger.info(f"  模型文件大小: {YOLO_MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB")
        try:
            yolo_model = YOLO(str(YOLO_MODEL_PATH))
            logger.info("YOLO 模型加载成功")
            logger.info(f"  模型类别: {yolo_model.names}")
        except Exception as e:
            logger.error(f"YOLO 模型加载失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yolo_model = None
    
    # 创建配置
    robot_config = create_robot_config()
    host_config = create_host_config()
    
    # 初始化机器人
    logger.info("初始化机器人...")
    robot = LeKiwi(robot_config)
    
    logger.info("连接机器人...")
    robot.connect()
    
    # 创建 ZMQ Host
    logger.info("启动 ZMQ Host...")
    context = zmq.Context()
    
    cmd_socket = context.socket(zmq.PULL)
    cmd_socket.setsockopt(zmq.CONFLATE, 1)  # 只保留最新消息
    cmd_socket.bind(f"tcp://*:{ZMQ_CMD_PORT}")
    
    obs_socket = context.socket(zmq.PUSH)
    obs_socket.setsockopt(zmq.CONFLATE, 1)  # 只保留最新消息
    obs_socket.bind(f"tcp://*:{ZMQ_OBS_PORT}")
    
    logger.info(f"等待客户端连接...")
    logger.info(f"  命令端口: {ZMQ_CMD_PORT}")
    logger.info(f"  图像端口: {ZMQ_OBS_PORT}")
    if yolo_model is not None:
        logger.info("  YOLO 检测: 已启用")
    else:
        logger.info("  YOLO 检测: 未启用（模型加载失败）")
    logger.info("按 Ctrl+C 停止")
    
    # 主循环变量
    last_cmd_time = time.time()
    watchdog_active = False
    no_command_logged = False
    frame_counter = 0          # 帧计数器（用于控制检测频率）
    last_detections = []       # 缓存上一次的检测结果
    
    try:
        while True:
            loop_start = time.time()
            
            # 1. 接收命令（非阻塞）
            try:
                msg = cmd_socket.recv_string(zmq.NOBLOCK)
                data = json.loads(msg)
                
                # 只在有运动命令时打印（减少日志刷屏）
                has_movement = any(
                    k.startswith(('base.', 'x.', 'y.', 'theta.')) and v != 0 
                    for k, v in data.items() if isinstance(v, (int, float))
                )
                if has_movement:
                    logger.info(f"收到命令: {data}")
                
                # 补齐从臂默认位置（如果不存在）
                # 关节角度定义：
                # - 关节向上转时: -x度
                # - 关节向下转时: +x度
                # - 左转时: +x度
                # - 右转时: -x度
                arm_defaults = {
                    "arm_shoulder_pan.pos": 0.0,
                    "arm_shoulder_lift.pos": -100.0,
                    "arm_elbow_flex.pos": 90.0,
                    "arm_wrist_flex.pos": 70.0,
                    "arm_wrist_roll.pos": 0.0,
                    "arm_gripper.pos": 0.0,
                }
                for key, default_val in arm_defaults.items():
                    if key not in data:
                        data[key] = default_val
                
                robot.send_action(data)
                last_cmd_time = time.time()
                watchdog_active = False
                no_command_logged = False
            except (zmq.Again, StopIteration):
                # 没有命令，这是正常的
                if not watchdog_active and not no_command_logged:
                    logger.info("等待命令中...")
                    no_command_logged = True
            except Exception as e:
                import traceback
                logger.error(f"命令错误: {type(e).__name__}: {e}")
                logger.debug(traceback.format_exc())
            
            # 2. 看门狗（超时停止底盘）
            if (time.time() - last_cmd_time > WATCHDOG_MS / 1000) and not watchdog_active:
                logger.warning("看门狗超时，停止底盘")
                watchdog_active = True
                robot.stop_base()
            
            # 3. 获取图像、推理并发送
            try:
                obs = robot.get_observation()
                
                # 编码 front 相机图像
                if "front" in obs:
                    frame = obs["front"]
                    if isinstance(frame, cv2.UMat):
                        frame = frame.get()
                    
                    # YOLO 推理 - 识别纸团（每3帧检测一次）
                    frame_counter += 1
                    if yolo_model is not None and frame_counter % (YOLO_SKIP_FRAMES + 1) == 0:
                        try:
                            infer_start = time.time()
                            # 降低分辨率推理以提升性能
                            results = yolo_model(
                                frame, 
                                conf=YOLO_CONFIDENCE, 
                                verbose=False,        # 关闭ultralytics内部日志
                                imgsz=YOLO_INFER_SIZE # 推理输入尺寸320x320
                            )
                            infer_time = time.time() - infer_start
                            
                            # 检查是否有检测结果
                            if len(results) > 0 and len(results[0].boxes) > 0:
                                boxes = results[0].boxes
                                last_detections = []
                                for i, box in enumerate(boxes):
                                    # 获取框的坐标 [x1, y1, x2, y2]
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    # 获取置信度
                                    conf = float(box.conf[0].cpu().numpy())
                                    # 获取类别索引和名称
                                    cls = int(box.cls[0].cpu().numpy())
                                    cls_name = results[0].names[cls]
                                    
                                    # 构建检测结果字典
                                    last_detections.append({
                                        "class": cls_name,
                                        "confidence": round(conf, 4),
                                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                        "center": [float((x1+x2)/2), float((y1+y2)/2)],
                                        "size": [float(x2-x1), float(y2-y1)]
                                    })
                                
                                # 每10秒输出一次性能统计（300帧）
                                if frame_counter % 300 == 0:
                                    logger.info(
                                        f"YOLO推理性能: {infer_time*1000:.1f}ms/帧, "
                                        f"输入尺寸: {YOLO_INFER_SIZE}x{YOLO_INFER_SIZE}, "
                                        f"检测频率: 每{YOLO_SKIP_FRAMES+1}帧"
                                    )
                            else:
                                # 没有检测到目标，清空缓存
                                last_detections = []
                        except Exception as e:
                            logger.error(f"YOLO 推理错误: {e}")
                    
                    # 使用缓存的检测结果（保持显示连贯性）
                    # 非检测帧直接复用上一次的检测结果
                    obs["detections"] = last_detections
                    
                    # 图像编码为 JPEG（降低带宽）
                    ret, buffer = cv2.imencode(
                        ".jpg", 
                        frame, 
                        [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # 质量85%
                    )
                    if ret:
                        obs["front"] = base64.b64encode(buffer).decode("utf-8")
                    else:
                        obs["front"] = ""
                
                # 发送观测数据（包含图像和检测结果）
                obs_socket.send_string(json.dumps(obs), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            except Exception as e:
                logger.error(f"图像发送错误: {e}")
            
            # 4. 控制帧率
            elapsed = time.time() - loop_start
            sleep_time = max(1 / FPS - elapsed, 0)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        logger.info("用户停止")
    finally:
        logger.info("关闭中...")
        robot.disconnect()
        cmd_socket.close()
        obs_socket.close()
        context.term()
        logger.info("已关闭")


if __name__ == "__main__":
    main()
```

### 关键实现说明

#### 1. 日志配置前置
```python
# 在导入 ultralytics 之前配置好 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
```
**原因**：ultralytics 导入时会覆盖 logging 配置，`force=True` 确保我们的配置生效。

#### 2. 检测频率控制
```python
# 每3帧检测一次
frame_counter += 1
if yolo_model is not None and frame_counter % (YOLO_SKIP_FRAMES + 1) == 0:
    results = yolo_model(frame, ...)
```
- `YOLO_SKIP_FRAMES = 2`：每3帧检测一次
- 可以调大减少卡顿，调小提升实时性

#### 3. 推理分辨率优化
```python
results = yolo_model(frame, imgsz=320, ...)
```
- `imgsz=320`：使用 320x320 分辨率推理
- 原始图像 640x480，缩小后推理速度提升约 75%

#### 4. 检测结果缓存
```python
last_detections = []  # 缓存检测结果
obs["detections"] = last_detections  # 每帧都发送（非检测帧用缓存）
```
- 检测帧：更新 `last_detections`
- 非检测帧：复用 `last_detections`
- 保证客户端显示连贯，不会出现闪烁

#### 5. 图像压缩传输
```python
ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
obs["front"] = base64.b64encode(buffer).decode("utf-8")
```
- JPEG 编码，质量 85%
- Base64 编码后通过 ZMQ 发送

## client_pc.py 详细实现

### 文件位置
`test/client_pc.py`

### 完整代码实现

```python
#!/usr/bin/env python
"""
电脑端 Client - 简化版
功能：
- 连接 LeKiwi Host（树莓派）
- Xbox 手柄控制底盘
- 显示摄像头画面（带YOLO检测框）
- 手柄 X 键拍照保存到 data 目录
- 在控制台打印检测信息
"""

import sys
import time
import json
import base64
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import pygame
import zmq


# ==================== 配置区域 ====================

DEFAULT_IP = "192.168.3.176"      # 树莓派默认IP
CMD_PORT = 5555                   # 命令发送端口
OBS_PORT = 5556                   # 图像接收端口
FPS = 30                          # 帧率
DATA_DIR = Path(__file__).parent / "data"  # 照片保存目录

# 确保数据目录存在
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 速度档位配置
SPEED_LEVELS = [
    {"xy": 0.1, "theta": 30},   # Slow（慢速）
    {"xy": 0.3, "theta": 60},   # Medium（中速）
    {"xy": 0.5, "theta": 90},   # Fast（快速）
]


# ==================== 手柄控制器 ====================

class GamepadController:
    """Xbox 手柄控制器"""
    
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        
        self.joystick = None
        self.connected = False
        self.speed_index = 1  # 默认 Medium（中速）
        
        # 按钮编号（Xbox 手柄标准映射）
        self.BTN_RB = 7       # 右肩键
        self.BTN_LB = 6       # 左肩键（切换速度）
        self.BTN_START = 11   # START键（退出）
        self.BTN_X = 3        # X键（蓝色，拍照）
        
        # 上一帧按钮状态（用于边沿检测）
        self.prev_states = {}
    
    def connect(self):
        """连接手柄"""
        retry = 0
        max_retry = 50
        
        while pygame.joystick.get_count() == 0 and retry < max_retry:
            time.sleep(0.1)
            pygame.joystick.init()
            retry += 1
        
        if pygame.joystick.get_count() == 0:
            print("未检测到手柄")
            return False
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.connected = True
        print(f"手柄已连接: {self.joystick.get_name()}")
        return True
    
    def get_action(self):
        """获取手柄动作"""
        if not self.connected:
            return {}
        
        pygame.event.pump()
        
        speed = SPEED_LEVELS[self.speed_index]
        xy_speed = speed["xy"]
        theta_speed = speed["theta"]
        
        x_cmd = 0.0
        y_cmd = 0.0
        theta_cmd = 0.0
        
        # 读取 D-pad（方向键）
        if self.joystick.get_numhats() > 0:
            hat = self.joystick.get_hat(0)
            hat_x, hat_y = hat
            rb_pressed = self.joystick.get_button(self.BTN_RB)
            
            if rb_pressed and hat_x < 0:
                # RB + 左 = 逆时针旋转
                theta_cmd = theta_speed
            elif rb_pressed and hat_x > 0:
                # RB + 右 = 顺时针旋转
                theta_cmd = -theta_speed
            else:
                # 平移控制
                if hat_y > 0:
                    x_cmd = xy_speed       # 前进
                elif hat_y < 0:
                    x_cmd = -xy_speed      # 后退
                
                if hat_x < 0:
                    y_cmd = xy_speed       # 左平移
                elif hat_x > 0:
                    y_cmd = -xy_speed      # 右平移
        
        # LB 切换速度（边沿检测：按下瞬间触发）
        lb_current = self.joystick.get_button(self.BTN_LB)
        lb_prev = self.prev_states.get("LB", False)
        if lb_current and not lb_prev:
            self.speed_index = (self.speed_index + 1) % len(SPEED_LEVELS)
            names = ["Slow", "Medium", "Fast"]
            print(f"  速度: {names[self.speed_index]} (xy={SPEED_LEVELS[self.speed_index]['xy']})")
        self.prev_states["LB"] = lb_current
        
        # X 按钮拍照（边沿检测）
        x_current = self.joystick.get_button(self.BTN_X)
        x_prev = self.prev_states.get("X", False)
        capture_image = x_current and not x_prev
        if capture_image:
            print("  拍照命令已发送")
        self.prev_states["X"] = x_current
        
        # 返回动作字典
        return {
            "x.vel": x_cmd,
            "y.vel": y_cmd,
            "theta.vel": theta_cmd,
            "capture_image": capture_image,  # True 表示需要拍照
        }
    
    def check_exit(self):
        """检查是否退出"""
        if not self.connected:
            return False
        return self.joystick.get_button(self.BTN_START)
    
    def disconnect(self):
        """断开手柄"""
        if self.joystick:
            self.joystick.quit()
        pygame.quit()
        print("手柄已断开")


# ==================== 图像显示函数 ====================

def display_frame(frame_b64, window_name="Camera", save_path=None, detections=None):
    """
    显示图像（RGB转换显示，绘制检测框，原始BGR保存）
    
    Args:
        frame_b64: Base64编码的图像字符串
        window_name: 窗口标题
        save_path: 保存路径（如果为None则不保存）
        detections: 检测结果列表 [{"class": ..., "confidence": ..., "bbox": [...], ...}]
    
    Returns:
        frame: 解码后的OpenCV图像（BGR格式）
    """
    if not frame_b64:
        return None
    
    try:
        # Base64 解码为二进制
        frame_data = base64.b64decode(frame_b64)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            # 绘制检测结果
            if detections:
                for i, det in enumerate(detections):
                    x1, y1, x2, y2 = det["bbox"]      # 边界框坐标
                    cx, cy = det["center"]             # 中心点坐标
                    conf = det["confidence"]           # 置信度
                    cls_name = det["class"]            # 类别名称
                    
                    # 绘制矩形框（绿色，线宽2）
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # 绘制中心点（红色实心圆，半径5）
                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                    
                    # 绘制标签（类别+置信度）
                    label = f"{cls_name}: {conf:.2%}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 显示时转换为 RGB（颜色正确）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow(window_name, frame_rgb)
            cv2.waitKey(1)
            
            # 保存原始帧（BGR格式，不做转换）
            if save_path:
                cv2.imwrite(str(save_path), frame)
                print(f"  图像已保存(BGR): {save_path}")
            
            return frame
    except Exception as e:
        print(f"  图像处理错误: {e}")
    return None


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="LeKiwi PC Client - 底盘遥操作")
    parser.add_argument("--ip", default=DEFAULT_IP, help="树莓派 IP")
    parser.add_argument("--display", action="store_true", default=True, help="显示摄像头画面")
    args = parser.parse_args()
    
    print("=" * 60)
    print("LeKiwi Client - 底盘遥操作")
    print("=" * 60)
    print(f"目标: {args.ip}:{CMD_PORT}/{OBS_PORT}")
    print("=" * 60)
    
    # 连接手柄
    print("\n[1/3] 连接手柄...")
    gamepad = GamepadController()
    if not gamepad.connect():
        print("错误：未检测到手柄")
        return
    
    # 连接 ZMQ
    print("\n[2/3] 连接树莓派...")
    context = zmq.Context()
    
    cmd_socket = context.socket(zmq.PUSH)
    cmd_socket.connect(f"tcp://{args.ip}:{CMD_PORT}")
    
    obs_socket = context.socket(zmq.PULL)
    obs_socket.setsockopt(zmq.CONFLATE, 1)  # 只保留最新帧
    obs_socket.connect(f"tcp://{args.ip}:{OBS_PORT}")
    
    print("已连接")
    
    # 显示帮助信息
    print("\n" + "=" * 60)
    print("控制说明:")
    print("  D-pad 上/下     : 前进/后退")
    print("  D-pad 左/右     : 左平移/右平移")
    print("  RB + 左/右      : 原地旋转")
    print("  LB              : 切换速度档")
    print("  X               : 拍照保存到 data 目录")
    print("  START           : 退出")
    print("=" * 60)
    print(f"\n图像保存目录: {DATA_DIR.absolute()}")
    
    # 主循环
    print("\n[3/3] 开始遥操作...")
    running = True
    frame_count = 0
    capture_requested = False  # 拍照请求标志
    
    try:
        while running:
            t0 = time.perf_counter()
            
            # 1. 获取手柄动作
            action = gamepad.get_action()
            
            # 检测拍照请求
            if action.get("capture_image", False):
                capture_requested = True
                print("  拍照请求已记录，下一帧将保存")
            
            # 2. 发送命令到树莓派
            # 过滤掉值为0的项（减少网络传输）
            if any(v != 0 for v in action.values() if isinstance(v, (int, float))):
                print(f"发送: {action}")
            cmd_socket.send_string(json.dumps(action), flags=zmq.NOBLOCK)
            
            # 3. 接收图像和检测结果
            if args.display:
                try:
                    msg = obs_socket.recv_string(zmq.NOBLOCK)
                    obs = json.loads(msg)
                    
                    if "front" in obs:
                        # 准备保存路径
                        save_path = None
                        if capture_requested:
                            # 生成时间戳文件名
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            save_path = DATA_DIR / f"capture_{timestamp}.jpg"
                            capture_requested = False
                        
                        # 获取检测结果
                        detections = obs.get("detections", [])
                        
                        # 打印检测信息到控制台
                        if detections:
                            print(f"检测到 {len(detections)} 个目标:")
                            for i, det in enumerate(detections):
                                x1, y1, x2, y2 = det["bbox"]
                                cx, cy = det["center"]
                                print(f"  [{i+1}] 类别: {det['class']}, "
                                      f"置信度: {det['confidence']:.2%}, "
                                      f"位置: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f}), "
                                      f"中心: ({cx:.1f}, {cy:.1f})")
                        
                        # 显示图像（带检测框）
                        display_frame(obs["front"], "Front Camera", save_path, detections)
                except zmq.Again:
                    pass
            
            # 4. 检查退出
            if gamepad.check_exit():
                print("\nSTART 按下，退出...")
                running = False
            
            # 5. 帧率控制
            elapsed = time.perf_counter() - t0
            sleep_time = max(1.0 / FPS - elapsed, 0)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        print("\n清理中...")
        gamepad.disconnect()
        cmd_socket.close()
        obs_socket.close()
        context.term()
        if args.display:
            cv2.destroyAllWindows()
        print("已退出")


if __name__ == "__main__":
    main()
```

### 关键实现说明

#### 1. 检测结果绘制
```python
def display_frame(frame_b64, window_name="Camera", save_path=None, detections=None):
    # 绘制矩形框（绿色）
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # 绘制中心点（红色）
    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
    
    # 绘制标签
    label = f"{cls_name}: {conf:.2%}"
    cv2.putText(frame, label, (int(x1), int(y1) - 10), ...)
```
- 绿色矩形框：目标边界
- 红色圆点：目标中心
- 文字标签：类别 + 置信度

#### 2. 显示与保存的区别
```python
# 显示：转换为 RGB（颜色正确）
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cv2.imshow(window_name, frame_rgb)

# 保存：保持原始 BGR（供 YOLO 训练使用）
cv2.imwrite(str(save_path), frame)
```
- **显示用 RGB**：OpenCV 默认用 BGR，转成 RGB 显示颜色才正确
- **保存用 BGR**：YOLO 训练通常使用 BGR 格式，保持一致性

#### 3. 拍照功能实现
```python
# 手柄端：边沿检测
x_current = self.joystick.get_button(self.BTN_X)
x_prev = self.prev_states.get("X", False)
capture_image = x_current and not x_prev  # 只在按下瞬间为 True

# 客户端：记录请求，下一帧保存
if action.get("capture_image", False):
    capture_requested = True

# 接收图像时保存
if capture_requested:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    save_path = DATA_DIR / f"capture_{timestamp}.jpg"
    capture_requested = False
```
- 边沿检测：避免按住不放时连续拍照
- 异步保存：记录请求，接收到下一帧时保存

## 使用流程

### 1. 启动树莓派端

在树莓派上执行：
```bash
cd ~/lerobot-workspace/lekiwi-pi/test
python host_pi.py
```

启动后输出：
```
============================================================
LeKiwi Host - Robot ID: lekiwi
============================================================
加载 YOLO 模型...
  模型路径: /home/pi/lerobot-workspace/lekiwi-pi/models/...
  模型文件大小: 12.34 MB
YOLO 模型加载成功
  模型类别: {0: 'paper_ball'}
初始化机器人...
连接机器人...
启动 ZMQ Host...
等待客户端连接...
  命令端口: 5555
  图像端口: 5556
  YOLO 检测: 已启用
按 Ctrl+C 停止
```

### 2. 启动电脑端

在电脑上执行：
```bash
cd D:\work\lerobot-workspace\lekiwi-pi\test
python client_pc.py
```

启动后输出：
```
============================================================
LeKiwi Client - 底盘遥操作
============================================================
目标: 192.168.3.176:5555/5556
============================================================

[1/3] 连接手柄...
手柄已连接: Xbox Controller

[2/3] 连接树莓派...
已连接

============================================================
控制说明:
  D-pad 上/下     : 前进/后退
  D-pad 左/右     : 左平移/右平移
  RB + 左/右      : 原地旋转
  LB              : 切换速度档
  X               : 拍照保存到 data 目录
  START           : 退出
============================================================

图像保存目录: D:\work\lerobot-workspace\lekiwi-pi\test\data

[3/3] 开始遥操作...
```

### 3. 操作说明

| 按键 | 功能 |
|------|------|
| D-pad 上 | 前进 |
| D-pad 下 | 后退 |
| D-pad 左 | 左平移 |
| D-pad 右 | 右平移 |
| RB + 左 | 逆时针旋转 |
| RB + 右 | 顺时针旋转 |
| LB | 切换速度档（慢/中/快） |
| **X** | **拍照保存到 data 目录** |
| START | 退出程序 |

### 4. 检测结果显示

当检测到纸团时，客户端控制台输出：
```
检测到 2 个目标:
  [1] 类别: paper_ball, 置信度: 92.50%, 位置: (120.5, 80.3) - (340.2, 280.7), 中心: (230.4, 180.5)
  [2] 类别: paper_ball, 置信度: 78.30%, 位置: (400.1, 150.2) - (520.6, 290.8), 中心: (460.4, 220.5)
```

同时图像窗口会显示：
- 🟩 绿色矩形框：目标边界
- 🔴 红色圆点：目标中心
- 🏷️ 标签文字：类别和置信度

## 常见问题

### 1. ultralytics 与 OpenCV 冲突

**现象**：安装 ultralytics 后 LeRobot 报错 `cv2 模块找不到`、`cv2 版本不兼容` 或 `ImportError`

**根本原因**：
- ultralytics 依赖 `opencv-python`（PyPI 包）
- LeRobot 使用 `opencv-python-headless` 或系统 `cv2`
- 两者同时存在会导致冲突

**解决**（按顺序执行）：
```bash
# 1. 卸载所有 opencv 相关包
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python

# 2. 卸载 ultralytics 和 numpy
pip uninstall -y ultralytics numpy

# 3. 重新安装 LeRobot 依赖（安装正确的 opencv 和 numpy）
cd ~/lerobot-workspace/lerobot
pip install -e ".[feetech]"

# 4. 安装 ultralytics（不安装依赖）
pip install ultralytics --no-deps
```

**验证修复**：
```bash
python -c "import cv2; import numpy; from ultralytics import YOLO; from lerobot.robots.lekiwi import LeKiwi; print('全部导入成功')"
```

### 2. 模型加载失败

**现象**：`模型文件不存在` 或 `YOLO 模型加载失败`

**排查**：
```bash
# 检查文件是否存在
ls ~/lerobot-workspace/lekiwi-pi/models/paper_ball_detection-1-8/weights/best.pt

# 检查文件大小（正常应该几十MB）
ls -lh ~/lerobot-workspace/lekiwi-pi/models/paper_ball_detection-1-8/weights/best.pt

# 如果不存在，查找 best.pt
find ~ -name "best.pt" 2>/dev/null
```

### 3. 运行时卡顿

**优化方法**：
1. 降低检测频率：修改 `YOLO_SKIP_FRAMES = 4`（每5帧检测一次）
2. 降低推理分辨率：修改 `YOLO_INFER_SIZE = 224`
3. 降低图像质量：修改 `cv2.IMWRITE_JPEG_QUALITY` 为 70

### 4. 检测不到目标

**排查**：
1. 检查摄像头画面是否正常（客户端能否看到图像）
2. 检查纸团是否在画面中
3. 降低置信度阈值：`YOLO_CONFIDENCE = 0.3`
4. 检查模型类别名是否正确（应该包含 `paper_ball`）

## 文件结构

```
lekiwi-pi/
├── models/                          # 模型目录
│   └── paper_ball_detection-1-8/    # YOLO训练结果
│       └── weights/
│           └── best.pt              # 最佳模型
├── test/                            # 测试代码
│   ├── host_pi.py                   # 树莓派端（本文档）
│   ├── client_pc.py                 # 电脑端（本文档）
│   ├── test_gamepad.py              # 手柄测试工具
│   └── data/                        # 拍照保存目录
│       └── capture_20260526_143052_123.jpg
└── docs/                            # 文档
    └── yolo_detection_guide.md      # 本指南
```

## 性能参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `YOLO_CONFIDENCE` | 0.5 | 置信度阈值（0-1） |
| `YOLO_INFER_SIZE` | 320 | 推理分辨率（320/416/640） |
| `YOLO_SKIP_FRAMES` | 2 | 跳过帧数（0=每帧检测） |
| `FPS` | 30 | 目标帧率 |
| JPEG质量 | 85 | 图像传输质量（0-100） |

**推理性能**（树莓派5实测）：
- 320x320 分辨率：约 50-80ms/帧
- 640x640 分辨率：约 150-200ms/帧

## 版本记录

| 日期 | 版本 | 说明 |
|------|------|------|
| 2025-05-26 | v1.0 | 初始版本，集成YOLO检测 |
| 2025-05-26 | v1.1 | 优化推理性能（降频+降分辨率） |

---

**作者**：AI Assistant
**项目**：LeKiwi Robot Platform
**用途**：YOLO目标检测入门与纸团识别
