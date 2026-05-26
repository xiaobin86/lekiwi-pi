#!/usr/bin/env python
"""
树莓派端 Host - 简化版
功能：
- 创建 LeKiwi Host
- 只使用 front 相机 (/dev/video0)
- robot id = lekiwi
- 不控制从臂，只处理底盘命令
- 等待电脑端连接
"""

import sys
import time
import logging
import json
import base64
from pathlib import Path

# 先配置日志（在导入其他库之前）
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


# 配置
ROBOT_ID = "lekiwi"
FRONT_CAMERA = "/dev/video2"
SERIAL_PORT = "/dev/ttyACM0"
ZMQ_CMD_PORT = 5555
ZMQ_OBS_PORT = 5556
WATCHDOG_MS = 2000
FPS = 30

# YOLO 模型配置
YOLO_MODEL_PATH = Path.home() / "lerobot-workspace/lekiwi-pi/models/paper_ball_detection-1-8/weights/best.pt"
YOLO_CONFIDENCE = 0.5  # 置信度阈值
YOLO_INFER_SIZE = 320  # 推理分辨率（降低以提升性能）
YOLO_SKIP_FRAMES = 2   # 每3帧检测一次（跳过2帧）

# 自动导航配置
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_AREA = IMAGE_WIDTH * IMAGE_HEIGHT
TARGET_AREA_RATIO = 0.20  # 目标占视野20%时到达
CENTER_THRESHOLD = 0.15   # 中心偏差阈值（图像宽度的15%）
NAV_SPEED = 0.3          # 自动导航前进速度
ROT_SPEED = 45           # 自动导航旋转速度


def get_logger():
    """获取logger实例"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger


class AutoNavigator:
    """自动导航控制器 - 固化在树莓派端"""
    
    def __init__(self):
        self.image_center_x = IMAGE_WIDTH / 2
        self.image_center_y = IMAGE_HEIGHT / 2
        self.target_area = IMAGE_AREA * TARGET_AREA_RATIO
        self.state = "idle"  # idle, searching, aligning, approaching, arrived
        
    def calculate_velocity(self, detections):
        """根据检测结果计算底盘速度"""
        if not detections:
            # 没有检测到纸团，原地旋转寻找
            self.state = "searching"
            return {
                "x.vel": 0.0,
                "y.vel": 0.0,
                "theta.vel": ROT_SPEED,
                "arrived": False,
                "state": self.state
            }
        
        # 获取第一个纸团（置信度最高的）
        target = detections[0]
        x1, y1, x2, y2 = target["bbox"]
        cx, cy = target["center"]
        w, h = target["size"]
        
        # 计算纸团面积占比
        ball_area = w * h
        area_ratio = ball_area / IMAGE_AREA
        
        # 检查是否到达目标（纸团占视野20%以上）
        if area_ratio >= TARGET_AREA_RATIO:
            self.state = "arrived"
            return {
                "x.vel": 0.0,
                "y.vel": 0.0,
                "theta.vel": 0.0,
                "arrived": True,
                "state": self.state
            }
        
        # 计算纸团中心与图像中心的偏差
        dx = cx - self.image_center_x
        
        # 归一化偏差（-1 到 1）
        nx = dx / (IMAGE_WIDTH / 2)
        
        # 如果水平偏差大，先旋转对准
        if abs(nx) > CENTER_THRESHOLD:
            # 纸团在右边(nx>0)，底盘需要顺时针旋转（负方向）
            # 纸团在左边(nx<0)，底盘需要逆时针旋转（正方向）
            theta_cmd = -ROT_SPEED * nx
            self.state = "aligning"
            return {
                "x.vel": 0.0,
                "y.vel": 0.0,
                "theta.vel": theta_cmd,
                "arrived": False,
                "state": self.state
            }
        
        # 对准后前进
        # 距离越近速度越慢
        speed_factor = 1.0 - (area_ratio / TARGET_AREA_RATIO)
        x_cmd = NAV_SPEED * speed_factor
        self.state = "approaching"
        
        return {
            "x.vel": x_cmd,
            "y.vel": 0.0,
            "theta.vel": 0.0,
            "arrived": False,
            "state": self.state
        }


def create_robot_config():
    """创建机器人配置（只使用 front 相机）"""
    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=FRONT_CAMERA,
            fps=FPS,
            width=640,
            height=480,
            rotation=Cv2Rotation.ROTATE_180,
            warmup_s=3.0,
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
        logger.error(f"❌ 模型文件不存在: {YOLO_MODEL_PATH}")
        logger.info("  请检查路径是否正确，或运行以下命令查找:")
        logger.info("  find ~ -name 'best.pt' 2>/dev/null")
        yolo_model = None
    else:
        logger.info(f"  模型文件大小: {YOLO_MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB")
        try:
            yolo_model = YOLO(str(YOLO_MODEL_PATH))
            logger.info("✅ YOLO 模型加载成功")
            logger.info(f"  模型类别: {yolo_model.names}")
        except Exception as e:
            logger.error(f"❌ YOLO 模型加载失败: {e}")
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
    cmd_socket.setsockopt(zmq.CONFLATE, 1)
    cmd_socket.bind(f"tcp://*:{ZMQ_CMD_PORT}")
    
    obs_socket = context.socket(zmq.PUSH)
    obs_socket.setsockopt(zmq.CONFLATE, 1)
    obs_socket.bind(f"tcp://*:{ZMQ_OBS_PORT}")
    
    logger.info(f"等待客户端连接...")
    logger.info(f"  命令端口: {ZMQ_CMD_PORT}")
    logger.info(f"  图像端口: {ZMQ_OBS_PORT}")
    if yolo_model is not None:
        logger.info("  YOLO 检测: ✅ 已启用")
    else:
        logger.info("  YOLO 检测: ❌ 未启用（模型加载失败）")
    logger.info("按 Ctrl+C 停止")
    
    # 主循环
    last_cmd_time = time.time()
    watchdog_active = False
    no_command_logged = False
    frame_counter = 0  # 帧计数器
    last_detections = []  # 缓存上一次的检测结果
    auto_mode = False  # 自动导航模式标志
    navigator = AutoNavigator()  # 自动导航控制器
    
    try:
        while True:
            loop_start = time.time()
            
            # 1. 接收命令
            try:
                msg = cmd_socket.recv_string(zmq.NOBLOCK)
                data = json.loads(msg)
                
                # 处理自动导航切换命令
                if data.get("toggle_auto", False):
                    auto_mode = not auto_mode
                    if auto_mode:
                        logger.info("🤖 切换到自动导航模式")
                    else:
                        logger.info("🎮 切换到手动控制模式")
                        navigator.state = "idle"
                    last_cmd_time = time.time()
                    watchdog_active = False
                    no_command_logged = False
                    continue  # 切换模式后不处理其他命令
                
                # 只在有运动命令时打印（减少日志刷屏）
                if not auto_mode:
                    has_movement = any(k.startswith(('base.', 'x.', 'y.', 'theta.')) and v != 0 
                                    for k, v in data.items() if isinstance(v, (int, float)))
                    if has_movement:
                        logger.info(f"收到命令: {data}")
                
                # 补齐从臂默认位置（如果不存在）
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
                
                # 如果是自动导航模式，不执行手动命令（看门狗除外）
                if not auto_mode:
                    robot.send_action(data)
                    last_cmd_time = time.time()
                    watchdog_active = False
                    no_command_logged = False
            except (zmq.Again, StopIteration):
                # No command available, this is normal
                if not watchdog_active and not no_command_logged:
                    if auto_mode:
                        logger.info("自动导航运行中...")
                    else:
                        logger.info("等待命令中...")
                    no_command_logged = True
            except Exception as e:
                import traceback
                logger.error(f"命令错误: {type(e).__name__}: {e}")
                logger.debug(traceback.format_exc())
            
            # 2. 看门狗
            if (time.time() - last_cmd_time > WATCHDOG_MS / 1000) and not watchdog_active:
                logger.warning("看门狗超时，停止底盘")
                watchdog_active = True
                robot.stop_base()
            
            # 3. 获取图像并发送
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
                                verbose=False,
                                imgsz=YOLO_INFER_SIZE
                            )
                            infer_time = time.time() - infer_start
                            
                            # 检查是否有检测结果
                            if len(results) > 0 and len(results[0].boxes) > 0:
                                boxes = results[0].boxes
                                last_detections = []
                                for i, box in enumerate(boxes):
                                    # 获取框的坐标
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    # 获取置信度
                                    conf = float(box.conf[0].cpu().numpy())
                                    # 获取类别
                                    cls = int(box.cls[0].cpu().numpy())
                                    cls_name = results[0].names[cls]
                                    
                                    last_detections.append({
                                        "class": cls_name,
                                        "confidence": round(conf, 4),
                                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                        "center": [float((x1+x2)/2), float((y1+y2)/2)],
                                        "size": [float(x2-x1), float(y2-y1)]
                                    })
                                
                                # 每10秒输出一次性能统计
                                if frame_counter % 300 == 0:
                                    logger.info(f"YOLO推理性能: {infer_time*1000:.1f}ms/帧, "
                                              f"输入尺寸: {YOLO_INFER_SIZE}x{YOLO_INFER_SIZE}, "
                                              f"检测频率: 每{YOLO_SKIP_FRAMES+1}帧")
                            else:
                                last_detections = []
                        except Exception as e:
                            logger.error(f"YOLO 推理错误: {e}")
                    
                    # 使用缓存的检测结果（保持显示连贯性）
                    obs["detections"] = last_detections
                    
                    # 自动导航：根据检测结果计算并发送底盘速度
                    if auto_mode and yolo_model is not None:
                        nav_cmd = navigator.calculate_velocity(last_detections)
                        
                        # 构建底盘动作命令
                        nav_action = {
                            "x.vel": nav_cmd["x.vel"],
                            "y.vel": nav_cmd["y.vel"],
                            "theta.vel": nav_cmd["theta.vel"],
                            "arm_shoulder_pan.pos": 0.0,
                            "arm_shoulder_lift.pos": -100.0,
                            "arm_elbow_flex.pos": 90.0,
                            "arm_wrist_flex.pos": 70.0,
                            "arm_wrist_roll.pos": 0.0,
                            "arm_gripper.pos": 0.0,
                        }
                        
                        robot.send_action(nav_action)
                        last_cmd_time = time.time()
                        watchdog_active = False
                        
                        # 每3秒打印一次导航状态
                        if frame_counter % 90 == 0:
                            state_icon = {
                                "searching": "🔍",
                                "aligning": "🎯",
                                "approaching": "🚀",
                                "arrived": "✅"
                            }
                            icon = state_icon.get(nav_cmd["state"], "🤖")
                            logger.info(
                                f"{icon} 自动导航: {nav_cmd['state']}, "
                                f"x={nav_cmd['x.vel']:.2f}, "
                                f"theta={nav_cmd['theta.vel']:.1f}"
                            )
                    
                    # 添加导航状态到观测数据
                    obs["auto_mode"] = auto_mode
                    obs["nav_state"] = navigator.state if auto_mode else "manual"
                    
                    ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if ret:
                        obs["front"] = base64.b64encode(buffer).decode("utf-8")
                    else:
                        obs["front"] = ""
                
                # 发送
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
