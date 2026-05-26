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
CENTER_THRESHOLD = 0.10   # 中心偏差阈值（图像宽度的10%，更精细）
NAV_SPEED = 0.25         # 自动导航前进速度（降低避免丢失）
ROT_SPEED = 20           # 自动导航旋转速度（大幅降低，避免移出视野）
SEARCH_ROT_ANGLE = 20    # 每次步进旋转角度（度）
SEARCH_STOP_TIME = 0.5   # 停止后等待检测时间（秒）
TRACK_LOST_FRAMES = 5    # 允许连续丢失多少帧仍继续跟踪（约0.17秒）

# PID控制器参数（用于旋转对准）
PID_KP = 15.0             # 比例系数（基础响应）
PID_KI = 0.1              # 积分系数（消除静态误差）
PID_KD = 8.0              # 微分系数（抑制超调震荡）
PID_MAX_I = 5.0           # 积分上限（防止积分饱和）


# 全局 logger（供 AutoNavigator 使用）
logger = None

def get_logger():
    """获取logger实例"""
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger


class PIDController:
    """PID控制器 - 用于平滑控制旋转速度"""
    
    def __init__(self, kp, ki, kd, max_i=5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_i = max_i
        self.reset()
    
    def reset(self):
        """重置积分和上次误差"""
        self.integral = 0.0
        self.last_error = 0.0
        self.first_run = True
    
    def compute(self, error, dt):
        """
        计算PID输出
        
        Args:
            error: 当前误差
            dt: 时间间隔（秒）
        
        Returns:
            float: 控制输出
        """
        # 比例项
        p = self.kp * error
        
        # 积分项（累积误差）
        self.integral += error * dt
        # 限制积分，防止饱和
        self.integral = max(-self.max_i, min(self.max_i, self.integral))
        i = self.ki * self.integral
        
        # 微分项（变化率）
        if self.first_run:
            d = 0
            self.first_run = False
        else:
            d = self.kd * (error - self.last_error) / dt
        
        self.last_error = error
        
        # PID输出
        output = p + i + d
        return output


class AutoNavigator:
    """自动导航控制器 - 固化在树莓派端"""
    
    def __init__(self):
        self.image_center_x = IMAGE_WIDTH / 2
        self.image_center_y = IMAGE_HEIGHT / 2
        self.target_area = IMAGE_AREA * TARGET_AREA_RATIO
        self.state = "idle"  # idle, searching, aligning, approaching, arrived
        
        # 步进式搜索状态
        self.search_phase = "rotate"  # rotate, stop, detect
        self.search_timer = 0
        self.search_start_time = 0
        
        # 追踪缓冲
        self.last_target = None       # 最后一次检测到的目标
        self.lost_count = 0           # 连续丢失帧数
        self.tracking = False         # 是否正在跟踪
        
        # PID控制器（用于平滑旋转对准）
        self.pid = PIDController(PID_KP, PID_KI, PID_KD, PID_MAX_I)
        self.last_time = time.time()  # 上一次调用时间
        
    def calculate_velocity(self, detections, current_time):
        """
        根据检测结果计算底盘速度
        
        Args:
            detections: 检测结果列表
            current_time: 当前时间戳（time.time()）
            
        Returns:
            dict: {"x.vel": ..., "y.vel": ..., "theta.vel": ..., "arrived": ...}
        """
        # 如果检测到纸团，立即切换到跟踪模式
        if detections:
            # 如果从搜索模式切换到跟踪模式，重置PID控制器
            if not self.tracking:
                self.pid.reset()
                logger.info("🎯 发现目标，开始跟踪")
            self.last_target = detections[0]
            self.lost_count = 0
            self.tracking = True
            return self._track_target(self.last_target)
        
        # 没有检测到纸团，但正在跟踪中（可能短暂丢失）
        if self.tracking and self.last_target is not None:
            self.lost_count += 1
            
            if self.lost_count <= TRACK_LOST_FRAMES:
                # 在允许范围内，继续向最后已知位置移动
                logger.info(f"⚠️ 目标丢失 {self.lost_count}/{TRACK_LOST_FRAMES} 帧，继续跟踪")
                return self._track_target(self.last_target)
            else:
                # 丢失太久，放弃跟踪，进入搜索模式
                logger.info("❌ 目标丢失太久，开始搜索")
                self.tracking = False
                self.last_target = None
                self.lost_count = 0
        
        # 没有检测到纸团，执行步进式搜索
        return self._step_search(current_time)
    
    def _track_target(self, target):
        """跟踪检测到的目标（使用比例控制）"""
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
        
        # 计算前进速度（根据距离）
        speed_factor = 1.0 - (area_ratio / TARGET_AREA_RATIO)
        x_cmd = NAV_SPEED * speed_factor
        
        # 使用PID控制器计算旋转速度（平滑控制）
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # 误差：目标在中心左侧(nx<0)需要逆时针(正)，右侧(nx>0)需要顺时针(负)
        # PID输入：nx是归一化偏差 [-1, 1]
        pid_output = self.pid.compute(nx, dt)
        
        # 限制旋转速度（最大ROT_SPEED）
        theta_cmd = -pid_output  # 负号：目标在右(nx>0)需要顺时针(负速度)
        if abs(theta_cmd) > ROT_SPEED:
            theta_cmd = ROT_SPEED if theta_cmd > 0 else -ROT_SPEED
        
        # 判断状态
        if abs(nx) > CENTER_THRESHOLD:
            self.state = "aligning"
            # 偏差大时，降低前进速度，主要旋转
            x_cmd = x_cmd * 0.3  # 降低至30%
            logger.info(f"  [PID对准] 偏差nx={nx:.3f}, PID输出={pid_output:.2f}, theta={theta_cmd:.1f}")
        else:
            self.state = "approaching"
            # 偏差小时，主要前进，微调旋转（PID会自动减小）
            # 当接近中心时，PID输出会很小，底盘几乎不转
        
        return {
            "x.vel": x_cmd,
            "y.vel": 0.0,
            "theta.vel": theta_cmd,
            "arrived": False,
            "state": self.state
        }
    
    def _step_search(self, current_time):
        """
        步进式搜索：旋转一段 → 停止 → 检测 → 没找到继续
        
        解决连续旋转导致的图像模糊问题
        """
        if self.search_phase == "rotate":
            # 开始旋转
            if self.search_start_time == 0:
                self.search_start_time = current_time
                logger.info("🔍 开始步进搜索：旋转")
            
            # 计算已旋转的时间
            elapsed = current_time - self.search_start_time
            
            # 旋转指定角度后停止（根据速度计算时间）
            # 旋转角度 = 速度 * 时间 → 时间 = 角度 / 速度
            rotate_duration = SEARCH_ROT_ANGLE / ROT_SPEED
            
            if elapsed < rotate_duration:
                # 继续旋转
                return {
                    "x.vel": 0.0,
                    "y.vel": 0.0,
                    "theta.vel": ROT_SPEED,
                    "arrived": False,
                    "state": "searching"
                }
            else:
                # 旋转完成，进入停止检测阶段
                self.search_phase = "stop"
                self.search_timer = current_time
                logger.info("🛑 停止旋转，准备检测")
        
        elif self.search_phase == "stop":
            # 停止一段时间，让图像稳定
            elapsed = current_time - self.search_timer
            
            if elapsed < SEARCH_STOP_TIME:
                # 保持停止，等待检测
                return {
                    "x.vel": 0.0,
                    "y.vel": 0.0,
                    "theta.vel": 0.0,
                    "arrived": False,
                    "state": "searching"
                }
            else:
                # 停止时间到，进入检测阶段（由主循环完成检测后再次调用）
                self.search_phase = "detect"
                logger.info("📷 检测阶段")
        
        elif self.search_phase == "detect":
            # 检测阶段：如果还是没检测到，重新开始旋转
            self.search_phase = "rotate"
            self.search_start_time = 0
            logger.info("🔄 未检测到目标，继续搜索")
        
        # 默认返回搜索状态
        self.state = "searching"
        return {
            "x.vel": 0.0,
            "y.vel": 0.0,
            "theta.vel": 0.0,
            "arrived": False,
            "state": "searching"
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
                    
                    # YOLO 推理 - 识别纸团
                    frame_counter += 1
                    # 自动导航模式下每帧都检测（避免丢失目标）
                    # 手动模式下每3帧检测一次（降低负载）
                    should_detect = (auto_mode or 
                                   (yolo_model is not None and frame_counter % (YOLO_SKIP_FRAMES + 1) == 0))
                    
                    if yolo_model is not None and should_detect:
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
                        nav_cmd = navigator.calculate_velocity(last_detections, time.time())
                        
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
