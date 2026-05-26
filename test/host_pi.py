#!/usr/bin/env python
"""
树莓派端 Host - 优化版
功能：底盘控制 + YOLO检测 + 自动导航
优化：减少日志、简化逻辑、提升性能
"""

import sys
import time
import logging
import json
import base64
from pathlib import Path

# 配置日志（在导入其他库之前）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path.home() / "lerobot-workspace/lerobot/src"))

import cv2
import zmq
from ultralytics import YOLO
from lerobot.robots.lekiwi import LeKiwi
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig, LeKiwiHostConfig
from lerobot.cameras import Cv2Rotation
from lerobot.cameras.opencv import OpenCVCameraConfig


# ==================== 配置 ====================
ROBOT_ID = "lekiwi"
FRONT_CAMERA = "/dev/video2"
SERIAL_PORT = "/dev/ttyACM0"
ZMQ_CMD_PORT = 5555
ZMQ_OBS_PORT = 5556
WATCHDOG_MS = 2000
FPS = 30

# YOLO配置
YOLO_MODEL_PATH = Path.home() / "lerobot-workspace/lekiwi-pi/models/paper_ball_detection-1-8/weights/best.pt"
YOLO_CONFIDENCE = 0.5
YOLO_INFER_SIZE = 320
YOLO_SKIP_FRAMES = 2

# 导航配置
IMG_W, IMG_H = 640, 480
IMG_AREA = IMG_W * IMG_H
TARGET_RATIO = 0.20
CENTER_THRESH = 0.10
NAV_SPEED = 0.25
ROT_SPEED = 20
SEARCH_ANGLE = 20
SEARCH_STOP = 0.5
LOST_MAX = 5

# PID参数
PID_KP, PID_KI, PID_KD, PID_MAX_I = 15.0, 0.1, 8.0, 5.0

# 从臂默认值（预计算，避免每帧创建）
ARM_DEFAULTS = {
    "arm_shoulder_pan.pos": 0.0,
    "arm_shoulder_lift.pos": -100.0,
    "arm_elbow_flex.pos": 90.0,
    "arm_wrist_flex.pos": 70.0,
    "arm_wrist_roll.pos": 0.0,
    "arm_gripper.pos": 0.0,
}


# ==================== PID控制器 ====================
class PID:
    def __init__(self, kp, ki, kd, max_i):
        self.kp, self.ki, self.kd, self.max_i = kp, ki, kd, max_i
        self.reset()
    
    def reset(self):
        self.integral = self.last_err = 0.0
        self.first = True
    
    def compute(self, err, dt):
        self.integral = max(-self.max_i, min(self.max_i, self.integral + err * dt))
        if self.first:
            d = 0.0
            self.first = False
        else:
            d = self.kd * (err - self.last_err) / dt if dt > 0 else 0.0
        self.last_err = err
        return self.kp * err + self.ki * self.integral + d


# ==================== 自动导航 ====================
class Navigator:
    def __init__(self):
        self.cx, self.cy = IMG_W / 2, IMG_H / 2
        self.pid = PID(PID_KP, PID_KI, PID_KD, PID_MAX_I)
        self.last_t = time.time()
        self.reset()
    
    def reset(self):
        self.state = "idle"
        self.target = None
        self.lost = 0
        self.search_t = 0
        self.search_phase = "rotate"
    
    def update(self, detections):
        now = time.time()
        
        # 检测到目标
        if detections:
            if self.state in ("idle", "searching"):
                logger.info("🎯 发现目标，开始跟踪")
                self.pid.reset()
            self.target = detections[0]
            self.lost = 0
            return self._track(self.target, now)
        
        # 丢失目标，缓冲跟踪
        if self.target and self.lost < LOST_MAX:
            self.lost += 1
            return self._track(self.target, now)
        
        # 完全丢失，搜索
        if self.target:
            logger.info("❌ 目标丢失，开始搜索")
            self.target = None
            self.pid.reset()
        
        return self._search(now)
    
    def _track(self, t, now):
        x1, y1, x2, y2 = t["bbox"]
        cx, w, h = t["center"][0], t["size"][0], t["size"][1]
        ratio = (w * h) / IMG_AREA
        
        # 到达检查
        if ratio >= TARGET_RATIO:
            if self.state != "arrived":
                logger.info("✅ 已到达目标")
                self.state = "arrived"
            return {"x": 0.0, "y": 0.0, "theta": 0.0, "state": "arrived"}
        
        # 计算偏差
        nx = (cx - self.cx) / (IMG_W / 2)
        
        # PID旋转
        dt = now - self.last_t
        self.last_t = now
        theta = -self.pid.compute(nx, dt)
        theta = max(-ROT_SPEED, min(ROT_SPEED, theta))
        
        # 前进速度
        x_vel = NAV_SPEED * (1.0 - ratio / TARGET_RATIO)
        
        # 偏差大时降低前进
        if abs(nx) > CENTER_THRESH:
            self.state = "aligning"
            x_vel *= 0.3
        else:
            self.state = "approaching"
        
        return {"x": x_vel, "y": 0.0, "theta": theta, "state": self.state}
    
    def _search(self, now):
        self.state = "searching"
        
        if self.search_phase == "rotate":
            if self.search_t == 0:
                self.search_t = now
                logger.info("🔍 搜索中...")
            
            if now - self.search_t < SEARCH_ANGLE / ROT_SPEED:
                return {"x": 0.0, "y": 0.0, "theta": ROT_SPEED, "state": "searching"}
            
            self.search_phase = "stop"
            self.search_t = now
        
        elif self.search_phase == "stop":
            if now - self.search_t < SEARCH_STOP:
                return {"x": 0.0, "y": 0.0, "theta": 0.0, "state": "searching"}
            self.search_phase = "detect"
        
        elif self.search_phase == "detect":
            self.search_phase = "rotate"
            self.search_t = 0
        
        return {"x": 0.0, "y": 0.0, "theta": 0.0, "state": "searching"}


# ==================== 主程序 ====================
def main():
    logger.info("=" * 50)
    logger.info(f"LeKiwi Host - {ROBOT_ID}")
    logger.info("=" * 50)
    
    # 加载模型
    logger.info("加载YOLO模型...")
    if not YOLO_MODEL_PATH.exists():
        logger.error(f"模型不存在: {YOLO_MODEL_PATH}")
        yolo_model = None
    else:
        try:
            yolo_model = YOLO(str(YOLO_MODEL_PATH))
            logger.info(f"✅ 模型加载成功: {list(yolo_model.names.values())}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            yolo_model = None
    
    # 初始化机器人
    logger.info("初始化机器人...")
    robot = LeKiwi(LeKiwiConfig(
        port=SERIAL_PORT,
        id=ROBOT_ID,
        cameras={"front": OpenCVCameraConfig(
            index_or_path=FRONT_CAMERA, fps=FPS,
            width=IMG_W, height=IMG_H,
            rotation=Cv2Rotation.ROTATE_180, warmup_s=3.0
        )},
    ))
    robot.connect()
    
    # ZMQ
    ctx = zmq.Context()
    cmd_sock = ctx.socket(zmq.PULL)
    cmd_sock.setsockopt(zmq.CONFLATE, 1)
    cmd_sock.bind(f"tcp://*:{ZMQ_CMD_PORT}")
    
    obs_sock = ctx.socket(zmq.PUSH)
    obs_sock.setsockopt(zmq.CONFLATE, 1)
    obs_sock.bind(f"tcp://*:{ZMQ_OBS_PORT}")
    
    logger.info(f"等待连接... 命令:{ZMQ_CMD_PORT} 图像:{ZMQ_OBS_PORT}")
    logger.info("按 Ctrl+C 停止")
    
    # 状态
    auto_mode = False
    navigator = Navigator()
    last_cmd_t = time.time()
    watchdog_on = False
    frame_cnt = 0
    detections = []
    last_nav_state = ""
    
    try:
        while True:
            loop_start = time.time()
            
            # 1. 接收命令
            try:
                msg = cmd_sock.recv_string(zmq.NOBLOCK)
                data = json.loads(msg)
                
                # 切换自动导航
                if data.get("toggle_auto"):
                    auto_mode = not auto_mode
                    navigator.reset()
                    logger.info(f"{'🤖 自动导航' if auto_mode else '🎮 手动控制'}")
                    last_cmd_t = time.time()
                    watchdog_on = False
                    continue
                
                # 手动模式执行命令
                if not auto_mode:
                    # 补齐从臂默认值
                    cmd = {**ARM_DEFAULTS, **data}
                    robot.send_action(cmd)
                    last_cmd_t = time.time()
                    watchdog_on = False
            
            except zmq.Again:
                pass
            except Exception as e:
                logger.error(f"命令错误: {e}")
            
            # 2. 看门狗（自动导航模式下不触发，因为本机在发命令）
            if not auto_mode and not watchdog_on and (time.time() - last_cmd_t > WATCHDOG_MS / 1000):
                logger.warning("看门狗超时，停止底盘")
                watchdog_on = True
                robot.stop_base()
            
            # 3. 获取图像
            try:
                obs = robot.get_observation()
                
                if "front" in obs:
                    frame = obs["front"]
                    if isinstance(frame, cv2.UMat):
                        frame = frame.get()
                    
                    # YOLO检测
                    frame_cnt += 1
                    should_detect = auto_mode or (yolo_model and frame_cnt % (YOLO_SKIP_FRAMES + 1) == 0)
                    
                    if yolo_model and should_detect:
                        try:
                            results = yolo_model(frame, conf=YOLO_CONFIDENCE, verbose=False, imgsz=YOLO_INFER_SIZE)
                            if len(results) > 0 and len(results[0].boxes) > 0:
                                detections = []
                                for box in results[0].boxes:
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    detections.append({
                                        "class": results[0].names[int(box.cls[0])],
                                        "confidence": round(float(box.conf[0]), 4),
                                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                        "center": [float((x1+x2)/2), float((y1+y2)/2)],
                                        "size": [float(x2-x1), float(y2-y1)]
                                    })
                            else:
                                detections = []
                        except Exception:
                            pass
                    
                    # 自动导航
                    if auto_mode and yolo_model:
                        nav = navigator.update(detections)
                        
                        # 只在状态变化时打印
                        if nav["state"] != last_nav_state:
                            last_nav_state = nav["state"]
                            icons = {"searching": "🔍", "aligning": "🎯", "approaching": "🚀", "arrived": "✅"}
                            logger.info(f"{icons.get(nav['state'], '🤖')} {nav['state']}")
                        
                        # 发送导航命令
                        robot.send_action({
                            "x.vel": nav["x"], "y.vel": nav["y"], "theta.vel": nav["theta"],
                            **ARM_DEFAULTS
                        })
                        last_cmd_t = time.time()
                        watchdog_on = False
                    
                    # 编码发送
                    obs["detections"] = detections
                    obs["auto_mode"] = auto_mode
                    obs["nav_state"] = navigator.state if auto_mode else "manual"
                    
                    ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    obs["front"] = base64.b64encode(buf).decode() if ret else ""
                    
                    obs_sock.send_string(json.dumps(obs), flags=zmq.NOBLOCK)
            
            except zmq.Again:
                pass
            except Exception as e:
                logger.error(f"图像错误: {e}")
            
            # 4. 帧率控制
            elapsed = time.time() - loop_start
            sleep_t = max(1 / FPS - elapsed, 0)
            if sleep_t > 0:
                time.sleep(sleep_t)
    
    except KeyboardInterrupt:
        logger.info("停止")
    finally:
        robot.disconnect()
        cmd_sock.close()
        obs_sock.close()
        ctx.term()
        logger.info("已关闭")


if __name__ == "__main__":
    main()
