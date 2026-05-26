#!/usr/bin/env python
"""
树莓派端 Host - 多进程优化版
架构：
  - 主进程：ZMQ通信 + 图像采集 + 自动导航 + 图像编码发送
  - 控制进程：底盘控制（100Hz高频，实时性最高）
  - 推理进程：YOLO推理（独立运行，不阻塞控制）

优化点：
  1. 底盘控制独立进程，不受YOLO推理阻塞
  2. 共享内存传递图像，零拷贝
  3. 控制频率稳定在100Hz，推理频率30Hz
"""

import sys
import time
import logging
import json
import base64
import multiprocessing as mp
from multiprocessing import Process, Queue, shared_memory
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path.home() / "lerobot-workspace/lerobot/src"))

import cv2
import numpy as np
import zmq
from ultralytics import YOLO
from lerobot.robots.lekiwi import LeKiwi
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig
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

# 从臂默认值
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


# ==================== 导航器 ====================
class Navigator:
    def __init__(self):
        self.cx = IMG_W / 2
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
        
        if detections:
            # 选择距离中心最近的目标
            if len(detections) > 1:
                for det in detections:
                    dx = det["center"][0] - self.cx
                    det["_dist"] = abs(dx)
                detections.sort(key=lambda x: x["_dist"])
                for det in detections:
                    det.pop("_dist", None)
            
            target = detections[0]
            
            if self.state in ("idle", "searching"):
                count = len(detections)
                if count > 1:
                    logger.info(f"🎯 发现 {count} 个目标，选择最近的一个跟踪")
                else:
                    logger.info("🎯 发现目标，开始跟踪")
                self.pid.reset()
            
            self.target = target
            self.lost = 0
            return self._track(self.target, now)
        
        if self.target and self.lost < LOST_MAX:
            self.lost += 1
            return self._track(self.target, now)
        
        if self.target:
            logger.info("❌ 目标丢失，开始搜索")
            self.target = None
            self.pid.reset()
        
        return self._search(now)
    
    def _track(self, t, now):
        cx, w, h = t["center"][0], t["size"][0], t["size"][1]
        ratio = (w * h) / IMG_AREA
        
        if ratio >= TARGET_RATIO:
            if self.state != "arrived":
                logger.info("✅ 已到达目标")
                self.state = "arrived"
            return {"x": 0.0, "y": 0.0, "theta": 0.0, "state": "arrived"}
        
        nx = (cx - self.cx) / (IMG_W / 2)
        dt = now - self.last_t
        self.last_t = now
        theta = -self.pid.compute(nx, dt)
        theta = max(-ROT_SPEED, min(ROT_SPEED, theta))
        x_vel = NAV_SPEED * (1.0 - ratio / TARGET_RATIO)
        
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


# ==================== 进程1：底盘控制 ====================
def controller_worker(cmd_queue):
    """
    底盘控制进程
    - 高频循环（100Hz）
    - 从Queue读取命令并执行
    - 看门狗保护
    - 独立运行，不受推理阻塞
    """
    logger.info("[Controller] 初始化底盘连接...")
    try:
        # 初始化LeRobot（包含相机和底盘）
        # 注意：相机也会初始化，但控制进程不读取图像
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
        logger.info("[Controller] 底盘连接成功")
    except Exception as e:
        logger.error(f"[Controller] 底盘连接失败: {e}")
        return
    
    last_cmd_t = time.time()
    watchdog_on = False
    
    try:
        while True:
            loop_start = time.time()
            
            # 非阻塞读取命令（最多等待1ms）
            try:
                cmd = cmd_queue.get(timeout=0.001)
                if cmd is None:  # 退出信号
                    break
                
                # 补齐从臂默认值并发送
                full_cmd = {**ARM_DEFAULTS, **cmd}
                robot.send_action(full_cmd)
                last_cmd_t = time.time()
                watchdog_on = False
            except:
                pass
            
            # 看门狗：2秒无命令则停止底盘
            if not watchdog_on and (time.time() - last_cmd_t > WATCHDOG_MS / 1000):
                logger.warning("[Controller] 看门狗超时，停止底盘")
                robot.stop_base()
                watchdog_on = True
            
            # 100Hz控制频率（10ms周期）
            elapsed = time.time() - loop_start
            sleep_t = max(0.01 - elapsed, 0)
            if sleep_t > 0:
                time.sleep(sleep_t)
    
    except KeyboardInterrupt:
        pass
    finally:
        robot.stop_base()
        robot.disconnect()
        logger.info("[Controller] 已关闭")


# ==================== 进程2：YOLO推理 ====================
def inference_worker(shm_name, img_shape, det_queue, model_path, conf, infer_size):
    """
    YOLO推理进程
    - 从共享内存读取图像
    - 独立推理，不阻塞控制
    - 结果放入Queue（maxsize=1，旧结果自动丢弃）
    """
    logger.info("[Inference] 加载YOLO模型...")
    try:
        if not model_path.exists():
            logger.error(f"[Inference] 模型不存在: {model_path}")
            return
        model = YOLO(str(model_path))
        logger.info(f"[Inference] 模型加载成功: {list(model.names.values())}")
    except Exception as e:
        logger.error(f"[Inference] 模型加载失败: {e}")
        return
    
    # 连接共享内存
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        logger.info(f"[Inference] 已连接共享内存: {shm_name}")
    except Exception as e:
        logger.error(f"[Inference] 共享内存连接失败: {e}")
        return
    
    try:
        while True:
            loop_start = time.time()
            
            # 读取共享内存中的图像（零拷贝）
            frame = np.ndarray(img_shape, dtype=np.uint8, buffer=shm.buf)
            
            # YOLO推理
            try:
                results = model(frame, conf=conf, verbose=False, imgsz=infer_size)
                
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
                    
                    # 放入Queue（maxsize=1，旧结果自动丢弃）
                    try:
                        det_queue.put_nowait(detections)
                    except:
                        pass
                else:
                    try:
                        det_queue.put_nowait([])
                    except:
                        pass
            except Exception as e:
                logger.error(f"[Inference] 推理错误: {e}")
            
            # 控制推理频率（约30fps）
            elapsed = time.time() - loop_start
            sleep_t = max(1/30 - elapsed, 0)
            if sleep_t > 0:
                time.sleep(sleep_t)
    
    except KeyboardInterrupt:
        pass
    finally:
        shm.close()
        logger.info("[Inference] 已关闭")


# ==================== 主进程 ====================
def main():
    logger.info("=" * 50)
    logger.info("LeKiwi Host - 多进程版")
    logger.info("=" * 50)
    
    # 创建共享内存（用于主进程→推理进程传递图像）
    # 大小：640*480*3 = 921,600 bytes
    img_size = IMG_W * IMG_H * 3
    shm = shared_memory.SharedMemory(create=True, size=img_size)
    logger.info(f"[Main] 创建共享内存: {shm.name}, 大小: {img_size} bytes")
    
    # 创建进程间通信队列
    cmd_queue = Queue(maxsize=10)      # Main → Controller（底盘命令）
    det_queue = Queue(maxsize=1)       # Inference → Main（检测结果，只保留最新）
    
    # 启动底盘控制进程
    logger.info("[Main] 启动底盘控制进程...")
    ctrl_proc = Process(target=controller_worker, args=(cmd_queue,), name="Controller")
    ctrl_proc.start()
    
    # 启动YOLO推理进程
    logger.info("[Main] 启动YOLO推理进程...")
    inf_proc = Process(
        target=inference_worker,
        args=(shm.name, (IMG_H, IMG_W, 3), det_queue, YOLO_MODEL_PATH, YOLO_CONFIDENCE, YOLO_INFER_SIZE),
        name="Inference"
    )
    inf_proc.start()
    
    # 等待子进程初始化完成
    logger.info("[Main] 等待子进程初始化...")
    time.sleep(3)
    
    # 初始化ZMQ通信
    logger.info("[Main] 初始化ZMQ...")
    ctx = zmq.Context()
    cmd_sock = ctx.socket(zmq.PULL)
    cmd_sock.setsockopt(zmq.CONFLATE, 1)
    cmd_sock.bind(f"tcp://*:{ZMQ_CMD_PORT}")
    
    obs_sock = ctx.socket(zmq.PUSH)
    obs_sock.setsockopt(zmq.CONFLATE, 1)
    obs_sock.bind(f"tcp://*:{ZMQ_OBS_PORT}")
    
    logger.info(f"[Main] 等待连接... 命令:{ZMQ_CMD_PORT} 图像:{ZMQ_OBS_PORT}")
    logger.info("[Main] 按 Ctrl+C 停止")
    
    # 状态变量
    auto_mode = False
    navigator = Navigator()
    last_nav_state = ""
    client_detected = False
    host_start_time = time.time()
    auto_triggered = False
    detections = []
    
    # 初始化OpenCV摄像头（主进程独占）
    logger.info("[Main] 初始化摄像头...")
    try:
        camera_idx = int(FRONT_CAMERA.replace("/dev/video", ""))
        cap = cv2.VideoCapture(camera_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        logger.info("[Main] 摄像头初始化成功")
    except Exception as e:
        logger.error(f"[Main] 摄像头初始化失败: {e}")
        cap = None
    
    try:
        while True:
            loop_start = time.time()
            
            # 1. 接收ZMQ命令（非阻塞）
            try:
                msg = cmd_sock.recv_string(zmq.NOBLOCK)
                data = json.loads(msg)
                
                if not client_detected:
                    client_detected = True
                    logger.info("[Main] 📱 Client已连接")
                
                # 切换自动导航（A键）
                if data.get("toggle_auto"):
                    auto_mode = not auto_mode
                    navigator.reset()
                    logger.info(f"{'🤖 自动导航' if auto_mode else '🎮 手动控制'}")
                
                # 手动模式：直接发送命令到底盘控制进程
                if not auto_mode:
                    cmd = {
                        "x.vel": data.get("x.vel", 0.0),
                        "y.vel": data.get("y.vel", 0.0),
                        "theta.vel": data.get("theta.vel", 0.0),
                    }
                    try:
                        cmd_queue.put_nowait(cmd)
                    except:
                        pass
            
            except zmq.Again:
                pass
            except Exception as e:
                logger.error(f"[Main] 命令错误: {e}")
            
            # 2. 20秒超时检查：无client则自动进入自动导航
            if not client_detected and not auto_mode and not auto_triggered:
                if time.time() - host_start_time >= 20:
                    logger.info("[Main] ⏱️ 20秒内无client连接，自动进入自动导航模式")
                    auto_mode = True
                    auto_triggered = True
                    navigator.reset()
            
            # 3. 从推理进程获取最新结果（非阻塞，清空旧数据）
            try:
                while not det_queue.empty():
                    detections = det_queue.get_nowait()
            except:
                pass
            
            # 4. 自动导航：计算速度命令
            if auto_mode:
                nav = navigator.update(detections)
                
                # 状态变化时打印
                if nav["state"] != last_nav_state:
                    last_nav_state = nav["state"]
                    icons = {"searching": "🔍", "aligning": "🎯", "approaching": "🚀", "arrived": "✅"}
                    logger.info(f"{icons.get(nav['state'], '🤖')} {nav['state']}")
                
                # 发送导航命令到底盘控制进程
                cmd = {
                    "x.vel": nav["x"],
                    "y.vel": nav["y"],
                    "theta.vel": nav["theta"],
                }
                try:
                    cmd_queue.put_nowait(cmd)
                except:
                    pass
            
            # 5. 读取摄像头图像
            frame = None
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    # 旋转180度
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                    
                    # 写入共享内存（给推理进程，零拷贝）
                    try:
                        shm_array = np.ndarray((IMG_H, IMG_W, 3), dtype=np.uint8, buffer=shm.buf)
                        shm_array[:] = frame[:]
                    except Exception as e:
                        logger.error(f"[Main] 共享内存写入错误: {e}")
                    
                    # 编码并发送给client
                    try:
                        ret_jpg, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        if ret_jpg:
                            obs = {
                                "front": base64.b64encode(buf).decode(),
                                "detections": detections,
                                "auto_mode": auto_mode,
                                "nav_state": navigator.state if auto_mode else "manual"
                            }
                            try:
                                obs_sock.send_string(json.dumps(obs), flags=zmq.NOBLOCK)
                            except:
                                pass
                    except Exception as e:
                        logger.error(f"[Main] 图像编码错误: {e}")
            
            # 6. 帧率控制（目标30fps）
            elapsed = time.time() - loop_start
            sleep_t = max(1 / FPS - elapsed, 0)
            if sleep_t > 0:
                time.sleep(sleep_t)
    
    except KeyboardInterrupt:
        logger.info("[Main] 停止")
    finally:
        # 发送退出信号给子进程
        try:
            cmd_queue.put(None)
        except:
            pass
        
        # 等待子进程结束
        time.sleep(1)
        if ctrl_proc.is_alive():
            ctrl_proc.terminate()
        if inf_proc.is_alive():
            inf_proc.terminate()
        
        # 释放摄像头
        if cap:
            cap.release()
        
        # 清理共享内存
        shm.close()
        shm.unlink()
        
        # 关闭ZMQ
        cmd_sock.close()
        obs_sock.close()
        ctx.term()
        
        logger.info("[Main] 已关闭")


if __name__ == "__main__":
    main()
