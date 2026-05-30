#!/usr/bin/env python
"""树莓派端 Host - 多进程优化版
架构：主进程(ZMQ+图像+导航) | 控制进程(底盘100Hz) | 推理进程(YOLO)"""

import sys, time, logging, json, base64
import multiprocessing as mp
from multiprocessing import Process, Queue, shared_memory
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path.home() / "lerobot-workspace/lerobot/src"))
import cv2, numpy as np, zmq
from ultralytics import YOLO
from lerobot.robots.lekiwi import LeKiwi
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig

# ==================== 配置 ====================
ROBOT_ID = "lekiwi"
FRONT_CAMERA = "/dev/video2"
WRIST_CAMERA = "/dev/video0"  # 腕部摄像头
SERIAL_PORT = "/dev/ttyACM0"
CMD_PORT, OBS_PORT = 5555, 5556
WATCHDOG_MS, FPS = 2000, 30

YOLO_MODEL = Path.home() / "lerobot-workspace/lekiwi-pi/models/paper_ball_detection-1-8/weights/best.pt"
YOLO_CONF, YOLO_SIZE = 0.5, 320

IMG_W, IMG_H = 640, 480
IMG_AREA = IMG_W * IMG_H
TARGET_RATIO = 0.20
CENTER_THRESH = 0.10
NAV_SPEED = 0.25
ROT_SPEED = 20
SEARCH_ANGLE = 20
SEARCH_STOP = 0.5
LOST_MAX = 5

PID_KP, PID_KI, PID_KD, PID_MAX_I = 15.0, 0.1, 8.0, 5.0

# ACT抓取配置
ACT_GRASP_DURATION = 5.0  # ACT抓取持续时间（秒）
ACT_ARRIVED_DELAY = 2.0   # 到达后延迟多久开始抓取

ARM_DEFAULTS = {
    "arm_shoulder_pan.pos": 0.0, "arm_shoulder_lift.pos": -100.0,
    "arm_elbow_flex.pos": 90.0, "arm_wrist_flex.pos": 70.0,
    "arm_wrist_roll.pos": 0.0, "arm_gripper.pos": 0.0,
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
        d = 0.0 if self.first else (self.kd * (err - self.last_err) / dt if dt > 0 else 0.0)
        self.first = False
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
        self.arrived_time = 0
        self.grasp_start_time = 0
    
    def update(self, detections):
        now = time.time()
        
        # === ACT抓取状态 ===
        if self.state == "grasping":
            elapsed = now - self.grasp_start_time
            if elapsed >= ACT_GRASP_DURATION:
                logger.info("✅ ACT抓取完成，重置状态")
                self.reset()
                return {"x": 0.0, "y": 0.0, "theta": 0.0, "state": "idle", "request_act": False}
            
            # 继续抓取，发送观测请求
            progress = elapsed / ACT_GRASP_DURATION
            return {
                "x": 0.0, "y": 0.0, "theta": 0.0,
                "state": "grasping",
                "request_act": True,
                "grasp_progress": progress
            }
        
        # === 已到达，准备进入抓取 ===
        if self.state == "arrived":
            if self.arrived_time == 0:
                self.arrived_time = now
                logger.info("✅ 已到达目标，准备抓取...")
            elif now - self.arrived_time >= ACT_ARRIVED_DELAY:
                logger.info("🤖 进入ACT抓取状态")
                self.state = "grasping"
                self.grasp_start_time = now
                return {
                    "x": 0.0, "y": 0.0, "theta": 0.0,
                    "state": "grasping",
                    "request_act": True,
                    "grasp_progress": 0.0
                }
            
            return {"x": 0.0, "y": 0.0, "theta": 0.0, "state": "arrived", "request_act": False}
        
        # === 目标跟踪 ===
        if detections:
            # 多目标时选择距离中心最近的
            if len(detections) > 1:
                for det in detections:
                    det["_dist"] = abs(det["center"][0] - self.cx)
                detections.sort(key=lambda x: x["_dist"])
                for det in detections:
                    det.pop("_dist", None)
            
            target = detections[0]
            
            if self.state in ("idle", "searching"):
                logger.info(f"🎯 发现{' ' + str(len(detections)) + '个' if len(detections) > 1 else ''}目标，开始跟踪")
                self.pid.reset()
            
            self.target = target
            self.lost = 0
            return self._track(self.target, now)
        
        # 丢失缓冲
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
                self.state = "arrived"
                self.arrived_time = 0  # 将在update中设置
            return {"x": 0.0, "y": 0.0, "theta": 0.0, "state": "arrived", "request_act": False}
        
        nx = (cx - self.cx) / (IMG_W / 2)
        dt = now - self.last_t
        self.last_t = now
        theta = max(-ROT_SPEED, min(ROT_SPEED, -self.pid.compute(nx, dt)))
        x_vel = NAV_SPEED * (1.0 - ratio / TARGET_RATIO)
        
        if abs(nx) > CENTER_THRESH:
            self.state = "aligning"
            x_vel *= 0.3
        else:
            self.state = "approaching"
        
        return {"x": x_vel, "y": 0.0, "theta": theta, "state": self.state, "request_act": False}
    
    def _search(self, now):
        self.state = "searching"
        
        if self.search_phase == "rotate":
            if self.search_t == 0:
                self.search_t = now
                logger.info("🔍 搜索中...")
            
            if now - self.search_t < SEARCH_ANGLE / ROT_SPEED:
                return {"x": 0.0, "y": 0.0, "theta": ROT_SPEED, "state": "searching", "request_act": False}
            
            self.search_phase = "stop"
            self.search_t = now
        
        elif self.search_phase == "stop":
            if now - self.search_t < SEARCH_STOP:
                return {"x": 0.0, "y": 0.0, "theta": 0.0, "state": "searching", "request_act": False}
            self.search_phase = "detect"
        
        elif self.search_phase == "detect":
            self.search_phase = "rotate"
            self.search_t = 0
        
        return {"x": 0.0, "y": 0.0, "theta": 0.0, "state": "searching", "request_act": False}


# ==================== 进程1：底盘控制 ====================
def controller_worker(cmd_queue):
    """底盘控制进程 - 100Hz高频，独立运行"""
    logger.info("[Controller] 初始化...")
    try:
        robot = LeKiwi(LeKiwiConfig(port=SERIAL_PORT, id=ROBOT_ID, cameras={}))
        robot.connect()
        logger.info("[Controller] 已连接")
    except Exception as e:
        logger.error(f"[Controller] 连接失败: {e}")
        return
    
    last_cmd_t = time.time()
    watchdog_on = False
    
    # 维护当前机械臂状态，避免被默认值覆盖
    current_arm_state = dict(ARM_DEFAULTS)
    
    try:
        while True:
            loop_start = time.time()
            
            try:
                cmd = cmd_queue.get(timeout=0.001)
                if cmd is None:
                    break
                
                # 更新当前机械臂状态（只更新 cmd 中提供的键）
                for key in ["arm_shoulder_pan.pos", "arm_shoulder_lift.pos", 
                           "arm_elbow_flex.pos", "arm_wrist_flex.pos",
                           "arm_wrist_roll.pos", "arm_gripper.pos"]:
                    if key in cmd:
                        current_arm_state[key] = cmd[key]
                
                # 构建完整命令
                full_cmd = dict(current_arm_state)
                
                # 添加底盘速度
                for vel_key in ["x.vel", "y.vel", "theta.vel"]:
                    if vel_key in cmd:
                        full_cmd[vel_key] = cmd[vel_key]
                
                # 检查是否是ACT命令并记录
                has_arm_cmd = any(k.startswith("arm_") for k in cmd.keys())
                if has_arm_cmd:
                    logger.info(f"[Controller] 📤 发送机械臂命令: pan={full_cmd.get('arm_shoulder_pan.pos', 0):.1f}, "
                              f"lift={full_cmd.get('arm_shoulder_lift.pos', 0):.1f}, "
                              f"elbow={full_cmd.get('arm_elbow_flex.pos', 0):.1f}, "
                              f"wrist_flex={full_cmd.get('arm_wrist_flex.pos', 0):.1f}, "
                              f"wrist_roll={full_cmd.get('arm_wrist_roll.pos', 0):.1f}, "
                              f"gripper={full_cmd.get('arm_gripper.pos', 0):.1f}")
                
                robot.send_action(full_cmd)
                last_cmd_t = time.time()
                watchdog_on = False
            except:
                # queue.Empty is normal when no command available
                pass
            
            if not watchdog_on and (time.time() - last_cmd_t > WATCHDOG_MS / 1000):
                logger.warning("[Controller] 看门狗超时")
                robot.stop_base()
                watchdog_on = True
            
            # 100Hz
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
    """YOLO推理进程 - 独立运行，零拷贝读取图像"""
    logger.info("[Inference] 加载模型...")
    try:
        if not model_path.exists():
            logger.error(f"[Inference] 模型不存在")
            return
        model = YOLO(str(model_path))
        logger.info(f"[Inference] 已加载: {list(model.names.values())}")
    except Exception as e:
        logger.error(f"[Inference] 加载失败: {e}")
        return
    
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
    except Exception as e:
        logger.error(f"[Inference] 共享内存连接失败: {e}")
        return
    
    try:
        while True:
            loop_start = time.time()
            
            frame = np.ndarray(img_shape, dtype=np.uint8, buffer=shm.buf)
            
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
            
            # 30fps
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
    
    # 创建共享内存
    img_size = IMG_W * IMG_H * 3
    shm = shared_memory.SharedMemory(create=True, size=img_size)
    logger.info(f"[Main] 共享内存: {shm.name}, {img_size} bytes")
    
    # 创建队列
    cmd_queue = Queue(maxsize=10)
    det_queue = Queue(maxsize=1)
    
    # 在主进程中进行校准（子进程没有stdin，无法交互）
    logger.info("[Main] 检查机器人校准...")
    try:
        cal_robot = LeKiwi(LeKiwiConfig(port=SERIAL_PORT, id=ROBOT_ID, cameras={}))
        cal_robot.connect(calibrate=True)
        cal_robot.disconnect()
        logger.info("[Main] ✅ 校准检查完成")
    except Exception as e:
        logger.warning(f"[Main] 校准检查跳过: {e}")
    
    # 启动子进程
    logger.info("[Main] 启动进程...")
    ctrl_proc = Process(target=controller_worker, args=(cmd_queue,), name="Controller")
    inf_proc = Process(target=inference_worker, args=(shm.name, (IMG_H, IMG_W, 3), det_queue, YOLO_MODEL, YOLO_CONF, YOLO_SIZE), name="Inference")
    ctrl_proc.start()
    inf_proc.start()
    time.sleep(3)
    
    # ZMQ
    ctx = zmq.Context()
    cmd_sock = ctx.socket(zmq.PULL)
    cmd_sock.setsockopt(zmq.CONFLATE, 1)
    cmd_sock.bind(f"tcp://*:{CMD_PORT}")
    
    obs_sock = ctx.socket(zmq.PUSH)
    obs_sock.setsockopt(zmq.CONFLATE, 1)
    obs_sock.bind(f"tcp://*:{OBS_PORT}")
    
    logger.info(f"[Main] 等待连接... CMD:{CMD_PORT} OBS:{OBS_PORT}")
    logger.info("[Main] Ctrl+C 停止")
    
    # 状态
    auto_mode = False
    navigator = Navigator()
    last_nav_state = ""
    client_detected = False
    detections = []
    
    # 摄像头
    logger.info("[Main] 初始化摄像头...")
    cap = None
    cap_wrist = None
    
    try:
        # 前视摄像头
        camera_idx = int(FRONT_CAMERA.replace("/dev/video", ""))
        cap = cv2.VideoCapture(camera_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        
        # 腕部摄像头
        wrist_idx = int(WRIST_CAMERA.replace("/dev/video", ""))
        cap_wrist = cv2.VideoCapture(wrist_idx)
        cap_wrist.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
        cap_wrist.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)
        cap_wrist.set(cv2.CAP_PROP_FPS, FPS)
        
        logger.info("[Main] 双摄像头已就绪")
    except Exception as e:
        logger.error(f"[Main] 摄像头失败: {e}")
        if cap is None:
            cap = None
        if cap_wrist is None:
            cap_wrist = None
    
    try:
        while True:
            loop_start = time.time()
            
            # 1. 接收命令
            try:
                msg = cmd_sock.recv_string(zmq.NOBLOCK)
                data = json.loads(msg)
                
                if not client_detected:
                    client_detected = True
                    logger.info("[Main] 📱 Client已连接")
                
                if data.get("toggle_auto"):
                    auto_mode = not auto_mode
                    navigator.reset()
                    logger.info(f"{'🤖 自动导航' if auto_mode else '🎮 手动控制'}")
                
                # 检查是否是ACT动作（来自PC端推理）
                if data.get("source") == "act":
                    # 提取ACT动作（包含机械臂位置+底盘速度）
                    act_cmd = {
                        "arm_shoulder_pan.pos": data.get("arm_shoulder_pan.pos", ARM_DEFAULTS["arm_shoulder_pan.pos"]),
                        "arm_shoulder_lift.pos": data.get("arm_shoulder_lift.pos", ARM_DEFAULTS["arm_shoulder_lift.pos"]),
                        "arm_elbow_flex.pos": data.get("arm_elbow_flex.pos", ARM_DEFAULTS["arm_elbow_flex.pos"]),
                        "arm_wrist_flex.pos": data.get("arm_wrist_flex.pos", ARM_DEFAULTS["arm_wrist_flex.pos"]),
                        "arm_wrist_roll.pos": data.get("arm_wrist_roll.pos", ARM_DEFAULTS["arm_wrist_roll.pos"]),
                        "arm_gripper.pos": data.get("arm_gripper.pos", ARM_DEFAULTS["arm_gripper.pos"]),
                        "x.vel": data.get("x.vel", 0.0),
                        "y.vel": data.get("y.vel", 0.0),
                        "theta.vel": data.get("theta.vel", 0.0)
                    }
                    logger.info(f"[Main] 📥 收到ACT命令: pan={act_cmd['arm_shoulder_pan.pos']:.1f}, "
                              f"lift={act_cmd['arm_shoulder_lift.pos']:.1f}, "
                              f"elbow={act_cmd['arm_elbow_flex.pos']:.1f}, "
                              f"wrist_flex={act_cmd['arm_wrist_flex.pos']:.1f}, "
                              f"wrist_roll={act_cmd['arm_wrist_roll.pos']:.1f}, "
                              f"gripper={act_cmd['arm_gripper.pos']:.1f}")
                    try:
                        cmd_queue.put_nowait(act_cmd)
                        logger.info(f"[Main] ✅ ACT命令已发送到controller队列")
                    except Exception as e:
                        logger.error(f"[Main] ❌ ACT命令发送失败: {e}")
                elif not auto_mode:
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
            
            # 2. 获取检测结果
            try:
                while not det_queue.empty():
                    detections = det_queue.get_nowait()
            except:
                pass
            
            # 4. 自动导航 / ACT抓取
            request_act = False
            grasp_progress = 0.0
            
            if auto_mode:
                nav = navigator.update(detections)
                
                if nav["state"] != last_nav_state:
                    last_nav_state = nav["state"]
                    icons = {"searching": "🔍", "aligning": "🎯", "approaching": "🚀", 
                            "arrived": "✅", "grasping": "🦾", "idle": "⏹️"}
                    logger.info(f"{icons.get(nav['state'], '🤖')} {nav['state']}")
                
                # 获取ACT请求标志和进度
                request_act = nav.get("request_act", False)
                grasp_progress = nav.get("grasp_progress", 0.0)
                
                # 发送底盘命令（仅在非grasping状态，避免覆盖ACT动作）
                if nav["state"] != "grasping":
                    try:
                        cmd_queue.put_nowait({
                            "x.vel": nav["x"], 
                            "y.vel": nav["y"], 
                            "theta.vel": nav["theta"]
                        })
                    except:
                        pass
            
            # 5. 读取图像
            front_b64 = None
            wrist_b64 = None
            
            # 读取前视摄像头
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                    
                    # 写入共享内存
                    try:
                        shm_array = np.ndarray((IMG_H, IMG_W, 3), dtype=np.uint8, buffer=shm.buf)
                        shm_array[:] = frame[:]
                    except Exception as e:
                        logger.error(f"[Main] 共享内存错误: {e}")
                    
                    # 编码前视图像
                    try:
                        ret_jpg, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        if ret_jpg:
                            front_b64 = base64.b64encode(buf).decode()
                    except Exception as e:
                        logger.error(f"[Main] 前视图像编码错误: {e}")
            
            # 读取腕部摄像头
            if cap_wrist and cap_wrist.isOpened():
                ret_wrist, frame_wrist = cap_wrist.read()
                if ret_wrist and frame_wrist is not None:
                    try:
                        ret_jpg_wrist, buf_wrist = cv2.imencode(".jpg", frame_wrist, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        if ret_jpg_wrist:
                            wrist_b64 = base64.b64encode(buf_wrist).decode()
                    except Exception as e:
                        logger.error(f"[Main] 腕部图像编码错误: {e}")
            
            # 发送观测数据
            if front_b64:
                obs = {
                    "front": front_b64,
                    "wrist": wrist_b64,
                    "detections": detections,
                    "auto_mode": auto_mode,
                    "nav_state": navigator.state if auto_mode else "manual",
                    "request_act": request_act,
                    "grasp_progress": grasp_progress
                }
                try:
                    obs_sock.send_string(json.dumps(obs), flags=zmq.NOBLOCK)
                except:
                    pass
            
            # 6. 帧率控制
            elapsed = time.time() - loop_start
            sleep_t = max(1 / FPS - elapsed, 0)
            if sleep_t > 0:
                time.sleep(sleep_t)
    
    except KeyboardInterrupt:
        logger.info("[Main] 停止")
    finally:
        try:
            cmd_queue.put(None)
        except:
            pass
        
        time.sleep(1)
        if ctrl_proc.is_alive():
            ctrl_proc.terminate()
        if inf_proc.is_alive():
            inf_proc.terminate()
        
        if cap:
            cap.release()
        if cap_wrist:
            cap_wrist.release()
        
        shm.close()
        shm.unlink()
        
        cmd_sock.close()
        obs_sock.close()
        ctx.term()
        
        logger.info("[Main] 已关闭")


if __name__ == "__main__":
    main()
