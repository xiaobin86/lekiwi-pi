#!/usr/bin/env python
"""
树莓派端 录制专用 Host（简化版）
功能：
1. 读取front+wrist两个摄像头
2. 运行YOLO检测纸团
3. 读取机械臂状态
4. 发送图像+检测+状态到PC
5. 接收PC命令控制录制流程

启动方式：
    python src/host_pi_record.py

流程：
    等待PC命令 -> 开始录制 -> 发送图像+状态 -> 结束录制 -> 等待重置
"""

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
WRIST_CAMERA = "/dev/video0"
SERIAL_PORT = "/dev/ttyACM0"
CMD_PORT, OBS_PORT = 5555, 5556
FPS = 30

YOLO_MODEL = Path.home() / "lerobot-workspace/lekiwi-pi/models/paper_ball_detection-1-8/weights/best.pt"
YOLO_CONF, YOLO_SIZE = 0.5, 320

IMG_W, IMG_H = 640, 480


def controller_worker(cmd_queue):
    """底盘+机械臂控制进程"""
    logger.info("[Controller] 初始化...")
    try:
        robot = LeKiwi(LeKiwiConfig(port=SERIAL_PORT, id=ROBOT_ID, cameras={}))
        robot.connect()
        logger.info("[Controller] 已连接")
    except Exception as e:
        logger.error(f"[Controller] 连接失败: {e}")
        return
    
    last_cmd_t = time.time()
    
    try:
        while True:
            loop_start = time.time()
            
            try:
                cmd = cmd_queue.get(timeout=0.001)
                if cmd is None:
                    break
                
                if cmd.get("type") == "action":
                    # 执行动作（来自录制数据）
                    robot.send_action(cmd["data"])
                    last_cmd_t = time.time()
                elif cmd.get("type") == "stop":
                    # 停止底盘
                    robot.stop_base()
            except:
                pass
            
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


def inference_worker(shm_name, img_shape, det_queue, model_path, conf, infer_size):
    """YOLO推理进程"""
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
            
            elapsed = time.time() - loop_start
            sleep_t = max(1/30 - elapsed, 0)
            if sleep_t > 0:
                time.sleep(sleep_t)
    
    except KeyboardInterrupt:
        pass
    finally:
        shm.close()


class RecordHost:
    """录制专用Host（简化版）"""
    
    def __init__(self):
        self.state = "waiting"  # waiting, recording, reset
        self.episode_count = 0
        self.recording_data = []
        
        # 创建共享内存
        img_size = IMG_W * IMG_H * 3
        self.shm = shared_memory.SharedMemory(create=True, size=img_size)
        
        # 创建队列
        self.cmd_queue = Queue(maxsize=10)
        self.det_queue = Queue(maxsize=1)
        
        # 启动子进程
        self.ctrl_proc = Process(target=controller_worker, args=(self.cmd_queue,), name="Controller")
        self.inf_proc = Process(target=inference_worker, args=(self.shm.name, (IMG_H, IMG_W, 3), self.det_queue, YOLO_MODEL, YOLO_CONF, YOLO_SIZE), name="Inference")
        self.ctrl_proc.start()
        self.inf_proc.start()
        time.sleep(3)
        
        # ZMQ
        self.ctx = zmq.Context()
        self.cmd_sock = self.ctx.socket(zmq.PULL)
        self.cmd_sock.setsockopt(zmq.CONFLATE, 1)
        self.cmd_sock.bind(f"tcp://*:{CMD_PORT}")
        
        self.obs_sock = self.ctx.socket(zmq.PUSH)
        self.obs_sock.setsockopt(zmq.CONFLATE, 1)
        self.obs_sock.bind(f"tcp://*:{OBS_PORT}")
        
        # 摄像头
        self.cap_front = cv2.VideoCapture(2)
        self.cap_front.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
        self.cap_front.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)
        
        self.cap_wrist = cv2.VideoCapture(0)
        self.cap_wrist.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.cap_wrist.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        
        logger.info("=" * 60)
        logger.info("LeKiwi 录制Host（简化版）")
        logger.info("流程: 手动定位 -> PC提示录制 -> 录制 -> 重置")
        logger.info("=" * 60)
    
    def read_cameras(self):
        """读取两个摄像头"""
        front = wrist = None
        
        if self.cap_front and self.cap_front.isOpened():
            ret, frame = self.cap_front.read()
            if ret:
                front = cv2.rotate(frame, cv2.ROTATE_180)
        
        if self.cap_wrist and self.cap_wrist.isOpened():
            ret, frame = self.cap_wrist.read()
            if ret:
                wrist = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        return front, wrist
    
    def encode_image(self, frame):
        """编码图像"""
        if frame is None:
            return None
        ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ret:
            return base64.b64encode(buf).decode()
        return None
    
    def get_arm_state(self):
        """获取机械臂状态"""
        # TODO: 从控制进程读取真实状态
        # 临时返回默认值
        return {
            "arm_shoulder_pan.pos": 0.0,
            "arm_shoulder_lift.pos": -90.0,
            "arm_elbow_flex.pos": 80.0,
            "arm_wrist_flex.pos": 60.0,
            "arm_wrist_roll.pos": 0.0,
            "arm_gripper.pos": 0.0,
        }
    
    def send_observation(self, front_b64, wrist_b64, detections):
        """发送观测到PC"""
        obs = {
            "front": front_b64,
            "wrist": wrist_b64,
            "detections": detections,
            "host_state": self.state,
            "episode_count": self.episode_count,
            "arm_state": self.get_arm_state(),
        }
        
        try:
            self.obs_sock.send_string(json.dumps(obs), flags=zmq.NOBLOCK)
        except:
            pass
    
    def run(self):
        """主循环"""
        logger.info("[Host] 开始录制循环...")
        detections = []
        
        try:
            while True:
                loop_start = time.perf_counter()
                
                # 1. 获取YOLO检测结果
                try:
                    while not self.det_queue.empty():
                        detections = self.det_queue.get_nowait()
                except:
                    pass
                
                # 2. 读取摄像头
                front, wrist = self.read_cameras()
                
                # 3. 写入共享内存
                if front is not None:
                    try:
                        shm_array = np.ndarray((IMG_H, IMG_W, 3), dtype=np.uint8, buffer=self.shm.buf)
                        shm_array[:] = front[:]
                    except:
                        pass
                
                # 4. 录制时保存数据
                if self.state == "recording":
                    self.recording_data.append({
                        "timestamp": time.time(),
                        "arm_state": self.get_arm_state(),
                    })
                
                # 5. 发送观测
                self.send_observation(
                    self.encode_image(front),
                    self.encode_image(wrist),
                    detections
                )
                
                # 6. 接收PC命令
                try:
                    msg = self.cmd_sock.recv_string(zmq.NOBLOCK)
                    data = json.loads(msg)
                    
                    if data.get("start_recording"):
                        self.state = "recording"
                        self.recording_data = []
                        logger.info(f"🎬 开始录制 Episode {self.episode_count + 1}!")
                    
                    elif data.get("stop_recording"):
                        self.state = "reset"
                        self.episode_count += 1
                        logger.info(f"✅ Episode {self.episode_count} 完成!")
                        logger.info(f"   录制了 {len(self.recording_data)} 帧")
                        logger.info("🔄 请移动底盘到新位置，然后在PC上按R键继续...")
                    
                    elif data.get("confirm_reset"):
                        self.state = "waiting"
                        logger.info(f"🔍 Episode {self.episode_count + 1} 准备就绪，等待开始...")
                
                except zmq.Again:
                    pass
                
                # 7. 帧率控制
                elapsed = time.perf_counter() - loop_start
                sleep_t = max(1 / FPS - elapsed, 0)
                if sleep_t > 0:
                    time.sleep(sleep_t)
        
        except KeyboardInterrupt:
            logger.info("[Host] 停止")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """清理资源"""
        logger.info("[Host] 清理资源...")
        
        try:
            self.cmd_queue.put(None)
        except:
            pass
        
        time.sleep(1)
        if self.ctrl_proc.is_alive():
            self.ctrl_proc.terminate()
        if self.inf_proc.is_alive():
            self.inf_proc.terminate()
        
        if self.cap_front:
            self.cap_front.release()
        if self.cap_wrist:
            self.cap_wrist.release()
        
        self.shm.close()
        self.shm.unlink()
        
        self.cmd_sock.close()
        self.obs_sock.close()
        self.ctx.term()
        
        logger.info(f"[Host] 已关闭，共录制 {self.episode_count} 个 episodes")


def main():
    host = RecordHost()
    host.run()


if __name__ == "__main__":
    main()
