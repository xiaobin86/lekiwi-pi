#!/usr/bin/env python
"""
树莓派端 Host - ACT策略推理支持版
架构：主进程(观测采集+ZMQ通信) | 控制进程(底盘100Hz+机械臂)
模式：
  - inference: 接收PC端推理动作，执行
  - teleop: 手柄遥操作（保留原有功能）
"""

import sys, time, logging, json, base64, argparse
import multiprocessing as mp
from multiprocessing import Process, Queue
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path.home() / "lerobot-workspace/lerobot/src"))
import cv2, numpy as np, zmq
from lerobot.robots.lekiwi import LeKiwi
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig

# ==================== 配置 ====================
ROBOT_ID = "lekiwi"
SERIAL_PORT = "/dev/ttyACM0"
CMD_PORT, OBS_PORT = 5555, 5556
WATCHDOG_MS, FPS = 2000, 30

IMG_W, IMG_H = 640, 480

# 机械臂关节名称
ARM_JOINTS = [
    "arm_shoulder_pan", "arm_shoulder_lift", "arm_elbow_flex",
    "arm_wrist_flex", "arm_wrist_roll", "arm_gripper"
]

# 状态特征顺序（与训练时一致）
STATE_KEYS = [
    "arm_shoulder_pan.pos", "arm_shoulder_lift.pos", "arm_elbow_flex.pos",
    "arm_wrist_flex.pos", "arm_wrist_roll.pos", "arm_gripper.pos",
    "x.vel", "y.vel", "theta.vel"
]


def controller_worker(cmd_queue, robot_config):
    """控制进程 - 100Hz高频，控制底盘+机械臂"""
    logger.info("[Controller] 初始化...")
    try:
        robot = LeKiwi(robot_config)
        robot.connect()
        logger.info("[Controller] 机器人已连接")
    except Exception as e:
        logger.error(f"[Controller] 连接失败: {e}")
        return
    
    last_cmd_t = time.time()
    watchdog_on = False
    
    try:
        while True:
            loop_start = time.time()
            
            try:
                cmd = cmd_queue.get(timeout=0.001)
                if cmd is None:
                    break
                
                # 构建动作字典
                action = {}
                for key in STATE_KEYS:
                    action[key] = cmd.get(key, 0.0)
                
                robot.send_action(action)
                last_cmd_t = time.time()
                watchdog_on = False
            except:
                pass
            
            # 看门狗
            if not watchdog_on and (time.time() - last_cmd_t > WATCHDOG_MS / 1000):
                logger.warning("[Controller] 看门狗超时，停止底盘")
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


class ACTHost:
    """ACT策略Host - 支持推理模式和遥操作模式"""
    
    def __init__(self, mode="inference"):
        self.mode = mode
        self.running = True
        
        # 机器人配置
        self.robot_config = LeKiwiConfig(
            port=SERIAL_PORT,
            id=ROBOT_ID,
            cameras={}  # 控制进程不初始化摄像头
        )
        
        # 主进程单独读取摄像头
        logger.info("[Host] 初始化摄像头...")
        self.cap = cv2.VideoCapture(2)  # /dev/video2
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        
        # ZMQ
        self.ctx = zmq.Context()
        self.cmd_sock = self.ctx.socket(zmq.PULL)
        self.cmd_sock.setsockopt(zmq.CONFLATE, 1)
        self.cmd_sock.bind(f"tcp://*:{CMD_PORT}")
        
        self.obs_sock = self.ctx.socket(zmq.PUSH)
        self.obs_sock.setsockopt(zmq.CONFLATE, 1)
        self.obs_sock.bind(f"tcp://*:{OBS_PORT}")
        
        # 通信队列
        self.cmd_queue = Queue(maxsize=10)
        
        # 启动控制进程
        logger.info("[Host] 启动控制进程...")
        self.ctrl_proc = Process(
            target=controller_worker,
            args=(self.cmd_queue, self.robot_config),
            name="Controller"
        )
        self.ctrl_proc.start()
        time.sleep(2)
        
        # 状态
        self.client_connected = False
        self.episode_count = 0
        
        logger.info("=" * 60)
        logger.info(f"LeKiwi ACT Host - 模式: {mode}")
        logger.info("=" * 60)
        logger.info(f"[Host] 等待连接... CMD:{CMD_PORT} OBS:{OBS_PORT}")
        logger.info("[Host] Ctrl+C 停止")
    
    def read_camera(self):
        """读取摄像头图像"""
        ret, frame = self.cap.read()
        if ret and frame is not None:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            return frame
        return None
    
    def encode_image(self, frame):
        """编码图像为base64"""
        ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ret:
            return base64.b64encode(buf).decode()
        return None
    
    def get_observation(self):
        """获取观测（图像 + 状态占位）"""
        frame = self.read_camera()
        if frame is None:
            return None
        
        # 编码图像
        img_b64 = self.encode_image(frame)
        if img_b64 is None:
            return None
        
        # 构建观测字典
        # 注意：机械臂状态由控制进程维护，这里发送占位值
        # 实际使用时可以通过共享内存获取真实状态
        obs = {
            "front": img_b64,
            "timestamp": time.time(),
        }
        
        return obs
    
    def run_inference_mode(self):
        """推理模式：接收PC端动作，执行"""
        logger.info("[Host] 推理模式启动")
        
        while self.running:
            loop_start = time.perf_counter()
            
            # 1. 获取观测并发送
            obs = self.get_observation()
            if obs:
                try:
                    self.obs_sock.send_string(json.dumps(obs), flags=zmq.NOBLOCK)
                except:
                    pass
            
            # 2. 接收动作
            try:
                msg = self.cmd_sock.recv_string(zmq.NOBLOCK)
                action = json.loads(msg)
                
                if not self.client_connected:
                    self.client_connected = True
                    logger.info("[Host] PC推理客户端已连接")
                
                # 发送动作到控制进程
                try:
                    self.cmd_queue.put_nowait(action)
                except:
                    pass
                    
            except zmq.Again:
                pass
            
            # 3. 帧率控制
            elapsed = time.perf_counter() - loop_start
            sleep_t = max(1 / FPS - elapsed, 0)
            if sleep_t > 0:
                time.sleep(sleep_t)
    
    def run_teleop_mode(self):
        """遥操作模式（保留原有功能）"""
        logger.info("[Host] 遥操作模式启动")
        # TODO: 集成原有的手柄遥操作逻辑
        pass
    
    def run(self):
        """主循环"""
        try:
            if self.mode == "inference":
                self.run_inference_mode()
            else:
                self.run_teleop_mode()
        except KeyboardInterrupt:
            logger.info("[Host] 停止")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """清理资源"""
        logger.info("[Host] 清理资源...")
        self.running = False
        
        # 停止控制进程
        try:
            self.cmd_queue.put(None)
        except:
            pass
        
        time.sleep(1)
        if self.ctrl_proc.is_alive():
            self.ctrl_proc.terminate()
        
        # 释放资源
        if self.cap:
            self.cap.release()
        
        self.cmd_sock.close()
        self.obs_sock.close()
        self.ctx.term()
        
        logger.info("[Host] 已关闭")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inference", "teleop"], default="inference")
    args = parser.parse_args()
    
    host = ACTHost(mode=args.mode)
    host.run()


if __name__ == "__main__":
    main()
