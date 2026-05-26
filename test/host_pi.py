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


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


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
    logger = setup_logging()
    
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
    
    try:
        while True:
            loop_start = time.time()
            
            # 1. 接收命令
            try:
                msg = cmd_socket.recv_string(zmq.NOBLOCK)
                data = json.loads(msg)
                logger.info(f"收到命令: {data}")
                
                # 补齐从臂默认位置（如果不存在）
                # 关节向上转时-x度，向下转时+x度，左转时+x度，右转是-x度
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
                # No command available, this is normal
                if not watchdog_active and not no_command_logged:
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
                    if yolo_model is not None:
                        try:
                            infer_start = time.time()
                            results = yolo_model(frame, conf=YOLO_CONFIDENCE, verbose=False)
                            infer_time = time.time() - infer_start
                            
                            # 检查是否有检测结果
                            if len(results) > 0 and len(results[0].boxes) > 0:
                                boxes = results[0].boxes
                                logger.info(f"🎯 检测到 {len(boxes)} 个目标 (推理耗时: {infer_time*1000:.1f}ms):")
                                for i, box in enumerate(boxes):
                                    # 获取框的坐标
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    # 获取置信度
                                    conf = float(box.conf[0].cpu().numpy())
                                    # 获取类别
                                    cls = int(box.cls[0].cpu().numpy())
                                    cls_name = results[0].names[cls]
                                    
                                    logger.info(f"  [{i+1}] 类别: {cls_name}, "
                                              f"置信度: {conf:.2%}, "
                                              f"位置: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f}), "
                                              f"中心: ({(x1+x2)/2:.1f}, {(y1+y2)/2:.1f}), "
                                              f"大小: {x2-x1:.1f}x{y2-y1:.1f}")
                            else:
                                # 每5秒输出一次"未检测到目标"的提示（避免日志刷屏）
                                if int(time.time()) % 5 == 0:
                                    logger.info(f"🔍 未检测到目标 (推理耗时: {infer_time*1000:.1f}ms)")
                        except Exception as e:
                            logger.error(f"YOLO 推理错误: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                    
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
