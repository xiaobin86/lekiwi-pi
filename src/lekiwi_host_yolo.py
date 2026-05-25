#!/usr/bin/env python
"""
LeKiwi Host with YOLO Detection for Raspberry Pi

在标准 LeKiwi Host 基础上集成 YOLO 纸团检测，在回传图像上绘制检测框和中点。

Usage:
    # 基础用法（使用默认模型路径）
    python src/lekiwi_host_yolo.py
    
    # 指定自定义模型
    python src/lekiwi_host_yolo.py --yolo-model /path/to/best.pt
    
    # 调整检测置信度阈值
    python src/lekiwi_host_yolo.py --conf 0.5
    
Features:
    - 实时 YOLO 纸团检测
    - 绿色检测框 + 红色中心点标记
    - 遥操作功能完全不受影响
    - 支持 front/wrist 双摄像头
"""

import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass

# Add lekiwi-pi src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import LeRobot components
from lerobot.robots.lekiwi import LeKiwi
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig, LeKiwiHostConfig
from lerobot.cameras import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv import OpenCVCameraConfig

# YOLO import
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "ultralytics not installed. Run: pip install ultralytics"
    )


# =============================================================================
# Custom Configuration
# =============================================================================

@dataclass
class LeKiwiYoloConfig:
    """Configuration for LeKiwi YOLO Host."""
    
    # Robot
    port: str = "/dev/ttyACM0"
    
    # Camera
    front_camera: str = "/dev/video2"
    enable_wrist: bool = True
    camera_warmup: float = 3.0
    camera_flip: int = 0
    
    # Network
    zmq_cmd_port: int = 5555
    zmq_obs_port: int = 5556
    
    # Timing (0 = infinite)
    duration: int = 0
    watchdog_ms: int = 2000
    max_freq: int = 30
    
    # YOLO
    yolo_model: str = "models/paper_ball_detection-1-8/weights/best.pt"
    conf_threshold: float = 0.3
    
    # Debug
    verbose: bool = False


def get_camera_config(config: LeKiwiYoloConfig) -> dict[str, CameraConfig]:
    """Create camera configuration."""
    rotation_map = {
        0: Cv2Rotation.NO_ROTATION,
        1: Cv2Rotation.ROTATE_90,
        2: Cv2Rotation.ROTATE_180,
        3: Cv2Rotation.ROTATE_270,
    }
    rotation = rotation_map.get(config.camera_flip, Cv2Rotation.NO_ROTATION)
    
    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=config.front_camera,
            fps=30,
            width=640,
            height=480,
            rotation=rotation,
            warmup_s=config.camera_warmup,
        ),
    }
    
    if config.enable_wrist:
        cameras["wrist"] = OpenCVCameraConfig(
            index_or_path="/dev/video0",
            fps=30,
            width=480,
            height=640,
            rotation=Cv2Rotation.ROTATE_90,
            warmup_s=config.camera_warmup,
        )
    
    return cameras


# =============================================================================
# YOLO Detection
# =============================================================================

class PaperBallDetector:
    """YOLO detector for paper ball detection with visualization."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.3):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"YOLO model loaded: {model_path}")
        self.logger.info(f"Confidence threshold: {conf_threshold}")
    
    def detect_and_draw(self, frame, cam_name: str = "front"):
        """
        Run YOLO detection on frame and draw bounding boxes + center points.
        
        Args:
            frame: OpenCV BGR image (numpy array)
            cam_name: Camera name for logging
            
        Returns:
            Annotated frame with detection boxes and center points
        """
        import cv2
        import numpy as np
        
        # Run YOLO inference
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        
        # Draw detections
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Calculate center point
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Draw green bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw red center point
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
                # Draw confidence label
                label = f"paper: {conf:.2f}"
                cv2.putText(
                    frame, label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2
                )
                
                # Draw center coordinates
                center_text = f"({cx}, {cy})"
                cv2.putText(
                    frame, center_text,
                    (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 255), 1
                )
        
        # Draw detection count
        num_detections = len(results[0].boxes) if results and results[0].boxes else 0
        cv2.putText(
            frame,
            f"Detections: {num_detections}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )
        
        return frame


# =============================================================================
# Host Logic with YOLO
# =============================================================================

def run_host(config: LeKiwiYoloConfig):
    """Run LeKiwi host with YOLO detection."""
    
    # Setup logging
    level = logging.DEBUG if config.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("LeKiwi Host with YOLO Detection")
    logger.info("=" * 60)
    
    # Initialize YOLO detector
    logger.info("Loading YOLO model...")
    detector = PaperBallDetector(config.yolo_model, config.conf_threshold)
    
    # Create configurations
    logger.info("Configuring robot...")
    camera_config = get_camera_config(config)
    
    robot_config = LeKiwiConfig(
        port=config.port,
        cameras=camera_config,
    )
    
    host_config = LeKiwiHostConfig(
        port_zmq_cmd=config.zmq_cmd_port,
        port_zmq_observations=config.zmq_obs_port,
        connection_time_s=config.duration,
        watchdog_timeout_ms=config.watchdog_ms,
        max_loop_freq_hz=config.max_freq,
    )
    
    # Initialize robot
    logger.info("Initializing robot...")
    robot = LeKiwi(robot_config)
    
    logger.info("Connecting...")
    robot.connect()
    
    logger.info("Starting host...")
    from lerobot.robots.lekiwi.lekiwi_host import LeKiwiHost
    host = LeKiwiHost(host_config)
    
    logger.info("Ready! Waiting for client...")
    logger.info(f"Duration: {config.duration}s | Ctrl+C to stop")
    logger.info("YOLO detection active on camera frames")
    
    try:
        import base64
        import json
        import zmq
        import cv2
        
        last_cmd_time = time.time()
        watchdog_active = False
        no_command_logged = False
        start = time.perf_counter()
        duration = 0
        frame_count = 0
        
        while host.connection_time_s == 0 or duration < host.connection_time_s:
            loop_start = time.time()
            
            # Receive command
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                robot.send_action(data)
                last_cmd_time = time.time()
                watchdog_active = False
                no_command_logged = False
            except zmq.Again:
                if not watchdog_active and not no_command_logged:
                    logger.info("No command available (waiting...)")
                    no_command_logged = True
            except Exception as e:
                logger.error(f"Command error: {e}")
            
            # Watchdog
            if (time.time() - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logger.warning("Watchdog: stopping base")
                watchdog_active = True
                robot.stop_base()
            
            # Get observation
            obs = robot.get_observation()
            
            # Process camera frames with YOLO
            for cam_key in robot.cameras.keys():
                frame = obs[cam_key]
                if isinstance(frame, numpy.ndarray) and frame.size > 0:
                    # Run YOLO detection and draw boxes
                    frame = detector.detect_and_draw(frame, cam_key)
                    obs[cam_key] = frame
                
                # Encode to JPEG for transmission
                ret, buffer = cv2.imencode(".jpg", obs[cam_key], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                obs[cam_key] = base64.b64encode(buffer).decode("utf-8") if ret else ""
            
            # Send observation
            try:
                host.zmq_observation_socket.send_string(json.dumps(obs), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            
            # Maintain frequency
            elapsed = time.time() - loop_start
            sleep_time = max(1 / host.max_loop_freq_hz - elapsed, 0)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            frame_count += 1
            duration = time.perf_counter() - start
        
        logger.info(f"Time limit reached. Total frames: {frame_count}")
        
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        logger.info("Shutting down...")
        robot.disconnect()
        host.disconnect()


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LeKiwi Host with YOLO Paper Ball Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default usage (paper ball detection)
    python src/lekiwi_host_yolo.py
    
    # Custom model path
    python src/lekiwi_host_yolo.py --yolo-model /path/to/best.pt
    
    # Higher confidence threshold
    python src/lekiwi_host_yolo.py --conf 0.5
    
    # Disable wrist camera
    python src/lekiwi_host_yolo.py --no-wrist
        """
    )
    
    # Robot
    parser.add_argument("--port", "-p", default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--front-camera", default="/dev/video2", help="Front camera")
    parser.add_argument("--no-wrist", action="store_true", help="Disable wrist cam")
    parser.add_argument("--warmup", type=float, default=3.0, help="Camera warmup (s)")
    parser.add_argument("--flip", type=int, default=0, choices=[0, 1, 2, 3], help="Rotation: 0=none, 1=90°, 2=180°, 3=270°")
    
    # Network
    parser.add_argument("--cmd-port", type=int, default=5555)
    parser.add_argument("--obs-port", type=int, default=5556)
    
    # Timing
    parser.add_argument("--duration", "-d", type=int, default=0, help="Duration in seconds, 0=infinite")
    parser.add_argument("--watchdog", type=int, default=2000, help="Watchdog (ms)")
    parser.add_argument("--freq", type=int, default=30, help="Max freq (Hz)")
    
    # YOLO
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="models/paper_ball_detection-1-8/weights/best.pt",
        help="Path to YOLO model weights"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="YOLO confidence threshold (default: 0.3)"
    )
    
    # Debug
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    config = LeKiwiYoloConfig(
        port=args.port,
        front_camera=args.front_camera,
        enable_wrist=not args.no_wrist,
        camera_warmup=args.warmup,
        camera_flip=args.flip,
        zmq_cmd_port=args.cmd_port,
        zmq_obs_port=args.obs_port,
        duration=args.duration,
        watchdog_ms=args.watchdog,
        max_freq=args.freq,
        yolo_model=args.yolo_model,
        conf_threshold=args.conf,
        verbose=args.verbose,
    )
    
    run_host(config)


if __name__ == "__main__":
    main()
