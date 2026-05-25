#!/usr/bin/env python
"""
LeKiwi Host for Raspberry Pi (Custom Configuration)

This is a customized version of lekiwi_host that doesn't require modifying
lerobot source code. It uses custom camera configuration for Raspberry Pi 5.

Usage:
    # Run on Raspberry Pi
    python src/lekiwi_host_pi.py
    
    # Or with custom duration
    python src/lekiwi_host_pi.py --duration 300

Features:
    - Front camera: /dev/video2 (USB camera)
    - Wrist camera: disabled (optional)
    - Custom warmup time for USB camera
    - No need to modify lerobot source code
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


# =============================================================================
# Custom Configuration
# =============================================================================

@dataclass
class LeKiwiPiConfig:
    """Configuration for LeKiwi Raspberry Pi Host."""
    
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
    
    # Debug
    verbose: bool = False


def get_camera_config(config: LeKiwiPiConfig) -> dict[str, CameraConfig]:
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
# Host Logic (Simplified)
# =============================================================================

def run_host(config: LeKiwiPiConfig):
    """Run LeKiwi host with custom configuration."""
    
    # Setup logging
    level = logging.DEBUG if config.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("LeKiwi Host for Raspberry Pi")
    logger.info("=" * 60)
    
    # Create configurations
    logger.info("Configuring...")
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
    
    # Initialize
    logger.info("Initializing robot...")
    robot = LeKiwi(robot_config)
    
    logger.info("Connecting...")
    robot.connect()
    
    logger.info("Starting host...")
    from lerobot.robots.lekiwi.lekiwi_host import LeKiwiHost
    host = LeKiwiHost(host_config)
    
    logger.info("Ready! Waiting for client...")
    logger.info(f"Duration: {config.duration}s | Ctrl+C to stop")
    
    try:
        import base64
        import json
        import zmq
        import cv2
        
        last_cmd_time = time.time()
        watchdog_active = False
        start = time.perf_counter()
        duration = 0
        
        while host.connection_time_s == 0 or duration < host.connection_time_s:
            loop_start = time.time()
            
            # Receive command
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                robot.send_action(data)
                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                pass
            except Exception as e:
                logger.error(f"Command error: {e}")
            
            # Watchdog
            if (time.time() - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logger.warning("Watchdog: stopping base")
                watchdog_active = True
                robot.stop_base()
            
            # Get and send observation
            obs = robot.get_observation()
            for cam_key in robot.cameras.keys():
                ret, buffer = cv2.imencode(".jpg", obs[cam_key], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                obs[cam_key] = base64.b64encode(buffer).decode("utf-8") if ret else ""
            
            try:
                host.zmq_observation_socket.send_string(json.dumps(obs), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            
            # Maintain frequency
            elapsed = time.time() - loop_start
            sleep_time = max(1 / host.max_loop_freq_hz - elapsed, 0)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            duration = time.perf_counter() - start
        
        logger.info("Time limit reached")
        
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
        description="LeKiwi Host for Raspberry Pi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/lekiwi_host_pi.py
    python src/lekiwi_host_pi.py --duration 300
    python src/lekiwi_host_pi.py --enable-wrist --warmup 5
        """
    )
    
    parser.add_argument("--port", "-p", default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--front-camera", default="/dev/video2", help="Front camera")
    parser.add_argument("--enable-wrist", action="store_true", help="Enable wrist cam")
    parser.add_argument("--warmup", type=float, default=3.0, help="Camera warmup (s)")
    parser.add_argument("--flip", type=int, default=0, choices=[0, 1, 2, 3], help="Rotation: 0=none, 1=90°, 2=180°, 3=270°")
    parser.add_argument("--cmd-port", type=int, default=5555)
    parser.add_argument("--obs-port", type=int, default=5556)
    parser.add_argument("--duration", "-d", type=int, default=0, help="Duration in seconds, 0=infinite (default: 0)")
    parser.add_argument("--watchdog", type=int, default=500, help="Watchdog (ms)")
    parser.add_argument("--freq", type=int, default=30, help="Max freq (Hz)")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    config = LeKiwiPiConfig(
        port=args.port,
        front_camera=args.front_camera,
        enable_wrist=args.enable_wrist,
        camera_warmup=args.warmup,
        camera_flip=args.flip,
        zmq_cmd_port=args.cmd_port,
        zmq_obs_port=args.obs_port,
        duration=args.duration,
        watchdog_ms=args.watchdog,
        max_freq=args.freq,
        verbose=args.verbose,
    )
    
    run_host(config)


if __name__ == "__main__":
    main()
