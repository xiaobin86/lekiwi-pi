#!/usr/bin/env python
"""
PC端 录制专用 Client
功能：
1. 显示front+wrist两个摄像头图像
2. front上显示YOLO检测框和辅助信息
3. 纸团框水平居中(±15%)且占比20%-30%即为合格
4. 提示用户手动调整底盘位置
5. 按空格键开始/结束录制
6. 按R键确认重置

使用方式：
    python src/client_record.py --ip 192.168.3.176
"""

import sys, time, json, base64, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import zmq


DEFAULT_IP = "192.168.3.176"
CMD_PORT, OBS_PORT = 5555, 5556
FPS = 30
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 合格标准（带容错）
CENTER_THRESHOLD = 0.15  # 水平偏差 ±15%
RATIO_MIN = 0.20         # 最小占比 20%
RATIO_MAX = 0.30         # 最大占比 30%
RATIO_TARGET = 0.25      # 目标占比 25%


class RecordClient:
    """录制客户端"""
    
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        self.frame_count = 0
        self.recording = False
        self.recorded_frames = []
        self.episode_count = 0
        
        # ZMQ
        print(f"[Client] Connecting to {robot_ip}...")
        self.ctx = zmq.Context()
        
        self.cmd_sock = self.ctx.socket(zmq.PUSH)
        self.cmd_sock.connect(f"tcp://{robot_ip}:{CMD_PORT}")
        
        self.obs_sock = self.ctx.socket(zmq.PULL)
        self.obs_sock.setsockopt(zmq.CONFLATE, 1)
        self.obs_sock.connect(f"tcp://{robot_ip}:{OBS_PORT}")
        
        print("[Client] Connected")
        print("\n" + "=" * 60)
        print("Recording Client")
        print("=" * 60)
        print("Workflow:")
        print("  1. Position base manually (watch guides)")
        print("  2. Press SPACE to start recording")
        print("  3. Teleop with leader arm to grasp")
        print("  4. Press SPACE to stop recording")
        print("  5. Move base to new position, press R")
        print("=" * 60)
        print("Keys:")
        print("  SPACE: Start/Stop recording")
        print("  R: Confirm reset, next round")
        print("  Q: Quit")
        print("=" * 60 + "\n")
    
    def decode_image(self, img_b64):
        """Decode base64 image"""
        if not img_b64:
            return None
        try:
            data = base64.b64decode(img_b64)
            arr = np.frombuffer(data, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except:
            return None
    
    def draw_detection(self, frame, detections):
        """Draw detection box and guides
        
        Pass criteria:
        1. Box center x within ±15% of image center
        2. Box area ratio between 20%-30%
        """
        if frame is None:
            return None, False
        
        h, w = frame.shape[:2]
        is_position_good = False
        
        # Draw center zone (±15%)
        center_zone_left = int(w * (0.5 - CENTER_THRESHOLD))
        center_zone_right = int(w * (0.5 + CENTER_THRESHOLD))
        cv2.rectangle(frame, (center_zone_left, 0), (center_zone_right, h), (255, 255, 0), 2)
        cv2.putText(frame, "Center Zone (±15%)", (center_zone_left + 5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw detection boxes
        if detections:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cx, cy = det["center"]
                conf = det["confidence"]
                
                # Detection box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Center point
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                
                # Label
                label = f"{det['class']}: {conf:.1%}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Calculate ratio
                det_area = (x2 - x1) * (y2 - y1)
                img_area = w * h
                actual_ratio = det_area / img_area
                
                # Check centered (±15%)
                center_offset = abs(cx - w//2) / (w / 2)
                is_centered = center_offset <= CENTER_THRESHOLD
                
                # Check ratio (20%-30%)
                is_good_ratio = RATIO_MIN <= actual_ratio <= RATIO_MAX
                
                is_position_good = is_centered and is_good_ratio
                
                # Draw center guide line
                if not is_centered:
                    # Show direction to move
                    direction = "Move Right -->" if cx < w//2 else "<-- Move Left"
                    line_color = (0, 165, 255)
                    cv2.line(frame, (int(cx), int(cy)), (w//2, int(cy)), line_color, 2)
                    cv2.putText(frame, direction, (w//2 - 80, int(cy) - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)
                else:
                    cv2.line(frame, (int(cx), int(cy)), (w//2, int(cy)), (0, 255, 0), 2)
                    cv2.putText(frame, "Centered OK", (w//2 - 50, int(cy) - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show ratio info
                ratio_color = (0, 255, 0) if is_good_ratio else (0, 0, 255)
                ratio_status = "OK" if is_good_ratio else "BAD"
                ratio_text = f"Ratio: {actual_ratio:.1%} [{ratio_status}] (Target: {RATIO_MIN:.0%}-{RATIO_MAX:.0%})"
                cv2.putText(frame, ratio_text, (int(x1), int(y2) + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, ratio_color, 2)
                
                # Draw ratio bar at bottom
                bar_y = h - 35
                bar_w = int(w * 0.4)
                bar_h = 25
                bar_x = (w - bar_w) // 2
                
                # Background
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                
                # Fill (current ratio mapped to bar)
                # Map 0%-50% to full bar
                fill_ratio = min(actual_ratio / 0.5, 1.0)
                fill_w = int(bar_w * fill_ratio)
                bar_color = (0, 255, 0) if is_good_ratio else (0, 165, 255)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
                
                # Target zone markers (20% and 30%)
                target_min_x = bar_x + int(bar_w * (RATIO_MIN / 0.5))
                target_max_x = bar_x + int(bar_w * (RATIO_MAX / 0.5))
                cv2.line(frame, (target_min_x, bar_y - 5), (target_min_x, bar_y + bar_h + 5), (255, 255, 0), 2)
                cv2.line(frame, (target_max_x, bar_y - 5), (target_max_x, bar_y + bar_h + 5), (255, 255, 0), 2)
                cv2.putText(frame, f"{RATIO_MIN:.0%}", (target_min_x - 15, bar_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                cv2.putText(frame, f"{RATIO_MAX:.0%}", (target_max_x - 15, bar_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return frame, is_position_good
    
    def draw_info_panel(self, frame, obs, is_position_good):
        """Draw info panel"""
        if frame is None:
            return None
        
        h, w = frame.shape[:2]
        host_state = obs.get("host_state", "waiting")
        episode = obs.get("episode_count", 0)
        detections = obs.get("detections", [])
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (420, 170), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (10, 10), (420, 170), (255, 255, 255), 1)
        
        # Title
        cv2.putText(frame, "Recording Panel", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Episode
        cv2.putText(frame, f"Episode: {episode + 1}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Status
        state_colors = {
            "waiting": (255, 255, 0),
            "recording": (0, 0, 255),
            "reset": (255, 0, 255)
        }
        state_color = state_colors.get(host_state, (255, 255, 255))
        cv2.putText(frame, f"State: {host_state.upper()}", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        
        # Position evaluation
        if host_state == "waiting":
            if len(detections) == 0:
                pos_text = "No paper ball detected"
                pos_color = (0, 0, 255)
            elif is_position_good:
                pos_text = "Position OK! Ready to record"
                pos_color = (0, 255, 0)
            else:
                pos_text = "Adjust base position"
                pos_color = (0, 165, 255)
            
            cv2.putText(frame, pos_text, (20, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, pos_color, 2)
        
        # Recording frame count
        if self.recording:
            cv2.putText(frame, f"Frames: {len(self.recorded_frames)}", (20, 155),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Big prompt at bottom
        if host_state == "waiting" and is_position_good:
            frame = self.draw_big_text(frame, "Press SPACE to Start Recording", (0, 255, 0))
        elif host_state == "waiting" and len(detections) == 0:
            frame = self.draw_big_text(frame, "Place paper ball in view", (0, 0, 255))
        elif host_state == "recording":
            frame = self.draw_big_text(frame, "Recording... Press SPACE to Stop", (0, 0, 255))
        elif host_state == "reset":
            frame = self.draw_big_text(frame, "Move base, then press R to continue", (255, 0, 255))
        
        return frame
    
    def draw_big_text(self, frame, text, color):
        """Draw big prompt text"""
        h, w = frame.shape[:2]
        
        # Background
        overlay = frame.copy()
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
        box_w, box_h = text_size[0] + 40, text_size[1] + 40
        x1 = (w - box_w) // 2
        y1 = h - 100
        
        cv2.rectangle(overlay, (x1, y1), (x1 + box_w, y1 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), color, 2)
        
        # Text
        text_x = x1 + 20
        text_y = y1 + box_h - 15
        cv2.putText(frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        
        return frame
    
    def display_dual_camera(self, front, wrist):
        """Display dual camera"""
        if front is None:
            front = np.zeros((480, 640, 3), dtype=np.uint8)
        if wrist is None:
            wrist = np.zeros((480, 640, 3), dtype=np.uint8)
        
        front_display = cv2.resize(front, (640, 480))
        wrist_display = cv2.resize(wrist, (640, 480))
        
        cv2.putText(front_display, "Front Camera", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(wrist_display, "Wrist Camera", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        combined = np.hstack([front_display, wrist_display])
        scale = 0.8
        combined = cv2.resize(combined, (int(combined.shape[1] * scale), int(combined.shape[0] * scale)))
        
        cv2.imshow("LeKiwi Record", combined)
    
    def send_command(self, cmd_dict):
        """Send command"""
        try:
            self.cmd_sock.send_string(json.dumps(cmd_dict), flags=zmq.NOBLOCK)
        except:
            pass
    
    def run(self):
        """Main loop"""
        running = True
        
        try:
            while running:
                # Receive observation
                try:
                    msg = self.obs_sock.recv_string(zmq.NOBLOCK)
                    obs = json.loads(msg)
                    self.frame_count += 1
                except zmq.Again:
                    time.sleep(0.001)
                    continue
                
                # Decode images
                front = self.decode_image(obs.get("front"))
                wrist = self.decode_image(obs.get("wrist"))
                
                # Draw detection and guides
                is_position_good = False
                if front is not None:
                    front, is_position_good = self.draw_detection(front, obs.get("detections", []))
                    front = self.draw_info_panel(front, obs, is_position_good)
                
                # Display
                self.display_dual_camera(front, wrist)
                
                # Save frames when recording
                if self.recording and front is not None:
                    frame_data = {
                        "timestamp": time.time(),
                        "front": front.copy(),
                        "wrist": wrist.copy() if wrist is not None else None,
                        "arm_state": obs.get("arm_state", {}),
                    }
                    self.recorded_frames.append(frame_data)
                
                # Keyboard handling
                key = cv2.waitKey(1) & 0xFF
                host_state = obs.get("host_state", "waiting")
                
                if key == ord(' '):  # SPACE
                    if host_state == "waiting" and not self.recording:
                        self.send_command({"start_recording": True})
                        self.recording = True
                        self.recorded_frames = []
                        print(f"Episode {self.episode_count + 1} started!")
                    
                    elif host_state == "recording" and self.recording:
                        self.send_command({"stop_recording": True})
                        self.recording = False
                        self.episode_count += 1
                        print(f"Episode {self.episode_count} completed!")
                        self.save_episode()
                
                elif key == ord('r') or key == ord('R'):  # R
                    if host_state == "reset":
                        self.send_command({"confirm_reset": True})
                        print("Reset confirmed, next round...")
                
                elif key == ord('q') or key == ord('Q'):  # Q
                    if self.recording:
                        self.send_command({"stop_recording": True})
                        self.save_episode()
                    running = False
        
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            if self.recording:
                self.save_episode()
            cv2.destroyAllWindows()
            self.cmd_sock.close()
            self.obs_sock.close()
            self.ctx.term()
            print(f"\nDone! Recorded {self.episode_count} episodes")
            print(f"Data saved to: {DATA_DIR}")
    
    def save_episode(self):
        """Save episode"""
        if not self.recorded_frames:
            return
        
        episode_dir = DATA_DIR / f"episode_{self.episode_count:04d}"
        episode_dir.mkdir(exist_ok=True)
        
        for i, frame in enumerate(self.recorded_frames):
            front_path = episode_dir / f"front_{i:04d}.jpg"
            cv2.imwrite(str(front_path), frame["front"])
            
            if frame["wrist"] is not None:
                wrist_path = episode_dir / f"wrist_{i:04d}.jpg"
                cv2.imwrite(str(wrist_path), frame["wrist"])
        
        import pickle
        actions = [f["arm_state"] for f in self.recorded_frames]
        with open(episode_dir / "actions.pkl", "wb") as f:
            pickle.dump(actions, f)
        
        print(f"  Saved: {episode_dir} ({len(self.recorded_frames)} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=DEFAULT_IP)
    args = parser.parse_args()
    
    client = RecordClient(robot_ip=args.ip)
    client.run()


if __name__ == "__main__":
    main()
