#!/usr/bin/env python
"""
PC端 录制专用 Client（简化版）
功能：
1. 显示front+wrist两个摄像头图像
2. front上显示YOLO检测框和辅助信息
3. 显示25%占比参考框（帮助定位）
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

TARGET_RATIO = 0.25  # 目标占比25%


class RecordClient:
    """录制客户端（简化版）"""
    
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        self.frame_count = 0
        self.recording = False
        self.recorded_frames = []
        self.episode_count = 0
        
        # ZMQ
        print(f"[Client] 连接树莓派 {robot_ip}...")
        self.ctx = zmq.Context()
        
        self.cmd_sock = self.ctx.socket(zmq.PUSH)
        self.cmd_sock.connect(f"tcp://{robot_ip}:{CMD_PORT}")
        
        self.obs_sock = self.ctx.socket(zmq.PULL)
        self.obs_sock.setsockopt(zmq.CONFLATE, 1)
        self.obs_sock.connect(f"tcp://{robot_ip}:{OBS_PORT}")
        
        print("[Client] 已连接")
        print("\n" + "=" * 60)
        print("录制客户端")
        print("=" * 60)
        print("流程:")
        print("  1. 手动调整底盘位置（观察辅助线）")
        print("  2. 位置合适后按空格键开始录制")
        print("  3. 用主臂遥操作控制从臂抓取")
        print("  4. 抓取完成后按空格键结束录制")
        print("  5. 移动底盘到新位置，按R键继续")
        print("=" * 60)
        print("按键:")
        print("  空格键: 开始/结束录制")
        print("  R键: 确认重置，开始下一轮")
        print("  Q键: 退出程序")
        print("=" * 60 + "\n")
    
    def decode_image(self, img_b64):
        """解码base64图像"""
        if not img_b64:
            return None
        try:
            data = base64.b64decode(img_b64)
            arr = np.frombuffer(data, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except:
            return None
    
    def draw_detection(self, frame, detections):
        """绘制检测框和辅助信息"""
        if frame is None:
            return None
        
        h, w = frame.shape[:2]
        
        # 绘制25%占比参考框（中心区域）
        ref_w = int(np.sqrt(TARGET_RATIO) * w)
        ref_h = int(np.sqrt(TARGET_RATIO) * h)
        ref_x1 = (w - ref_w) // 2
        ref_y1 = (h - ref_h) // 2
        ref_x2 = ref_x1 + ref_w
        ref_y2 = ref_y1 + ref_h
        
        cv2.rectangle(frame, (ref_x1, ref_y1), (ref_x2, ref_y2), (255, 255, 0), 2)
        cv2.putText(frame, f"目标: {TARGET_RATIO:.0%}", (ref_x1, ref_y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # 绘制检测框
        if detections:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cx, cy = det["center"]
                conf = det["confidence"]
                
                # 检测框
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # 中心点
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                
                # 标签
                label = f"{det['class']}: {conf:.1%}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 绘制从中心到检测框中心的线
                cv2.line(frame, (w//2, h//2), (int(cx), int(cy)), (255, 255, 0), 2)
                
                # 计算实际占比
                det_area = (x2 - x1) * (y2 - y1)
                img_area = w * h
                actual_ratio = det_area / img_area
                
                # 在检测框下方显示占比
                ratio_text = f"占比: {actual_ratio:.1%}"
                ratio_color = (0, 255, 0) if actual_ratio >= TARGET_RATIO * 0.8 else (0, 165, 255)
                cv2.putText(frame, ratio_text, (int(x1), int(y2) + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, ratio_color, 2)
        
        # 绘制中心十字线
        cv2.line(frame, (w//2, 0), (w//2, h), (255, 0, 0), 1)
        cv2.line(frame, (0, h//2), (w, h//2), (255, 0, 0), 1)
        cv2.circle(frame, (w//2, h//2), 5, (255, 0, 0), -1)
        
        return frame
    
    def draw_info_panel(self, frame, obs):
        """绘制信息面板"""
        if frame is None:
            return None
        
        h, w = frame.shape[:2]
        host_state = obs.get("host_state", "waiting")
        episode = obs.get("episode_count", 0)
        detections = obs.get("detections", [])
        
        # 检查位置是否合适
        is_position_good = False
        if detections:
            det = detections[0]
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["center"]
            
            # 检查居中
            center_x, center_y = w // 2, h // 2
            is_centered = abs(cx - center_x) < w * 0.1 and abs(cy - center_y) < h * 0.1
            
            # 检查占比
            det_area = (x2 - x1) * (y2 - y1)
            img_area = w * h
            ratio = det_area / img_area
            is_good_ratio = ratio >= TARGET_RATIO * 0.8
            
            is_position_good = is_centered and is_good_ratio
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (380, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (10, 10), (380, 150), (255, 255, 255), 1)
        
        # 标题
        cv2.putText(frame, "录制控制面板", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Episode
        cv2.putText(frame, f"Episode: {episode + 1}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 状态
        state_text = {
            "waiting": "等待开始",
            "recording": "录制中",
            "reset": "重置环境"
        }.get(host_state, host_state)
        
        state_colors = {
            "waiting": (255, 255, 0),
            "recording": (0, 0, 255),
            "reset": (255, 0, 255)
        }
        state_color = state_colors.get(host_state, (255, 255, 255))
        cv2.putText(frame, f"状态: {state_text}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        
        # 位置评估
        if host_state == "waiting":
            if len(detections) == 0:
                pos_text = "未检测到纸团"
                pos_color = (0, 0, 255)
            elif is_position_good:
                pos_text = "位置合适 ✓"
                pos_color = (0, 255, 0)
            else:
                pos_text = "请调整底盘位置"
                pos_color = (0, 165, 255)
            
            cv2.putText(frame, pos_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pos_color, 2)
        
        # 录制帧数
        if self.recording:
            cv2.putText(frame, f"已录制: {len(self.recorded_frames)} 帧", (20, 135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 大提示
        if host_state == "waiting" and is_position_good:
            self.draw_big_text(frame, "位置合适！按空格键开始录制", (0, 255, 0))
        elif host_state == "waiting" and len(detections) == 0:
            self.draw_big_text(frame, "请将纸团放入视野", (0, 0, 255))
        elif host_state == "recording":
            self.draw_big_text(frame, "录制中... 按空格键结束", (0, 0, 255))
        elif host_state == "reset":
            self.draw_big_text(frame, "请移动底盘，按R键继续", (255, 0, 255))
        
        return frame
    
    def draw_big_text(self, frame, text, color):
        """绘制大提示文字"""
        h, w = frame.shape[:2]
        
        # 背景
        overlay = frame.copy()
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        box_w, box_h = text_size[0] + 40, text_size[1] + 40
        x1 = (w - box_w) // 2
        y1 = h - 100
        cv2.rectangle(overlay, (x1, y1), (x1 + box_w, y1 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), color, 2)
        
        # 文字
        text_x = x1 + 20
        text_y = y1 + box_h - 20
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    def display_dual_camera(self, front, wrist):
        """显示双摄像头"""
        if front is None:
            front = np.zeros((480, 640, 3), dtype=np.uint8)
        if wrist is None:
            wrist = np.zeros((480, 640, 3), dtype=np.uint8)
        
        front_display = cv2.resize(front, (640, 480))
        wrist_display = cv2.resize(wrist, (640, 480))
        
        cv2.putText(front_display, "Front Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(wrist_display, "Wrist Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        combined = np.hstack([front_display, wrist_display])
        scale = 0.8
        combined = cv2.resize(combined, (int(combined.shape[1] * scale), int(combined.shape[0] * scale)))
        
        cv2.imshow("LeKiwi Record", combined)
    
    def send_command(self, cmd_dict):
        """发送命令"""
        try:
            self.cmd_sock.send_string(json.dumps(cmd_dict), flags=zmq.NOBLOCK)
        except:
            pass
    
    def run(self):
        """主循环"""
        running = True
        
        try:
            while running:
                # 接收观测
                try:
                    msg = self.obs_sock.recv_string(zmq.NOBLOCK)
                    obs = json.loads(msg)
                    self.frame_count += 1
                except zmq.Again:
                    time.sleep(0.001)
                    continue
                
                # 解码图像
                front = self.decode_image(obs.get("front"))
                wrist = self.decode_image(obs.get("wrist"))
                
                # 绘制检测和辅助信息
                if front is not None:
                    front = self.draw_detection(front, obs.get("detections", []))
                    front = self.draw_info_panel(front, obs)
                
                # 显示
                self.display_dual_camera(front, wrist)
                
                # 录制时保存数据
                if self.recording and front is not None:
                    frame_data = {
                        "timestamp": time.time(),
                        "front": front.copy(),
                        "wrist": wrist.copy() if wrist is not None else None,
                        "arm_state": obs.get("arm_state", {}),
                    }
                    self.recorded_frames.append(frame_data)
                
                # 键盘处理
                key = cv2.waitKey(1) & 0xFF
                host_state = obs.get("host_state", "waiting")
                
                if key == ord(' '):  # 空格键
                    if host_state == "waiting" and not self.recording:
                        # 开始录制
                        self.send_command({"start_recording": True})
                        self.recording = True
                        self.recorded_frames = []
                        print(f"🎬 开始录制 Episode {self.episode_count + 1}!")
                    
                    elif host_state == "recording" and self.recording:
                        # 结束录制
                        self.send_command({"stop_recording": True})
                        self.recording = False
                        self.episode_count += 1
                        print(f"✅ Episode {self.episode_count} 完成!")
                        self.save_episode()
                
                elif key == ord('r') or key == ord('R'):  # R键
                    if host_state == "reset":
                        self.send_command({"confirm_reset": True})
                        print("🔄 确认重置，准备下一轮...")
                
                elif key == ord('q') or key == ord('Q'):  # Q键
                    if self.recording:
                        self.send_command({"stop_recording": True})
                        self.save_episode()
                    running = False
        
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断")
        finally:
            if self.recording:
                self.save_episode()
            cv2.destroyAllWindows()
            self.cmd_sock.close()
            self.obs_sock.close()
            self.ctx.term()
            print(f"\n✅ 完成！共录制 {self.episode_count} 个 episodes")
            print(f"   数据保存路径: {DATA_DIR}")
    
    def save_episode(self):
        """保存episode"""
        if not self.recorded_frames:
            return
        
        episode_dir = DATA_DIR / f"episode_{self.episode_count:04d}"
        episode_dir.mkdir(exist_ok=True)
        
        # 保存图像
        for i, frame in enumerate(self.recorded_frames):
            front_path = episode_dir / f"front_{i:04d}.jpg"
            cv2.imwrite(str(front_path), frame["front"])
            
            if frame["wrist"] is not None:
                wrist_path = episode_dir / f"wrist_{i:04d}.jpg"
                cv2.imwrite(str(wrist_path), frame["wrist"])
        
        # 保存动作序列
        import pickle
        actions = [f["arm_state"] for f in self.recorded_frames]
        with open(episode_dir / "actions.pkl", "wb") as f:
            pickle.dump(actions, f)
        
        print(f"  💾 已保存: {episode_dir} ({len(self.recorded_frames)} 帧)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=DEFAULT_IP)
    args = parser.parse_args()
    
    client = RecordClient(robot_ip=args.ip)
    client.run()


if __name__ == "__main__":
    main()
