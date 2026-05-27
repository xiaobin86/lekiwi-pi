#!/usr/bin/env python
"""
PC端 完整录制客户端（SO101 + LeRobotDataset视频格式）

功能：
1. 连接主臂（SO101 Leader）
2. 通过ZMQ发送动作到树莓派控制从臂
3. 从树莓派接收双摄像头图像+从臂状态
4. 使用LeRobotDataset保存为视频格式（MP4）
5. 显示辅助线提示调整底盘位置
6. 键盘控制录制流程

使用方式：
    python src/client_record.py --ip 192.168.3.176 --arm-port COM5

硬件：
    - 主臂：SO101 Leader（PC串口，如COM5）
    - 从臂：LeKiwi（树莓派）
    - 树莓派IP: 192.168.3.176
"""

import sys, time, json, base64, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import zmq
import torch

sys.path.insert(0, str(Path.home() / "lerobot-workspace/lerobot/src"))
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.datasets import LeRobotDataset
from lerobot.configs import VideoEncoderConfig
from lerobot.utils.feature_utils import hw_to_dataset_features


DEFAULT_IP = "192.168.3.176"
CMD_PORT, OBS_PORT = 5555, 5556
FPS = 30

# 数据集配置（存储在项目目录中）
DATASET_ROOT = Path(__file__).parent.parent / "data"
DATASET_REPO_ID = "acelan/lekiwi_grasp_paper_ball"

# 合格标准
CENTER_THRESHOLD = 0.15
RATIO_MIN = 0.20
RATIO_MAX = 0.30

# 主臂配置
LEADER_ARM_ID = "L07252802"
DEFAULT_ARM_PORT = "COM5"

# 图像分辨率（必须与树莓派一致）
IMG_WIDTH, IMG_HEIGHT = 640, 480


class RecordClient:
    """SO101主臂遥操作 + LeRobotDataset视频录制"""
    
    def __init__(self, robot_ip, arm_port, repo_id=DATASET_REPO_ID, root=DATASET_ROOT):
        self.robot_ip = robot_ip
        self.arm_port = arm_port
        self.repo_id = repo_id
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        
        self.frame_count = 0
        self.recording = False
        self.episode_count = 0
        self.teleop_active = True  # 遥操作始终开启
        
        # 1. 连接主臂（SO101）
        print("[Client] 连接SO101主臂...")
        leader_config = SO101LeaderConfig(port=arm_port, id=LEADER_ARM_ID)
        self.leader_arm = SO101Leader(leader_config)
        self.leader_arm.connect()
        print(f"[Client] ✅ 主臂已连接: {self.leader_arm.is_connected}")
        
        # 2. 连接树莓派
        print(f"[Client] 连接树莓派 {robot_ip}...")
        self.ctx = zmq.Context()
        
        self.cmd_sock = self.ctx.socket(zmq.PUSH)
        self.cmd_sock.connect(f"tcp://{robot_ip}:{CMD_PORT}")
        
        self.obs_sock = self.ctx.socket(zmq.PULL)
        self.obs_sock.setsockopt(zmq.CONFLATE, 1)
        self.obs_sock.connect(f"tcp://{robot_ip}:{OBS_PORT}")
        
        # 3. 创建LeRobotDataset（视频格式）
        print("[Client] 创建数据集...")
        self._init_dataset()
        
        print("[Client] ✅ 初始化完成")
        self._print_instructions()
    
    def _init_dataset(self):
        """初始化LeRobotDataset（自动使用时间戳命名，防止重复）"""
        # 使用时间戳命名数据集: lekiwi_grasp_20260527_152030
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.repo_id = f"{self.repo_id}_{timestamp}"
        
        # 更新root路径（不预先创建，让LeRobotDataset.create自己创建）
        self.root = self.root / self.repo_id.replace("/", "_")
        
        # 如果目录已存在（同一秒内多次运行），删除旧目录
        if self.root.exists():
            import shutil
            shutil.rmtree(self.root)
            print(f"[Client] 清理旧目录: {self.root}")
        
        print(f"[Client] 数据集名称: {self.repo_id}")
        print(f"[Client] 保存路径: {self.root}")
        
        # 定义硬件features
        action_features = {
            "arm_shoulder_pan.pos": float,
            "arm_shoulder_lift.pos": float,
            "arm_elbow_flex.pos": float,
            "arm_wrist_flex.pos": float,
            "arm_wrist_roll.pos": float,
            "arm_gripper.pos": float,
            "x.vel": float,
            "y.vel": float,
            "theta.vel": float,
        }
        
        obs_state_features = {
            "arm_shoulder_pan.pos": float,
            "arm_shoulder_lift.pos": float,
            "arm_elbow_flex.pos": float,
            "arm_wrist_flex.pos": float,
            "arm_wrist_roll.pos": float,
            "arm_gripper.pos": float,
        }
        
        obs_image_features = {
            "front": (IMG_HEIGHT, IMG_WIDTH, 3),
            "wrist": (IMG_HEIGHT, IMG_WIDTH, 3),
        }
        
        action_dict = hw_to_dataset_features(action_features, prefix="action", use_video=False)
        obs_state_dict = hw_to_dataset_features(obs_state_features, prefix="observation", use_video=False)
        obs_image_dict = hw_to_dataset_features(obs_image_features, prefix="observation", use_video=True)
        
        features = {**action_dict, **obs_state_dict, **obs_image_dict}
        
        camera_encoder = VideoEncoderConfig(
            vcodec="h264",
            preset="fast",
            extra_options={"tune": "film", "profile:v": "high", "bf": 2}
        )
        
        # 创建数据集（启用流式编码）
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=FPS,
            features=features,
            root=self.root,
            robot_type="lekiwi",
            use_videos=True,
            streaming_encoding=True,
            encoder_threads=2,
            camera_encoder=camera_encoder,
        )
        
        print(f"[Client] ✅ 数据集创建成功!")
    
    def _print_instructions(self):
        """打印使用说明"""
        print("\n" + "=" * 60)
        print("LeKiwi SO101 录制客户端（视频格式）")
        print("=" * 60)
        print("流程：")
        print("  1. 手动调整底盘位置（观察辅助线）")
        print("  2. 按空格键开始录制")
        print("  3. 移动主臂控制从臂抓取纸团")
        print("  4. 按空格键结束录制")
        print("  5. 移动底盘到新位置，按R键继续")
        print("=" * 60)
        print("按键：")
        print("  空格: 开始/结束录制")
        print("  R: 确认重置，下一轮")
        print("  Q: 退出并保存数据集")
        print("=" * 60 + "\n")
    
    def get_leader_action(self):
        """获取主臂动作"""
        try:
            return self.leader_arm.get_action()
        except Exception as e:
            print(f"[Client] 获取主臂动作失败: {e}")
            return None
    
    def send_action(self, action):
        """发送主臂动作到树莓派（含完整9维数据）"""
        try:
            # SO101主臂返回的键名是: shoulder_pan.pos, shoulder_lift.pos, ...
            # 需要添加 "arm_" 前缀以匹配 LeKiwi 的期望格式: arm_shoulder_pan.pos
            arm_action = {}
            for k, v in action.items():
                if k.endswith(".pos"):
                    # 将 "shoulder_pan.pos" 转换为 "arm_shoulder_pan.pos"
                    motor_name = k.replace(".pos", "")
                    arm_action[f"arm_{motor_name}.pos"] = float(v)
            
            # 添加底盘静止
            full_action = {
                **arm_action,
                "x.vel": 0.0,
                "y.vel": 0.0,
                "theta.vel": 0.0,
            }
            
            self.cmd_sock.send_string(json.dumps(full_action), flags=zmq.NOBLOCK)
            return full_action
        except Exception as e:
            print(f"[Client] 发送动作失败: {e}")
            return None
    
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
        """绘制检测框和辅助线"""
        if frame is None:
            return None, False
        
        h, w = frame.shape[:2]
        is_position_good = False
        
        # 绘制中心区域（±15%）
        center_zone_left = int(w * (0.5 - CENTER_THRESHOLD))
        center_zone_right = int(w * (0.5 + CENTER_THRESHOLD))
        cv2.rectangle(frame, (center_zone_left, 0), (center_zone_right, h), (255, 255, 0), 2)
        cv2.putText(frame, "Center Zone (±15%)", (center_zone_left + 5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        if detections:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cx, cy = det["center"]
                conf = det["confidence"]
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                
                label = f"{det['class']}: {conf:.1%}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                det_area = (x2 - x1) * (y2 - y1)
                img_area = w * h
                actual_ratio = det_area / img_area
                
                center_offset = abs(cx - w//2) / (w / 2)
                is_centered = center_offset <= CENTER_THRESHOLD
                is_good_ratio = RATIO_MIN <= actual_ratio <= RATIO_MAX
                is_position_good = is_centered and is_good_ratio
                
                # 绘制指引线
                if not is_centered:
                    direction = "Move Right -->" if cx < w//2 else "<-- Move Left"
                    line_color = (0, 165, 255)
                    cv2.line(frame, (int(cx), int(cy)), (w//2, int(cy)), line_color, 2)
                    cv2.putText(frame, direction, (w//2 - 80, int(cy) - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)
                else:
                    cv2.line(frame, (int(cx), int(cy)), (w//2, int(cy)), (0, 255, 0), 2)
                    cv2.putText(frame, "Centered OK", (w//2 - 50, int(cy) - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 显示占比
                ratio_color = (0, 255, 0) if is_good_ratio else (0, 0, 255)
                ratio_status = "OK" if is_good_ratio else "BAD"
                ratio_text = f"Ratio: {actual_ratio:.1%} [{ratio_status}]"
                cv2.putText(frame, ratio_text, (int(x1), int(y2) + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, ratio_color, 2)
        
        return frame, is_position_good
    
    def draw_info_panel(self, frame, obs, is_position_good, action_info):
        """绘制信息面板"""
        if frame is None:
            return None
        
        h, w = frame.shape[:2]
        host_state = obs.get("host_state", "waiting")
        episode = obs.get("episode_count", 0)
        detections = obs.get("detections", [])
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (420, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (10, 10), (420, 200), (255, 255, 255), 1)
        
        cv2.putText(frame, "SO101 Record + Teleop", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Episode: {episode + 1}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        state_colors = {"waiting": (255, 255, 0), "recording": (0, 0, 255), "reset": (255, 0, 255)}
        state_color = state_colors.get(host_state, (255, 255, 255))
        cv2.putText(frame, f"State: {host_state.upper()}", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        
        if host_state == "waiting":
            if len(detections) == 0:
                pos_text, pos_color = "No paper ball detected", (0, 0, 255)
            elif is_position_good:
                pos_text, pos_color = "Position OK! Press SPACE", (0, 255, 0)
            else:
                pos_text, pos_color = "Adjust base position", (0, 165, 255)
            cv2.putText(frame, pos_text, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pos_color, 2)
        
        if self.recording:
            cv2.putText(frame, f"Recording...", (20, 155),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if action_info:
            cv2.putText(frame, f"Gripper: {action_info.get('gripper', 0):.1f}", (20, 185),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 底部提示
        if host_state == "waiting" and is_position_good:
            frame = self._draw_big_text(frame, "Press SPACE to Start", (0, 255, 0))
        elif host_state == "waiting" and len(detections) == 0:
            frame = self._draw_big_text(frame, "Place paper ball in view", (0, 0, 255))
        elif host_state == "recording":
            frame = self._draw_big_text(frame, "Recording... Press SPACE to Stop", (0, 0, 255))
        elif host_state == "reset":
            frame = self._draw_big_text(frame, "Move base, then press R", (255, 0, 255))
        
        return frame
    
    def _draw_big_text(self, frame, text, color):
        """绘制大提示文字"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
        box_w, box_h = text_size[0] + 40, text_size[1] + 40
        x1 = (w - box_w) // 2
        y1 = h - 100
        cv2.rectangle(overlay, (x1, y1), (x1 + box_w, y1 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), color, 2)
        cv2.putText(frame, text, (x1 + 20, y1 + box_h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        return frame
    
    def display_dual_camera(self, front, wrist):
        """显示双摄像头"""
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
        combined = cv2.resize(combined, (int(combined.shape[1] * 0.8), int(combined.shape[0] * 0.8)))
        cv2.imshow("LeKiwi Record", combined)
    
    def build_frame(self, action, obs, front_img, wrist_img):
        """构建LeRobotDataset frame"""
        arm_state = obs.get("arm_state", {})
        
        # action: 9维数组
        action_values = [
            action.get("arm_shoulder_pan.pos", 0),
            action.get("arm_shoulder_lift.pos", 0),
            action.get("arm_elbow_flex.pos", 0),
            action.get("arm_wrist_flex.pos", 0),
            action.get("arm_wrist_roll.pos", 0),
            action.get("arm_gripper.pos", 0),
            action.get("x.vel", 0),
            action.get("y.vel", 0),
            action.get("theta.vel", 0),
        ]
        
        # observation state: 6维数组
        state_values = [
            arm_state.get("arm_shoulder_pan.pos", 0),
            arm_state.get("arm_shoulder_lift.pos", 0),
            arm_state.get("arm_elbow_flex.pos", 0),
            arm_state.get("arm_wrist_flex.pos", 0),
            arm_state.get("arm_wrist_roll.pos", 0),
            arm_state.get("arm_gripper.pos", 0),
        ]
        
        frame = {
            "action": np.array(action_values, dtype=np.float32),
            "observation.state": np.array(state_values, dtype=np.float32),
            "observation.images.front": front_img,
            "observation.images.wrist": wrist_img,
            "task": "Grasp the paper ball",
        }
        
        return frame
    
    def send_command(self, cmd_dict):
        """发送控制命令"""
        try:
            self.cmd_sock.send_string(json.dumps(cmd_dict), flags=zmq.NOBLOCK)
        except:
            pass
    
    def run(self):
        """主循环"""
        running = True
        action_frame_count = 0
        
        try:
            while running:
                loop_start = time.perf_counter()
                
                # 1. 获取主臂动作并发送（遥操作始终开启）
                action = self.get_leader_action()
                action_info = None
                full_action = None
                
                if action:
                    full_action = self.send_action(action)
                    action_frame_count += 1
                    action_info = {"gripper": action.get("gripper.pos", 0)}
                
                # 2. 接收树莓派观测
                try:
                    msg = self.obs_sock.recv_string(zmq.NOBLOCK)
                    obs = json.loads(msg)
                    self.frame_count += 1
                except zmq.Again:
                    obs = {"host_state": "waiting", "detections": []}
                
                # 3. 解码图像
                front = self.decode_image(obs.get("front"))
                wrist = self.decode_image(obs.get("wrist"))
                
                # 4. 显示图像（wrist旋转90度）
                is_position_good = False
                if front is not None:
                    front_display = front.copy()
                    front_display, is_position_good = self.draw_detection(front_display, obs.get("detections", []))
                    front_display = self.draw_info_panel(front_display, obs, is_position_good, action_info)
                    
                    # wrist摄像头旋转90度
                    wrist_rotated = wrist
                    if wrist is not None:
                        wrist_rotated = cv2.rotate(wrist, cv2.ROTATE_90_CLOCKWISE)
                    
                    self.display_dual_camera(front_display, wrist_rotated)
                
                # 5. 录制时保存到数据集
                if self.recording and full_action and front is not None and wrist is not None:
                    # 确保图像尺寸正确 (height, width, channels) = (480, 640, 3)
                    front_resized = cv2.resize(front, (IMG_WIDTH, IMG_HEIGHT))
                    
                    # wrist旋转90度并调整尺寸
                    wrist_rotated = cv2.rotate(wrist, cv2.ROTATE_90_CLOCKWISE)
                    wrist_resized = cv2.resize(wrist_rotated, (IMG_WIDTH, IMG_HEIGHT))
                    
                    frame = self.build_frame(full_action, obs, front_resized, wrist_resized)
                    self.dataset.add_frame(frame)
                
                # 6. 键盘处理
                key = cv2.waitKey(1) & 0xFF
                host_state = obs.get("host_state", "waiting")
                
                if key == ord(' '):  # 空格
                    if not self.recording:
                        # 开始录制（不依赖host_state，直接开始）
                        self.send_command({"start_recording": True})
                        self.recording = True
                        self.episode_count += 1
                        print(f"🎬 Episode {self.episode_count} 开始录制!")
                    else:
                        # 停止录制（不依赖host_state，直接停止）
                        self.send_command({"stop_recording": True})
                        self.recording = False
                        print(f"✅ Episode {self.episode_count} 完成!")
                        
                        # 保存episode（触发视频编码）
                        self.dataset.save_episode()
                        print(f"💾 已保存 Episode {self.episode_count}")
                
                elif key == ord('r') or key == ord('R'):
                    if host_state == "reset":
                        self.send_command({"confirm_reset": True})
                        print("🔄 重置确认，开始下一轮...")
                
                elif key == ord('q') or key == ord('Q'):
                    if self.recording:
                        self.send_command({"stop_recording": True})
                        self.dataset.save_episode()
                    running = False
                
                # 7. 帧率控制
                elapsed = time.perf_counter() - loop_start
                sleep_t = max(1.0 / FPS - elapsed, 0)
                if sleep_t > 0:
                    time.sleep(sleep_t)
        
        except KeyboardInterrupt:
            print("\n[Client] 用户停止")
        finally:
            if self.recording:
                self.dataset.save_episode()
            
            cv2.destroyAllWindows()
            self.leader_arm.disconnect()
            
            # 完成数据集（写入footer metadata）
            print("[Client] 完成数据集...")
            self.dataset.finalize()
            
            self.cmd_sock.close()
            self.obs_sock.close()
            self.ctx.term()
            
            print(f"\n[Client] 已关闭，共录制 {self.episode_count} 个 episodes")
            print(f"[Client] 数据集位置: {self.root}")
            print(f"[Client] 可使用: lerobot-train --dataset.repo_id={self.repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=DEFAULT_IP, help="树莓派IP")
    parser.add_argument("--arm-port", default=DEFAULT_ARM_PORT, help="主臂串口")
    parser.add_argument("--repo-id", default=DATASET_REPO_ID, help="数据集ID")
    parser.add_argument("--root", default=str(DATASET_ROOT), help="数据集根目录")
    args = parser.parse_args()
    
    client = RecordClient(
        robot_ip=args.ip,
        arm_port=args.arm_port,
        repo_id=args.repo_id,
        root=args.root,
    )
    client.run()


if __name__ == "__main__":
    main()
