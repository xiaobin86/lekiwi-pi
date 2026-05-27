#!/usr/bin/env python
"""
PC端 ACT 推理客户端
功能：加载ACT模型，接收树莓派观测，推理动作，发送回树莓派执行
"""

import sys, time, json, base64, argparse
from pathlib import Path

import numpy as np
import cv2
import zmq
import torch

# LeRobot imports
sys.path.insert(0, str(Path.home() / "lerobot-workspace/lerobot/src"))
from lerobot.policies import make_policy_from_config
from lerobot.configs import PreTrainedConfig
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.constants import ACTION, OBS_STR


# ==================== 配置 ====================
DEFAULT_ROBOT_IP = "192.168.3.176"
CMD_PORT, OBS_PORT = 5555, 5556
FPS = 30

# 状态特征顺序（必须与训练时一致）
STATE_KEYS = [
    "arm_shoulder_pan.pos", "arm_shoulder_lift.pos", "arm_elbow_flex.pos",
    "arm_wrist_flex.pos", "arm_wrist_roll.pos", "arm_gripper.pos",
    "x.vel", "y.vel", "theta.vel"
]

# 图像尺寸
IMG_H, IMG_W = 480, 640


class ACTInferenceClient:
    """ACT策略推理客户端"""
    
    def __init__(self, model_path, robot_ip, device="cuda"):
        self.model_path = model_path
        self.robot_ip = robot_ip
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 1. 加载策略模型
        print(f"[Inference] 加载ACT模型: {model_path}")
        self.config = PreTrainedConfig.from_pretrained(model_path)
        self.config.pretrained_path = model_path
        self.policy = make_policy_from_config(self.config)
        self.policy.eval()
        self.policy.to(self.device)
        print(f"[Inference] 模型已加载到 {self.device}")
        
        # 2. 获取特征信息
        self.input_features = self.config.input_features
        self.output_features = self.config.output_features
        self.action_keys = list(self.output_features.keys())
        print(f"[Inference] 动作维度: {len(self.action_keys)}")
        
        # 3. 连接ZMQ
        print(f"[Inference] 连接树莓派 {robot_ip}...")
        self.ctx = zmq.Context()
        
        self.obs_sock = self.ctx.socket(zmq.PULL)
        self.obs_sock.setsockopt(zmq.CONFLATE, 1)
        self.obs_sock.connect(f"tcp://{robot_ip}:{OBS_PORT}")
        
        self.cmd_sock = self.ctx.socket(zmq.PUSH)
        self.cmd_sock.setsockopt(zmq.CONFLATE, 1)
        self.cmd_sock.connect(f"tcp://{robot_ip}:{CMD_PORT}")
        
        print("[Inference] ZMQ已连接")
        
        # 4. 统计数据
        self.inference_times = []
        self.frame_count = 0
        
    def decode_image(self, img_b64):
        """解码base64图像"""
        if not img_b64:
            return None
        try:
            data = base64.b64decode(img_b64)
            arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            print(f"[Inference] 图像解码错误: {e}")
            return None
    
    def preprocess_observation(self, obs_dict):
        """预处理观测为模型输入"""
        # 解码图像
        frame = self.decode_image(obs_dict.get("front"))
        if frame is None:
            return None
        
        # 图像预处理: BGR → RGB, HWC → CHW, /255
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1).contiguous()  # HWC → CHW
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)  # 添加batch维度
        
        # 构建观测字典
        obs = {
            "observation.images.front": frame_tensor,
            "task": "Grasp paper ball with follower arm",
            "robot_type": "lekiwi",
        }
        
        # 如果有状态信息，添加状态
        if "state" in obs_dict:
            state = np.array(obs_dict["state"], dtype=np.float32)
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            obs["observation.state"] = state_tensor
        else:
            # 使用零状态作为占位（实际应从树莓派获取）
            state_tensor = torch.zeros(1, len(STATE_KEYS), device=self.device)
            obs["observation.state"] = state_tensor
        
        return obs
    
    def postprocess_action(self, action_tensor):
        """后处理动作输出为字典"""
        # action_tensor: [1, action_dim] 或 [1, chunk_size, action_dim]
        action_np = action_tensor.squeeze(0).cpu().numpy()
        
        # 如果输出是动作序列，取第一步
        if action_np.ndim == 2:
            action_np = action_np[0]
        
        # 构建动作字典
        action_dict = {}
        for i, key in enumerate(self.action_keys):
            if i < len(action_np):
                # 去掉前缀 "action."
                clean_key = key.replace("action.", "")
                action_dict[clean_key] = float(action_np[i])
        
        return action_dict
    
    def send_action(self, action_dict):
        """发送动作到树莓派"""
        try:
            self.cmd_sock.send_string(json.dumps(action_dict), flags=zmq.NOBLOCK)
        except Exception as e:
            print(f"[Inference] 发送动作失败: {e}")
    
    def run(self):
        """主推理循环"""
        print("\n" + "=" * 60)
        print("ACT推理客户端已启动")
        print("=" * 60)
        print("等待树莓派观测数据...")
        print("按 Ctrl+C 停止")
        print("=" * 60 + "\n")
        
        running = True
        
        try:
            while running:
                loop_start = time.perf_counter()
                
                # 1. 接收观测
                try:
                    msg = self.obs_sock.recv_string(zmq.NOBLOCK)
                    obs_dict = json.loads(msg)
                except zmq.Again:
                    # 没有新数据，继续等待
                    time.sleep(0.001)
                    continue
                
                # 2. 预处理观测
                obs_tensor = self.preprocess_observation(obs_dict)
                if obs_tensor is None:
                    continue
                
                # 3. 策略推理
                inference_start = time.perf_counter()
                with torch.inference_mode():
                    action = self.policy.select_action(obs_tensor)
                inference_time = (time.perf_counter() - inference_start) * 1000
                self.inference_times.append(inference_time)
                
                # 4. 后处理动作
                action_dict = self.postprocess_action(action)
                
                # 5. 发送动作
                self.send_action(action_dict)
                
                self.frame_count += 1
                
                # 6. 打印统计
                if self.frame_count % 30 == 0:
                    avg_time = np.mean(self.inference_times[-30:])
                    print(f"[Inference] 帧 {self.frame_count}, "
                          f"推理延迟: {avg_time:.1f}ms, "
                          f"动作: {list(action_dict.keys())[:3]}...")
                
                # 7. 帧率控制
                elapsed = time.perf_counter() - loop_start
                sleep_t = max(1 / FPS - elapsed, 0)
                if sleep_t > 0:
                    time.sleep(sleep_t)
        
        except KeyboardInterrupt:
            print("\n[Inference] 用户停止")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """清理资源"""
        print("\n[Inference] 清理资源...")
        
        # 打印统计
        if self.inference_times:
            avg_time = np.mean(self.inference_times)
            max_time = np.max(self.inference_times)
            min_time = np.min(self.inference_times)
            print(f"[Inference] 统计:")
            print(f"  总帧数: {self.frame_count}")
            print(f"  平均推理延迟: {avg_time:.1f}ms")
            print(f"  最大推理延迟: {max_time:.1f}ms")
            print(f"  最小推理延迟: {min_time:.1f}ms")
            print(f"  等效帧率: {1000/avg_time:.1f}fps")
        
        self.obs_sock.close()
        self.cmd_sock.close()
        self.ctx.term()
        print("[Inference] 已关闭")


def main():
    parser = argparse.ArgumentParser(description="ACT策略推理客户端")
    parser.add_argument("--model_path", required=True, help="训练好的ACT模型路径")
    parser.add_argument("--robot_ip", default=DEFAULT_ROBOT_IP, help="树莓派IP地址")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="推理设备")
    args = parser.parse_args()
    
    print("=" * 60)
    print("LeKiwi ACT 推理客户端")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print(f"树莓派IP: {args.robot_ip}")
    print(f"推理设备: {args.device}")
    print("=" * 60)
    
    client = ACTInferenceClient(
        model_path=args.model_path,
        robot_ip=args.robot_ip,
        device=args.device
    )
    
    client.run()


if __name__ == "__main__":
    main()
