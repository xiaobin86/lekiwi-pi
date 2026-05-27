#!/usr/bin/env python
"""
PC端 ACT 抓取推理客户端（6-DOF机械臂版本）
功能：
1. 连接树莓派 ZMQ，接收观测数据
2. 当树莓派 request_act=True 时，运行 ACT 模型推理
3. 只输出机械臂6维动作，底盘速度始终为0

使用方式：
    python src/act_grasp_client.py --model_path outputs/lekiwi_grasp_act/checkpoints/last

注意：
- 需要先训练好 ACT 模型（6-DOF机械臂）
- 树莓派需要运行 host_pi.py 并进入自动导航模式
- 当底盘到达纸团附近时，会自动进入 grasping 状态并请求 ACT 动作
"""

import sys, time, json, base64, argparse
from pathlib import Path

import numpy as np
import cv2
import zmq
import torch

# 可选：LeRobot imports（当模型训练好后启用）
# sys.path.insert(0, str(Path.home() / "lerobot-workspace/lerobot/src"))
# from lerobot.policies import make_policy_from_config
# from lerobot.configs import PreTrainedConfig

# ==================== 配置 ====================
DEFAULT_ROBOT_IP = "192.168.3.176"
CMD_PORT, OBS_PORT = 5555, 5556

# 6-DOF机械臂关节（不再包含底盘3维）
ARM_KEYS = [
    "arm_shoulder_pan.pos", "arm_shoulder_lift.pos", "arm_elbow_flex.pos",
    "arm_wrist_flex.pos", "arm_wrist_roll.pos", "arm_gripper.pos"
]

# 默认机械臂位置（rest位置）
DEFAULT_ARM_POS = {
    "arm_shoulder_pan.pos": 0.0,
    "arm_shoulder_lift.pos": -90.0,
    "arm_elbow_flex.pos": 80.0,
    "arm_wrist_flex.pos": 60.0,
    "arm_wrist_roll.pos": 0.0,
    "arm_gripper.pos": 0.0,  # 完全闭合
}

# 预抓取姿态
PREGRASP_POS = {
    "arm_shoulder_pan.pos": 0.0,
    "arm_shoulder_lift.pos": -80.0,
    "arm_elbow_flex.pos": 70.0,
    "arm_wrist_flex.pos": 50.0,
    "arm_wrist_roll.pos": 0.0,
    "arm_gripper.pos": 100.0,  # 完全打开
}


class ACTArmClient:
    """ACT 6-DOF机械臂推理客户端"""
    
    def __init__(self, robot_ip, model_path=None, device="cuda"):
        self.robot_ip = robot_ip
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 统计数据
        self.frame_count = 0
        self.act_request_count = 0
        self.inference_times = []
        
        # 连接ZMQ
        print(f"[ACT] 连接树莓派 {robot_ip}...")
        self.ctx = zmq.Context()
        
        # 接收观测（树莓派发送）
        self.obs_sock = self.ctx.socket(zmq.PULL)
        self.obs_sock.setsockopt(zmq.CONFLATE, 1)
        self.obs_sock.connect(f"tcp://{robot_ip}:{OBS_PORT}")
        
        # 发送动作（到树莓派）
        self.cmd_sock = self.ctx.socket(zmq.PUSH)
        self.cmd_sock.setsockopt(zmq.CONFLATE, 1)
        self.cmd_sock.connect(f"tcp://{robot_ip}:{CMD_PORT}")
        
        print("[ACT] ZMQ已连接")
        
        # 加载模型
        self.policy = None
        if model_path:
            self.load_model(model_path)
        else:
            print("[ACT] ⚠️ 未提供模型路径，将使用默认动作序列")
    
    def load_model(self, model_path):
        """加载6-DOF ACT模型"""
        print(f"[ACT] 加载6-DOF模型: {model_path}")
        try:
            # TODO: 当模型训练好后，启用以下代码
            # self.config = PreTrainedConfig.from_pretrained(model_path)
            # self.config.pretrained_path = model_path
            # self.policy = make_policy_from_config(self.config)
            # self.policy.eval()
            # self.policy.to(self.device)
            # print(f"[ACT] ✅ 6-DOF模型已加载到 {self.device}")
            
            # 临时：标记为未加载
            print("[ACT] ⚠️ 模型加载代码待实现（需先训练6-DOF模型）")
            self.policy = None
        except Exception as e:
            print(f"[ACT] ❌ 模型加载失败: {e}")
            self.policy = None
    
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
            print(f"[ACT] 图像解码错误: {e}")
            return None
    
    def preprocess_observation(self, obs_dict):
        """预处理观测为模型输入格式（6-DOF）"""
        # 解码图像
        frame = self.decode_image(obs_dict.get("front"))
        if frame is None:
            return None
        
        # 图像预处理: BGR → RGB, HWC → CHW, /255
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1).contiguous()  # HWC → CHW
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)  # 添加batch维度
        
        # 构建观测字典（6-DOF）
        obs = {
            "observation.images.front": frame_tensor,
            "task": "Grasp paper ball with follower arm",
            "robot_type": "lekiwi",
        }
        
        # 获取机械臂状态（如果树莓派发送了）
        if "arm_state" in obs_dict:
            arm_state = np.array(obs_dict["arm_state"], dtype=np.float32)
        else:
            # 使用默认状态
            arm_state = np.array([
                DEFAULT_ARM_POS[k] for k in ARM_KEYS
            ], dtype=np.float32)
        
        state_tensor = torch.from_numpy(arm_state).unsqueeze(0).to(self.device)
        obs["observation.state"] = state_tensor  # [1, 6]
        
        return obs
    
    def inference_placeholder(self, obs_dict):
        """6-DOF占位推理函数（模型未训练时使用）
        
        返回机械臂抓取动作序列：
        1. 移动到预抓取位置，夹爪打开
        2. 向下移动到抓取位置
        3. 夹爪闭合（抓取）
        4. 抬起机械臂
        5. 保持抓取姿态
        """
        progress = obs_dict.get("grasp_progress", 0.0)
        
        # 根据进度生成不同动作（只有机械臂6-DOF）
        if progress < 0.2:
            # 阶段1: 预抓取，夹爪打开
            action = {
                "arm_shoulder_pan.pos": 0.0,
                "arm_shoulder_lift.pos": -80.0,
                "arm_elbow_flex.pos": 70.0,
                "arm_wrist_flex.pos": 50.0,
                "arm_wrist_roll.pos": 0.0,
                "arm_gripper.pos": 100.0,  # 打开
            }
        elif progress < 0.4:
            # 阶段2: 向下抓取
            action = {
                "arm_shoulder_pan.pos": 0.0,
                "arm_shoulder_lift.pos": -100.0,
                "arm_elbow_flex.pos": 90.0,
                "arm_wrist_flex.pos": 30.0,
                "arm_wrist_roll.pos": 0.0,
                "arm_gripper.pos": 100.0,  # 保持打开
            }
        elif progress < 0.6:
            # 阶段3: 夹爪闭合
            action = {
                "arm_shoulder_pan.pos": 0.0,
                "arm_shoulder_lift.pos": -100.0,
                "arm_elbow_flex.pos": 90.0,
                "arm_wrist_flex.pos": 30.0,
                "arm_wrist_roll.pos": 0.0,
                "arm_gripper.pos": 0.0,  # 闭合
            }
        elif progress < 0.8:
            # 阶段4: 抬起
            action = {
                "arm_shoulder_pan.pos": 0.0,
                "arm_shoulder_lift.pos": -70.0,
                "arm_elbow_flex.pos": 60.0,
                "arm_wrist_flex.pos": 60.0,
                "arm_wrist_roll.pos": 0.0,
                "arm_gripper.pos": 0.0,  # 保持闭合
            }
        else:
            # 阶段5: 保持
            action = {
                "arm_shoulder_pan.pos": 0.0,
                "arm_shoulder_lift.pos": -70.0,
                "arm_elbow_flex.pos": 60.0,
                "arm_wrist_flex.pos": 60.0,
                "arm_wrist_roll.pos": 0.0,
                "arm_gripper.pos": 0.0,
            }
        
        return action
    
    def run_inference(self, obs_dict):
        """运行ACT推理（6-DOF）"""
        if self.policy is not None:
            # TODO: 实现真实6-DOF推理
            # obs_tensor = self.preprocess_observation(obs_dict)
            # with torch.inference_mode():
            #     action = self.policy.select_action(obs_tensor)  # [1, 6]
            # # 转换为dict
            # action_np = action.squeeze(0).cpu().numpy()
            # return {k: float(action_np[i]) for i, k in enumerate(ARM_KEYS)}
            pass
        
        # 使用占位推理
        return self.inference_placeholder(obs_dict)
    
    def send_act_action(self, action_dict):
        """发送6-DOF机械臂动作到树莓派"""
        # 添加来源标记和底盘静止命令
        full_action = {
            **action_dict,  # 6-DOF机械臂
            "source": "act",
            "x.vel": 0.0,   # 底盘静止
            "y.vel": 0.0,
            "theta.vel": 0.0
        }
        
        try:
            self.cmd_sock.send_string(json.dumps(full_action), flags=zmq.NOBLOCK)
        except Exception as e:
            print(f"[ACT] 发送动作失败: {e}")
    
    def run(self):
        """主循环"""
        print("\n" + "=" * 60)
        print("LeKiwi ACT 6-DOF机械臂推理客户端")
        print("=" * 60)
        print(f"树莓派IP: {self.robot_ip}")
        print(f"模型路径: {self.model_path or '未提供（使用默认动作）'}")
        print(f"推理设备: {self.device}")
        print(f"控制维度: 6-DOF机械臂（底盘自动静止）")
        print("=" * 60)
        print("\n等待树莓派进入 grasping 状态...")
        print("按 Ctrl+C 停止\n")
        
        running = True
        
        try:
            while running:
                # 1. 接收观测
                try:
                    msg = self.obs_sock.recv_string(zmq.NOBLOCK)
                    obs = json.loads(msg)
                    self.frame_count += 1
                except zmq.Again:
                    time.sleep(0.001)
                    continue
                
                # 2. 检查是否需要ACT动作
                if obs.get("request_act"):
                    self.act_request_count += 1
                    
                    # 运行推理（只输出6-DOF机械臂）
                    inference_start = time.perf_counter()
                    action = self.run_inference(obs)
                    inference_time = (time.perf_counter() - inference_start) * 1000
                    self.inference_times.append(inference_time)
                    
                    # 发送动作（自动添加底盘静止）
                    self.send_act_action(action)
                    
                    # 打印状态
                    if self.act_request_count % 10 == 0:
                        avg_time = np.mean(self.inference_times[-10:]) if self.inference_times else 0
                        gripper = action.get('arm_gripper.pos', 0)
                        print(f"[ACT] 请求 #{self.act_request_count}, "
                              f"进度: {obs.get('grasp_progress', 0):.0%}, "
                              f"推理: {avg_time:.1f}ms, "
                              f"夹爪: {gripper:.0f}")
        
        except KeyboardInterrupt:
            print("\n[ACT] 用户停止")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """清理资源"""
        print("\n" + "=" * 60)
        print("ACT 6-DOF客户端统计")
        print("=" * 60)
        print(f"总帧数: {self.frame_count}")
        print(f"ACT请求数: {self.act_request_count}")
        
        if self.inference_times:
            avg_time = np.mean(self.inference_times)
            max_time = np.max(self.inference_times)
            print(f"平均推理延迟: {avg_time:.1f}ms")
            print(f"最大推理延迟: {max_time:.1f}ms")
        
        print("=" * 60)
        
        self.obs_sock.close()
        self.cmd_sock.close()
        self.ctx.term()
        print("[ACT] 已关闭")


def main():
    parser = argparse.ArgumentParser(description="LeKiwi ACT 6-DOF机械臂推理客户端")
    parser.add_argument("--robot_ip", default=DEFAULT_ROBOT_IP, help="树莓派IP地址")
    parser.add_argument("--model_path", default=None, help="训练好的6-DOF ACT模型路径（可选）")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="推理设备")
    args = parser.parse_args()
    
    client = ACTArmClient(
        robot_ip=args.robot_ip,
        model_path=args.model_path,
        device=args.device
    )
    
    client.run()


if __name__ == "__main__":
    main()
