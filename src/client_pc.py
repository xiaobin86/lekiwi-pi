#!/usr/bin/env python
"""
PC端 Client - 手柄控制 + ACT推理 + 图像显示

功能：
1. 手柄控制底盘移动（D-pad）
2. A键切换自动/手动模式
3. X键拍照
4. 当树莓派进入grasping状态时，自动进行ACT推理
5. 显示摄像头图像和状态信息

使用方式：
    # 带图像显示（默认）
    python src/client_pc.py --ip 192.168.3.176 --model_path outputs/train/grasp_paper_ball/checkpoints/004000/pretrained_model
    
    # 仅控制台输出（无图像窗口）
    python src/client_pc.py --ip 192.168.3.176 --model_path outputs/train/grasp_paper_ball/checkpoints/004000/pretrained_model --no-display
    
    # 使用CPU推理
    python src/client_pc.py --ip 192.168.3.176 --model_path outputs/train/grasp_paper_ball/checkpoints/004000/pretrained_model --device cpu

工作流程：
    1. 按A键切换自动模式
    2. 树莓派自动导航寻找纸团
    3. 到达纸团附近后，自动进入grasping状态
    4. PC端收到request_act=True，开始ACT推理
    5. 机械臂自动执行抓取动作
    6. 完成后回到idle状态

按键说明：
    D-pad:      移动底盘
    RB+左右:     旋转底盘
    LB:         切换速度（慢/中/快）
    A:          切换自动/手动模式
    X:          拍照保存
    START:      退出程序
    ESC:        退出程序（键盘）
"""

import sys, time, json, base64, argparse, logging
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import pygame
import zmq
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path.home() / "lerobot-workspace/lerobot/src"))
from lerobot.policies import make_policy
from lerobot.datasets import LeRobotDataset

# ==================== 配置 ====================
DEFAULT_IP = "192.168.3.176"
CMD_PORT, OBS_PORT = 5555, 5556
FPS = 30
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SPEED_LEVELS = [
    {"xy": 0.1, "theta": 30},
    {"xy": 0.3, "theta": 60},
    {"xy": 0.5, "theta": 90},
]

ARM_DEFAULTS = {
    "arm_shoulder_pan.pos": 0.0,
    "arm_shoulder_lift.pos": -100.0,
    "arm_elbow_flex.pos": 90.0,
    "arm_wrist_flex.pos": 70.0,
    "arm_wrist_roll.pos": 0.0,
    "arm_gripper.pos": 0.0,
}


# ==================== ACT推理器 ====================
class ACTInference:
    """ACT策略推理器"""
    
    def __init__(self, model_path, device="cuda"):
        self.model_path = Path(model_path)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.policy = None
        self.dataset = None
        self.action_mean = None
        self.action_std = None
        
        self._load_model()
    
    def _load_model(self):
        """加载训练好的ACT模型（推理时不需要完整数据集）"""
        try:
            logger.info(f"[ACT] 加载模型: {self.model_path}")
            
            # 检查模型文件是否存在
            config_path = self.model_path / "config.json"
            model_path = self.model_path / "model.safetensors"
            
            if not config_path.exists():
                logger.error(f"[ACT] 模型配置不存在: {config_path}")
                return False
            
            if not model_path.exists():
                logger.error(f"[ACT] 模型权重不存在: {model_path}")
                return False
            
            # 从训练配置获取数据集路径（仅用于日志显示）
            train_config_path = self.model_path / "train_config.json"
            dataset_root = None
            if train_config_path.exists():
                import json
                with open(train_config_path, 'r') as f:
                    train_config = json.load(f)
                dataset_root = train_config.get('dataset', {}).get('root', None)
                if dataset_root:
                    logger.info(f"[ACT] 训练数据集: {dataset_root}")
            
            # 创建策略
            from lerobot.configs import PreTrainedConfig
            config = PreTrainedConfig.from_pretrained(str(self.model_path))
            
            logger.info(f"[ACT] 配置类型: {config.type}")
            logger.info(f"[ACT] 输入特征: {list(config.input_features.keys())}")
            logger.info(f"[ACT] 输出特征: {list(config.output_features.keys())}")
            
            # 加载数据集元信息（make_policy需要ds_meta）
            logger.info(f"[ACT] 加载数据集元信息...")
            if dataset_root and Path(dataset_root).exists():
                self.dataset = LeRobotDataset(repo_id='acelan', root=dataset_root)
            else:
                # 尝试使用默认路径
                default_data_path = Path.home() / "lerobot-workspace/lekiwi-pi/data"
                if default_data_path.exists():
                    data_dirs = list(default_data_path.glob("acelan_*"))
                    if data_dirs:
                        dataset_root = str(data_dirs[0])
                        logger.info(f"[ACT] 使用默认数据集: {dataset_root}")
                        self.dataset = LeRobotDataset(repo_id='acelan', root=dataset_root)
                    else:
                        logger.error("[ACT] 找不到数据集目录")
                        return False
                else:
                    logger.error("[ACT] 数据集路径不存在")
                    return False
            
            self.policy = make_policy(cfg=config, ds_meta=self.dataset.meta)
            self.policy = self.policy.from_pretrained(str(self.model_path))
            self.policy.eval()
            self.policy.to(self.device)
            
            # 加载后处理器参数（用于手动反归一化）
            postprocessor_path = self.model_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
            if postprocessor_path.exists():
                from safetensors.torch import load_file
                pp_state = load_file(str(postprocessor_path))
                self.action_mean = pp_state.get("action.mean", None)
                self.action_std = pp_state.get("action.std", None)
                if self.action_mean is not None and self.action_std is not None:
                    self.action_mean = self.action_mean.cpu().numpy()
                    self.action_std = self.action_std.cpu().numpy()
                    logger.info(f"[ACT] 反归一化参数已加载")
                    logger.info(f"[ACT]   action.mean: {self.action_mean[:6]}")
                    logger.info(f"[ACT]   action.std:  {self.action_std[:6]}")
            
            logger.info(f"[ACT] ✅ 模型加载完成，设备: {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"[ACT] 加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def infer(self, front_image, wrist_image=None, arm_state=None):
        """运行ACT推理"""
        if self.policy is None:
            logger.error("[ACT] 模型未加载，无法推理")
            return None
        
        try:
            obs = {}
            
            # 处理图像
            if front_image is not None:
                # BGR -> RGB
                front_rgb = cv2.cvtColor(front_image, cv2.COLOR_BGR2RGB)
                # (H, W, C) -> (C, H, W) 并归一化到 [0, 1]
                front_tensor = torch.from_numpy(front_rgb).permute(2, 0, 1).float() / 255.0
                front_tensor = front_tensor.to(self.device)
                obs["observation.images.front"] = front_tensor.unsqueeze(0)  # 添加batch维度
                logger.debug(f"[ACT] 前视图像: {front_tensor.shape}")
            else:
                # 如果没有前视图像，使用零张量
                obs["observation.images.front"] = torch.zeros(1, 3, 480, 640, device=self.device)
            
            # 处理腕部图像（如果没有，使用零张量）
            if wrist_image is not None:
                wrist_rgb = cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB)
                wrist_tensor = torch.from_numpy(wrist_rgb).permute(2, 0, 1).float() / 255.0
                wrist_tensor = wrist_tensor.to(self.device)
                obs["observation.images.wrist"] = wrist_tensor.unsqueeze(0)
                logger.debug(f"[ACT] 腕部图像: {wrist_tensor.shape}")
            else:
                # 如果没有腕部图像，使用零张量（模型训练时要求）
                obs["observation.images.wrist"] = torch.zeros(1, 3, 480, 640, device=self.device)
                logger.debug("[ACT] 腕部图像: 使用零张量")
            
            # 处理状态
            if arm_state is None:
                arm_state = ARM_DEFAULTS
            
            # 构建状态向量 [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
            state_values = [
                arm_state.get("arm_shoulder_pan.pos", 0.0),
                arm_state.get("arm_shoulder_lift.pos", 0.0),
                arm_state.get("arm_elbow_flex.pos", 0.0),
                arm_state.get("arm_wrist_flex.pos", 0.0),
                arm_state.get("arm_wrist_roll.pos", 0.0),
                arm_state.get("arm_gripper.pos", 0.0),
            ]
            state_tensor = torch.tensor([state_values], dtype=torch.float32, device=self.device)
            obs["observation.state"] = state_tensor
            logger.debug(f"[ACT] 状态: {state_values}")
            
            # 运行推理
            logger.debug("[ACT] 开始推理...")
            with torch.no_grad():
                action = self.policy.select_action(obs)
            
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            logger.info(f"[ACT] 原始输出(归一化): {action}")
            
            # 反归一化：actual = normalized * std + mean
            if self.action_mean is not None and self.action_std is not None:
                action = action * self.action_std[:action.shape[-1]] + self.action_mean[:action.shape[-1]]
                logger.info(f"[ACT] 反归一化后(角度): {action}")
            else:
                logger.warning("[ACT] 未加载反归一化参数，输出可能不正确")
            
            return action
            
        except Exception as e:
            logger.error(f"[ACT] 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return None


# ==================== 手柄控制器 ====================
class Gamepad:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        self.connected = False
        self.speed_idx = 1
        self.prev = {}
        
        self.BTN_A = 0
        self.BTN_B = 1
        self.BTN_X = 3
        self.BTN_Y = 4
        self.BTN_LB = 6
        self.BTN_RB = 7
        self.BTN_BACK = 10
        self.BTN_START = 11
    
    def connect(self):
        retry = 0
        while pygame.joystick.get_count() == 0 and retry < 50:
            time.sleep(0.1)
            pygame.joystick.init()
            retry += 1
        
        if pygame.joystick.get_count() == 0:
            logger.warning("❌ 未检测到手柄")
            return False
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.connected = True
        logger.info(f"✅ 手柄: {self.joystick.get_name()}")
        return True
    
    def get_action(self):
        if not self.connected:
            return {}
        
        pygame.event.pump()
        
        speed = SPEED_LEVELS[self.speed_idx]
        x_cmd = y_cmd = theta_cmd = 0.0
        
        # D-pad
        if self.joystick.get_numhats() > 0:
            hat_x, hat_y = self.joystick.get_hat(0)
            rb = self.joystick.get_button(self.BTN_RB)
            
            if rb and hat_x < 0:
                theta_cmd = speed["theta"]
            elif rb and hat_x > 0:
                theta_cmd = -speed["theta"]
            else:
                if hat_y > 0:
                    x_cmd = speed["xy"]
                elif hat_y < 0:
                    x_cmd = -speed["xy"]
                
                if hat_x < 0:
                    y_cmd = speed["xy"]
                elif hat_x > 0:
                    y_cmd = -speed["xy"]
        
        # LB切换速度
        lb = self.joystick.get_button(self.BTN_LB)
        if lb and not self.prev.get("LB", False):
            self.speed_idx = (self.speed_idx + 1) % len(SPEED_LEVELS)
            names = ["慢速", "中速", "快速"]
            logger.info(f"速度: {names[self.speed_idx]}")
        self.prev["LB"] = lb
        
        # A切换自动导航
        a = self.joystick.get_button(self.BTN_A)
        toggle_auto = a and not self.prev.get("A", False)
        if toggle_auto:
            logger.info("🔄 切换自动导航")
        self.prev["A"] = a
        
        # X拍照
        x = self.joystick.get_button(self.BTN_X)
        capture = x and not self.prev.get("X", False)
        if capture:
            logger.info("📷 拍照")
        self.prev["X"] = x
        
        # START退出
        start_pressed = self.joystick.get_button(self.BTN_START)
        start_trigger = start_pressed and not self.prev.get("START", False)
        if start_trigger:
            logger.info("🚪 按下START键，准备退出")
        self.prev["START"] = start_pressed
        
        return {
            "x.vel": x_cmd,
            "y.vel": y_cmd,
            "theta.vel": theta_cmd,
            "capture_image": capture,
            "toggle_auto": toggle_auto,
            "exit": start_trigger,
        }
    
    def check_exit(self):
        if not self.connected:
            return False
        pygame.event.pump()
        return self.joystick.get_button(self.BTN_START)
    
    def disconnect(self):
        if self.joystick:
            self.joystick.quit()
        pygame.quit()


# ==================== 图像显示 ====================
# 创建窗口（只创建一次）
cv2.namedWindow("LeKiwi ACT", cv2.WINDOW_NORMAL)
cv2.resizeWindow("LeKiwi ACT", 1280, 480)  # 双摄像头宽度

def show_frame(front_b64, wrist_b64=None, detections=None, auto_mode=False, nav_state="idle",
               grasp_progress=0.0, request_act=False, save_path=None, act_action=None):
    if not front_b64:
        return False
    
    try:
        # 解码前视图像
        data = base64.b64decode(front_b64)
        arr = np.frombuffer(data, np.uint8)
        front_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if front_frame is None:
            logger.warning("前视图像解码失败")
            return False
        
        # 解码腕部图像（如果有）
        if wrist_b64:
            try:
                wrist_data = base64.b64decode(wrist_b64)
                wrist_arr = np.frombuffer(wrist_data, np.uint8)
                wrist_frame = cv2.imdecode(wrist_arr, cv2.IMREAD_COLOR)
                if wrist_frame is None:
                    wrist_frame = np.zeros_like(front_frame)
            except:
                wrist_frame = np.zeros_like(front_frame)
        else:
            wrist_frame = np.zeros_like(front_frame)
        
        # 确保尺寸一致
        if wrist_frame.shape != front_frame.shape:
            wrist_frame = cv2.resize(wrist_frame, (front_frame.shape[1], front_frame.shape[0]))
        
        # 绘制检测框（仅在前视图像上）
        if detections:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cx, cy = det["center"]
                cv2.rectangle(front_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(front_frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)
                label = f"{det['class']}: {det['confidence']:.1%}"
                cv2.putText(front_frame, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 添加标签
        cv2.putText(front_frame, "FRONT", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(wrist_frame, "WRIST", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 合并图像（水平拼接）
        combined = np.hstack([front_frame, wrist_frame])
        
        # 状态面板
        mode_text = "AUTO" if auto_mode else "MANUAL"
        mode_color = (0, 165, 255) if auto_mode else (0, 255, 0)
        
        state_colors = {
            "idle": (128, 128, 128),
            "searching": (255, 255, 0),
            "aligning": (255, 165, 0),
            "approaching": (0, 255, 0),
            "arrived": (0, 255, 0),
            "grasping": (255, 0, 255),
            "manual": (0, 255, 0),
        }
        state_color = state_colors.get(nav_state, (255, 255, 255))
        
        # 绘制信息面板（在合并图像的左半部分）
        panel_h = 140 if nav_state == "grasping" else 110
        cv2.rectangle(combined, (10, 10), (360, panel_h), (0, 0, 0), -1)
        cv2.rectangle(combined, (10, 10), (360, panel_h), (255, 255, 255), 1)
        
        cv2.putText(combined, f"Mode: {mode_text}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        state_text = nav_state.upper() if auto_mode else "MANUAL CONTROL"
        cv2.putText(combined, f"State: {state_text}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        
        det_count = len(detections) if detections else 0
        cv2.putText(combined, f"Objects: {det_count}", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 抓取进度
        if nav_state == "grasping":
            progress_text = f"Grasp: {grasp_progress:.0%}"
            cv2.putText(combined, progress_text, (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # 进度条
            bar_x, bar_y, bar_w, bar_h = 10, combined.shape[0] - 30, 200, 20
            cv2.rectangle(combined, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            fill_w = int(bar_w * grasp_progress)
            cv2.rectangle(combined, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (255, 0, 255), -1)
            cv2.rectangle(combined, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
            
            # 显示ACT动作
            if act_action is not None:
                gripper_val = act_action[5] if len(act_action) > 5 else 0
                cv2.putText(combined, f"Gripper: {gripper_val:.2f}", (220, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 显示图像
        cv2.imshow("LeKiwi ACT", combined)
        key = cv2.waitKey(1) & 0xFF
        
        # 保存
        if save_path:
            cv2.imwrite(str(save_path), combined)
            logger.info(f"💾 已保存: {save_path}")
        
        # 返回按键状态（ESC退出）
        return key == 27
        
    except Exception as e:
        logger.error(f"图像显示错误: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description="LeKiwi PC端整合客户端")
    parser.add_argument("--ip", default=DEFAULT_IP, help="树莓派IP地址")
    parser.add_argument("--model_path", required=True, help="ACT模型路径")
    parser.add_argument("--display", action="store_true", default=True, help="显示图像窗口")
    parser.add_argument("--no-display", action="store_true", help="不显示图像窗口（仅控制台）")
    parser.add_argument("--device", default="cuda", help="推理设备 (cuda/cpu)")
    args = parser.parse_args()
    
    # 处理 no-display 参数
    if args.no_display:
        args.display = False
    
    logger.info("=" * 60)
    logger.info("LeKiwi PC端整合客户端")
    logger.info(f"树莓派: {args.ip}:{CMD_PORT}/{OBS_PORT}")
    logger.info(f"模型: {args.model_path}")
    logger.info(f"设备: {args.device}")
    logger.info(f"显示: {'开启' if args.display else '关闭'}")
    logger.info("=" * 60)
    
    # 1. 加载ACT模型
    logger.info("\n[1/4] 加载ACT模型...")
    act_inference = ACTInference(args.model_path, device=args.device)
    if act_inference.policy is None:
        logger.error("模型加载失败，退出")
        return
    
    # 2. 连接手柄
    logger.info("\n[2/4] 连接手柄...")
    gamepad = Gamepad()
    if not gamepad.connect():
        logger.warning("未连接手柄，使用键盘控制（功能受限）")
    
    # 3. 连接树莓派
    logger.info("\n[3/4] 连接树莓派...")
    ctx = zmq.Context()
    
    cmd_sock = ctx.socket(zmq.PUSH)
    cmd_sock.connect(f"tcp://{args.ip}:{CMD_PORT}")
    
    obs_sock = ctx.socket(zmq.PULL)
    obs_sock.setsockopt(zmq.CONFLATE, 1)
    obs_sock.connect(f"tcp://{args.ip}:{OBS_PORT}")
    
    logger.info("✅ 已连接")
    
    # 4. 帮助信息
    logger.info("\n" + "=" * 60)
    logger.info("控制说明:")
    logger.info("  D-pad: 移动底盘")
    logger.info("  RB+左右: 旋转底盘")
    logger.info("  LB: 切换速度")
    logger.info("  A: 切换自动/手动模式")
    logger.info("  X: 拍照")
    logger.info("  START: 退出")
    logger.info("=" * 60)
    logger.info(f"\n📁 保存目录: {DATA_DIR}")
    
    # 主循环
    logger.info("\n[4/4] 运行中...")
    running = True
    capture_req = False
    last_act_time = 0
    act_interval = 0.1  # 100ms, 10Hz 推理频率
    current_front_image = None
    current_wrist_image = None
    act_action = None  # 当前ACT动作
    
    try:
        while running:
            loop_start = time.perf_counter()
            
            # 1. 接收观测数据（先接收，了解当前状态）
            obs_data = None
            try:
                msg = obs_sock.recv_string(zmq.NOBLOCK)
                obs_data = json.loads(msg)
            except zmq.Again:
                pass
            except Exception as e:
                logger.error(f"接收观测错误: {e}")
            
            # 2. 处理观测数据
            request_act = False
            nav_state = "idle"
            front_b64 = None
            detections = []
            auto_mode = False
            grasp_progress = 0.0
            
            if obs_data:
                front_b64 = obs_data.get("front")
                wrist_b64 = obs_data.get("wrist")
                detections = obs_data.get("detections", [])
                auto_mode = obs_data.get("auto_mode", False)
                nav_state = obs_data.get("nav_state", "idle")
                request_act = obs_data.get("request_act", False)
                grasp_progress = obs_data.get("grasp_progress", 0.0)
                
                # 如果进入grasping状态，记住这个状态直到完成
                if nav_state == "grasping":
                    request_act = True
                
                # 解码图像用于ACT推理
                if front_b64:
                    try:
                        img_data = base64.b64decode(front_b64)
                        img_arr = np.frombuffer(img_data, np.uint8)
                        current_front_image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    except:
                        pass
                
                if wrist_b64:
                    try:
                        wrist_data = base64.b64decode(wrist_b64)
                        wrist_arr = np.frombuffer(wrist_data, np.uint8)
                        current_wrist_image = cv2.imdecode(wrist_arr, cv2.IMREAD_COLOR)
                    except:
                        pass
            
            # 3. 手柄输入（获取控制命令）
            action = gamepad.get_action()
            
            if action.get("capture_image"):
                capture_req = True
            
            # 检查退出
            if action.get("exit"):
                logger.info("\n🚪 退出...")
                running = False
                continue
            
            # 4. 发送命令（根据状态决定发送什么）
            cmd = {}
            
            if request_act:
                time_since_last = time.time() - last_act_time
                logger.info(f"🤖 request_act=True, front_image={current_front_image is not None}, time_since_last={time_since_last:.3f}s, will_infer={time_since_last >= act_interval}")
            
            if request_act and current_front_image is not None:
                # === ACT抓取阶段：运行推理并发送机械臂动作 ===
                now = time.time()
                if now - last_act_time >= act_interval:
                    last_act_time = now
                    
                    logger.info("🤖 运行ACT推理...")
                    logger.info(f"   front_image: {current_front_image is not None}, wrist_image: {current_wrist_image is not None}")
                    act_result = act_inference.infer(
                        front_image=current_front_image,
                        wrist_image=current_wrist_image,
                        arm_state=ARM_DEFAULTS
                    )
                    
                    if act_result is not None:
                        act_action = act_result[0] if len(act_result.shape) > 1 else act_result
                        logger.info(f"✅ ACT原始推理角度: {act_action[:6]}")
                        
                        # 安全限制：机械臂关节限制在 ±45 度范围内（平衡安全与抓取能力）
                        SAFE_ANGLE_LIMIT = 45.0
                        if len(act_action) >= 6:
                            # 只限制前6个关节（不包括底盘速度）
                            clamped_angles = np.clip(act_action[:6], -SAFE_ANGLE_LIMIT, SAFE_ANGLE_LIMIT)
                            logger.info(f"🔒 安全限制后(±{SAFE_ANGLE_LIMIT}°): {clamped_angles}")
                            
                            cmd = {
                                "source": "act",
                                "arm_shoulder_pan.pos": float(clamped_angles[0]),
                                "arm_shoulder_lift.pos": float(clamped_angles[1]),
                                "arm_elbow_flex.pos": float(clamped_angles[2]),
                                "arm_wrist_flex.pos": float(clamped_angles[3]),
                                "arm_wrist_roll.pos": float(clamped_angles[4]),
                                "arm_gripper.pos": float(clamped_angles[5]),
                                "x.vel": 0.0,
                                "y.vel": 0.0,
                                "theta.vel": 0.0,
                            }
                            logger.info(f"📤 发送ACT命令: {cmd}")
                    else:
                        logger.warning("❌ ACT推理返回 None")
            else:
                # === 导航/手动阶段：发送手柄命令 ===
                if action.get("toggle_auto"):
                    cmd["toggle_auto"] = True
                    logger.info("🔄 发送切换自动模式命令")
                else:
                    cmd = {
                        "x.vel": action.get("x.vel", 0.0),
                        "y.vel": action.get("y.vel", 0.0),
                        "theta.vel": action.get("theta.vel", 0.0),
                    }
            
            # 发送命令（如果有）
            if cmd:
                try:
                    cmd_sock.send_string(json.dumps(cmd), flags=zmq.NOBLOCK)
                except:
                    pass
            
            # 5. 显示图像
            if args.display:
                if obs_data and front_b64:
                    save_path = None
                    if capture_req:
                        save_path = DATA_DIR / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
                        capture_req = False
                    
                    should_exit = show_frame(
                        front_b64,
                        wrist_b64=wrist_b64,
                        detections=detections,
                        auto_mode=auto_mode,
                        nav_state=nav_state,
                        grasp_progress=grasp_progress,
                        request_act=request_act,
                        save_path=save_path,
                        act_action=act_action if request_act else None
                    )
                    
                    if should_exit:
                        logger.info("ESC pressed, exiting...")
                        running = False
                elif obs_data and not front_b64:
                    logger.warning("收到观测数据但没有图像")
            else:
                # 不显示图像，只在控制台打印信息
                if obs_data:
                    if nav_state == "grasping":
                        logger.info(f"🦾 Grasping... Progress: {grasp_progress:.0%}, ACT: {act_action is not None}")
                    elif auto_mode:
                        logger.info(f"🤖 Auto mode: {nav_state}, detections: {len(detections)}")
            
            # 打印状态（限制频率）
            if nav_state == "grasping" and obs_data:
                logger.info(f"🦾 抓取中... 进度: {grasp_progress:.0%}, ACT: {act_action is not None}")
            
            # 7. 帧率控制
            elapsed = time.perf_counter() - loop_start
            sleep_t = max(1.0 / FPS - elapsed, 0)
            if sleep_t > 0:
                time.sleep(sleep_t)
    
    except KeyboardInterrupt:
        logger.info("\n用户中断")
    except Exception as e:
        logger.error(f"运行时错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("\n清理资源...")
        gamepad.disconnect()
        cmd_sock.close()
        obs_sock.close()
        ctx.term()
        if args.display:
            cv2.destroyAllWindows()
        logger.info("已退出")


if __name__ == "__main__":
    main()
