#!/usr/bin/env python
"""
电脑端 Client - 简化版
功能：
- 连接 LeKiwi Host（树莓派）
- Xbox 手柄控制底盘
- 键位：
  * D-pad 上/下/左/右：前后左右平移
  * RB + 左/右：原地旋转
  * LB：切换速度档
  * START：退出
- 不做主臂遥操作
"""

import sys
import time
import json
import base64
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import pygame
import zmq


# 配置
DEFAULT_IP = "192.168.3.176"
CMD_PORT = 5555
OBS_PORT = 5556
FPS = 30
DATA_DIR = Path(__file__).parent / "data"

# 图像尺寸（用于自动导航计算）
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_AREA = IMAGE_WIDTH * IMAGE_HEIGHT

# 自动导航配置
TARGET_AREA_RATIO = 0.20  # 目标占视野20%时到达
CENTER_THRESHOLD = 0.15   # 中心偏差阈值（图像宽度的15%）
NAV_SPEED = 0.3          # 自动导航速度
ROT_SPEED = 45           # 自动旋转速度


# 确保数据目录存在
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 速度档位
SPEED_LEVELS = [
    {"xy": 0.1, "theta": 30},   # Slow
    {"xy": 0.3, "theta": 60},   # Medium
    {"xy": 0.5, "theta": 90},   # Fast
]


class AutoNavigator:
    """自动导航控制器 - 根据YOLO检测结果自动寻路"""
    
    def __init__(self):
        self.image_center_x = IMAGE_WIDTH / 2
        self.image_center_y = IMAGE_HEIGHT / 2
        self.target_area = IMAGE_AREA * TARGET_AREA_RATIO
        
    def calculate_velocity(self, detections):
        """
        根据检测结果计算底盘速度
        
        Args:
            detections: 检测结果列表
            
        Returns:
            dict: {"x.vel": ..., "y.vel": ..., "theta.vel": ..., "arrived": ...}
        """
        if not detections:
            # 没有检测到纸团，原地旋转寻找
            return {
                "x.vel": 0.0,
                "y.vel": 0.0,
                "theta.vel": ROT_SPEED,
                "arrived": False,
                "state": "searching"
            }
        
        # 获取第一个纸团（置信度最高的）
        target = detections[0]
        x1, y1, x2, y2 = target["bbox"]
        cx, cy = target["center"]
        w, h = target["size"]
        
        # 计算纸团面积
        ball_area = w * h
        area_ratio = ball_area / IMAGE_AREA
        
        print(f"  [自动导航] 纸团面积占比: {area_ratio:.1%} (目标: {TARGET_AREA_RATIO:.0%})")
        
        # 检查是否到达目标（纸团占视野20%以上）
        if area_ratio >= TARGET_AREA_RATIO:
            return {
                "x.vel": 0.0,
                "y.vel": 0.0,
                "theta.vel": 0.0,
                "arrived": True,
                "state": "arrived"
            }
        
        # 计算纸团中心与图像中心的偏差
        dx = cx - self.image_center_x
        dy = cy - self.image_center_y
        
        # 归一化偏差（-1 到 1）
        nx = dx / (IMAGE_WIDTH / 2)
        ny = dy / (IMAGE_HEIGHT / 2)
        
        print(f"  [自动导航] 偏差: dx={dx:.1f}, dy={dy:.1f}, nx={nx:.2f}, ny={ny:.2f}")
        
        # 如果水平偏差大，先旋转对准
        if abs(nx) > CENTER_THRESHOLD:
            # 纸团在左边，需要逆时针旋转（正方向）
            # 纸团在右边，需要顺时针旋转（负方向）
            theta_cmd = ROT_SPEED * nx  # nx 为正（右边）时顺时针转
            return {
                "x.vel": 0.0,
                "y.vel": 0.0,
                "theta.vel": -theta_cmd,  # 反转：目标在右边时，底盘需要顺时针转（负）
                "arrived": False,
                "state": "aligning"
            }
        
        # 对准后前进
        # 距离越近速度越慢（根据面积比例）
        speed_factor = 1.0 - (area_ratio / TARGET_AREA_RATIO)
        x_cmd = NAV_SPEED * speed_factor
        
        return {
            "x.vel": x_cmd,
            "y.vel": 0.0,
            "theta.vel": 0.0,
            "arrived": False,
            "state": "approaching"
        }


class GamepadController:
    """Xbox 手柄控制器"""
    
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        
        self.joystick = None
        self.connected = False
        self.speed_index = 1  # 默认 Medium
        
        # 按钮编号（根据实际手柄调整）
        self.BTN_A = 0  # Xbox A 按钮（绿色），切换自动导航
        self.BTN_RB = 7
        self.BTN_LB = 6
        self.BTN_START = 11
        self.BTN_X = 3  # Xbox X 按钮（蓝色），用于拍照
        
        # 上一帧按钮状态（用于边沿检测）
        self.prev_states = {}
    
    def connect(self):
        """连接手柄"""
        retry = 0
        max_retry = 50
        
        while pygame.joystick.get_count() == 0 and retry < max_retry:
            time.sleep(0.1)
            pygame.joystick.init()
            retry += 1
        
        if pygame.joystick.get_count() == 0:
            print("❌ 未检测到手柄")
            return False
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.connected = True
        print(f"✅ 手柄已连接: {self.joystick.get_name()}")
        return True
    
    def get_action(self):
        """获取手柄动作"""
        if not self.connected:
            return {}
        
        pygame.event.pump()
        
        speed = SPEED_LEVELS[self.speed_index]
        xy_speed = speed["xy"]
        theta_speed = speed["theta"]
        
        x_cmd = 0.0
        y_cmd = 0.0
        theta_cmd = 0.0
        
        # 读取 D-pad
        if self.joystick.get_numhats() > 0:
            hat = self.joystick.get_hat(0)
            hat_x, hat_y = hat
            rb_pressed = self.joystick.get_button(self.BTN_RB)
            
            if rb_pressed and hat_x < 0:
                # RB + 左右 = 旋转
                theta_cmd = theta_speed
            elif  rb_pressed and hat_x > 0:
                    theta_cmd = -theta_speed
            else:
                # 平移
                if hat_y > 0:
                    x_cmd = xy_speed
                elif hat_y < 0:
                    x_cmd = -xy_speed
                
                if hat_x < 0:
                    y_cmd = xy_speed
                elif hat_x > 0:
                    y_cmd = -xy_speed
        
        # LB 切换速度（边沿检测）
        lb_current = self.joystick.get_button(self.BTN_LB)
        lb_prev = self.prev_states.get("LB", False)
        if lb_current and not lb_prev:
            self.speed_index = (self.speed_index + 1) % len(SPEED_LEVELS)
            names = ["Slow", "Medium", "Fast"]
            print(f"  速度: {names[self.speed_index]} (xy={SPEED_LEVELS[self.speed_index]['xy']})")
        self.prev_states["LB"] = lb_current
        
        # A 按钮切换自动导航（边沿检测）
        a_current = self.joystick.get_button(self.BTN_A)
        a_prev = self.prev_states.get("A", False)
        toggle_auto = a_current and not a_prev
        if toggle_auto:
            print("  🔄 切换自动导航模式")
        self.prev_states["A"] = a_current
        
        # X 按钮拍照（边沿检测）
        x_current = self.joystick.get_button(self.BTN_X)
        x_prev = self.prev_states.get("X", False)
        capture_image = x_current and not x_prev
        if capture_image:
            print("  📷 拍照命令已发送")
        self.prev_states["X"] = x_current
        
        return {
            "x.vel": x_cmd,
            "y.vel": y_cmd,
            "theta.vel": theta_cmd,
            "capture_image": capture_image,
            "toggle_auto": toggle_auto,
        }
    
    def check_exit(self):
        """检查是否退出"""
        if not self.connected:
            return False
        return self.joystick.get_button(self.BTN_START)
    
    def disconnect(self):
        """断开手柄"""
        if self.joystick:
            self.joystick.quit()
        pygame.quit()
        print("手柄已断开")


def display_frame(frame_b64, window_name="Camera", save_path=None, detections=None):
    """显示图像（RGB转换显示，绘制检测框，原始BGR保存）"""
    if not frame_b64:
        return None
    
    try:
        frame_data = base64.b64decode(frame_b64)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            # 绘制检测结果
            if detections:
                for i, det in enumerate(detections):
                    x1, y1, x2, y2 = det["bbox"]
                    cx, cy = det["center"]
                    conf = det["confidence"]
                    cls_name = det["class"]
                    
                    # 绘制矩形框（绿色）
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # 绘制中心点（红色）
                    cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)
                    
                    # 绘制标签
                    label = f"{cls_name}: {conf:.2%}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 显示时转换为 RGB（颜色正确）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow(window_name, frame_rgb)
            cv2.waitKey(1)
            
            # 保存原始帧（BGR格式，不做转换）
            if save_path:
                cv2.imwrite(str(save_path), frame)
                print(f"  💾 图像已保存(BGR): {save_path}")
            
            return frame
    except Exception as e:
        print(f"  图像处理错误: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description="LeKiwi PC Client - 底盘遥操作")
    parser.add_argument("--ip", default=DEFAULT_IP, help="树莓派 IP")
    parser.add_argument("--display", action="store_true", default=True, help="显示摄像头画面")
    args = parser.parse_args()
    
    print("=" * 60)
    print("LeKiwi Client - 底盘遥操作")
    print("=" * 60)
    print(f"目标: {args.ip}:{CMD_PORT}/{OBS_PORT}")
    print("=" * 60)
    
    # 连接手柄
    print("\n[1/3] 连接手柄...")
    gamepad = GamepadController()
    if not gamepad.connect():
        print("错误：未检测到手柄")
        return
    
    # 连接 ZMQ
    print("\n[2/3] 连接树莓派...")
    context = zmq.Context()
    
    cmd_socket = context.socket(zmq.PUSH)
    cmd_socket.connect(f"tcp://{args.ip}:{CMD_PORT}")
    
    obs_socket = context.socket(zmq.PULL)
    obs_socket.setsockopt(zmq.CONFLATE, 1)
    obs_socket.connect(f"tcp://{args.ip}:{OBS_PORT}")
    
    print("✅ 已连接")
    
    # 显示帮助
    print("\n" + "=" * 60)
    print("控制说明:")
    print("  D-pad 上/下     : 前进/后退")
    print("  D-pad 左/右     : 左平移/右平移")
    print("  RB + 左/右      : 原地旋转")
    print("  LB              : 切换速度档")
    print("  A               : 切换自动导航/手动控制")
    print("  X               : 拍照保存到 data 目录")
    print("  START           : 退出")
    print("=" * 60)
    print(f"\n📁 图像保存目录: {DATA_DIR.absolute()}")
    
    # 主循环
    print("\n[3/3] 开始遥操作...")
    print("当前模式: 🎮 手动控制")
    running = True
    frame_count = 0
    capture_requested = False
    auto_mode = False  # 自动导航模式标志
    navigator = AutoNavigator()  # 自动导航控制器
    current_detections = []  # 当前检测结果
    
    try:
        while running:
            t0 = time.perf_counter()
            
            # 1. 获取手柄动作
            action = gamepad.get_action()
            
            # 检测模式切换
            if action.get("toggle_auto", False):
                auto_mode = not auto_mode
                if auto_mode:
                    print("\n🤖 切换到自动导航模式")
                    print("  将根据YOLO检测结果自动寻路")
                else:
                    print("\n🎮 切换到手动控制模式")
            
            # 检测拍照请求
            if action.get("capture_image", False):
                capture_requested = True
                print("  📷 拍照请求已记录，下一帧将保存")
            
            # 2. 根据模式生成命令
            if auto_mode:
                # 自动导航模式：根据检测结果计算速度
                nav_cmd = navigator.calculate_velocity(current_detections)
                
                if nav_cmd["arrived"]:
                    print("\n✅ 已到达目标！持续搜索中...")
                    print("  等待新目标或继续跟踪当前目标")
                    # 保持自动导航模式，继续搜索（不退出）
                    cmd = {
                        "x.vel": 0.0,
                        "y.vel": 0.0,
                        "theta.vel": 0.0,
                        "auto_state": "arrived_waiting"
                    }
                else:
                    cmd = {
                        "x.vel": nav_cmd["x.vel"],
                        "y.vel": nav_cmd["y.vel"],
                        "theta.vel": nav_cmd["theta.vel"],
                        "auto_state": nav_cmd["state"]
                    }
                    print(f"  [自动导航] 状态: {nav_cmd['state']}, "
                          f"速度: x={nav_cmd['x.vel']:.2f}, theta={nav_cmd['theta.vel']:.1f}")
            else:
                # 手动控制模式：使用手柄输入
                cmd = {
                    "x.vel": action["x.vel"],
                    "y.vel": action["y.vel"],
                    "theta.vel": action["theta.vel"],
                }
                # 只在有运动时打印
                if any(v != 0 for v in [action["x.vel"], action["y.vel"], action["theta.vel"]]):
                    print(f"发送: {cmd}")
            
            cmd_socket.send_string(json.dumps(cmd), flags=zmq.NOBLOCK)
            
            # 3. 接收图像
            if args.display:
                try:
                    msg = obs_socket.recv_string(zmq.NOBLOCK)
                    obs = json.loads(msg)
                    if "front" in obs:
                        # 准备保存路径
                        save_path = None
                        if capture_requested:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            save_path = DATA_DIR / f"capture_{timestamp}.jpg"
                            capture_requested = False
                        
                        # 获取检测结果
                        current_detections = obs.get("detections", [])
                        
                        # 打印检测信息（仅在手动模式或检测到时）
                        if not auto_mode and current_detections:
                            print(f"🎯 检测到 {len(current_detections)} 个目标:")
                            for i, det in enumerate(current_detections):
                                x1, y1, x2, y2 = det["bbox"]
                                cx, cy = det["center"]
                                print(f"  [{i+1}] 类别: {det['class']}, "
                                      f"置信度: {det['confidence']:.2%}, "
                                      f"位置: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f}), "
                                      f"中心: ({cx:.1f}, {cy:.1f})")
                        
                        # 在图像上添加模式指示
                        frame_data = base64.b64decode(obs["front"])
                        nparr = np.frombuffer(frame_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            # 添加模式文字
                            mode_text = "AUTO NAV" if auto_mode else "MANUAL"
                            color = (0, 255, 255) if auto_mode else (255, 255, 255)
                            cv2.putText(frame, mode_text, (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            
                            # 添加到达指示
                            if auto_mode and current_detections:
                                target = current_detections[0]
                                area_ratio = (target["size"][0] * target["size"][1]) / IMAGE_AREA
                                progress_text = f"Progress: {area_ratio/TARGET_AREA_RATIO:.0%}"
                                cv2.putText(frame, progress_text, (10, 70),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                            # 重新编码
                            ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                            if ret:
                                obs["front"] = base64.b64encode(buffer).decode("utf-8")
                        
                        display_frame(obs["front"], "Front Camera", save_path, current_detections)
                except zmq.Again:
                    pass
            
            # 4. 检查退出
            if gamepad.check_exit():
                print("\nSTART 按下，退出...")
                running = False
            
            # 5. 帧率控制
            elapsed = time.perf_counter() - t0
            sleep_time = max(1.0 / FPS - elapsed, 0)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        print("\n清理中...")
        gamepad.disconnect()
        cmd_socket.close()
        obs_socket.close()
        context.term()
        if args.display:
            cv2.destroyAllWindows()
        print("已退出")


if __name__ == "__main__":
    main()
