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

import numpy as np
import cv2
import pygame
import zmq


# 配置
DEFAULT_IP = "192.168.3.176"
CMD_PORT = 5555
OBS_PORT = 5556
FPS = 30

# 速度档位
SPEED_LEVELS = [
    {"xy": 0.1, "theta": 30},   # Slow
    {"xy": 0.3, "theta": 60},   # Medium
    {"xy": 0.5, "theta": 90},   # Fast
]


class GamepadController:
    """Xbox 手柄控制器"""
    
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        
        self.joystick = None
        self.connected = False
        self.speed_index = 1  # 默认 Medium
        
        # 按钮编号（根据实际手柄调整）
        self.BTN_RB = 7
        self.BTN_LB = 6
        self.BTN_START = 11
        
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
        theta_speed = np.radians(speed["theta"])
        
        x_cmd = 0.0
        y_cmd = 0.0
        theta_cmd = 0.0
        
        # 读取 D-pad
        if self.joystick.get_numhats() > 0:
            hat = self.joystick.get_hat(0)
            hat_x, hat_y = hat
            rb_pressed = self.joystick.get_button(self.BTN_RB)
            
            if rb_pressed:
                # RB + 左右 = 旋转
                if hat_x < 0:
                    theta_cmd = theta_speed
                elif hat_x > 0:
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
        
        return {
            "x.vel": x_cmd,
            "y.vel": y_cmd,
            "theta.vel": theta_cmd,
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


def display_frame(frame_b64, window_name="Camera"):
    """显示图像"""
    if not frame_b64:
        return
    
    try:
        frame_data = base64.b64decode(frame_b64)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)
    except Exception as e:
        pass


def main():
    parser = argparse.ArgumentParser(description="LeKiwi PC Client - 底盘遥操作")
    parser.add_argument("--ip", default=DEFAULT_IP, help="树莓派 IP")
    parser.add_argument("--display", action="store_true", help="显示摄像头画面")
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
    print("  START           : 退出")
    print("=" * 60)
    
    # 主循环
    print("\n[3/3] 开始遥操作...")
    running = True
    frame_count = 0
    
    try:
        while running:
            t0 = time.perf_counter()
            
            # 1. 获取手柄动作
            action = gamepad.get_action()
            
            # 2. 发送命令
            cmd_socket.send_string(json.dumps(action), flags=zmq.NOBLOCK)
            
            # 3. 接收图像
            if args.display:
                try:
                    msg = obs_socket.recv_string(zmq.NOBLOCK)
                    obs = json.loads(msg)
                    if "front" in obs:
                        display_frame(obs["front"], "Front Camera")
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
