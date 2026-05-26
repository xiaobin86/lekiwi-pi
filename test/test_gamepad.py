#!/usr/bin/env python
"""
手柄按键测试工具
功能：检测并显示所有手柄按钮、轴、方向键的状态，帮助确定正确的按键编号
"""

import sys
import time
import pygame


def test_gamepad():
    """测试手柄所有输入"""
    print("=" * 60)
    print("手柄按键测试工具")
    print("=" * 60)
    
    pygame.init()
    pygame.joystick.init()
    
    # 等待手柄连接
    print("\n等待手柄连接...")
    retry = 0
    while pygame.joystick.get_count() == 0:
        time.sleep(0.5)
        pygame.joystick.init()
        retry += 1
        if retry % 4 == 0:
            print(f"  等待中... ({retry//2}s)")
    
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    
    print(f"\n✅ 手柄已连接: {joystick.get_name()}")
    print(f"   按钮数量: {joystick.get_numbuttons()}")
    print(f"   轴数量: {joystick.get_numaxes()}")
    print(f"   方向键数量: {joystick.get_numhats()}")
    
    print("\n" + "=" * 60)
    print("开始测试 - 按任意按钮查看编号")
    print("按 Ctrl+C 退出")
    print("=" * 60 + "\n")
    
    # 记录上一帧状态
    prev_buttons = [False] * joystick.get_numbuttons()
    prev_hat = (0, 0)
    
    try:
        while True:
            pygame.event.pump()
            
            # 检测按钮按下
            for i in range(joystick.get_numbuttons()):
                current = joystick.get_button(i)
                if current and not prev_buttons[i]:
                    print(f"[按钮] 编号 {i:2d} 被按下")
                prev_buttons[i] = current
            
            # 检测方向键变化
            if joystick.get_numhats() > 0:
                hat = joystick.get_hat(0)
                if hat != prev_hat:
                    if hat != (0, 0):
                        print(f"[方向键] 方向: {hat}")
                    prev_hat = hat
            
            # 检测轴变化（阈值0.5）
            for i in range(joystick.get_numaxes()):
                axis_val = joystick.get_axis(i)
                if abs(axis_val) > 0.5:
                    print(f"[摇杆/扳机] 轴 {i}: {axis_val:.3f}")
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\n测试结束")
    finally:
        pygame.quit()


if __name__ == "__main__":
    test_gamepad()
