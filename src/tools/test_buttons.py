#!/usr/bin/env python
"""
手柄按钮诊断工具
运行后按任意按钮查看编号
按 Ctrl+C 退出
"""

import pygame
import sys

def test_buttons():
    pygame.init()
    pygame.joystick.init()
    
    # 等待手柄连接
    print("等待手柄连接...")
    while pygame.joystick.get_count() == 0:
        pygame.time.wait(100)
        pygame.joystick.init()
    
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    
    print(f"\n手柄名称: {joystick.get_name()}")
    print(f"按钮总数: {joystick.get_numbuttons()}")
    print(f"摇杆总数: {joystick.get_numaxes()}")
    print(f"方向键总数: {joystick.get_numhats()}")
    print("\n" + "="*50)
    print("请依次按下以下按钮，查看对应编号：")
    print("  RB (右侧肩键)")
    print("  LB (左侧肩键)")
    print("  START (开始键)")
    print("  A/B/X/Y")
    print("="*50)
    print("\n按 Ctrl+C 退出\n")
    
    last_pressed = set()
    
    try:
        while True:
            pygame.event.pump()
            
            # 检测按钮按下
            for i in range(joystick.get_numbuttons()):
                if joystick.get_button(i):
                    if i not in last_pressed:
                        print(f"Button {i} pressed")
                        last_pressed.add(i)
                else:
                    if i in last_pressed:
                        last_pressed.remove(i)
            
            # 检测方向键
            if joystick.get_numhats() > 0:
                hat = joystick.get_hat(0)
                if hat != (0, 0):
                    print(f"D-pad: {hat}")
            
            pygame.time.wait(50)  # 20Hz 检测
            
    except KeyboardInterrupt:
        print("\n诊断结束")
    finally:
        pygame.quit()

if __name__ == "__main__":
    test_buttons()
