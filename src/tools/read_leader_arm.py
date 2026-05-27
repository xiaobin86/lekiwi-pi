#!/usr/bin/env python
"""
读取 SO101 主臂当前舵机位置

Usage:
    python test/read_leader_arm.py --port COM5
    python test/read_leader_arm.py --port COM5 --interval 0.5
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path.home() / "lerobot-workspace/lerobot/src"))

from lerobot.teleoperators.so_leader import SO101Leader, SOLeaderTeleopConfig


def read_arm(port: str, arm_id: str = "L07252802", interval: float = 1.0):
    """读取并显示主臂舵机位置"""
    
    print("=" * 60)
    print("SO101 主臂舵机读取工具")
    print("=" * 60)
    print(f"串口: {port}")
    print(f"ID: {arm_id}")
    print(f"刷新间隔: {interval}秒")
    print("=" * 60)
    
    # 配置
    config = SOLeaderTeleopConfig(port=port, id=arm_id)
    
    # 连接
    print("\n连接主臂...")
    arm = SO101Leader(config)
    arm.connect()
    
    if not arm.is_connected:
        print("❌ 连接失败")
        return
    
    print("✅ 已连接")
    print("\n按 Ctrl+C 停止\n")
    
    # 舵机名称映射
    joint_names = {
        1: "shoulder_pan",
        2: "shoulder_lift", 
        3: "elbow_flex",
        4: "wrist_flex",
        5: "wrist_roll",
        6: "gripper",
    }
    
    try:
        while True:
            # 读取当前位置
            action = arm.get_action()
            
            print(f"\r", end="")
            print(f"{time.strftime('%H:%M:%S')} | ", end="")
            
            for i, (key, value) in enumerate(action.items(), 1):
                name = joint_names.get(i, key)
                print(f"{name}={value:7.2f}°  ", end="")
            
            print("", end="", flush=True)
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n用户停止")
    finally:
        print("断开连接...")
        arm.disconnect()
        print("已断开")


def main():
    parser = argparse.ArgumentParser(description="读取 SO101 主臂舵机位置")
    parser.add_argument("--port", default="COM5", help="串口 (默认: COM5)")
    parser.add_argument("--id", default="L07252802", help="主臂 ID (默认: L07252802)")
    parser.add_argument("--interval", type=float, default=1.0, help="刷新间隔秒数 (默认: 1.0)")
    args = parser.parse_args()
    
    read_arm(args.port, args.id, args.interval)


if __name__ == "__main__":
    main()
