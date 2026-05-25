#!/usr/bin/env python
"""
Camera Diagnostic Tool for Raspberry Pi

检查摄像头支持的参数，帮助排查 OpenCVCamera 配置错误。
"""

import cv2
import sys

def check_camera(device_path="/dev/video2"):
    """检查摄像头支持的参数"""
    print(f"=" * 60)
    print(f"Camera Diagnostic: {device_path}")
    print(f"=" * 60)
    
    cap = cv2.VideoCapture(device_path)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头: {device_path}")
        return
    
    print(f"✅ 摄像头已打开")
    
    # 获取默认参数
    default_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    default_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    default_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n默认参数:")
    print(f"  宽度: {default_width}")
    print(f"  高度: {default_height}")
    print(f"  FPS: {default_fps}")
    
    # 测试常见分辨率
    test_resolutions = [
        (640, 480),
        (1280, 720),
        (1920, 1080),
        (320, 240),
        (800, 600),
    ]
    
    print(f"\n测试分辨率设置:")
    for width, height in test_resolutions:
        # 重新打开摄像头以测试不同分辨率
        cap.release()
        cap = cv2.VideoCapture(device_path)
        
        width_success = cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        height_success = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        status = "✅" if (actual_width == width and actual_height == height) else "⚠️"
        print(f"  {status} {width}x{height} -> 实际: {actual_width}x{actual_height} "
              f"(set成功: {width_success and height_success})")
    
    # 测试读取帧
    print(f"\n测试读取帧:")
    ret, frame = cap.read()
    if ret:
        print(f"  ✅ 成功读取帧: {frame.shape}")
    else:
        print(f"  ❌ 无法读取帧")
    
    cap.release()
    
    print(f"\n" + "=" * 60)
    print(f"建议:")
    print(f"  使用摄像头实际支持的分辨率")
    print(f"  或修改 lekiwi_host_yolo.py 中的 width/height 参数")
    print(f"=" * 60)


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "/dev/video2"
    check_camera(device)
