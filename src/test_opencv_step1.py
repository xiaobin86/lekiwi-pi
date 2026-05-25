#!/usr/bin/env python
"""
OpenCV 摄像头兼容性诊断工具
Step 1: 基础诊断 - 测试设备能否打开和读取帧
"""

import cv2
import sys

def test_basic_capture(device_path):
    """基础测试：打开摄像头并读取一帧"""
    print(f"\n{'='*60}")
    print(f"测试设备: {device_path}")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(device_path)
    print(f"1. 打开设备: {'成功' if cap.isOpened() else '失败'}")
    
    if not cap.isOpened():
        return False
    
    # 获取默认参数
    default_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    default_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    default_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"2. 默认参数: {default_w}x{default_h} @ {default_fps}fps")
    
    # 测试读取帧
    ret, frame = cap.read()
    print(f"3. 读取帧: {'成功' if ret else '失败'}", end="")
    if ret:
        print(f" (shape: {frame.shape})")
    else:
        print()
    
    cap.release()
    return ret

def test_set_properties(device_path):
    """测试设置参数"""
    print(f"\n{'='*60}")
    print(f"测试参数设置: {device_path}")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(device_path)
    if not cap.isOpened():
        print("无法打开")
        return
    
    # 测试设置 640x480
    w_ok = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    h_ok = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fps_ok = cap.set(cv2.CAP_PROP_FPS, 30)
    
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"设置 640x480 @ 30fps:")
    print(f"  Width set():  {w_ok} -> actual: {actual_w} {'OK' if actual_w==640 else 'FAIL'}")
    print(f"  Height set(): {h_ok} -> actual: {actual_h} {'OK' if actual_h==480 else 'FAIL'}")
    print(f"  FPS set():    {fps_ok} -> actual: {actual_fps} {'OK' if actual_fps==30 else 'FAIL'}")
    
    # 读取帧测试
    ret, frame = cap.read()
    print(f"  读取帧: {'成功' if ret else '失败'} {frame.shape if ret else ''}")
    
    cap.release()

if __name__ == "__main__":
    # 获取 OpenCV 版本
    print(f"OpenCV 版本: {cv2.__version__}")
    
    # 测试 /dev/video0
    test_basic_capture("/dev/video0")
    test_set_properties("/dev/video0")
    
    print("\n" + "="*60)
    print("Step 1 诊断完成")
    print("="*60)
