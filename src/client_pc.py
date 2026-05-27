#!/usr/bin/env python
"""电脑端 Client - 手柄控制底盘 + 显示图像"""

import sys, time, json, base64, argparse
from pathlib import Path
from datetime import datetime

import numpy as np, cv2, pygame, zmq


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


# ==================== 手柄控制器 ====================
class Gamepad:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        self.connected = False
        self.speed_idx = 1
        self.prev = {}
        
        # 按钮映射
        self.BTN_A = 0
        self.BTN_RB = 7
        self.BTN_LB = 6
        self.BTN_START = 11
        self.BTN_X = 3
    
    def connect(self):
        retry = 0
        while pygame.joystick.get_count() == 0 and retry < 50:
            time.sleep(0.1)
            pygame.joystick.init()
            retry += 1
        
        if pygame.joystick.get_count() == 0:
            print("❌ 未检测到手柄")
            return False
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.connected = True
        print(f"✅ 手柄: {self.joystick.get_name()}")
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
        
        # LB 切换速度
        lb = self.joystick.get_button(self.BTN_LB)
        if lb and not self.prev.get("LB", False):
            self.speed_idx = (self.speed_idx + 1) % len(SPEED_LEVELS)
            names = ["Slow", "Medium", "Fast"]
            print(f"  速度: {names[self.speed_idx]}")
        self.prev["LB"] = lb
        
        # A 切换自动导航
        a = self.joystick.get_button(self.BTN_A)
        toggle_auto = a and not self.prev.get("A", False)
        if toggle_auto:
            print("  🔄 切换自动导航")
        self.prev["A"] = a
        
        # X 拍照
        x = self.joystick.get_button(self.BTN_X)
        capture = x and not self.prev.get("X", False)
        if capture:
            print("  📷 拍照")
        self.prev["X"] = x
        
        return {
            "x.vel": x_cmd, "y.vel": y_cmd, "theta.vel": theta_cmd,
            "capture_image": capture, "toggle_auto": toggle_auto,
        }
    
    def check_exit(self):
        return self.connected and self.joystick.get_button(self.BTN_START)
    
    def disconnect(self):
        if self.joystick:
            self.joystick.quit()
        pygame.quit()
        print("手柄已断开")


# ==================== 显示图像 ====================
def show_frame(frame_b64, detections=None, auto_mode=False, nav_state="idle", 
               grasp_progress=0.0, request_act=False, save_path=None):
    if not frame_b64:
        return
    
    try:
        data = base64.b64decode(frame_b64)
        arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        
        # 绘制检测框
        if detections:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cx, cy = det["center"]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)
                label = f"{det['class']}: {det['confidence']:.2%}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制状态信息
        mode_text = "AUTO" if auto_mode else "MANUAL"
        mode_color = (0, 165, 255) if auto_mode else (0, 255, 0)  # Orange for AUTO, Green for MANUAL
        
        # 背景条
        info_h = 110 if nav_state == "grasping" else 90
        cv2.rectangle(frame, (10, 10), (280, info_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (280, info_h), (255, 255, 255), 1)
        
        # 模式状态
        cv2.putText(frame, f"Mode: {mode_text}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # 导航状态
        state_color = (255, 255, 255)
        if nav_state == "searching":
            state_color = (255, 255, 0)  # Cyan
        elif nav_state == "aligning":
            state_color = (255, 165, 0)  # Orange
        elif nav_state == "approaching":
            state_color = (0, 255, 0)    # Green
        elif nav_state == "arrived":
            state_color = (0, 255, 0)    # Green
        elif nav_state == "grasping":
            state_color = (255, 0, 255)  # Magenta
        
        state_text = nav_state.upper() if auto_mode else "MANUAL CONTROL"
        cv2.putText(frame, f"State: {state_text}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        
        # 检测信息
        det_count = len(detections) if detections else 0
        det_text = f"Objects: {det_count}"
        cv2.putText(frame, det_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # ACT抓取进度
        if nav_state == "grasping":
            progress_text = f"Grasp: {grasp_progress:.0%}"
            cv2.putText(frame, progress_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # 绘制进度条
            bar_x, bar_y, bar_w, bar_h = 10, frame.shape[0] - 30, 200, 20
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            fill_w = int(bar_w * grasp_progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (255, 0, 255), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
        
        # 显示 (OpenCV使用BGR格式)
        cv2.imshow("Front Camera", frame)
        cv2.waitKey(1)
        
        # 保存 (转为RGB保存)
        if save_path:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(save_path), rgb_frame)
            print(f"  💾 已保存: {save_path}")
    
    except Exception as e:
        print(f"  图像错误: {e}")


# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=DEFAULT_IP)
    parser.add_argument("--display", action="store_true", default=True)
    args = parser.parse_args()
    
    print("=" * 60)
    print("LeKiwi Client")
    print(f"目标: {args.ip}:{CMD_PORT}/{OBS_PORT}")
    print("=" * 60)
    
    # 手柄
    print("\n[1/3] 连接手柄...")
    gamepad = Gamepad()
    if not gamepad.connect():
        print("错误：无手柄")
        return
    
    # ZMQ
    print("\n[2/3] 连接树莓派...")
    ctx = zmq.Context()
    cmd_sock = ctx.socket(zmq.PUSH)
    cmd_sock.connect(f"tcp://{args.ip}:{CMD_PORT}")
    
    obs_sock = ctx.socket(zmq.PULL)
    obs_sock.setsockopt(zmq.CONFLATE, 1)
    obs_sock.connect(f"tcp://{args.ip}:{OBS_PORT}")
    
    print("✅ 已连接")
    
    # 帮助
    print("\n" + "=" * 60)
    print("控制:")
    print("  D-pad: 移动 | RB+左右: 旋转")
    print("  LB: 速度 | A: 自动导航 | X: 拍照 | START: 退出")
    print("=" * 60)
    print(f"\n📁 保存目录: {DATA_DIR}")
    
    # 主循环
    print("\n[3/3] 运行中...")
    running = True
    capture_req = False
    
    try:
        while running:
            t0 = time.perf_counter()
            
            # 1. 手柄
            action = gamepad.get_action()
            
            if action.get("capture_image"):
                capture_req = True
                print("  📷 记录拍照")
            
            # 2. 发送命令
            cmd = {}
            if action.get("toggle_auto"):
                cmd["toggle_auto"] = True
            elif not action.get("toggle_auto"):
                cmd = {
                    "x.vel": action["x.vel"],
                    "y.vel": action["y.vel"],
                    "theta.vel": action["theta.vel"],
                }
                if any(v != 0 for v in [action["x.vel"], action["y.vel"], action["theta.vel"]]):
                    print(f"发送: {cmd}")
            
            cmd_sock.send_string(json.dumps(cmd), flags=zmq.NOBLOCK)
            
            # 3. 接收图像
            if args.display:
                try:
                    msg = obs_sock.recv_string(zmq.NOBLOCK)
                    obs = json.loads(msg)
                    
                    if "front" in obs:
                        save_path = None
                        if capture_req:
                            save_path = DATA_DIR / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
                            capture_req = False
                        
                        dets = obs.get("detections", [])
                        auto_mode = obs.get("auto_mode", False)
                        nav_state = obs.get("nav_state", "idle")
                        grasp_progress = obs.get("grasp_progress", 0.0)
                        request_act = obs.get("request_act", False)
                        
                        # 打印检测信息
                        if dets:
                            print(f"🎯 {len(dets)} 个目标:")
                            for i, det in enumerate(dets):
                                print(f"  [{i+1}] {det['class']} {det['confidence']:.1%}")
                        
                        # 打印ACT状态
                        if nav_state == "grasping":
                            print(f"🦾 ACT抓取中... 进度: {grasp_progress:.0%}, 请求动作: {request_act}")
                        
                        show_frame(obs["front"], dets, auto_mode, nav_state, 
                                 grasp_progress, request_act, save_path)
                except zmq.Again:
                    pass
            
            # 4. 退出检查
            if gamepad.check_exit():
                print("\n退出...")
                running = False
            
            # 5. 帧率控制
            elapsed = time.perf_counter() - t0
            sleep_t = max(1.0 / FPS - elapsed, 0)
            if sleep_t > 0:
                time.sleep(sleep_t)
    
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        print("\n清理...")
        gamepad.disconnect()
        cmd_sock.close()
        obs_sock.close()
        ctx.term()
        if args.display:
            cv2.destroyAllWindows()
        print("已退出")


if __name__ == "__main__":
    main()
