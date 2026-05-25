#!/usr/bin/env python
"""
LekiWi Gamepad Teleoperation

PC connects to Raspberry Pi running kiwiclient via network.
Supports Xbox controller for base movement and SO101 leader arm for teleoperation.

Usage:
    python src/teleop_gamepad.py --remote-ip 192.168.31.165

Controls:
    D-pad Up/Down       → Forward / Backward
    D-pad Left/Right    → Strafe Left / Right
    RB + Left/Right     → Rotate Left / Right (in place)
    LB                  → Cycle speed (Slow → Medium → Fast)
    START               → Exit

Requirements:
    - LeRobot installed
    - pygame installed
    - Xbox controller connected (USB or Bluetooth)
    - Raspberry Pi running: python -m lerobot.robots.lekiwi.lekiwi_host
"""

import time
import argparse
from pathlib import Path
import sys

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent))

from gamepad_teleop import GamepadTeleop, GamepadTeleopConfig

# LeRobot imports
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep


FPS = 30


def main():
    parser = argparse.ArgumentParser(
        description="LekiWi Gamepad Teleoperation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Connect to Raspberry Pi at default IP
    python src/teleop_gamepad.py --remote-ip 192.168.31.165
    
    # Connect with custom arm port
    python src/teleop_gamepad.py --remote-ip 192.168.31.165 --arm-port COM5
    
    # Connect to local robot (debug)
    python src/teleop_gamepad.py --remote-ip 127.0.0.1
        """
    )
    parser.add_argument(
        "--remote-ip",
        type=str,
        required=True,
        help="Raspberry Pi IP address running kiwiclient (e.g., 192.168.31.165)",
    )
    parser.add_argument(
        "--arm-port",
        type=str,
        default="COM5",
        help="Serial port for SO101 leader arm (default: COM5)",
    )
    parser.add_argument(
        "--arm-id",
        type=str,
        default="L07252802",
        help="SO101 leader arm ID (default: L07252802)",
    )
    parser.add_argument(
        "--robot-id",
        type=str,
        default="my_lekiwi",
        help="Robot ID (default: my_lekiwi)",
    )
    parser.add_argument(
        "--zmq-cmd-port",
        type=int,
        default=5555,
        help="ZMQ command port (default: 5555)",
    )
    parser.add_argument(
        "--zmq-obs-port",
        type=int,
        default=5556,
        help="ZMQ observation port (default: 5556)",
    )
    parser.add_argument(
        "--no-arm",
        action="store_true",
        help="Disable leader arm teleoperation (base only)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Control loop frequency (default: 30)",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  LekiWi Gamepad Teleoperation")
    print("=" * 70)
    print(f"\n  Remote Pi:    {args.remote_ip}")
    print(f"  ZMQ Ports:    {args.zmq_cmd_port} / {args.zmq_obs_port}")
    print(f"  Leader Arm:   {args.arm_port} (ID: {args.arm_id})")
    print(f"  FPS:          {args.fps}")
    print()
    
    # =========================================================================
    # 1. Initialize configurations
    # =========================================================================
    print("[1/4] Initializing configurations...")
    
    # Robot client config (connects to Raspberry Pi)
    robot_config = LeKiwiClientConfig(
        remote_ip=args.remote_ip,
        port_zmq_cmd=args.zmq_cmd_port,
        port_zmq_observations=args.zmq_obs_port,
        id=args.robot_id,
    )
    
    # Leader arm config
    if not args.no_arm:
        arm_config = SO101LeaderConfig(
            port=args.arm_port,
            id=args.arm_id,
        )
    
    # Gamepad config
    gamepad_config = GamepadTeleopConfig()
    
    # =========================================================================
    # 2. Initialize hardware
    # =========================================================================
    print("[2/4] Connecting to hardware...")
    
    # Robot client
    print("  → Connecting to LekiWi client...")
    robot = LeKiwiClient(robot_config)
    robot.connect()
    if not robot.is_connected:
        raise RuntimeError(
            f"Failed to connect to robot at {args.remote_ip}.\n"
            f"  Make sure kiwiclient is running on the Pi:\n"
            f"  python -m lerobot.robots.lekiwi.lekiwi_host --robot.id={args.robot_id}"
        )
    print("    ✓ Robot connected")
    
    # Leader arm
    leader_arm = None
    if not args.no_arm:
        print("  → Connecting to SO101 leader arm...")
        leader_arm = SO101Leader(arm_config)
        leader_arm.connect()
        if not leader_arm.is_connected:
            print("    ⚠ Leader arm not connected (continuing without arm)")
            leader_arm = None
        else:
            print("    ✓ Leader arm connected")
    
    # Gamepad
    print("  → Connecting to Xbox controller...")
    gamepad = GamepadTeleop(gamepad_config)
    if not gamepad.connect():
        raise RuntimeError("Failed to connect Xbox controller")
    print("    ✓ Gamepad connected")
    
    # =========================================================================
    # 3. Print control reference
    # =========================================================================
    print("\n" + "=" * 70)
    print("  Control Reference")
    print("=" * 70)
    print("""
  Base Movement (D-pad):
    ┌─────────┬─────────┬─────────┐
    │  D-Up   │         │         │  → Forward
    │  (↑y)   │         │         │
    ├─────────┼─────────┼─────────┤
    │ D-Left  │ D-Down  │ D-Right │  → Backward / Strafe
    │ (←x)    │  (↓y)   │  (→x)   │
    └─────────┴─────────┴─────────┘
  
  Rotation (RB + D-pad):
    RB + D-Left   → Rotate Left (in place)
    RB + D-Right  → Rotate Right (in place)
  
  Speed Control:
    LB            → Cycle speed (Slow → Medium → Fast)
  
  Other:
    START         → Exit program
    """)
    print("=" * 70)
    
    # =========================================================================
    # 4. Main control loop
    # =========================================================================
    print("\n[3/4] Starting teleop loop...")
    print("   Press START on controller to exit\n")
    
    running = True
    frame_count = 0
    
    try:
        while running:
            t0 = time.perf_counter()
            
            # ---- Get observations ----
            observation = robot.get_observation()
            
            # ---- Get actions ----
            # 1. Leader arm action
            arm_action = {}
            if leader_arm is not None and leader_arm.is_connected:
                arm_action_raw = leader_arm.get_action()
                # Prefix with "arm_" to match robot action format
                arm_action = {f"arm_{k}": v for k, v in arm_action_raw.items()}
            
            # 2. Base action from gamepad
            base_action = gamepad.get_action()
            
            # Combine actions
            action = {**arm_action}
            if len(base_action) > 0:
                action.update(base_action)
            
            # ---- Send action ----
            if len(action) > 0:
                _ = robot.send_action(action)
            
            # ---- Check exit ----
            pygame.event.pump()
            if gamepad.joystick.get_button(gamepad.BTN_START):
                print("\n   START pressed - exiting...")
                running = False
            
            # ---- FPS control ----
            frame_count += 1
            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))
    
    except KeyboardInterrupt:
        print("\n   Interrupted by user")
    
    except Exception as e:
        print(f"\n   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # =========================================================================
        # 5. Cleanup
        # =========================================================================
        print("\n[4/4] Cleaning up...")
        
        # Stop robot
        print("  → Stopping robot...")
        robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        time.sleep(0.1)
        
        # Disconnect
        print("  → Disconnecting...")
        gamepad.disconnect()
        if leader_arm is not None:
            leader_arm.disconnect()
        robot.disconnect()
        
        print("\n" + "=" * 70)
        print("  Teleoperation ended")
        print(f"  Frames: {frame_count}")
        print("=" * 70)


if __name__ == "__main__":
    main()
