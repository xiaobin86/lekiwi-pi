"""
Gamepad Teleoperation Module for LekiWi

Supports Xbox controller for base movement with the following mapping:
- D-pad Up/Down: forward/backward (x.vel)
- D-pad Left/Right: strafe left/right (y.vel)
- RB + D-pad Left/Right: rotate in place (theta.vel)
- LB/LT: speed up / speed down
"""

import numpy as np
import pygame
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GamepadTeleopConfig:
    """Configuration for gamepad teleoperation"""
    id: str = "xbox_gamepad"
    
    # Speed levels (matching LeKiwiClient defaults)
    speed_levels: list = field(default_factory=lambda: [
        {"xy": 0.1, "theta": 30},   # slow
        {"xy": 0.3, "theta": 60},   # medium
        {"xy": 0.5, "theta": 90},   # fast
    ])
    
    # Button mappings (Xbox controller standard layout)
    # D-pad is accessed via joystick.get_hat(0)
    # RB (Right Bumper) is button 5
    # LB (Left Bumper) is button 4
    

class GamepadTeleop:
    """
    Xbox Gamepad teleoperator for LekiWi base control.
    
    Control Mapping:
        D-pad Up        → Forward
        D-pad Down      → Backward
        D-pad Left      → Strafe Left
        D-pad Right     → Strafe Right
        RB + D-pad Left → Rotate Left (in place)
        RB + D-pad Right→ Rotate Right (in place)
        LB              → Speed Up
        LT              → Speed Down
    """
    
    def __init__(self, config: Optional[GamepadTeleopConfig] = None):
        self.config = config or GamepadTeleopConfig()
        self.joystick = None
        self.is_connected = False
        self.speed_index = 1  # Start at medium speed
        self.speed_levels = self.config.speed_levels
        
        # Xbox button/axis indices
        self.BTN_RB = 5          # Right Bumper
        self.BTN_LB = 4          # Left Bumper
        self.BTN_A = 0           # A button
        self.BTN_B = 1           # B button
        self.BTN_X = 2           # X button
        self.BTN_Y = 3           # Y button
        self.BTN_BACK = 6        # Back/View button
        self.BTN_START = 7       # Start/Menu button
        
        self.AXIS_HAT = 0        # D-pad hat
        
        # Deadzone for analog inputs
        self.deadzone = 0.12
        
    def connect(self):
        """Initialize and connect to Xbox controller"""
        pygame.init()
        pygame.joystick.init()
        
        # Wait for controller with timeout
        retry_count = 0
        max_retries = 50  # 5 seconds
        
        while pygame.joystick.get_count() == 0 and retry_count < max_retries:
            print(f"  Waiting for gamepad... ({retry_count + 1}/{max_retries})")
            pygame.time.wait(100)
            pygame.joystick.init()
            retry_count += 1
        
        if pygame.joystick.get_count() == 0:
            print("❌ No gamepad detected!")
            print("   Please connect an Xbox controller via USB or Bluetooth")
            self.is_connected = False
            return False
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        name = self.joystick.get_name()
        print(f"🎮 Gamepad connected: {name}")
        
        if "bluetooth" in name.lower() or "wireless" in name.lower():
            print("   Connection: Bluetooth/Wireless")
        else:
            print("   Connection: USB")
        
        self.is_connected = True
        return True
    
    def disconnect(self):
        """Disconnect from gamepad"""
        if self.joystick:
            self.joystick.quit()
        pygame.quit()
        self.is_connected = False
        print("🎮 Gamepad disconnected")
    
    def get_action(self) -> dict:
        """
        Read gamepad state and convert to base action.
        
        Returns:
            dict: Action with keys "x.vel", "y.vel", "theta.vel"
                  Empty dict if no movement input
        """
        if not self.is_connected or self.joystick is None:
            return {}
        
        # Process events
        pygame.event.pump()
        
        # Get current speed setting
        speed = self.speed_levels[self.speed_index]
        xy_speed = speed["xy"]
        theta_speed = speed["theta"]
        
        # Initialize commands
        x_cmd = 0.0   # forward/backward
        y_cmd = 0.0   # left/right strafe
        theta_cmd = 0.0  # rotation
        
        # Read D-pad (hat)
        if self.joystick.get_numhats() > 0:
            hat = self.joystick.get_hat(self.AXIS_HAT)
            hat_x, hat_y = hat
            
            # Check if RB is held (rotation mode)
            rb_pressed = self.joystick.get_button(self.BTN_RB)
            
            if rb_pressed:
                # RB + D-pad: rotate in place
                if hat_x < 0:    # D-pad Left
                    theta_cmd += theta_speed   # Rotate left
                elif hat_x > 0:  # D-pad Right
                    theta_cmd -= theta_speed   # Rotate right
                # D-pad Up/Down with RB does nothing (or could be speed control)
            else:
                # Normal mode: translation
                if hat_y > 0:    # D-pad Up
                    x_cmd += xy_speed      # Forward
                elif hat_y < 0:  # D-pad Down
                    x_cmd -= xy_speed      # Backward
                
                if hat_x < 0:    # D-pad Left
                    y_cmd += xy_speed      # Strafe left
                elif hat_x > 0:  # D-pad Right
                    y_cmd -= xy_speed      # Strafe right
        
        # Speed control with LB/LT
        if self.joystick.get_button(self.BTN_LB):
            # Check if this is a new press (avoid rapid toggling)
            # Simple approach: only act on button down events
            pass  # Handled via events below
        
        # Process button events for speed control
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == self.BTN_LB:
                    # Speed up
                    self.speed_index = min(self.speed_index + 1, len(self.speed_levels) - 1)
                    speed_name = ["SLOW", "MEDIUM", "FAST"][self.speed_index]
                    print(f"   Speed: {speed_name} (xy={self.speed_levels[self.speed_index]['xy']})")
                elif event.button == self.BTN_A:
                    # A button could be used for gripper or other functions
                    pass
        
        # Build action dict
        action = {}
        if abs(x_cmd) > 0.001:
            action["x.vel"] = x_cmd
        if abs(y_cmd) > 0.001:
            action["y.vel"] = y_cmd
        if abs(theta_cmd) > 0.001:
            action["theta.vel"] = theta_cmd
        
        return action
    
    def get_speed_info(self) -> str:
        """Get current speed setting as string"""
        speed = self.speed_levels[self.speed_index]
        names = ["SLOW", "MEDIUM", "FAST"]
        return f"{names[self.speed_index]} (xy={speed['xy']}, rot={speed['theta']})"


if __name__ == "__main__":
    # Test the gamepad teleop
    print("=" * 60)
    print("Gamepad Teleop Test")
    print("=" * 60)
    print("\nControls:")
    print("  D-pad Up/Down     → Forward/Backward")
    print("  D-pad Left/Right  → Strafe Left/Right")
    print("  RB + Left/Right   → Rotate in place")
    print("  LB                → Speed Up")
    print("  START             → Exit")
    print("\nPress START to begin...")
    
    teleop = GamepadTeleop()
    if not teleop.connect():
        print("Failed to connect gamepad")
        exit(1)
    
    import time
    try:
        while True:
            action = teleop.get_action()
            
            if action:
                print(f"\rAction: x={action.get('x.vel', 0):+.2f} "
                      f"y={action.get('y.vel', 0):+.2f} "
                      f"θ={action.get('theta.vel', 0):+.1f}  "
                      f"| Speed: {teleop.get_speed_info()}", end="", flush=True)
            
            # Check for quit
            pygame.event.pump()
            if teleop.joystick.get_button(teleop.BTN_START):
                print("\n\nExit requested")
                break
            
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    
    finally:
        teleop.disconnect()
        print("Test complete")
