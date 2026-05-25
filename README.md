# LekiWi Gamepad Teleoperation

PC-based teleoperation for LekiWi robot using Xbox controller.

## Architecture

```
PC (Windows)
├── Xbox Controller ──→ GamepadTeleop ──┐
├── SO101 Leader Arm ─→ SO101Leader ────┼──→ LeKiwiClient ──→ ZMQ ──→ Raspberry Pi
└── Keyboard (quit)                      │                       (kiwiclient host)
                                         │
                                    Action: {x.vel, y.vel, theta.vel, arm_*.pos}
```

## Features

- **Xbox Controller**: D-pad base movement, RB+dpad rotation, LB speed control
- **SO101 Leader Arm**: Direct teleoperation via serial port
- **Network Control**: PC connects to Raspberry Pi via ZMQ
- **Cross-Platform**: Windows PC + Raspberry Pi kiwiclient

## Control Mapping

### Base Movement

| Input | Action | Speed |
|-------|--------|-------|
| D-pad Up | Forward (x.vel+) | Current level |
| D-pad Down | Backward (x.vel-) | Current level |
| D-pad Left | Strafe Left (y.vel+) | Current level |
| D-pad Right | Strafe Right (y.vel-) | Current level |
| RB + D-pad Left | Rotate Left (theta.vel+) | Current level |
| RB + D-pad Right | Rotate Right (theta.vel-) | Current level |
| LB | Cycle speed: Slow → Medium → Fast | - |

### Speed Levels

| Level | XY Speed | Rotation Speed |
|-------|----------|----------------|
| Slow | 0.1 m/s | 30 deg/s |
| Medium | 0.3 m/s | 60 deg/s |
| Fast | 0.5 m/s | 90 deg/s |

### Leader Arm

SO101 leader arm mirrors directly to follower arm on Raspberry Pi.

## Installation

```bash
# 1. Install LeRobot (with lekiwi support)
cd ~/lerobot
pip install -e ".[feetech]"

# 2. Install this project
cd ~/lekiwi-pi
pip install -r requirements.txt

# 3. Connect Xbox controller (USB or Bluetooth)
```

## Usage

### 1. Start kiwiclient on Raspberry Pi

```bash
# SSH into Raspberry Pi
ssh pi@raspberrypi.local

# Start kiwiclient host
cd ~/lerobot
python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_lekiwi
```

### 2. Start teleoperation on PC

```bash
cd ~/lekiwi-pi
python src/teleop_gamepad.py --remote-ip 192.168.31.165
```

### Options

```
python src/teleop_gamepad.py --help

Options:
  --remote-ip        Raspberry Pi IP (required)
  --arm-port         Leader arm serial port (default: COM5)
  --arm-id           Leader arm ID (default: L07252802)
  --robot-id         Robot ID (default: my_lekiwi)
  --zmq-cmd-port     ZMQ command port (default: 5555)
  --zmq-obs-port     ZMQ observation port (default: 5556)
  --no-arm           Disable leader arm (base only)
  --fps              Control loop frequency (default: 30)
```

## Troubleshooting

### Gamepad not detected

```bash
# Windows: check in Device Manager
# Bluetooth: ensure paired and connected
# USB: try different USB port
```

### Cannot connect to Raspberry Pi

```bash
# Check network connectivity
ping 192.168.31.165

# Check kiwiclient is running on Pi
ssh pi@raspberrypi.local "ps aux | grep lekiwi_host"

# Check ZMQ ports are open
ssh pi@raspberrypi.local "netstat -tlnp | grep 5555"
```

### Serial port issues (leader arm)

```bash
# Windows: check COM port in Device Manager
# Linux: check /dev/ttyACM0 or /dev/ttyUSB0
```

## File Structure

```
lekiwi-pi/
├── src/
│   ├── teleop_gamepad.py      # Main teleop script
│   └── gamepad_teleop.py      # Xbox controller class
├── requirements.txt
└── README.md
```

## Development

```bash
# Git-flow
# Feature branch: feature/gamepad-teleop
# Main branch: master

git checkout -b feature/your-feature
# ... make changes ...
git add -A
git commit -m "feat: your feature"
git push origin feature/your-feature
```

## License

Same as LeRobot project.
