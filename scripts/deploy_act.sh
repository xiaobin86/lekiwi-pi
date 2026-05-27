#!/bin/bash
# ACT策略推理部署脚本
# 启动树莓派Host和PC端推理客户端

set -e

echo "=================================="
echo "LeKiwi ACT 策略推理部署"
echo "=================================="

# 配置
ROBOT_IP="192.168.3.176"
MODEL_PATH="outputs/lekiwi_grasp_act/checkpoints/last"
DEVICE="cuda"
MODE="inference"  # inference 或 teleop

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --ip)
            ROBOT_IP="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "树莓派IP: $ROBOT_IP"
echo "模型路径: $MODEL_PATH"
echo "推理设备: $DEVICE"
echo "运行模式: $MODE"
echo ""

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    echo "请确保已完成训练或提供正确的模型路径"
    exit 1
fi

# 激活环境
conda activate lerobot

cd ~/lerobot-workspace/lekiwi-pi

# 步骤1: 启动树莓派Host（在后台）
echo "[1/3] 启动树莓派Host..."
ssh acelan@$ROBOT_IP "
    cd ~/lerobot-workspace/lekiwi-pi
    pkill -f host_pi_act.py || true
    sleep 1
    nohup python src/host_pi_act.py --mode=$MODE > logs/host_pi.log 2>&1 &
    sleep 3
    echo '树莓派Host已启动'
"

echo "等待树莓派初始化..."
sleep 5

# 步骤2: 启动PC端推理客户端
echo "[2/3] 启动PC端推理客户端..."
python src/act_inference_client.py \
    --model_path=$MODEL_PATH \
    --robot_ip=$ROBOT_IP \
    --device=$DEVICE

# 步骤3: 停止树莓派Host
echo "[3/3] 停止树莓派Host..."
ssh acelan@$ROBOT_IP "
    pkill -f host_pi_act.py || true
    echo '树莓派Host已停止'
"

echo ""
echo "部署完成！"
