#!/bin/bash
# ACT策略训练脚本
# 使用LeRobot训练ACT策略模型

set -e

echo "=================================="
echo "LeKiwi ACT 策略训练"
echo "=================================="

# 配置
DATASET_REPO="your_username/lekiwi_grasp_paper_ball"  # 替换为你的数据集
OUTPUT_DIR="outputs/lekiwi_grasp_act"
JOB_NAME="lekiwi_grasp"
DEVICE="cuda"

# 检查参数
if [ "$#" -ge 1 ]; then
    DATASET_REPO=$1
fi

if [ "$#" -ge 2 ]; then
    OUTPUT_DIR=$2
fi

echo "数据集: $DATASET_REPO"
echo "输出目录: $OUTPUT_DIR"
echo "设备: $DEVICE"
echo ""

# 激活环境
conda activate lerobot

cd ~/lerobot-workspace/lerobot

# 训练ACT策略
echo "开始训练ACT策略..."
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=$DATASET_REPO \
    --output_dir=$OUTPUT_DIR \
    --job_name=$JOB_NAME \
    --device=$DEVICE \
    --wandb.enable=true \
    --batch_size=16 \
    --num_workers=4 \
    \
    # ACT策略配置
    --policy.n_action_steps=8 \
    --policy.chunk_size=8 \
    --policy.n_obs_steps=1 \
    --policy.dim_model=512 \
    --policy.n_heads=8 \
    --policy.n_encoder_layers=4 \
    --policy.n_decoder_layers=7 \
    --policy.use_vae=true \
    --policy.kl_weight=10.0 \
    --policy.latent_dim=32 \
    \
    # 图像编码器
    --policy.vision_backbone=resnet18 \
    --policy.pretrained_backbone=true \
    \
    # 训练配置
    --training.lr=1e-5 \
    --training.num_epochs=3000 \
    --training.save_freq=1000 \
    --training.eval_freq=1000 \
    \
    # 数据增强
    --training.image_transforms.enable=true

echo ""
echo "训练完成！"
echo "模型保存路径: $OUTPUT_DIR/checkpoints/last"
echo ""

# 评估模型
echo "开始评估..."
lerobot-eval \
    --policy.type=act \
    --policy.pretrained_path=$OUTPUT_DIR/checkpoints/last \
    --dataset.repo_id=$DATASET_REPO \
    --output_dir=$OUTPUT_DIR/eval \
    --device=$DEVICE

echo "评估完成！"
