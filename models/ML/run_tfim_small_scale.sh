#!/bin/bash
# 小规模测试 - 使用更少测量次数的数据集

GRID_SIZE="5x5"
LAMBDA=1000
RFF_R=20  # 减少RFF维度
RFF_GAMMA=0.6


if [ "$GRID_SIZE" = "5x5" ]; then
    TRAIN_FILE="/home/ubuntu/code/python/DeepModelFusion/ml4quantum/dataset_generation/dataset_results/tfim_2d/n20|X(coupling, meas512)_y(energy,entropy,corrs)_q(5, 5).csv"
    TEST_FILE="/home/ubuntu/code/python/DeepModelFusion/ml4quantum/dataset_generation/dataset_results/tfim_2d/n200|X(coupling, meas512)_y(energy,entropy,corrs)_q(5, 5).csv"
else
    TRAIN_FILE="/home/ubuntu/code/python/DeepModelFusion/ml4quantum/dataset_generation/dataset_results/tfim_2d/n100|X(coupling, meas64)_y(energy,entropy,corrs)_q(8, 8).csv"
    TEST_FILE="/home/ubuntu/code/python/DeepModelFusion/ml4quantum/dataset_generation/dataset_results/tfim_2d/n100|X(coupling, meas64)_y(energy,entropy,corrs)_q(8, 8).csv"
fi

echo "====================================="
echo "小规模测试 (meas=64, R=10)"
echo "  预计特征维度: ~1600 (5x5) 或 ~4096 (8x8)"
echo "  训练集: n=100, 测试集: n=100"
echo "====================================="

mkdir -p ./results_tfim

# 只运行entropy任务
echo "Running entropy task..."
time python3 train_tfim.py \
    --train-file "$TRAIN_FILE" \
    --test-file "$TEST_FILE" \
    --task entropy \
    --lasso-alpha $LAMBDA \
    --num-rff $RFF_R \
    --rff-gamma $RFF_GAMMA \
    --save-results \
    --output-dir ./results_tfim

echo ""
echo "====================================="
echo "完成!"
echo "====================================="

