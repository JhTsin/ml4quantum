#!/bin/bash
# TFIM 2D Lasso Regression - Entropy Task
# Single experiment runner

# Parameters
GRID_SIZE=$1      # "5" or "8" (for 5x5 or 8x8)
MEAS_NUM=$2       # e.g., 64, 128, 256, 512
TRAIN_SIZE=$3     # e.g., 20, 40, 60, 80, 100

# Fixed hyperparameters
LAMBDA=1000
RFF_R=20
RFF_GAMMA=0.6

# Validate parameters
if [ -z "$GRID_SIZE" ] || [ -z "$MEAS_NUM" ] || [ -z "$TRAIN_SIZE" ]; then
    echo "Usage: $0 <grid_size> <meas_num> <train_size>"
    echo "Example: $0 5 128 100"
    exit 1
fi

# Set file paths
DATASET_DIR="/home/ubuntu/code/python/DeepModelFusion/ml4quantum/dataset_generation/dataset_results/tfim_2d_new"
TRAIN_FILE="${DATASET_DIR}/n${TRAIN_SIZE}|X(coupling, meas${MEAS_NUM})_y(energy,entropy,corrs)_q(${GRID_SIZE}, ${GRID_SIZE}).csv"
TEST_FILE="${DATASET_DIR}/n200|X(coupling, meas${MEAS_NUM})_y(energy,entropy,corrs)_q(${GRID_SIZE}, ${GRID_SIZE}).csv"

# Check if files exist
if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Training file not found: $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file not found: $TEST_FILE"
    exit 1
fi

# Create results directory
mkdir -p ./results_tfim

# Run entropy prediction
python3 train_tfim.py \
    --train-file "$TRAIN_FILE" \
    --test-file "$TEST_FILE" \
    --task entropy \
    --lasso-alpha $LAMBDA \
    --num-rff $RFF_R \
    --rff-gamma $RFF_GAMMA \
    --save-results \
    --output-dir ./results_tfim

