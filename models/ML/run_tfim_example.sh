#!/bin/bash
# TFIM 2D Lasso Regression Experiment (ICML 2025 Paper Settings)
# Aligned with: 'Rethink the Role of Deep Learning towards Large-scale Quantum Systems'

# Configuration
GRID_SIZE="5x5"  # Options: "5x5" or "8x8"
LAMBDA=1000      # Fixed λ = 10³ (paper setting)
RFF_R=20         # RFF dimension R (paper: ≈ 10-40)
RFF_GAMMA=0.6    # RFF scaling γ (paper: ≈ 0.5-0.7)

# Set paths based on grid size
if [ "$GRID_SIZE" = "5x5" ]; then
    TRAIN_FILE="/home/ubuntu/code/python/DeepModelFusion/ml4quantum/dataset_generation/dataset_results/tfim_2d/n20|X(coupling, meas512)_y(energy,entropy,corrs)_q(5, 5).csv"
    TEST_FILE="/home/ubuntu/code/python/DeepModelFusion/ml4quantum/dataset_generation/dataset_results/tfim_2d/n200|X(coupling, meas512)_y(energy,entropy,corrs)_q(5, 5).csv"
elif [ "$GRID_SIZE" = "8x8" ]; then
    TRAIN_FILE="/home/ubuntu/code/python/DeepModelFusion/ml4quantum/dataset_generation/dataset_results/tfim_2d/n20|X(coupling, meas512)_y(energy,entropy,corrs)_q(8, 8).csv"
    TEST_FILE="/home/ubuntu/code/python/DeepModelFusion/ml4quantum/dataset_generation/dataset_results/tfim_2d/n200|X(coupling, meas512)_y(energy,entropy,corrs)_q(8, 8).csv"
else
    echo "Error: Invalid GRID_SIZE. Use '5x5' or '8x8'"
    exit 1
fi

echo "====================================="
echo "ICML 2025 Paper Settings:"
echo "  Grid size: $GRID_SIZE"
echo "  λ (Lasso): $LAMBDA"
echo "  RFF R: $RFF_R"
echo "  RFF γ: $RFF_GAMMA"
echo "  Train: $(basename "$TRAIN_FILE")"
echo "  Test: $(basename "$TEST_FILE")"
echo "====================================="
echo ""

# Run correlation prediction
echo "====================================="
echo "Running correlation prediction (ICML 2025 settings)..."
echo "====================================="

# Generate log filename based on train file name
TRAIN_BASENAME=$(basename "$TRAIN_FILE" .csv)
TEST_BASENAME=$(basename "$TEST_FILE" .csv)
LOG_FILE="./results_tfim/lasso_tfim2d_correlation_${TRAIN_BASENAME}_test_${TEST_BASENAME}_R${RFF_R}_gamma${RFF_GAMMA}_multi_seed.log"

# Create results directory if it doesn't exist
mkdir -p ./results_tfim

python3 train_tfim.py \
    --train-file "$TRAIN_FILE" \
    --test-file "$TEST_FILE" \
    --task correlation \
    --lasso-alpha $LAMBDA \
    --num-rff $RFF_R \
    --rff-gamma $RFF_GAMMA \
    --save-results \
    --output-dir ./results_tfim \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "====================================="
echo "Running entropy prediction (ICML 2025 settings)..."
echo "====================================="

# Generate log filename for entropy task
LOG_FILE_ENTROPY="./results_tfim/lasso_tfim2d_entropy_${TRAIN_BASENAME}_test_${TEST_BASENAME}_R${RFF_R}_gamma${RFF_GAMMA}_multi_seed.log"

# Run entropy prediction
python3 train_tfim.py \
    --train-file "$TRAIN_FILE" \
    --test-file "$TEST_FILE" \
    --task entropy \
    --lasso-alpha $LAMBDA \
    --num-rff $RFF_R \
    --rff-gamma $RFF_GAMMA \
    --save-results \
    --output-dir ./results_tfim \
    2>&1 | tee "$LOG_FILE_ENTROPY"

echo ""
echo "====================================="
echo "Experiment completed!"
echo "Results saved in ./results_tfim/"
echo "Log files:"
echo "  Correlation: $LOG_FILE"
echo "  Entropy: $LOG_FILE_ENTROPY"
echo "Settings: λ=$LAMBDA, R=$RFF_R, γ=$RFF_GAMMA"
echo "====================================="
