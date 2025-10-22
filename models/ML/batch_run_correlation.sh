#!/bin/bash
# Batch runner for correlation experiments
# Iterates through different grid sizes, measurement numbers, and training sizes

# Define experiment configurations
############## Experiment 1 ##############
# GRID_SIZES=(5 8)                    # 5x5 and 8x8 grids
# MEAS_NUMS=(512)          # Different measurement numbers
# TRAIN_SIZES=(20 40 60 80 100)      # Different training sample sizes
############## Experiment 2 ##############
GRID_SIZES=(5)                    # 5x5 and 8x8 grids
MEAS_NUMS=(64 128 256 512)          # Different measurement numbers
TRAIN_SIZES=(100)      # Different training sample sizes


# Log directory
LOG_DIR="/home/ubuntu/code/python/DeepModelFusion/ml4quantum/models/ML/results_log/exp1"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Batch Correlation Experiments"
echo "========================================"
echo "Grid sizes: ${GRID_SIZES[@]}"
echo "Measurement numbers: ${MEAS_NUMS[@]}"
echo "Training sizes: ${TRAIN_SIZES[@]}"
echo "========================================"
echo ""

# Counter for tracking experiments
TOTAL_EXPERIMENTS=$((${#GRID_SIZES[@]} * ${#MEAS_NUMS[@]} * ${#TRAIN_SIZES[@]}))
CURRENT_EXP=0

# Nested loops: grid_size -> meas_num -> train_size
for GRID_SIZE in "${GRID_SIZES[@]}"; do
    for MEAS_NUM in "${MEAS_NUMS[@]}"; do
        for TRAIN_SIZE in "${TRAIN_SIZES[@]}"; do
            CURRENT_EXP=$((CURRENT_EXP + 1))
            
            # Generate log filename
            LOG_FILE="${LOG_DIR}/tfim_example${GRID_SIZE}${GRID_SIZE}_${MEAS_NUM}_${TRAIN_SIZE}_correlation.log"
            
            echo "[$CURRENT_EXP/$TOTAL_EXPERIMENTS] Running experiment: ${GRID_SIZE}x${GRID_SIZE}, meas=${MEAS_NUM}, n=${TRAIN_SIZE}"
            echo "  Log file: $LOG_FILE"
            
            # Run experiment in background with nohup
            nohup bash run_tfim_correlation.sh "$GRID_SIZE" "$MEAS_NUM" "$TRAIN_SIZE" > "$LOG_FILE" 2>&1 &

            # Store PID
            PID=$!
            echo "  Started with PID: $PID"
            
            # Optional: Add a small delay to avoid overwhelming the system
            sleep 2
        done
    done
done

echo ""
echo "========================================"
echo "All correlation experiments submitted!"
echo "Total: $TOTAL_EXPERIMENTS experiments"
echo "Check log files in: $LOG_DIR"
echo "========================================"
echo ""
echo "To monitor running jobs, use:"
echo "  ps aux | grep run_tfim_correlation.sh"
echo "  ps aux | grep train_tfim.py"

