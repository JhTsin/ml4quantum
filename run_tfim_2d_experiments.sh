#!/bin/bash

# 2D TFIM Dataset Generation Script
# This script runs generation_tfim_2d_digital.jl with various parameter combinations

echo "Starting 2D TFIM dataset generation experiments..."
echo "Start time: $(date)"
echo "======================================================"

# Grid sizes to test
GRID_SIZES=("5 5" "8 8")

# Part 1: Run with -n 100 for different shot numbers
SHOTS_N100=(64 128 256 512)
echo "Part 1: Running with -n 100 for different shot numbers..."
for grid in "${GRID_SIZES[@]}"; do
    read -r nx ny <<< "$grid"
    for shots in "${SHOTS_N100[@]}"; do
        echo "Running: Nx=$nx, Ny=$ny, samples=100, shots=$shots"
        julia dataset_generation/generation_tfim_2d_digital.jl -n 100 -s $shots --Nx $nx --Ny $ny
        echo "Completed: Nx=$nx, Ny=$ny, samples=100, shots=$shots at $(date)"
        echo "------------------------------------------------------"
    done
done

# Part 2: Run with -s 512 for different sample numbers
SAMPLES_S512=(20 40 60 80 100)
echo "Part 2: Running with -s 512 for different sample numbers..."
for grid in "${GRID_SIZES[@]}"; do
    read -r nx ny <<< "$grid"
    for samples in "${SAMPLES_S512[@]}"; do
        echo "Running: Nx=$nx, Ny=$ny, samples=$samples, shots=512"
        julia dataset_generation/generation_tfim_2d_digital.jl -n $samples -s 512 --Nx $nx --Ny $ny
        echo "Completed: Nx=$nx, Ny=$ny, samples=$samples, shots=512 at $(date)"
        echo "------------------------------------------------------"
    done
done

echo "======================================================"
echo "All experiments completed!"
echo "End time: $(date)"
