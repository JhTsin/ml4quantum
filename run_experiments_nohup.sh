#!/bin/bash

# 2D TFIM Experiments Runner with nohup
# This script runs experiments with maximum parallelism in background

echo "🚀 Starting 2D TFIM experiments with maximum parallelism..."
echo "Running in background with nohup"
echo "All output will be saved to multiprocessor.log"
echo "======================================================"

# Check if we're in the right directory
if [ ! -f "run_tfim_2d_experiments_parallel.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Display system info
echo "System Information:"
echo "  CPU cores: $(nproc)"
echo "  Memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "  Available memory: $(free -h | awk '/^Mem:/ {print $7}')"
echo ""

echo "Starting experiments with nohup..."
echo "Start time: $(date)"
echo ""

# Run with nohup, redirect all output to multiprocessor.log
nohup python3 run_tfim_2d_experiments_parallel.py > multiprocessor.log 2>&1 &

# Get the process ID
PID=$!

echo "✅ Experiments started successfully!"
echo "Process ID: $PID"
echo ""
echo "📊 Monitoring commands:"
echo "  - View live output: tail -f multiprocessor.log"
echo "  - Check process status: ps -p $PID"
echo "  - Kill process if needed: kill $PID"
echo ""
echo "📁 Output file: multiprocessor.log"
echo ""

# Save PID to file for easy management
echo $PID > experiment.pid
echo "Process ID saved to: experiment.pid"
