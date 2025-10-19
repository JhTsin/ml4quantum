#!/bin/bash

# Startup script to run the 2D TFIM experiments with nohup

cd /home/ubuntu/code/python/DeepModelFusion/ml4quantum

echo "Starting 2D TFIM experiments in background with nohup..."
echo "All output will be logged to 2dHFIM.log"

nohup ./run_tfim_2d_experiments.sh > 2dHFIM.log 2>&1 &

PID=$!
echo "Experiments started with PID: $PID"
echo "You can monitor progress with: tail -f 2dHFIM.log"
echo "To check if it's still running: ps -p $PID"
