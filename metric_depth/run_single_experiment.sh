#!/bin/bash

# Simple script to run a single ablation experiment
# Usage: ./run_single_experiment.sh <trainable> <gt>

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <trainable> <gt>"
    echo ""
    echo "Parameters:"
    echo "  trainable: full or head"
    echo "  gt: pseudo or chm"
    echo "  loss: silog+l1 (fixed for all experiments)"
    echo ""
    echo "Examples:"
    echo "  $0 full chm"
    echo "  $0 head pseudo"
    echo ""
    echo "Note: All experiments use combined silog+l1 loss for consistency"
    exit 1
fi

TRAINABLE=$1
GT=$2

# Validate parameters
if [[ ! "$TRAINABLE" =~ ^(full|head)$ ]]; then
    echo "Error: trainable must be 'full' or 'head'"
    exit 1
fi

if [[ ! "$GT" =~ ^(pseudo|chm)$ ]]; then
    echo "Error: gt must be 'pseudo' or 'chm'"
    exit 1
fi

echo "=========================================="
echo "Running experiment:"
echo "  trainable: $TRAINABLE"
echo "  gt: $GT"
echo "  loss: silog+l1 (combined loss)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Get timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/experiment_${TRAINABLE}_${GT}_silog_l1_${TIMESTAMP}.log"

# Run the experiment with silog+l1 loss
python depth_chm_optimize.py \
    --trainable $TRAINABLE \
    --gt $GT \
    --loss l1 \
    --encoder vitb \
    --epochs 50 \
    --bs 8 \
    --save-path /home/dao2/segment/Depth-Anything-V2/checkpoints \
    2>&1 | tee "$LOG_FILE"

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Experiment completed successfully!"
    echo "📄 Log saved to: $LOG_FILE"
    echo "💡 Note: This experiment used combined silog+l1 loss"
else
    echo "❌ Experiment failed!"
    echo "📄 Check log file: $LOG_FILE"
    exit 1
fi




