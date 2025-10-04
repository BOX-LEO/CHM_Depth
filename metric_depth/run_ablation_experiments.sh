#!/bin/bash

# Ablation Experiment Script for Depth Anything V2
# This script runs experiments with all combinations of:
# - trainable: full, head
# - gt: pseudo, chm  
# - loss: silog+l1 (combined loss for all experiments)

# Set the base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Function to run a single experiment
run_experiment() {
    local trainable=$1
    local gt=$2
    
    echo "=========================================="
    echo "Starting experiment: trainable=$trainable, gt=$gt, loss=silog+l1"
    echo "=========================================="
    
    # Create log file name
    log_file="logs/ablation_${trainable}_${gt}_silog_l1_${TIMESTAMP}.log"
    
    # Run the experiment with silog+l1 loss
    python depth_chm_optimize.py \
        --trainable $trainable \
        --gt $gt \
        --loss l1 \
        --encoder vitb \
        --epochs 50 \
        --bs 8 \
        --save-path /home/boxiang/work/dao2/segment/Depth-Anything-V2/checkpoints \
        2>&1 | tee "$log_file"
    
    # Check if the experiment completed successfully
    if [ $? -eq 0 ]; then
        echo "✅ Experiment completed successfully: trainable=$trainable, gt=$gt, loss=silog+l1"
        echo "📄 Log saved to: $log_file"
    else
        echo "❌ Experiment failed: trainable=$trainable, gt=$gt, loss=silog+l1"
        echo "📄 Check log file: $log_file"
    fi
    
    echo ""
}

# Function to run all experiments
run_all_experiments() {
    echo "🚀 Starting Ablation Experiments"
    echo "Timestamp: $TIMESTAMP"
    echo "Total experiments: 4 (2 x 2 combinations with silog+l1 loss)"
    echo ""
    
    # Define all parameter values
    trainable_options=("full" "head")
    gt_options=("chm" "pseudo" )
    
    # Counter for experiments
    experiment_count=0
    total_experiments=4
    
    # Run all combinations
    for trainable in "${trainable_options[@]}"; do
        for gt in "${gt_options[@]}"; do
            experiment_count=$((experiment_count + 1))
            echo "🔄 Running experiment $experiment_count/$total_experiments"
            run_experiment $trainable $gt
            
            # Optional: Add a small delay between experiments
            sleep 5
        done
    done
    
    echo "🎉 All ablation experiments completed!"
    echo "📁 Check the 'logs' directory for detailed logs"
    echo "📁 Check the 'checkpoints' directory for saved models"
}

# Function to run specific experiment combinations
run_specific_experiments() {
    echo "🎯 Running specific experiments..."
    
    # Example: Run only experiments with full training
    # run_experiment "full" "chm"
    # run_experiment "full" "pseudo"
    
    # Example: Run only experiments with head training
    # run_experiment "head" "chm"
    # run_experiment "head" "pseudo"
    
    echo "Please uncomment the experiments you want to run in the script."
}

# Function to show experiment summary
show_summary() {
    echo "📊 Ablation Experiment Summary"
    echo "=============================="
    echo ""
    echo "Parameter combinations:"
    echo "  - trainable: full, head"
    echo "  - gt: pseudo, chm"
    echo "  - loss: silog+l1 (combined loss for all experiments)"
    echo ""
    echo "Total experiments: 4"
    echo ""
    echo "Experiment matrix:"
    echo "┌─────────┬─────────┬─────────┬─────────┐"
    echo "│ trainable │ gt      │ loss     │ exp #   │"
    echo "├─────────┼─────────┼─────────┼─────────┤"
    echo "│ full     │ pseudo  │ silog+l1 │   1     │"
    echo "│ full     │ chm     │ silog+l1 │   2     │"
    echo "│ head     │ pseudo  │ silog+l1 │   3     │"
    echo "│ head     │ chm     │ silog+l1 │   4     │"
    echo "└─────────┴─────────┴─────────┴─────────┘"
    echo ""
    echo "Note: All experiments use the combined silog+l1 loss function"
    echo "      (SiLogLoss + 0.1 * L1Loss) for better training stability"
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  all       Run all 4 ablation experiments (silog+l1 loss)"
    echo "  specific  Run specific experiments (edit script to choose)"
    echo "  summary   Show experiment summary"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 all              # Run all experiments with silog+l1 loss"
    echo "  $0 summary          # Show experiment summary"
    echo "  $0 specific         # Run specific experiments"
    echo ""
    echo "Note: All experiments use combined silog+l1 loss for consistency"
}

# Main script logic
case "${1:-help}" in
    "all")
        run_all_experiments
        ;;
    "specific")
        run_specific_experiments
        ;;
    "summary")
        show_summary
        ;;
    "help"|*)
        show_help
        ;;
esac




