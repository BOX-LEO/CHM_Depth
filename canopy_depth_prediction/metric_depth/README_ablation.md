# Ablation Experiments for Depth Anything V2

This directory contains scripts to run comprehensive ablation experiments for the Depth Anything V2 model.

## Overview

The ablation experiments test the effect of two key components:
1. **Trainable Components** (`--trainable`): `full` vs `head`
2. **Ground Truth Type** (`--gt`): `pseudo` vs `chm`
3. **Loss Function**: All experiments use **combined silog+l1 loss** for consistency

## Files

- `depth_chm_optimize.py`: Main training script with ablation support
- `run_ablation_experiments.sh`: Comprehensive script to run all 4 experiments
- `run_single_experiment.sh`: Simple script to run individual experiments
- `README_ablation.md`: This file

## Usage

### Option 1: Run All Experiments (Recommended)

```bash
cd Depth-Anything-V2/metric_depth
./run_ablation_experiments.sh all
```

This will run all 4 combinations:
1. full + pseudo + silog+l1
2. full + chm + silog+l1
3. head + pseudo + silog+l1
4. head + chm + silog+l1

### Option 2: Run Individual Experiments

```bash
cd Depth-Anything-V2/metric_depth

# Run a single experiment
./run_single_experiment.sh full chm

# Examples
./run_single_experiment.sh head pseudo
./run_single_experiment.sh full chm
```

### Option 3: Show Experiment Summary

```bash
./run_ablation_experiments.sh summary
```

### Option 4: Manual Python Execution

```bash
python depth_chm_optimize.py \
    --trainable full \
    --gt chm \
    --loss l1 \
    --encoder vitb \
    --epochs 50 \
    --bs 8
```

## Experiment Matrix

| Experiment | Trainable | GT Type | Loss | Description |
|------------|-----------|---------|------|-------------|
| 1 | full | pseudo | silog+l1 | Full model, pseudo GT, combined loss |
| 2 | full | chm | silog+l1 | Full model, CHM GT, combined loss |
| 3 | head | pseudo | silog+l1 | Head-only, pseudo GT, combined loss |
| 4 | head | chm | silog+l1 | Head-only, CHM GT, combined loss |

## Loss Function Details

**All experiments use the combined silog+l1 loss function:**
- **Primary Loss**: SiLogLoss (scale-invariant logarithmic loss)
- **Secondary Loss**: 0.1 × L1Loss (mean absolute error)
- **Combined**: `SiLogLoss + 0.1 * L1Loss`

This combination provides:
- Better training stability
- Improved convergence
- More robust depth estimation
- Consistent loss landscape across experiments

## Output

### Logs
- All experiments log to the `logs/` directory
- Log files are named with timestamp: `ablation_<trainable>_<gt>_silog_l1_<timestamp>.log`

### Checkpoints
- Model checkpoints are saved to the specified `--save-path`
- Checkpoint names follow the pattern: `martell_<gt>_<trainable>_<loss>_ablation.pth`

### Example Output Structure
```
logs/
├── ablation_full_chm_silog_l1_20241201_143022.log
├── ablation_full_pseudo_silog_l1_20241201_143156.log
├── ablation_head_chm_silog_l1_20241201_143230.log
└── ablation_head_pseudo_silog_l1_20241201_143304.log

checkpoints/
├── martell_chm_full_l1_ablation.pth
├── martell_pseudo_full_l1_ablation.pth
├── martell_chm_head_l1_ablation.pth
└── martell_pseudo_head_l1_ablation.pth
```

## Configuration

### Default Parameters
- **Encoder**: `vitb` (Vision Transformer Base)
- **Epochs**: 50
- **Batch Size**: 8
- **Image Size**: 500x500
- **Min Depth**: 0.00001
- **Max Depth**: 40
- **Loss**: Combined silog+l1 (fixed for all experiments)

### Data Paths
- **CHM Data**: `/home/dao2/USGS_LiDAR/martell_data/trainset/gt`
- **Pseudo GT**: `/home/dao2/USGS_LiDAR/martell_data/trainset/pseudo_gt` (adjust as needed)
- **Images**: `/home/dao2/USGS_LiDAR/martell_data/trainset/image`

## Monitoring

### Real-time Monitoring
```bash
# Watch log files in real-time
tail -f logs/ablation_*.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Progress Tracking
The scripts provide progress indicators:
- ✅ Success messages
- ❌ Error messages
- 📄 Log file locations
- 🔄 Progress counters

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size with `--bs 4` or `--bs 2`
2. **Data Path Errors**: Update paths in `depth_chm_optimize.py` for your setup
3. **CUDA Errors**: Ensure CUDA is properly installed and GPU is available

### Debug Mode
```bash
# Run with verbose output
python depth_chm_optimize.py --trainable full --gt chm --loss l1 --verbose
```

## Analysis

After running experiments, analyze results by:
1. Comparing validation losses across experiments
2. Examining checkpoint files for model performance
3. Plotting training curves from log files
4. Comparing final model metrics

## Notes

- Experiments run sequentially to avoid GPU memory conflicts
- Each experiment includes hyperparameter optimization via Optuna
- Checkpoints include experiment configuration for reproducibility
- Logs capture both stdout and stderr for complete debugging
- **All experiments use the same combined silog+l1 loss for fair comparison**
- Reduced from 8 to 4 experiments for more focused ablation study

