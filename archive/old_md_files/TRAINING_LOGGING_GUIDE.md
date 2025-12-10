# Training Logging & Visualization Guide

## Overview
Your training script now includes **comprehensive metric logging and visualization** to track training progress over time.

## What Was Added

### 1. CSV Metric Logging
**File:** `training_metrics.csv` (automatically created in output directory)

**Columns tracked:**
- `epoch` - Training epoch number
- `train_loss` - Training loss for the epoch
- `val_spearman` - Validation Spearman correlation (every 2 epochs)
- `val_recall_pkd9` - Recall for high-affinity binders (pKd ≥ 9)
- `best_spearman` - Best validation Spearman so far
- `learning_rate` - Current learning rate
- `timestamp` - When the epoch completed

### 2. Enhanced Checkpoint Saving
Checkpoints now include:
- All validation metrics (Spearman, Recall)
- Training loss
- Complete training state for resumption

**Files saved:**
- `best_model.pth` - Model with best validation Spearman
- `checkpoint_epoch.pth` - Latest epoch checkpoint
- `checkpoint_latest.pth` - Rolling checkpoint (every 500 batches)

### 3. Visualization Script
**File:** `visualize_training.py`

Creates 4-panel training curves showing:
1. **Training Loss** - with polynomial trend line
2. **Validation Spearman** - with best epoch marked
3. **Recall@pKd≥9** - high-affinity detection performance
4. **Learning Rate** - schedule visualization

## How to Use

### During Training
No changes needed! The script automatically:
- Creates `training_metrics.csv` in your output directory
- Logs metrics after each epoch
- Saves metrics in checkpoints

### After Training

#### View Metrics CSV
```bash
# View the metrics
cat output/training_metrics.csv

# Or in Python
import pandas as pd
df = pd.read_csv('output/training_metrics.csv')
print(df)
```

#### Generate Visualizations
```bash
# Basic usage (uses default path)
python visualize_training.py

# Specify CSV file
python visualize_training.py --csv output/training_metrics.csv

# Specify output directory
python visualize_training.py --csv output/training_metrics.csv --output_dir plots/
```

This will:
- Display training curves in a window
- Save `training_curves.png` to the output directory
- Print a training summary with statistics
- Check for overfitting and provide recommendations

## Your Recent Training Results

Based on your 50-epoch training:

### Performance Summary
- **Best Validation Spearman:** 0.4234 (achieved mid-training)
- **Final Validation Spearman:** 0.3777 (epoch 50)
- **Recall@pKd≥9:** 100.00% ✓ (Perfect!)
- **Final Training Loss:** 56.4958

### Key Observations

#### ✅ Strengths
1. **Perfect High-Affinity Detection:** Your model achieves 100% recall for strong binders (pKd ≥ 9)
   - Critical for drug discovery applications
   - No false negatives for important candidates

2. **Moderate Correlation:** Spearman ~0.38-0.42 shows the model learns binding affinity patterns

#### ⚠️ Issues Detected

1. **Overfitting:**
   - Best performance at 0.4234 (likely around epoch 20-30)
   - Dropped to 0.3777 by epoch 50
   - **12% performance degradation** from peak

2. **Training Not Converged:**
   - Loss still at 56.5 after 50 epochs
   - Model could benefit from more training OR better convergence

### Recommendations

#### 1. Implement Early Stopping
Stop training when validation performance plateaus:

```python
# Add to your training arguments
parser.add_argument('--patience', type=int, default=10,
                   help='Early stopping patience')
```

#### 2. Reduce Learning Rate Earlier
Your cosine annealing might not be aggressive enough:

```python
# Try OneCycleLR instead
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=args.lr,
    epochs=args.epochs,
    steps_per_epoch=len(train_loader)
)
```

#### 3. Increase Regularization
```python
# Try higher dropout
--dropout 0.4  # (currently 0.3)

# Or higher weight decay
--weight_decay 0.05  # (currently 0.01)
```

#### 4. Use More Validation Data
```python
# Current: 5% of validation set (quick validation)
val_df_quick = val_df.sample(frac=0.05, random_state=42)

# Try: 20% for more reliable metrics
val_df_quick = val_df.sample(frac=0.20, random_state=42)
```

## Future Training Runs

### Next Training Session
The improved script will automatically:
1. Create `training_metrics.csv` with full history
2. Save all metrics in checkpoints
3. Enable you to:
   - Plot training curves
   - Identify best epoch
   - Detect overfitting early
   - Make data-driven decisions

### Comparing Multiple Runs
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load multiple runs
run1 = pd.read_csv('run1/training_metrics.csv')
run2 = pd.read_csv('run2/training_metrics.csv')

# Compare
plt.plot(run1['epoch'], run1['val_spearman'], label='Run 1')
plt.plot(run2['epoch'], run2['val_spearman'], label='Run 2')
plt.legend()
plt.show()
```

## Troubleshooting

### CSV file not created?
Check that:
- Output directory exists and is writable
- Training script is the updated version
- No errors during initialization

### Plots look wrong?
- Ensure matplotlib is installed: `pip install matplotlib`
- Check CSV has data: `wc -l output/training_metrics.csv`
- Verify validation ran: Look for non-empty `val_spearman` column

### Want more frequent validation?
Change in training args:
```python
--validation_frequency 1  # Every epoch instead of every 2
```

## Files Modified
- `train_ultra_speed_v26.py` - Added CSV logging and enhanced checkpoints
- `visualize_training.py` - New visualization script
- `analyze_training_results.py` - Checkpoint analysis tool

## Files Created During Training
- `output/training_metrics.csv` - Epoch-by-epoch metrics
- `output/best_model.pth` - Best performing model
- `output/checkpoint_epoch.pth` - Latest epoch state
- `output/checkpoint_latest.pth` - Rolling checkpoint

## Example Workflow

```bash
# 1. Start training (metrics logged automatically)
python train_ultra_speed_v26.py --data data.csv --epochs 50

# 2. During training, monitor in another terminal
tail -f output/training_metrics.csv

# 3. After training, visualize
python visualize_training.py --csv output/training_metrics.csv

# 4. Analyze checkpoints
python analyze_training_results.py

# 5. Make decisions based on curves
# - If overfitting: Reduce epochs, add regularization
# - If underfitting: Train longer, increase capacity
# - If unstable: Reduce learning rate
```

## Next Steps

For your next training run, I recommend:

1. **Add early stopping** to prevent overfitting
2. **Increase validation frequency** to `--validation_frequency 1`
3. **Use 20% of validation set** for more stable metrics
4. **Try different learning rates** (2e-3, 3e-3, 5e-3)
5. **Monitor with visualization** during training

Would you like me to implement any of these improvements?
