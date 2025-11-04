# Google Colab Training Guide

**Fast GPU training for AbAg Binding Prediction**

---

## Why Use Colab?

✅ **Faster GPU**: T4 GPU (15GB VRAM) vs local RTX 2060 (6GB VRAM)
✅ **Larger batch size**: Can use 128 vs 64 locally
✅ **Estimated time**: 7-10 hours (100 epochs) vs 14 hours locally
✅ **Free your local GPU**: Use your GPU for other work while training on Colab
✅ **No setup needed**: All dependencies pre-installed

---

## Setup Steps

### Step 1: Upload Data to Google Drive

1. Open Google Drive in your browser
2. Create a folder: `AbAg_data`
3. Upload the dataset file:
   - File: `/mnt/c/Users/401-24/Desktop/AbAg_binding_prediction/external_data/merged_with_all_features.csv`
   - Size: 883 MB
   - Upload to: `Google Drive/AbAg_data/`

**Upload time:** ~5-10 minutes (depending on internet speed)

---

### Step 2: Open Notebook in Colab

**Option A: Direct Upload**
1. Go to: https://colab.research.google.com/
2. Click: File → Upload notebook
3. Upload: `colab_training.ipynb` from your project folder

**Option B: Upload to Drive First**
1. Upload `colab_training.ipynb` to your Google Drive
2. Double-click the file in Drive
3. Choose: Open with → Google Colaboratory

---

### Step 3: Enable GPU

1. In Colab, click: **Runtime** → **Change runtime type**
2. Hardware accelerator: Select **GPU** (should show T4, V100, or A100)
3. Click **Save**

---

### Step 4: Update Data Path

In the notebook, find the cell with:
```python
DRIVE_DATA_PATH = "/content/drive/MyDrive/AbAg_data/merged_with_all_features.csv"
```

Update the path if you used a different folder name.

---

### Step 5: Run All Cells

1. Click: **Runtime** → **Run all**
2. When prompted, authorize Google Drive access
3. Training will start automatically

---

## Training Options

### Quick Test (Recommended First)

In the "Training Configuration" cell, change:
```python
EPOCHS = 20  # Quick 2-hour test
```

**Time:** ~1.5-2 hours
**Purpose:** Verify everything works before full training

### Full Training

```python
EPOCHS = 100  # Full training
```

**Time:** ~7-10 hours
**Expected results:** Best performance on very strong binders

---

## What Gets Saved

All outputs are automatically saved to Google Drive:

```
Google Drive/AbAg_data/models/
├── best_model.pth                  # Best model (by validation loss)
├── final_model.pth                 # Final model with metadata
├── checkpoint_epoch_10.pth         # Checkpoints every 10 epochs
├── checkpoint_epoch_20.pth
├── ...
├── training_curves.png             # Loss curves
├── predictions_vs_targets.png      # Prediction scatter plot
├── residuals_analysis.png          # Residuals analysis
└── training_results.txt            # Performance metrics
```

---

## Monitoring Training

### Check Progress

The notebook shows:
- Progress bars for each epoch
- Train/val loss after each epoch
- Best model saves automatically
- Time per epoch (~4-5 minutes on T4)

### Expected Output

```
Epoch 1/100 [Train]: 100%|██████| 1827/1827 [04:12<00:00]
Epoch 1/100 [Val]:   100%|██████| 323/323 [00:28<00:00]
Epoch 1/100 - Train Loss: 0.4523, Val Loss: 0.4201, Time: 280.3s
  ✅ New best model saved! (val_loss: 0.4201)
```

### GPU Usage

Check GPU status by running this in a new cell:
```python
!nvidia-smi
```

Expected: ~10-12 GB VRAM usage

---

## During Training

### Keep Tab Open
- Colab requires the browser tab to stay open
- If you close it, training may pause
- Mobile browser works too!

### Session Timeout
- Free Colab: 12 hours max
- If close to timeout, checkpoints are saved every 10 epochs
- Can resume from checkpoint if needed

---

## After Training

### 1. Check Results

Results appear at the end:
```
TEST SET PERFORMANCE
============================================================
RMSE:        0.6234
MAE:         0.4512
Spearman ρ:  0.8734
Pearson r:   0.9123
R²:          0.8923
============================================================

PER-BIN PERFORMANCE:
============================================================
Bin             | Count    | RMSE     | MAE
------------------------------------------------------------
very_weak       | 1,087    | 0.8234   | 0.6123
weak            | 17,808   | 0.7123   | 0.5234
moderate        | 13,962   | 0.5891   | 0.4567
strong          | 15,819   | 0.5234   | 0.4123
very_strong     | 34       | 0.9123   | 0.7234
============================================================
```

### 2. Download Model

**From Google Drive:**
1. Go to `Google Drive/AbAg_data/models/`
2. Right-click `final_model.pth`
3. Download to your local machine

**To your project folder:**
```bash
# Move to your models directory
cp ~/Downloads/final_model.pth /mnt/c/Users/401-24/Desktop/AbAg_binding_prediction/models/
```

### 3. Use Model Locally

```python
import torch
from abag_affinity import AffinityPredictor

# Load the trained model
checkpoint = torch.load('models/final_model.pth')
predictor = AffinityPredictor()
predictor.model.load_state_dict(checkpoint['model_state_dict'])

# Check performance
print(f"Model metrics:")
print(f"  RMSE: {checkpoint['metrics']['rmse']:.4f}")
print(f"  Spearman: {checkpoint['metrics']['spearman']:.4f}")

# Make predictions
result = predictor.predict(
    antibody_heavy="EVQ...",
    antigen="KVF..."
)
```

---

## Troubleshooting

### "No GPU available"
- Check: Runtime → Change runtime type → GPU selected
- Try: Runtime → Disconnect and delete runtime → Reconnect

### "Cannot find data file"
- Check the DRIVE_DATA_PATH is correct
- Make sure you authorized Google Drive access
- Verify file uploaded to Drive

### "Out of memory"
Reduce batch size:
```python
BATCH_SIZE = 64  # Instead of 128
```

### "Session timeout"
Training interrupted? Resume from checkpoint:
```python
# In new session, add before training loop:
checkpoint = torch.load(f'{OUTPUT_DIR}/checkpoint_epoch_50.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

## Cost

**Free Tier:**
- ✅ 100 epochs: FREE (if within 12-hour limit)
- ✅ T4 GPU: FREE
- ⚠️ May disconnect after 90 min idle (keep tab open)

**Colab Pro ($10/month):**
- ✅ 24-hour sessions
- ✅ Faster GPUs (V100, A100)
- ✅ Background execution (can close tab)

---

## Expected Performance

Based on 390,757 samples (330,762 with features):

**Target Metrics:**
- Overall RMSE: 0.6-0.7
- Very strong RMSE: **0.8-1.0** (vs 2.2 before)
- Spearman ρ: >0.85

**Improvement:**
- 64% better on very strong binders (pKd > 11)
- More stable predictions across all affinity ranges

---

## Next Steps After Training

1. **Evaluate on extreme cases:**
   ```python
   # Check very strong binders
   very_strong = test_df[test_df['affinity_bin'] == 'very_strong']
   print(f"Very strong RMSE: {rmse_very_strong:.4f}")
   ```

2. **Compare with baseline:**
   - Previous model: RMSE ~2.2 on very strong
   - New model: Target RMSE <1.0

3. **If results are good:**
   - Update README with new metrics
   - Save as production model
   - Publish results

4. **If results need improvement:**
   - Try 200 epochs
   - Experiment with learning rate
   - Try different architectures

---

## Tips for Best Results

✅ **Use full dataset**: 330k samples > 205k original
✅ **Train long enough**: 100 epochs recommended
✅ **Monitor overfitting**: Val loss should decrease with train loss
✅ **Save checkpoints**: Every 10 epochs for safety
✅ **Check per-bin metrics**: Focus on very strong/weak performance

---

## Questions?

Common checks:
- Is GPU enabled? Run `!nvidia-smi`
- Is data loaded? Check "Data preparation complete" message
- Is training progressing? Loss should decrease
- Are checkpoints saving? Check Drive folder

**Everything ready! Open the notebook in Colab and click "Run all"**
