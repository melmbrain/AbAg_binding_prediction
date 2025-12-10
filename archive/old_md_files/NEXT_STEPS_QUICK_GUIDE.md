# Quick Start Guide - Next Session

## 🎯 Your Goal
Improve **Recall @ pKd≥9** from **13.7% → 40-60%** by using balanced dataset

---

## ⚡ 3-Step Quick Start

### Step 1: Update Notebook (1 minute)
Open in Colab: `MyDrive/AbAg_Training_02/colab_training_v2.7.ipynb`

**Cell 11** - Change this ONE line:
```python
# OLD (imbalanced)
CSV_FILENAME = 'agab_phase2_full.csv'

# NEW (balanced) - CHANGE TO THIS:
CSV_FILENAME = 'agab_phase2_full_v2_balanced.csv'
```

### Step 2: Delete Old Checkpoints (1 minute)
Add a new cell at the top and run:
```python
import os
for ckpt in [
    '/content/drive/MyDrive/AbAg_Training_02/training_output_v2.7/checkpoint_latest.pth',
    '/content/drive/MyDrive/AbAg_Training_02/training_output_v2.7/best_model.pth'
]:
    if os.path.exists(ckpt):
        os.remove(ckpt)
        print(f"Deleted: {ckpt}")
print("Ready for balanced dataset training!")
```

### Step 3: Start Training
1. Runtime → Change runtime type → **A100 GPU**
2. Runtime → Run all (Ctrl+F9)
3. Wait for training to start

---

## 📊 What to Monitor

### Key Metric: Recall @ pKd≥9 (Your Goal!)

**Current (imbalanced data)**: 13.7%
**Expected (balanced data)**: 40-60%

**Look for in the new enhanced output**:
```
CLASSIFICATION @ pKd>=9 (HIGH AFFINITY):
  Recall:    XX.X% (how many strong binders we catch)
  Precision: XX.X% (how accurate our predictions are)
```

### Expected Progress:
- Epoch 10: Recall ~25%
- Epoch 20: Recall ~35%
- Epoch 30: Recall ~45%
- Epoch 40: Recall ~50-60% ⭐ **TARGET!**

---

## ✅ Success Signs

- Recall increasing steadily (not stuck at 13-15%)
- Predictions NOT all clustering at 7.75
- Spearman improving (0.25 → 0.55-0.65)
- Overfitting ratio < 3.0

---

## ⚠️ Warning Signs

- Recall stuck at 13-15% after 5 epochs
  - → Check: `CSV_FILENAME = 'agab_phase2_full_v2_balanced.csv'`
  - → Run Cell 11 and verify: "Filtered dataset: 121,688 samples"

- All predictions = 7.75 (regression to mean)
  - → Dataset might not have loaded correctly
  - → Check Google Drive has the balanced CSV file

- Model collapse (all predictions = 4.0)
  - → Should NOT happen with v2.7 fixes
  - → If it happens, read `V2.6_VS_V2.7_CHANGES.md`

---

## 📁 Files You Need

**On Google Drive** (`MyDrive/AbAg_Training_02/`):
- ✅ `colab_training_v2.7.ipynb` (enhanced notebook)
- ✅ `agab_phase2_full_v2_balanced.csv` (balanced dataset, 76 MB)

**Local (for reference)**:
- `SESSION_SUMMARY_2025-11-28.md` (full details)
- `DATASET_SCALING_RECOMMENDATIONS.md` (why balanced data helps)
- `V2.6_VS_V2.7_CHANGES.md` (what changed in v2.7)

---

## 🔄 If Training Crashes

Training auto-resumes from last checkpoint. Just:
1. Open Colab notebook
2. Runtime → Run all
3. Will resume from where it stopped

---

## 💡 The Key Difference

**Old dataset** (agab_phase2_full.csv):
- Weak binders: 106 (0.1%) ❌
- Model can't learn what "NOT high-affinity" looks like
- Recall stuck at 13.7%

**New dataset** (agab_phase2_full_v2_balanced.csv):
- Weak binders: 34,694 (28.5%) ✅ **327x more!**
- Model learns proper decision boundary at pKd = 9
- Expected recall: 40-60% ⭐

---

## 📊 New Enhanced Output

You'll see this after every epoch:

```
================================================================================
EPOCH 10/50 COMPLETE - Training Time: 150.2s
================================================================================

TRAINING METRICS:
  Train Loss:    0.4521
  Learning Rate: 1.00e-03

VALIDATION METRICS:
  Val Loss (RMSE): 0.8234
  MAE:             0.6012
  R2:              0.5678

CORRELATION METRICS:
  Spearman:  0.3542 <- NEW BEST!
  Pearson:   0.4123

CLASSIFICATION @ pKd>=9 (HIGH AFFINITY):
  Recall:    28.5% (how many strong binders we catch) ← WATCH THIS!
  Precision: 72.3% (how accurate our predictions are)

PREDICTION DISTRIBUTION:
  Range: [4.12, 10.75]
  Mean:  7.82 +/- 1.45

OVERFITTING CHECK:
  Val/Train Loss Ratio: 1.82x <- Good
================================================================================
```

**Focus on**: "Recall: XX.X%" under "CLASSIFICATION @ pKd>=9"
This is your main goal metric!

---

## ⏱️ Timeline

- **Training time**: ~2.5 hours per epoch
- **Expected epochs**: 40-50
- **Total time**: ~4-5 days
- **Check progress**: Every 10 epochs

---

## 🎯 Done When...

Training is complete when you achieve:
1. **Recall @ pKd≥9 ≥ 40%** ⭐ Main goal!
2. Precision @ pKd≥9 ≥ 70%
3. Spearman ≥ 0.55

Or early stopping triggers (no improvement for 15 epochs)

---

*Quick reference for continuing training with balanced dataset*
*Full details in: SESSION_SUMMARY_2025-11-28.md*
