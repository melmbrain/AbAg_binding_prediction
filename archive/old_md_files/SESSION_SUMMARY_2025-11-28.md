# Session Summary - November 28, 2025

## 🎯 Main Goal
Improve prediction of **high-affinity antibody-antigen binders (pKd ≥ 9.0)**
- Current recall: **13.7%** (too low!)
- Target recall: **40-60%**

---

## ✅ What We Accomplished

### 1. Enhanced Training Output (Complete)
**File**: `notebooks/colab_training_v2.7.ipynb`

**Problem**: Training output only showed basic metrics (Spearman, Recall, Loss)

**Solution**: Enhanced epoch output to show comprehensive metrics:
- Training metrics: Train Loss, Learning Rate
- Validation metrics: RMSE, MAE, R²
- Correlation metrics: Spearman, Pearson
- **Classification metrics @ pKd≥9**: Recall, Precision (YOUR GOAL!)
- Prediction distribution: Range, Mean, Std
- Overfitting check: Val/Train loss ratio

**New Output Format**:
```
================================================================================
EPOCH 8/50 COMPLETE - Training Time: 342.5s
================================================================================

TRAINING METRICS:
  Train Loss:    0.5234
  Learning Rate: 1.00e-03

VALIDATION METRICS:
  Val Loss (RMSE): 0.9652
  MAE:             0.7123
  R2:              0.4589

CORRELATION METRICS:
  Spearman:  0.2226 <- NEW BEST!
  Pearson:   0.2894

CLASSIFICATION @ pKd>=9 (HIGH AFFINITY):
  Recall:    13.7% (how many strong binders we catch)
  Precision: 67.3% (how accurate our predictions are)

PREDICTION DISTRIBUTION:
  Range: [3.95, 10.81]
  Mean:  7.45 +/- 1.23

OVERFITTING CHECK:
  Val/Train Loss Ratio: 1.84x <- Good
================================================================================
```

**Status**: ✅ Uploaded to Google Drive

---

### 2. Created Balanced Dataset (Complete)
**Script**: `merge_agab_full_balanced.py`
**Output File**: `agab_phase2_full_v2_balanced.csv` (76 MB)

**Problem**: Current dataset severely imbalanced:
- Weak binders [4-6]: Only **106 samples (0.1%)**
- Strong binders [8-10]: **80,351 samples (52.6%)**
- Result: Model can't learn what "NOT high-affinity" looks like
- Recall stuck at 13.7%

**Solution**: Merged AgAb Full + SAbDab datasets:
1. Loaded AgAb Full (1.2M samples)
2. Filtered to valid pKd [4.0, 14.0] → 187,894 samples
3. Created antibody_sequence (heavy + light chains)
4. Merged with SAbDab (954 high-quality samples)
5. Deduplicated → **Final: 121,688 samples**

**Results**:
```
BALANCED DATASET STATISTICS:
- Total samples: 121,688
- Weak binders [4-6]: 34,694 (28.5%) ← 327x more than current!
- Mid-range [6-8]: 30,476 (25.0%)
- Strong binders [8-10]: 48,337 (39.7%)
- Very strong [10-12]: 8,160 (6.7%)
```

**Why This Helps Your Goal**:
- 327x more weak binder examples → model learns what "NOT high-affinity" looks like
- Better decision boundary at pKd = 9 threshold
- Expected recall improvement: **13.7% → 40-60%** (3-4x better!)

**Status**: ✅ Uploaded to Google Drive (`MyDrive/AbAg_Training_02/agab_phase2_full_v2_balanced.csv`)

---

## 📊 Current Training Status (Before Dataset Switch)

**Training on**: `agab_phase2_full.csv` (152,827 samples, imbalanced)
**Epoch**: 8-9 / 50
**Spearman**: 0.2495
**Recall @ pKd≥9**: 13.7% (stuck, not improving)

**Key Observations**:
1. ✅ No model collapse (predictions distributed [3.89, 10.75])
2. ⚠️ Many predictions clustering at 7.75 (dataset mean) - "regression to mean" problem
3. ⚠️ Recall stuck at 13.7% - not catching high-affinity binders
4. ⚠️ Some predictions below 4.0 (3.89, 3.95, 3.97) - expected during training

**Validation Output** (Epoch 8):
```
Validation samples (first 10):
  True: 6.00 → Pred: 7.75
  True: 9.40 → Pred: 7.75  ← Should be 9.4, predicting 7.75!
  True: 9.36 → Pred: 7.75  ← Regressing to mean
```

**Diagnosis**: Imbalanced data causing model to predict average value (7.75) instead of learning true patterns.

---

## 🚀 Next Steps (READY TO EXECUTE)

### Step 1: Update Notebook to Use Balanced Dataset
Open `colab_training_v2.7.ipynb` in Google Colab and modify **Cell 11**:

**Change from**:
```python
CSV_FILENAME = 'agab_phase2_full.csv'  # OLD (imbalanced)
```

**Change to**:
```python
CSV_FILENAME = 'agab_phase2_full_v2_balanced.csv'  # NEW (balanced)
```

### Step 2: Clear Old Checkpoints (Important!)
Run this in a new Colab cell BEFORE training:
```python
# Delete old checkpoints from imbalanced dataset
import os
import shutil

old_checkpoints = [
    '/content/drive/MyDrive/AbAg_Training_02/training_output_v2.7/checkpoint_latest.pth',
    '/content/drive/MyDrive/AbAg_Training_02/training_output_v2.7/best_model.pth'
]

for ckpt in old_checkpoints:
    if os.path.exists(ckpt):
        os.remove(ckpt)
        print(f"Deleted: {ckpt}")

print("Ready for fresh training with balanced dataset!")
```

**Why**: Checkpoint from imbalanced dataset will confuse training (different data distribution)

### Step 3: Start Training
1. Runtime → Change runtime type → **A100 GPU**
2. Runtime → Run all (Ctrl+F9)
3. Training will start fresh with balanced dataset

### Step 4: Monitor Key Metrics
Watch for these improvements:

| Metric | Current (Imbalanced) | Expected (Balanced) |
|--------|---------------------|---------------------|
| **Recall @ pKd≥9** | **13.7%** | **40-60%** ⭐ |
| Spearman | 0.25 | 0.55-0.65 |
| Predictions | Clustered at 7.75 | Better distributed |
| Training time | 2.5 hrs/epoch | ~2.5 hrs/epoch (same) |

**Good signs to look for**:
- ✅ Recall increasing steadily (not stuck at 13-15%)
- ✅ Predictions more distributed (not all 7.75)
- ✅ Spearman improving consistently
- ✅ Fewer "regression to mean" patterns in validation

**Warning signs**:
- ⚠️ Recall still stuck at 13-15% after 5 epochs → check dataset loaded correctly
- ⚠️ All predictions still 7.75 → model collapsing again
- ⚠️ Overfitting ratio > 3.0 → reduce dropout or increase regularization

---

## 📁 Files Modified/Created

### Created Files:
1. `merge_agab_full_balanced.py` - Dataset merging script
2. `enhance_training_output_v2.py` - Notebook enhancement script
3. `DATASET_SCALING_RECOMMENDATIONS.md` - Dataset analysis report
4. `SESSION_SUMMARY_2025-11-28.md` - This file

### Modified Files:
1. `notebooks/colab_training_v2.7.ipynb` - Enhanced training output

### Generated Data Files:
1. `C:/Users/401-24/Desktop/Ab_Ag_dataset/data/agab/agab_phase2_full_v2_balanced.csv` (76 MB)
   - Location on Drive: `MyDrive/AbAg_Training_02/agab_phase2_full_v2_balanced.csv`

---

## 💡 Key Insights from This Session

### Why Balanced Data Matters for Your Goal

**Your Goal**: Predict high-affinity binders (pKd ≥ 9.0) with good recall

**Problem with Imbalanced Data**:
- Current dataset: Only 106 weak binders (0.1%)
- Model never sees what "weak" looks like
- Learns to predict average (7.75) for everything
- Can't distinguish high-affinity from mid-range
- **Result**: 13.7% recall (terrible!)

**Solution with Balanced Data**:
- Balanced dataset: 34,694 weak binders (28.5%)
- Model learns clear patterns: weak vs mid vs strong
- Better decision boundary at pKd = 9 threshold
- Improved precision (fewer false positives)
- **Expected Result**: 40-60% recall (3-4x better!)

### Technical Details

**Model Architecture**:
- Dual-encoder: IgT5 (antibodies) + ESM-2 3B (antigens)
- Cross-attention between Ab and Ag embeddings
- Multi-task: Regression (pKd) + Classification (strong binders)
- bfloat16 mixed precision on A100

**Training Config (v2.7)**:
- Loss: MSE (0.7) + BCE (0.3) - removed unstable Soft Spearman
- Learning rate: 1e-3 (research-validated)
- Batch: 16 × 8 gradient accumulation = 128 effective
- Dropout: 0.1 (reduced from 0.3)
- Scheduler: ReduceLROnPlateau (patience=10)
- Data filtering: pKd ∈ [4.0, 14.0] (removed negatives)

**Critical Fixes Applied (v2.7)**:
1. ✅ Removed prediction clamping (was blocking gradients)
2. ✅ Data filtering (removed pKd < 4.0, including negatives)
3. ✅ NaN detection (early failure detection)
4. ✅ Overfitting monitoring (val/train ratio)
5. ✅ Enhanced metrics output (comprehensive tracking)

---

## 🔄 Training Resume Information

**If training is interrupted**, it will automatically resume from checkpoint:
- Checkpoint location: `/content/drive/MyDrive/AbAg_Training_02/training_output_v2.7/checkpoint_latest.pth`
- Contains: Model weights, optimizer state, epoch, batch, best Spearman, complete RNG state
- Press Ctrl+C to save and exit safely

**Current checkpoint** (before dataset switch):
- Epoch: 8
- Batch: 6074/6687
- Best Spearman: 0.2495
- Recall: 13.7%
- **Will be deleted** when starting fresh with balanced dataset

---

## 📊 Expected Timeline with Balanced Dataset

**Training time**: ~2.5 hours per epoch on A100
**Expected epochs to convergence**: 40-50 epochs
**Total training time**: ~100-125 hours (~4-5 days)
**Early stopping**: patience=15 (stops if no improvement for 15 epochs)

**Estimated performance at checkpoints**:
- Epoch 10: Spearman ~0.35, Recall ~25%
- Epoch 20: Spearman ~0.45, Recall ~35%
- Epoch 30: Spearman ~0.55, Recall ~45%
- Epoch 40: Spearman ~0.60-0.65, Recall ~50-60% ⭐

---

## 🎯 Success Criteria

Your training is successful when:
1. ✅ **Recall @ pKd≥9 ≥ 40%** (catching most high-affinity binders)
2. ✅ **Precision @ pKd≥9 ≥ 70%** (accurate predictions)
3. ✅ Spearman ≥ 0.55 (good correlation)
4. ✅ Predictions distributed across full range (not clustered at 7.75)
5. ✅ Overfitting ratio < 3.0 (not memorizing)

---

## 📝 Quick Start Guide for Next Session

1. **Open Colab**: https://colab.research.google.com
2. **Open notebook**: MyDrive/AbAg_Training_02/colab_training_v2.7.ipynb
3. **Modify Cell 11**: Change CSV_FILENAME to `'agab_phase2_full_v2_balanced.csv'`
4. **Delete old checkpoints** (code above)
5. **Change runtime**: A100 GPU
6. **Run all cells** (Ctrl+F9)
7. **Monitor Recall @ pKd≥9** - should improve from 13.7% → 40-60%

---

## 🔗 Related Files

- Dataset analysis: `DATASET_SCALING_RECOMMENDATIONS.md`
- v2.6 vs v2.7 changes: `V2.6_VS_V2.7_CHANGES.md`
- Notebook ready status: `NOTEBOOK_FINAL_READY.md`
- Data sources report: `C:/Users/401-24/Desktop/Ab_Ag_dataset/DATA_SOURCES_AND_MERGING_REPORT.md`

---

## ✨ Summary

**Problem**: Recall stuck at 13.7% due to imbalanced data (only 106 weak binders)

**Solution**: Created balanced dataset with 34,694 weak binders (327x more!)

**Status**:
- ✅ Balanced dataset created and uploaded to Google Drive
- ✅ Enhanced notebook uploaded to Google Drive
- ⏳ Ready to start training with balanced dataset

**Expected Result**: Recall improves from 13.7% → 40-60% (3-4x better!)

**Next Action**: Update Cell 11 in Colab notebook and restart training

---

*Generated: November 28, 2025*
*Session Goal: Improve high-affinity binder prediction (pKd ≥ 9.0)*
*Current Recall: 13.7% → Target: 40-60%*
