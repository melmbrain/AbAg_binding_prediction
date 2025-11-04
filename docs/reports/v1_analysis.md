# Colab Training Results - Analysis Report

**Date:** 2025-11-04
**Training:** 100 epochs on Google Colab T4 GPU
**Status:** ⚠️ COMPLETED - Performance Below Target

---

## 📊 Performance Summary

### Overall Metrics

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| **RMSE** | 1.4761 | <0.7 | ❌ 111% worse |
| **MAE** | 1.3011 | <0.5 | ❌ 160% worse |
| **Spearman ρ** | 0.3912 | >0.85 | ❌ 54% lower |
| **Pearson r** | 0.7265 | >0.90 | ❌ 19% lower |
| **R²** | 0.5188 | >0.85 | ❌ 39% lower |

**Conclusion:** Performance is significantly below target across all metrics.

---

### Per-Bin Performance

| Bin | Count | RMSE | MAE | Target RMSE | Status |
|-----|-------|------|-----|-------------|--------|
| Very Weak (<5) | 900 | 1.1183 | 0.5297 | <1.2 | ✅ Good |
| Weak (5-7) | 17,904 | 1.7306 | 1.6679 | <1.0 | ❌ 73% worse |
| Moderate (7-9) | 13,822 | 0.9875 | 0.6913 | <0.8 | ❌ 23% worse |
| Strong (9-11) | 15,809 | 1.5264 | 1.4633 | <0.8 | ❌ 91% worse |
| **Very Strong (>11)** | **50** | **2.9394** | **2.0261** | **<1.0** | **❌ 194% worse** |

**Key Issue:** Very strong binders have the worst performance (RMSE 2.94 vs target <1.0).

---

## 🔍 Diagnostic Analysis

### 1. Data Distribution

**Test Set:**
- Total samples: 49,615
- Very strong binders: Only **50 samples** (0.10%)

**Problem:** Extreme class imbalance - very strong binders are underrepresented.

### 2. Prediction Statistics

| Metric | True pKd | Predicted pKd |
|--------|----------|---------------|
| Mean | 7.5262 | 7.6911 |
| Std | 2.1279 | 1.6620 |
| Min | -4.1007 | -0.1260 |
| Max | 15.6990 | 20.9290 |
| Range | 19.80 | 21.06 |

**Analysis:**
- ✅ Prediction range is reasonable (1.06x true range)
- ❌ Predictions have lower std (1.66 vs 2.13) - model is underconfident on extremes
- ⚠️ Model predicts some values outside training range (max 20.9 vs true max 15.7)

### 3. Model Behavior

Looking at sample predictions:
```
True: 6.81 → Pred: 7.80 (error: +0.99)
True: 6.00 → Pred: 7.17 (error: +1.17)
True: 9.48 → Pred: 7.78 (error: -1.70)
True: 9.37 → Pred: 7.97 (error: -1.40)
```

**Pattern:** Model tends to predict values closer to the mean (~7.5-8.0), underestimating strong binders and overestimating weak ones.

---

## ❓ What Went Wrong?

### Possible Causes

1. **Class Imbalance Not Properly Handled**
   - Despite using weighted loss, the model still struggles with rare bins
   - Only 50 very strong binders in test set - not enough to learn from

2. **Feature Quality**
   - PCA reduced from 1,280 → 150 dimensions (99.9% variance)
   - But some critical information for extreme affinities may be lost
   - Need to check if full 1,280 dimensions perform better

3. **Model Capacity**
   - Current: 150 → 256 → 128 → 1
   - May be too simple for this complex task
   - Consider deeper/wider architecture

4. **Training Configuration**
   - Batch size 128 may be too large for rare classes
   - Learning rate may need tuning
   - May need more epochs or different optimizer

5. **Data Quality**
   - Check if the merged dataset has issues
   - Verify that samples with PCA features are representative
   - Only 84.65% of data has features (330k/390k)

---

## 🎯 Next Steps - Recommendations

### Option 1: Quick Fixes (Try First)

**1.1. Use Full-Dimensional Features (No PCA)**
```bash
# Train with original 1,280-dim embeddings (will need Colab Pro or local with more RAM)
# Expected improvement: +5-15% on extreme bins
```

**1.2. Increase Model Capacity**
```python
# Deeper network
hidden_dims=[512, 256, 128, 64]  # vs current [256, 128]
# Expected improvement: +10-20%
```

**1.3. Stronger Class Weights**
```python
# Increase penalty for rare classes
bin_weights['very_strong'] *= 5  # Much stronger weight
# Expected improvement: +20-40% on very strong
```

**1.4. Focal Loss with Higher Gamma**
```python
# Use focal loss instead of weighted MSE
gamma=3.0  # Down-weight easy examples even more
# Expected improvement: +15-25%
```

---

### Option 2: Data Improvements

**2.1. Filter to Only Samples with Features**
- Currently using 330k/390k (84.65%)
- The 60k without features may be creating noise
- Try training on only the 330k with complete features

**2.2. Oversample Very Strong Binders**
```python
# During training, sample very strong binders more frequently
# Duplicate very strong samples 5-10x in each epoch
```

**2.3. Two-Stage Training**
```python
# Stage 1: Train on all data (100 epochs)
# Stage 2: Fine-tune on extreme bins only (50 epochs)
# Focus final training on the hard cases
```

---

### Option 3: Different Architecture

**3.1. Ensemble Model**
- Train multiple models
- Average predictions
- Usually improves by 10-20%

**3.2. Multi-Task Learning**
- Predict both pKd and affinity bin simultaneously
- Forces model to learn categorical distinctions

**3.3. Attention Mechanism**
- Add attention layers to focus on important features
- May help identify critical patterns for extreme affinities

---

## 🚀 Immediate Action Plan

### Step 1: Verify the Data (Quick - 5 min)

Check if the PCA features are the issue:

```bash
python.exe -c "
import pandas as pd
import numpy as np

df = pd.read_csv('external_data/merged_with_all_features.csv', low_memory=False)
pca_cols = [f'esm2_pca_{i}' for i in range(150)]

# Check feature quality
df_with_features = df[df[pca_cols[0]].notna()]
print(f'Samples with features: {len(df_with_features):,}')

# Check very strong distribution
very_strong = df_with_features[df_with_features['pKd'] > 11]
print(f'Very strong binders: {len(very_strong):,}')
print(f'Very strong percentage: {len(very_strong)/len(df_with_features)*100:.3f}%')

# Check feature variance
features = df_with_features[pca_cols].values
print(f'\\nFeature statistics:')
print(f'  Mean: {features.mean():.6f}')
print(f'  Std: {features.std():.6f}')
print(f'  Min: {features.min():.6f}')
print(f'  Max: {features.max():.6f}')
"
```

---

### Step 2: Try Full Dimensions (Recommended)

The PCA compression might be losing critical information. Train with full 1,280 dimensions:

**Requirements:**
- Colab Pro ($10/month) for more RAM
- OR local training if you have 16GB+ RAM

**Expected:**
- Training time: +50% longer
- Performance: +10-30% better on extremes

---

### Step 3: Retrain with Better Configuration

Create an improved training notebook with:

1. **Stronger class weights:**
   ```python
   # Exponential weighting instead of linear
   for label in BIN_LABELS:
       count = bin_counts.get(label, 1)
       bin_weights[label] = (total_samples / count) ** 1.5  # Exponential penalty
   ```

2. **Focal loss:**
   ```python
   class FocalLoss(nn.Module):
       def __init__(self, alpha=1.0, gamma=2.0):
           # Focuses on hard examples
   ```

3. **Deeper model:**
   ```python
   hidden_dims=[512, 256, 128, 64]  # vs [256, 128]
   ```

4. **Lower learning rate:**
   ```python
   lr = 0.0001  # vs 0.001 (slower but more stable)
   ```

---

## 📁 Files Organized

Your results have been organized:

```
AbAg_binding_prediction/
├── models/
│   └── best_model.pth              ✅ Your trained model
├── results/
│   ├── evaluation_results.txt       ✅ Performance metrics
│   ├── test_predictions.csv         ✅ All predictions
│   ├── predictions_vs_targets.png   ✅ Scatter plot
│   ├── residuals_analysis.png       ✅ Residual analysis
│   ├── per_bin_analysis.png         ✅ Per-bin performance
│   └── training_curves.png          ✅ Loss curves
└── colab result/                    (original download)
```

---

## 🎓 What You Learned

✅ **Training completed successfully** - 100 epochs, no errors
✅ **Model saved properly** - Checkpoints every 10 epochs
✅ **Evaluation working** - All metrics calculated
❌ **Performance below target** - Need improvements

**This is normal!** First training rarely achieves target performance. Now you have:
1. A baseline model (RMSE 1.48)
2. Clear diagnostics of what's wrong
3. Multiple paths to improvement

---

## 💡 Recommended Next Action

**Try this immediately (30 min):**

1. **Retrain with stronger class weights and deeper model**
   - Modify training notebook
   - Change 3 parameters:
     - `hidden_dims=[512, 256, 128, 64]`
     - Multiply `very_strong` weight by 10
     - Reduce `lr=0.0001`
   - Retrain on Colab (~8-10 hours)

**Expected result:** RMSE on very strong: 2.94 → 1.5-2.0 (still not target, but 50% better)

**If that doesn't work:**
2. Try full-dimensional features (1,280 dims)
3. Consider two-stage training
4. Or try ensemble of models

---

## Summary

**Current Model:**
- ✅ Trains successfully
- ✅ Makes reasonable predictions
- ❌ Poor on extreme affinities (very strong/weak)
- ❌ Below target on all metrics

**Root Cause:** Class imbalance + possible information loss from PCA

**Solution:** Stronger regularization + better architecture + possibly full dimensions

**Next:** Retrain with improved config (see recommendations above)

Your training infrastructure is working perfectly - now it's about tuning the model!
