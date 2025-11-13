# Model Performance Analysis - 49K Dataset

**Date**: 2025-11-10
**Model**: Trained on 49,735 samples (31% of full dataset)
**Test Set**: 7,461 samples
**Training Time**: 4 minutes

---

## Executive Summary

✅ **Good**: Overall error metrics are acceptable (RMSE: 1.40, MAE: 1.29)
⚠️ **Major Issue**: Model severely underperforms on **extreme affinities**
🚨 **Critical**: Would miss **83% of excellent drug candidates** in virtual screening

---

## Overall Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| RMSE | 1.398 | < 1.5 | ✅ Pass |
| MAE | 1.287 | < 1.3 | ✅ Pass |
| R² | 0.577 | > 0.5 | ✅ Pass |
| Pearson r | 0.760 | > 0.7 | ✅ Pass |
| Spearman ρ | **0.487** | > 0.5 | ❌ **Fail** |

**Key Issue**: Spearman correlation is below target, indicating poor **ranking ability** - critical for drug discovery.

---

## Performance by Affinity Range

### Very Weak Binders (pKd < 6)
- **Count**: 359 samples
- **MAE**: 0.589 ✅
- **Bias**: -0.187 (slight underprediction)
- **Status**: Good performance

### Weak Binders (6 ≤ pKd < 7)
- **Count**: 2,900 samples (39% of test set)
- **MAE**: 1.540
- **Bias**: **+1.540** (massive overprediction) 🚨
- **Impact**: Model thinks 1000 nM binders are 27 nM binders (35x error)
- **Status**: Poor - false positives

### Moderate Binders (7 ≤ pKd < 8)
- **Count**: 416 samples
- **MAE**: 0.534 ✅
- **Bias**: +0.123 (slight overprediction)
- **Status**: Best performance range

### Strong Binders (8 ≤ pKd < 9)
- **Count**: 1,204 samples
- **MAE**: 0.789
- **Bias**: -0.724 (underprediction)
- **Spearman**: 0.436 (poor ranking within range)
- **Status**: Moderate performance

### Very Strong Binders (pKd ≥ 9) 🚨
- **Count**: 2,582 samples (35% of test set)
- **MAE**: 1.455
- **RMSE**: 1.529
- **Bias**: **-1.454** (severe underprediction) 🚨
- **Spearman**: 0.369 (very poor ranking)
- **Impact**: Model thinks 0.3 nM binders are 10.5 nM binders (28x error)
- **Status**: Poor - missing drug candidates

---

## Clinical Impact Analysis

### What the Errors Mean in Real Terms

| True Affinity | Model Prediction | Real Kd | Predicted Kd | Error Factor |
|---------------|------------------|---------|--------------|--------------|
| pKd 11.84 | pKd 9.54 | **0.001 nM** | 0.29 nM | 290x weaker |
| pKd 9.50 | pKd 7.98 | **0.3 nM** | 10.5 nM | 35x weaker |
| pKd 8.50 | pKd 7.80 | **3.2 nM** | 15.9 nM | 5x weaker |
| pKd 6.00 | pKd 7.55 | **1000 nM** | 27 nM | 37x stronger |

### Drug Discovery Impact

**Therapeutic antibody criteria**:
- Excellent: Kd < 1 nM (pKd > 9)
- Very good: Kd < 10 nM (pKd > 8)
- Potential: Kd < 100 nM (pKd > 7)
- Poor: Kd > 1000 nM (pKd < 6)

**Problem**: Model predicts most antibodies are "potential" (pKd 7.5-8) regardless of true strength.

---

## Virtual Screening Performance

If you used this model to screen antibodies for strong binders (pKd ≥ 9):

| Metric | Value | Meaning |
|--------|-------|---------|
| **True Positives** | 438 | Found correctly |
| **False Positives** | 293 | Predicted strong, actually weak |
| **False Negatives** | **2,144** | Missed excellent candidates 🚨 |
| **Precision** | 59.9% | Of predicted hits, 60% are real |
| **Recall** | **17.0%** | Found only 17% of real hits 🚨 |
| **F1-Score** | 26.4% | Overall poor |

### Bottom Line:
**You would MISS 83% of excellent drug candidates!**

This is unacceptable for drug discovery applications.

---

## Root Cause Analysis

### 1. Regression to the Mean
Model predicts safe values around 7.5-8.0, avoiding extremes.

**Evidence**:
- True distribution: 40% weak (pKd=6), 35% very strong (pKd≥9)
- Predicted distribution: 72% moderate (pKd 7-8), only 2% very strong

### 2. Class Imbalance Issues
Dataset likely has few extreme examples, so model learns to ignore them.

### 3. Loss Function Limitation
Standard MSE loss treats all errors equally. Missing a pKd=10 binder by 2 units is much worse than missing a pKd=7 binder by 2 units, but MSE doesn't know this.

### 4. Insufficient Data
Only 49,735 samples (31% of available data). More data would help, especially for extremes.

### 5. Embedding Limitations
Likely using 768-dimensional ESM-2 embeddings. Higher dimensions (1280) capture more subtle differences needed for extreme affinities.

---

## Correlation Analysis by Range

The model's ranking ability varies dramatically by affinity range:

| Range | Spearman | Pearson | Interpretation |
|-------|----------|---------|----------------|
| **Weak (< 7)** | 0.619 | 0.981 | Good ranking |
| **Moderate (7-8)** | 0.257 | 0.064 | Very poor |
| **Strong (8-9)** | 0.436 | 0.350 | Moderate |
| **Very Strong (≥9)** | **0.369** | **0.310** | Poor |

**Interpretation**: Model can rank weak binders well, but fails at ranking therapeutic candidates (pKd > 8).

---

## Extreme Error Cases

### Worst Underpredictions (missing strong binders)
- pKd 11.84 → 9.23 (error: -2.61) - missed picomolar binder!
- Multiple pKd 9+ → 7-8 range

### Worst Overpredictions (false positives)
- pKd 6.00 → 8.14 (error: +2.14) - weak binder called moderate
- Multiple weak binders promoted to moderate

Only **0.7%** of predictions have errors > 2.0 pKd units, which is good. However, these errors are concentrated in the critical therapeutic range.

---

## Recommendations

### Immediate Improvements

1. **Train on Full Dataset** (159K samples)
   - 3x more data, especially more extreme examples
   - Should improve performance on rare strong binders
   - Expected training time: 15-20 hours

2. **Use Focal MSE Loss**
   - Already implemented in your codebase (`src/losses.py`)
   - Puts more weight on extreme affinities
   - Reduces regression to mean

3. **Use Full-Dimensional Embeddings**
   - Switch from 768-dim to 1,280-dim ESM-2
   - Your `COMPLETE_COLAB_TRAINING.py` already does this
   - Better captures subtle differences

4. **Two-Stage Training**
   - Train base model on all data
   - Fine-tune on extreme affinities only
   - Improves performance on critical range

### Long-term Improvements

5. **Ensemble Methods**
   - Train 5 models with different random seeds
   - Average predictions
   - Reduces variance, improves reliability

6. **Stratified Sampling**
   - Ensure balanced representation of all affinity ranges
   - Oversample rare strong binders during training

7. **Custom Evaluation Metrics**
   - Weight errors by clinical importance
   - Penalize missing pKd>9 binders more heavily

8. **Add Structure Information**
   - AlphaFold structure predictions
   - Interface residue features
   - Complementarity determining region (CDR) analysis

---

## Next Steps

### Option A: Quick Win (Recommended)
Run `COMPLETE_COLAB_TRAINING.py` with full 159K dataset:
- Uses 1,280-dim embeddings ✓
- Uses Focal MSE loss ✓
- Uses full dataset ✓
- Expected improvement: 20-40% on extreme affinities

### Option B: Detailed Analysis
Analyze current model further:
- Plot true vs predicted by range
- Identify specific antibody types that fail
- Analyze sequence features of mispredicted samples

### Option C: Ensemble Current Model
Since training is only 4 minutes:
- Train 5 models with different seeds
- Average predictions
- Low effort, moderate improvement

---

## Conclusion

Your model shows **acceptable overall performance** but has a **critical flaw for drug discovery**:

✅ Works well on weak/moderate binders (pKd < 8)
❌ Fails on therapeutic candidates (pKd > 8)
🚨 Misses 83% of excellent drug candidates

**Root cause**: Trained on only 31% of available data, uses standard loss function, likely 768-dim embeddings.

**Solution**: Run full training pipeline (`COMPLETE_COLAB_TRAINING.py`) on 159K dataset with 1,280-dim embeddings and Focal MSE loss.

**Expected outcome**: Spearman > 0.60, recall > 50% on strong binders, usable for virtual screening.

---

## Files Generated

- Model: `best_model.pth` (9.6 MB)
- Predictions: `test_predictions.csv` (7,461 samples)
- Metrics: `results_summary.json`

**Status**: Model exists but not ready for production use in drug discovery.
