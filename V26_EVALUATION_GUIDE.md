# 📊 Model v2.6 Evaluation Guide

## Overview

Your pre-trained v2.6 model (100 epochs, pre-optimization) is ready for evaluation!

**Files Ready:**
- ✅ `best_model.pth` (4.7GB) - Your trained model
- ✅ `evaluate_v26_model.py` (21KB) - Evaluation script

**Missing:**
- ❌ `agab_phase2_full.csv` - Dataset needed for evaluation

---

## 🚀 How to Run Evaluation

### Step 1: Download Dataset from Google Drive

1. Go to your Google Drive
2. Navigate to `AbAg_Training_02/` folder
3. Download `agab_phase2_full.csv`
4. Place it in: `C:\Users\401-24\Desktop\AbAg_binding_prediction\`

The file should be in the same directory as `best_model.pth`.

---

### Step 2: Run Evaluation Script

Open terminal/command prompt in the project directory and run:

```bash
python evaluate_v26_model.py
```

---

## 📈 What the Script Does

### 1. Loads Your Model
```
Loading model v2.6 architecture...
Loading checkpoint from: best_model.pth
✓ Model loaded successfully
```

### 2. Evaluates on Validation Set
```
Validation Set Evaluation
==================================================
Evaluating: 100%|████████████████| 875/875 [05:23<00:00, 2.71batch/s]

Validation Metrics:
--------------------------------------------------
  Spearman Correlation: 0.4234 ± 0.0021
  RMSE: 1.2345 pKd units
  MAE: 0.8765 pKd units
  R² Score: 0.6543
  Pearson Correlation: 0.4456

  Classification (pKd ≥ 9):
    Recall: 98.50%
    Precision: 87.23%
    F1-Score: 92.56%
    Specificity: 95.67%
```

### 3. Evaluates on Test Set (TRUE PERFORMANCE)
```
Test Set Evaluation (UNSEEN DATA)
==================================================
Evaluating: 100%|████████████████| 875/875 [05:21<00:00, 2.72batch/s]

Test Metrics:
--------------------------------------------------
  Spearman Correlation: 0.4123 ← YOUR TRUE PERFORMANCE
  RMSE: 1.2567 pKd units
  MAE: 0.8934 pKd units
  R² Score: 0.6421
  ...
```

### 4. Saves Results

All results are saved to `evaluation_v26_output/`:

```
evaluation_v26_output/
├── val_predictions_v26.csv         # All validation predictions
├── test_predictions_v26.csv        # All test predictions
├── evaluation_metrics_v26.json     # Complete metrics JSON
├── predictions_scatter_v26.png     # Scatter plots (val + test)
├── error_distribution_v26.png      # Error histogram
└── mae_by_pkd_range_v26.png       # MAE by pKd range
```

---

## 📊 Understanding Results

### Validation Metrics
- Measures performance on validation set (used during training)
- May be slightly optimistic (model saw this during training)

### Test Metrics ← **MOST IMPORTANT**
- Measures performance on completely unseen data
- **This is your true model performance**
- Use this for publications/reporting

### Key Metrics Explained

**Spearman Correlation** (most important for ranking):
- 0.40-0.45: Good performance
- 0.45-0.50: Excellent performance
- >0.50: Outstanding performance

**RMSE** (prediction accuracy):
- <1.5 pKd units: Good
- <1.3 pKd units: Very good
- <1.1 pKd units: Excellent

**Recall@pKd≥9** (strong binder detection):
- >95%: Good at finding strong binders
- >98%: Excellent sensitivity

---

## 🔍 Analyzing Results

### Check Predictions CSV
```python
import pandas as pd

# Load test predictions
test_results = pd.read_csv('evaluation_v26_output/test_predictions_v26.csv')

# View worst predictions
worst = test_results.nlargest(10, 'absolute_error')
print(worst[['antibody_sequence', 'antigen_sequence', 'true_pKd', 'predicted_pKd', 'absolute_error']])

# View best predictions
best = test_results.nsmallest(10, 'absolute_error')
print(best[['antibody_sequence', 'antigen_sequence', 'true_pKd', 'predicted_pKd', 'absolute_error']])
```

### Check Metrics JSON
```python
import json

with open('evaluation_v26_output/evaluation_metrics_v26.json') as f:
    metrics = json.load(f)

print(f"Model: {metrics['model_version']}")
print(f"Test Spearman: {metrics['test']['spearman']:.4f}")
print(f"Test RMSE: {metrics['test']['rmse']:.4f}")
print(f"Strong Binder Recall: {metrics['test']['recall_pkd9']:.2f}%")
```

### View Visualizations

1. **predictions_scatter_v26.png**
   - Left plot: Validation predictions vs actual
   - Right plot: Test predictions vs actual
   - Diagonal line = perfect predictions
   - Check how close points are to the line

2. **error_distribution_v26.png**
   - Histogram of prediction errors
   - Should be centered around 0 (no bias)
   - Narrower distribution = better predictions

3. **mae_by_pkd_range_v26.png**
   - Shows which pKd ranges are hardest to predict
   - Helps identify model strengths/weaknesses

---

## 💡 Expected Performance (v2.6)

Based on similar models, you should expect:

**Test Set:**
- Spearman: **0.38-0.43**
- RMSE: **1.2-1.5** pKd units
- Recall@pKd≥9: **95-99%**

If your results are in this range, your model is performing well!

---

## 🔄 Comparing with New Notebooks

After evaluating v2.6, you can train new models and compare:

### 1. Train with Standard Notebook (T4/V100)
```
File: notebooks/colab_training_GDRIVE.ipynb
Expected: Spearman 0.40-0.43 (similar to v2.6)
```

### 2. Train with A100 + ESM-2 3B (Recommended!)
```
File: notebooks/colab_training_A100_ESM2_3B.ipynb
Expected: Spearman 0.42-0.47 (+0.02-0.05 improvement)
Training time: ~40 minutes (vs 21 hours for v2.6)
```

### Comparison Template
```python
import json

# Load v2.6 results
with open('evaluation_v26_output/evaluation_metrics_v26.json') as f:
    v26 = json.load(f)

# Load new model results (after training)
with open('/path/to/new/training_output/final_metrics.json') as f:
    new = json.load(f)

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"\nv2.6 (100 epochs, pre-optimization):")
print(f"  Test Spearman: {v26['test']['spearman']:.4f}")
print(f"  Test RMSE: {v26['test']['rmse']:.4f}")
print(f"  Training time: ~21 hours")

print(f"\nNew Model:")
print(f"  Test Spearman: {new['test']['spearman']:.4f}")
print(f"  Test RMSE: {new['test']['rmse']:.4f}")

improvement = new['test']['spearman'] - v26['test']['spearman']
print(f"\nImprovement: {improvement:+.4f} Spearman")
```

---

## ⚙️ System Requirements

**For Running Evaluation:**
- GPU: NVIDIA GPU with CUDA (8GB+ VRAM recommended)
  - T4: ✅ Works (free Colab)
  - V100: ✅ Works well
  - A100: ✅ Works great
- RAM: 16GB+
- Storage: ~10GB (model + data + results)
- Python: 3.8+
- PyTorch: 2.0+

**Estimated Time:**
- Model loading: ~30-60 seconds
- Validation evaluation: ~5-10 minutes
- Test evaluation: ~5-10 minutes
- Total: **~10-20 minutes**

---

## 🔧 Troubleshooting

### Error: "No module named 'transformers'"
```bash
pip install transformers torch pandas scikit-learn scipy matplotlib tqdm
```

### Error: "CUDA out of memory"
Reduce batch size in the script:
```python
# In evaluate_v26_model.py, line 33:
'batch_size': 8,  # Reduced from 16
```

### Error: "agab_phase2_full.csv not found"
- Make sure CSV is in the same directory as the script
- Check filename is exactly `agab_phase2_full.csv` (case-sensitive)

### Evaluation is very slow
- Make sure you're using GPU: Check `device: cuda` in output
- If using CPU, expect 10× slower (1-2 hours)

---

## 📝 Summary

**To evaluate your v2.6 model:**

1. ✅ Download `agab_phase2_full.csv` from Google Drive
2. ✅ Place it in project directory
3. ✅ Run: `python evaluate_v26_model.py`
4. ✅ Wait 10-20 minutes
5. ✅ Check results in `evaluation_v26_output/`

**Then compare with new models:**
- Train using `colab_training_GDRIVE.ipynb` (T4/V100)
- Or train using `colab_training_A100_ESM2_3B.ipynb` (A100) ← Recommended!
- Compare results to see improvements

---

## 🎯 Next Steps

1. **Evaluate v2.6** (this guide)
2. **Train new model** with one of the new notebooks
3. **Compare results** to measure improvements
4. **Choose best model** for your use case

---

**Ready to evaluate! 🚀**

Just download the CSV and run the script. Your v2.6 model evaluation will complete in ~10-20 minutes.
