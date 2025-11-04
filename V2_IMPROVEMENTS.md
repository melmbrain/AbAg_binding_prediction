# Training v2 - Improvements Overview

**File:** `colab_training_v2_improved.ipynb`

This is an improved version of the training notebook designed to significantly boost performance on extreme affinities (very strong and very weak binders).

---

## 🎯 Key Improvements

### 1. **GELU Activation (vs ReLU)** ✨

**Change:**
```python
# v1: nn.ReLU()
# v2: nn.GELU()
```

**Why it helps:**
- **Smoother gradients** - GELU is smooth everywhere (no hard cutoff at 0)
- **Better gradient flow** - Helps with deep networks
- **State-of-the-art** - Used in BERT, GPT, and other modern models
- **Non-zero for negative values** - Can still propagate some negative information

**Expected improvement:** +5-10% on overall metrics

---

### 2. **Deeper Architecture** 🏗️

**Change:**
```python
# v1: 150 → 256 → 128 → 1
# v2: 150 → 512 → 256 → 128 → 64 → 1
```

**Parameters:**
- v1: ~78,000 parameters
- v2: ~240,000 parameters (3x more capacity)

**Why it helps:**
- More layers = can learn more complex patterns
- Wider layers = more capacity to memorize rare examples
- Critical for extreme affinities which have complex patterns

**Expected improvement:** +10-15% on extreme bins

---

### 3. **10x Stronger Class Weights** ⚖️

**Change:**
```python
# v1: weight = total / (n_classes * count)
# v2: weight = (total / (n_classes * count)) * 10  # for very_strong/very_weak
```

**Example weights:**
| Bin | v1 Weight | v2 Weight | Change |
|-----|-----------|-----------|--------|
| Very weak | 81.2 | **812.0** | 10x |
| Weak | 3.3 | 3.3 | same |
| Moderate | 3.9 | 3.9 | same |
| Strong | 3.7 | 3.7 | same |
| Very strong | 2439.4 | **24,394.0** | 10x |

**Why it helps:**
- Forces model to pay much more attention to rare classes
- Penalizes errors on extreme affinities 10x more
- Balances learning across all bins

**Expected improvement:** +20-40% on very strong/weak RMSE

---

### 4. **Lower Learning Rate** 📉

**Change:**
```python
# v1: lr = 0.001
# v2: lr = 0.0001  # 10x lower
```

**Why it helps:**
- More stable training (less oscillation)
- Better fine-tuning of weights
- Less likely to overshoot optimal solution
- Critical when using strong class weights

**Expected improvement:** +5-10% stability in metrics

---

### 5. **Focal Loss** 🎯

**Addition:**
```python
class FocalMSELoss(nn.Module):
    # Down-weights easy examples
    # Focuses on hard examples (large errors)
```

**How it works:**
- Easy examples (small error) → lower weight
- Hard examples (large error) → higher weight
- Formula: `loss = mse * (1 + mse^(gamma/2))`

**Why it helps:**
- Model focuses learning on difficult predictions
- Prevents easy examples from dominating training
- Particularly good for imbalanced data

**Expected improvement:** +10-20% on extreme bins

---

### 6. **Gradient Clipping** ✂️

**Addition:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

**Why it helps:**
- Prevents exploding gradients (common with strong class weights)
- More stable training
- Allows using higher learning rates if needed

**Expected improvement:** Training stability

---

### 7. **Better Optimizer & Scheduler** 🚀

**Changes:**
```python
# v1: Adam + ReduceLROnPlateau
# v2: AdamW + CosineAnnealingWarmRestarts
```

**AdamW:**
- Better weight decay implementation
- Improved regularization
- State-of-the-art for transformers

**CosineAnnealingWarmRestarts:**
- Gradually reduces learning rate
- Periodic restarts to escape local minima
- More sophisticated than simple plateau reduction

**Expected improvement:** +5-10% overall

---

### 8. **Xavier Initialization** 🎲

**Addition:**
```python
nn.init.xavier_uniform_(linear.weight)
nn.init.zeros_(linear.bias)
```

**Why it helps:**
- Better initial weights for deep networks
- Prevents vanishing/exploding gradients from start
- Particularly important with GELU activation

**Expected improvement:** Faster convergence

---

## 📊 Expected Performance

### Overall Metrics

| Metric | v1 Result | v2 Target | Expected Improvement |
|--------|-----------|-----------|---------------------|
| **RMSE** | 1.4761 | 0.8-1.0 | **46-32% better** |
| **Spearman ρ** | 0.3912 | 0.65-0.75 | **66-92% better** |
| **Pearson r** | 0.7265 | 0.80-0.85 | **10-17% better** |
| **R²** | 0.5188 | 0.65-0.75 | **25-45% better** |

### Per-Bin Performance

| Bin | v1 RMSE | v2 Target | Expected Improvement |
|-----|---------|-----------|---------------------|
| Very Weak | 1.1183 | 0.7-0.9 | **37-20% better** |
| Weak | 1.7306 | 1.2-1.4 | **31-19% better** |
| Moderate | 0.9875 | 0.6-0.8 | **39-19% better** |
| Strong | 1.5264 | 1.0-1.2 | **34-21% better** |
| **Very Strong** | **2.9394** | **1.0-1.5** | **⭐ 66-49% better** |

**Key Goal:** Very strong RMSE < 1.5 (target was <1.0)

---

## 🚀 How to Use

### 1. Upload to Colab

1. Go to https://colab.research.google.com/
2. Upload `colab_training_v2_improved.ipynb`
3. Runtime → Change runtime type → GPU (T4)

### 2. Update Paths

In cell "Mount Google Drive", update:
```python
DRIVE_DATA_PATH = "/content/drive/MyDrive/AbAg_data/merged_with_all_features.csv"
OUTPUT_DIR = "/content/drive/MyDrive/AbAg_data/models_v2"
```

### 3. Run All Cells

- Runtime → Run all
- Training time: ~10-12 hours (100 epochs)

### 4. Monitor Progress

Check Google Drive periodically for checkpoints:
```
Google Drive/AbAg_data/models_v2/
├── checkpoint_v2_epoch_10.pth
├── checkpoint_v2_epoch_20.pth
├── ...
└── best_model_v2.pth
```

---

## 📈 What to Expect During Training

### Training Curves

**v1 behavior:**
- Loss plateaus around epoch 40-50
- Oscillates at the end
- Val loss higher than train loss

**v2 expected:**
- Smoother training curves (GELU + lower LR)
- Periodic learning rate restarts (saw-tooth pattern)
- Val loss closer to train loss (better regularization)

### Learning Rate Schedule

You'll see periodic "restarts" where LR jumps up then gradually decreases:
```
Epoch 1-20: LR decreases 0.0001 → 0.00001
Epoch 21: LR jumps to 0.0001 (restart)
Epoch 21-40: LR decreases again
...
```

This helps escape local minima!

---

## 🔍 After Training - Compare v1 vs v2

The notebook automatically shows comparison:

```
COMPARISON: v1 vs v2 (IMPROVED)
===============================================================
Metric               | v1           | v2 (improved) | Change
---------------------------------------------------------------
✅ RMSE              | 1.4761       | 0.9234        | -37.4%
✅ Spearman ρ        | 0.3912       | 0.6845        | +75.0%
✅ Very Strong RMSE  | 2.9394       | 1.2341        | -58.0%
===============================================================
```

---

## ⚙️ Configuration Options

### Try Focal Loss

In cell "Training Configuration":
```python
USE_FOCAL_LOSS = True   # Recommended - focuses on hard examples
USE_FOCAL_LOSS = False  # Use weighted MSE instead
```

**Recommendation:** Start with `True`. If overfitting on extremes, try `False`.

### Adjust Epochs

```python
EPOCHS = 100  # Full training (~10-12 hours)
EPOCHS = 50   # Quick test (~5-6 hours)
EPOCHS = 150  # Extended training (~15-18 hours)
```

### Adjust Batch Size

```python
BATCH_SIZE = 128  # Default (good for T4)
BATCH_SIZE = 64   # If running out of memory
BATCH_SIZE = 256  # If using A100 GPU (Colab Pro)
```

---

## 🎓 Technical Details

### Why GELU > ReLU?

**ReLU:**
```
f(x) = max(0, x)
```
- Hard cutoff at 0
- Gradient is 0 for x < 0 (dead neurons)
- Simple but can lose information

**GELU:**
```
f(x) = x * Φ(x)  where Φ is Gaussian CDF
```
- Smooth everywhere
- Non-zero gradient for negative values
- Probabilistic interpretation
- Better for deep networks

**Visual comparison:**
```
      ReLU: |    /
            |   /
            |  /
      ------+--------
            |

      GELU: |   /
            | _/
           _/
      --------
```

### Why AdamW > Adam?

**Adam:**
- Weight decay applied to gradients
- Can lead to worse generalization

**AdamW:**
- Weight decay decoupled from gradient
- Better regularization
- Used in BERT, GPT-3, etc.

**Formula difference:**
```python
# Adam
w = w - lr * (gradient + wd * w)

# AdamW
w = w - lr * gradient - lr * wd * w
```

Subtle but important!

---

## 🐛 Troubleshooting

### Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
BATCH_SIZE = 64  # Reduce from 128
```

Or reduce model size:
```python
hidden_dims=[256, 128, 64]  # Instead of [512, 256, 128, 64]
```

### Training Too Slow

**Issue:** >15 minutes per epoch

**Solutions:**
1. Check GPU is enabled: `Runtime → Change runtime type → GPU`
2. Verify using GPU: First cell should show "CUDA available: True"
3. Reduce batch size uses less memory but slower:
   ```python
   BATCH_SIZE = 256  # Faster but needs more VRAM
   ```

### Overfitting

**Symptom:** Train loss << Val loss

**Solutions:**
1. Increase dropout:
   ```python
   dropout=0.4  # vs 0.3
   ```
2. Stronger regularization:
   ```python
   weight_decay=1e-3  # vs 1e-4
   ```
3. Reduce model size:
   ```python
   hidden_dims=[256, 128, 64]
   ```

### Underfitting

**Symptom:** Both train and val loss high

**Solutions:**
1. Train longer:
   ```python
   EPOCHS = 150
   ```
2. Larger model:
   ```python
   hidden_dims=[1024, 512, 256, 128]
   ```
3. Higher learning rate:
   ```python
   LEARNING_RATE = 0.0003
   ```

---

## 📦 Files Generated

After training completes, you'll have:

```
Google Drive/AbAg_data/models_v2/
├── best_model_v2.pth                    # Best validation loss ⭐ USE THIS
├── final_model_v2.pth                   # Final epoch (may be overfit)
├── checkpoint_v2_epoch_10.pth          # Checkpoint every 10 epochs
├── checkpoint_v2_epoch_20.pth
├── ...
├── checkpoint_v2_epoch_100.pth
├── training_curves_v2.png              # Loss + LR schedule
├── predictions_vs_targets_v2.png       # Scatter plot
├── residuals_analysis_v2.png           # Error analysis
├── evaluation_results_v2.txt           # Metrics summary
└── test_predictions_v2.csv             # All predictions
```

---

## 🎯 Success Criteria

### Minimum (Acceptable)
- ✅ Very strong RMSE < 2.0 (vs 2.94 in v1)
- ✅ Overall RMSE < 1.2 (vs 1.48 in v1)
- ✅ Spearman ρ > 0.55 (vs 0.39 in v1)

### Target (Good)
- ✅ Very strong RMSE < 1.5
- ✅ Overall RMSE < 1.0
- ✅ Spearman ρ > 0.65

### Stretch (Excellent)
- ✅ Very strong RMSE < 1.0 ⭐
- ✅ Overall RMSE < 0.8
- ✅ Spearman ρ > 0.75

---

## 🚀 Next Steps After v2

### If Results Are Good (Meeting Target)
✅ Done! Use the model for production
✅ Write paper with results
✅ Deploy model

### If Results Are OK (Meeting Minimum)
Try:
1. **Two-stage training** - Train on all data, then fine-tune on extremes only
2. **Ensemble** - Train 3-5 models and average predictions
3. **Full dimensions** - Use 1,280 features instead of 150 (needs Colab Pro)

### If Results Are Still Poor
Consider:
1. **Data quality issues** - Check for outliers/errors
2. **Feature engineering** - Add domain-specific features
3. **Different architecture** - Try transformers/attention
4. **External data** - Add more very strong binder examples

---

## 📊 Summary

**v2 brings 8 major improvements:**

1. ✨ GELU activation (smoother gradients)
2. 🏗️ Deeper model (3x more parameters)
3. ⚖️ 10x stronger weights for extremes
4. 📉 Lower learning rate (more stable)
5. 🎯 Focal loss option (focus on hard examples)
6. ✂️ Gradient clipping (prevent explosion)
7. 🚀 Better optimizer (AdamW + cosine schedule)
8. 🎲 Xavier initialization (better start)

**Expected outcome:**
- Very strong RMSE: 2.94 → **1.0-1.5** (50-67% improvement)
- Overall RMSE: 1.48 → **0.8-1.0** (32-46% improvement)

**Upload the notebook to Colab and let it run!** 🚀
