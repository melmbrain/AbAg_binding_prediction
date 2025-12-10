# Optimized Training Configurations

Based on your 50-epoch training analysis, here are recommended configurations for different scenarios:

## 🏆 Configuration 1: BALANCED (Recommended)
**Goal:** Best performance with early stopping to prevent overfitting

```bash
python train_ultra_speed_v26.py \
  --data agab_phase2_full.csv \
  --output_dir output_balanced \
  --epochs 50 \
  --batch_size 16 \
  --accumulation_steps 3 \
  --lr 3e-3 \
  --weight_decay 0.02 \
  --dropout 0.35 \
  --validation_frequency 1 \
  --use_early_stopping True \
  --early_stopping_patience 10
```

**Expected:**
- Spearman: 0.42-0.45
- Training time: 1.5-2.5 hours
- Automatically stops when overfitting starts

---

## ⚡ Configuration 2: FAST EXPERIMENTAL
**Goal:** Quick iteration for hyperparameter tuning

```bash
python train_ultra_speed_v26.py \
  --data agab_phase2_full.csv \
  --output_dir output_fast \
  --epochs 30 \
  --batch_size 20 \
  --accumulation_steps 2 \
  --lr 5e-3 \
  --weight_decay 0.01 \
  --dropout 0.3 \
  --validation_frequency 1 \
  --use_early_stopping True \
  --early_stopping_patience 5
```

**Expected:**
- Spearman: 0.35-0.40
- Training time: <1 hour
- Good for testing changes quickly

---

## 🎯 Configuration 3: CONSERVATIVE (Best Generalization)
**Goal:** Maximum generalization, lower risk of overfitting

```bash
python train_ultra_speed_v26.py \
  --data agab_phase2_full.csv \
  --output_dir output_conservative \
  --epochs 50 \
  --batch_size 12 \
  --accumulation_steps 4 \
  --lr 2e-3 \
  --weight_decay 0.05 \
  --dropout 0.4 \
  --validation_frequency 1 \
  --use_early_stopping True \
  --early_stopping_patience 15
```

**Expected:**
- Spearman: 0.40-0.44
- Training time: 2-3 hours
- More stable, better for production

---

## 🔥 Configuration 4: AGGRESSIVE (Maximum Performance)
**Goal:** Push for highest possible Spearman

```bash
python train_ultra_speed_v26.py \
  --data agab_phase2_full.csv \
  --output_dir output_aggressive \
  --epochs 100 \
  --batch_size 16 \
  --accumulation_steps 3 \
  --lr 4e-3 \
  --weight_decay 0.01 \
  --dropout 0.25 \
  --validation_frequency 1 \
  --use_early_stopping True \
  --early_stopping_patience 12
```

**Expected:**
- Spearman: 0.43-0.48
- Training time: 2-4 hours (early stopping will kick in)
- Higher risk but potentially better results

---

## 📊 Configuration Comparison

| Config | LR | Dropout | Weight Decay | Patience | Expected Spearman | Time |
|--------|-------|---------|--------------|----------|-------------------|------|
| **Balanced** | 3e-3 | 0.35 | 0.02 | 10 | 0.42-0.45 | 1.5-2.5h |
| Fast | 5e-3 | 0.30 | 0.01 | 5 | 0.35-0.40 | <1h |
| Conservative | 2e-3 | 0.40 | 0.05 | 15 | 0.40-0.44 | 2-3h |
| Aggressive | 4e-3 | 0.25 | 0.01 | 12 | 0.43-0.48 | 2-4h |

---

## 🔬 Hyperparameter Tuning Guide

### Learning Rate (`--lr`)
- **Higher (4e-3 to 5e-3):** Faster convergence, risk of instability
- **Medium (3e-3):** Balanced (recommended)
- **Lower (1e-3 to 2e-3):** More stable, slower convergence

**Your previous run:** 4e-3 (might be too high, caused some oscillation)

### Dropout (`--dropout`)
- **Higher (0.4+):** More regularization, prevents overfitting
- **Medium (0.3-0.35):** Balanced (recommended)
- **Lower (0.2-0.25):** Less regularization, higher capacity

**Your previous run:** 0.3 (good, but could increase slightly)

### Weight Decay (`--weight_decay`)
- **Higher (0.05+):** Strong regularization
- **Medium (0.01-0.02):** Balanced (recommended)
- **Lower (0.001-0.005):** Weak regularization

**Your previous run:** 0.01 (adequate)

### Early Stopping Patience (`--early_stopping_patience`)
- **Short (5):** Stops quickly, might miss late improvements
- **Medium (10-12):** Balanced (recommended)
- **Long (15+):** More patient, good for noisy validation

**Your analysis suggests:** Patience of 10 would have been ideal

### Validation Frequency (`--validation_frequency`)
- **Every epoch (1):** Best for early stopping (recommended)
- **Every 2 epochs (2):** Saves time, less precise
- **Every 5 epochs (5):** Quick runs only

**Your previous run:** 2 (should change to 1)

---

## 🚀 How to Use These Configs

### Option 1: Use the provided scripts
```bash
# Windows
train_optimized_config.bat

# Linux/Mac
bash train_optimized_config.sh
```

### Option 2: Run directly with Python
```bash
# Balanced config (recommended)
python train_ultra_speed_v26.py \
  --lr 3e-3 \
  --dropout 0.35 \
  --weight_decay 0.02 \
  --validation_frequency 1 \
  --use_early_stopping True \
  --early_stopping_patience 10
```

### Option 3: Create your own config
```python
# Edit the default args in train_ultra_speed_v26.py (line 1084)
# Or pass custom arguments
```

---

## 📈 After Training

### 1. Visualize Results
```bash
python visualize_training.py --csv output_balanced/training_metrics.csv
```

### 2. Find Best Epoch
```bash
python find_best_epoch.py --checkpoint_dir output_balanced --plot
```

### 3. Compare Multiple Runs
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics from different runs
balanced = pd.read_csv('output_balanced/training_metrics.csv')
aggressive = pd.read_csv('output_aggressive/training_metrics.csv')

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(balanced['epoch'], balanced['val_spearman'], label='Balanced')
plt.plot(aggressive['epoch'], aggressive['val_spearman'], label='Aggressive')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Validation Spearman')
plt.title('Configuration Comparison')
plt.grid(True, alpha=0.3)
plt.savefig('config_comparison.png', dpi=300)
plt.show()
```

---

## 🎓 What We Learned From Your Training

### ✅ What Worked
1. **Perfect Recall@pKd≥9 (100%)** - Model successfully identifies all high-affinity binders
2. **Reasonable Spearman (0.42)** - Model learns binding patterns
3. **Stable training** - No crashes or divergence

### ⚠️ What Needs Improvement
1. **Overfitting** - Performance dropped from 0.42 to 0.38 after peak
2. **No early stopping** - Wasted ~15-20 epochs after peak
3. **Infrequent validation** - Only every 2 epochs, hard to catch peak

### 🔧 Improvements Made
1. ✓ **Early stopping** - Automatically stops when overfitting starts
2. ✓ **More frequent validation** - Every epoch for better tracking
3. ✓ **Better regularization** - Increased dropout & weight decay
4. ✓ **Comprehensive logging** - CSV metrics + visualizations
5. ✓ **Lower learning rate** - More stable convergence

---

## 💡 Next Steps

1. **Start with BALANCED config** - Best overall choice
2. **Monitor training with visualization** - Use the new tools
3. **If results are good:**
   - Try AGGRESSIVE for potential improvement
   - Fine-tune hyperparameters
4. **If results are poor:**
   - Try CONSERVATIVE for stability
   - Check data quality

---

## 🤔 FAQ

**Q: Which config should I start with?**
A: BALANCED - it's optimized based on your previous results.

**Q: Can I mix parameters from different configs?**
A: Yes! Experiment with combinations.

**Q: How do I know if a run is going well?**
A: Watch validation Spearman - should reach 0.35+ by epoch 10-15.

**Q: What if early stopping triggers too soon?**
A: Increase `--early_stopping_patience` to 15 or 20.

**Q: What if training is too slow?**
A: Try FAST config or reduce `--batch_size` to 12.

**Q: How can I compare different runs?**
A: Use the comparison script above with training_metrics.csv files.

---

**Good luck with your optimized training! 🚀**
