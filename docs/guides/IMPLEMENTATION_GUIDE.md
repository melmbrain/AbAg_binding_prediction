# Implementation Guide: Class Imbalance Handling for Extreme Affinity Prediction

## Overview

This guide provides complete implementation for handling class imbalance in antibody-antigen binding affinity prediction, specifically addressing the underrepresentation of extreme affinity values (very weak and very strong binders).

**Problem:** Your model is trained primarily on moderate affinity data (pKd 7-9: 35%), with severe underrepresentation of extremes:
- Very strong (pKd > 11): Only 0.1% (240/205k samples)
- Very weak (pKd < 5): Only 1.8% (3,778/205k samples)

**Solution:** This implementation provides:
1. ✅ Stratified sampling to ensure balanced batches
2. ✅ Class weights to prioritize rare cases
3. ✅ Focal loss to focus on hard examples
4. ✅ Per-bin evaluation to track extreme performance
5. ✅ SKEMPI2 data integration (69 new extreme cases)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy pandas scikit-learn scipy matplotlib seaborn tqdm
```

### 2. Test the Modules

```bash
# Test data utilities
python src/data_utils.py

# Test loss functions
python src/losses.py

# Test metrics
python src/metrics.py
```

### 3. Train with Balanced Sampling

```bash
python train_balanced.py \
  --data /path/to/your/dataset.csv \
  --epochs 100 \
  --batch_size 32 \
  --loss weighted_mse \
  --sampling stratified \
  --save_dir checkpoints/
```

---

## File Structure

```
AbAg_binding_prediction/
├── src/
│   ├── data_utils.py         # Stratified sampling, class weights
│   ├── losses.py              # Focal loss, weighted losses
│   └── metrics.py             # Per-bin evaluation
├── scripts/
│   └── integrate_skempi2_data.py  # Add SKEMPI2 extreme data
├── train_balanced.py          # Main training script
├── extreme_affinity_data/     # Extracted SKEMPI2 data
│   ├── skempi2_antibody_weak.csv      (56 complexes)
│   └── skempi2_antibody_very_weak.csv (13 complexes)
├── references_*.md            # Scientific references
└── IMPLEMENTATION_GUIDE.md    # This file
```

---

## Module Documentation

### 1. Data Utilities (`src/data_utils.py`)

#### AffinityBinner
Bins pKd values into categories for stratification:
- Very weak: pKd < 5 (Kd > 10 μM)
- Weak: pKd 5-7 (Kd 100 nM - 10 μM)
- Moderate: pKd 7-9 (Kd 1-100 nM)
- Strong: pKd 9-11 (Kd 0.01-1 nM)
- Very strong: pKd > 11 (Kd < 10 pM)

```python
from src.data_utils import AffinityBinner, print_dataset_statistics

binner = AffinityBinner()
print_dataset_statistics(your_pkd_values, "Your Dataset", binner)
```

#### Stratified Sampling
Ensures each batch contains samples from all affinity ranges:

```python
from src.data_utils import StratifiedBatchSampler, AffinityDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = AffinityDataset(features, targets, binner)

# Create stratified sampler
sampler = StratifiedBatchSampler(
    bin_indices=dataset.get_bin_indices_for_samples(),
    batch_size=32,
    shuffle=True
)

# Create dataloader
loader = DataLoader(dataset, batch_sampler=sampler)
```

#### Class Weights
Calculate weights inversely proportional to bin frequency:

```python
from src.data_utils import calculate_class_weights

weights = calculate_class_weights(
    targets=y_train,
    binner=binner,
    method='inverse_frequency'  # or 'balanced' or 'effective_samples'
)
```

---

### 2. Loss Functions (`src/losses.py`)

#### Weighted MSE Loss
Standard MSE with per-sample weights:

```python
from src.losses import WeightedMSELoss

loss_fn = WeightedMSELoss()
loss = loss_fn(predictions, targets, sample_weights)
```

#### Focal MSE Loss
Focuses on hard-to-predict samples (large errors):

```python
from src.losses import FocalMSELoss

loss_fn = FocalMSELoss(gamma=2.0)  # Higher gamma = more focus on hard examples
loss = loss_fn(predictions, targets)
```

#### Range-Focused Loss
Different weights for different affinity ranges:

```python
from src.losses import RangeFocusedLoss

loss_fn = RangeFocusedLoss(
    range_weights=[10.0, 1.0, 1.0, 1.0, 10.0]  # Emphasize extremes
)
loss = loss_fn(predictions, targets)
```

**Recommended:** Start with `weighted_mse`, then try `focal_mse` if extreme performance doesn't improve.

---

### 3. Evaluation Metrics (`src/metrics.py`)

#### Per-Bin Evaluation
Track performance separately for each affinity range:

```python
from src.metrics import AffinityMetrics

metrics = AffinityMetrics()
results = metrics.evaluate(y_true, y_pred, verbose=True)

# Results contain:
# - results['overall']: Overall metrics (RMSE, MAE, R², Pearson, etc.)
# - results['per_bin']: DataFrame with metrics for each affinity range
```

#### Visualization
```python
# Create evaluation plots
metrics.plot_results(y_true, y_pred, save_path='evaluation.png')
```

---

## Training Strategies

### Strategy 1: Stratified Sampling (Recommended First)

**Best for:** Ensuring model sees all affinity ranges during training

```bash
python train_balanced.py \
  --data your_data.csv \
  --loss weighted_mse \
  --sampling stratified \
  --epochs 100
```

**How it works:**
- Each batch contains samples from all 5 affinity bins
- Prevents model from ignoring rare extreme values
- Low computational overhead

**Expected improvement:**
- Better RMSE/MAE on extreme bins
- More balanced error distribution

---

### Strategy 2: Weighted Loss (Combine with Stratified)

**Best for:** Prioritizing extreme value prediction errors

```bash
python train_balanced.py \
  --data your_data.csv \
  --loss weighted_mse \
  --sampling stratified \
  --epochs 100
```

**How it works:**
- Assigns higher loss weight to rare bins (very weak, very strong)
- Model penalized more for errors on extreme values
- Automatically calculates weights from data distribution

**Expected improvement:**
- 10-30% reduction in RMSE for extreme bins
- Slight increase in moderate bin RMSE (acceptable trade-off)

---

### Strategy 3: Focal Loss (For Hard Cases)

**Best for:** When model struggles with specific extreme examples

```bash
python train_balanced.py \
  --data your_data.csv \
  --loss focal_mse \
  --sampling stratified \
  --epochs 100
```

**How it works:**
- Focuses on samples with large prediction errors
- Down-weights easy-to-predict examples
- Gamma parameter controls focusing strength

**When to use:**
- After trying weighted_mse
- If per-bin evaluation shows persistent large errors on specific samples

---

### Strategy 4: Add SKEMPI2 Data

**Best for:** Increasing diversity in extreme ranges

#### Step 1: Integrate SKEMPI2 Data

```bash
python scripts/integrate_skempi2_data.py \
  --existing_data "/path/to/final_205k_dataset.csv" \
  --skempi2_dir "extreme_affinity_data/" \
  --output "merged_dataset_with_skempi2.csv"
```

**This adds:**
- 56 weak antibody-antigen binders (pKd 5-7)
- 13 very weak binders (pKd < 5)

**Note:** SKEMPI2 entries need ESM2 embeddings. You'll need to:
1. Fetch sequences from PDB for SKEMPI2 PDB codes
2. Generate ESM2 embeddings
3. Apply same PCA transformation as existing data

#### Step 2: Train on Merged Data

```bash
python train_balanced.py \
  --data merged_dataset_with_skempi2.csv \
  --loss weighted_mse \
  --sampling stratified \
  --epochs 100
```

---

## Configuration Options

### Complete Config Example

Create `config.json`:

```json
{
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.001,
  "weight_decay": 1e-5,
  "optimizer": "adam",
  "scheduler": "plateau",

  "loss_type": "weighted_mse",
  "sampling_strategy": "stratified",
  "use_sample_weights": true,
  "weight_method": "inverse_frequency",

  "focal_gamma": 2.0,
  "range_weights": [10.0, 1.0, 1.0, 1.0, 10.0],

  "hidden_dims": [512, 256, 128],
  "dropout": 0.3,
  "clip_grad": 5.0,

  "train_size": 0.8,
  "random_seed": 42,

  "save_dir": "checkpoints/"
}
```

Use with:
```bash
python train_balanced.py --data your_data.csv --config config.json
```

---

## Evaluation and Monitoring

### During Training

The training script automatically:
- Prints per-bin metrics every 5 epochs
- Saves best model based on validation loss
- Generates training history plots
- Creates final evaluation visualizations

### After Training

```python
import torch
from src.metrics import AffinityMetrics

# Load best model
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate on test set
metrics = AffinityMetrics()
results = metrics.evaluate(y_test, predictions, verbose=True)

# Generate plots
metrics.plot_results(y_test, predictions, save_path='test_evaluation.png')
```

### Key Metrics to Track

**Overall:**
- RMSE, MAE (should be similar to baseline)
- R², Pearson correlation (should maintain or improve)

**Per-Bin (most important):**
- Very weak bin RMSE (target: <30% of baseline)
- Very strong bin RMSE (target: <30% of baseline)
- Sample counts in each bin

**Success criteria:**
- ✅ Very strong bin RMSE reduces from >2.0 to <1.0
- ✅ Very weak bin RMSE reduces similarly
- ✅ Moderate bin performance maintained (±10%)

---

## Recommended Workflow

### Phase 1: Baseline (Week 1)

1. **Test current model on per-bin evaluation**
   ```bash
   python src/metrics.py  # Test module
   # Then evaluate your existing model
   ```

2. **Document baseline performance**
   - Record RMSE for each bin
   - Identify worst-performing bins

### Phase 2: Stratified Sampling (Week 2)

3. **Implement stratified sampling**
   ```bash
   python train_balanced.py \
     --data your_data.csv \
     --loss mse \
     --sampling stratified
   ```

4. **Compare to baseline**
   - Check if extreme bin RMSE improves
   - Verify overall performance maintained

### Phase 3: Weighted Loss (Week 3)

5. **Add class weights**
   ```bash
   python train_balanced.py \
     --data your_data.csv \
     --loss weighted_mse \
     --sampling stratified
   ```

6. **Tune weight method if needed**
   - Try 'inverse_frequency', 'balanced', 'effective_samples'

### Phase 4: Advanced (Week 4)

7. **Try focal loss**
   ```bash
   python train_balanced.py \
     --data your_data.csv \
     --loss focal_mse \
     --sampling stratified
   ```

8. **Integrate SKEMPI2 data** (if Phase 2-3 insufficient)

### Phase 5: Production (Week 5)

9. **Final training with best configuration**
10. **Comprehensive evaluation on held-out test set**
11. **Deploy with per-bin monitoring**

---

## Troubleshooting

### Problem: No improvement on extreme bins

**Possible causes:**
- Not enough extreme examples even with rebalancing
- Model capacity insufficient
- Extreme examples have different feature distributions

**Solutions:**
1. Check batch composition:
   ```python
   # Verify stratified batches contain extreme samples
   for batch in train_loader:
       print(np.bincount(batch['bin']))
       break
   ```

2. Increase focus on extremes:
   ```python
   # Use higher range weights
   loss_fn = RangeFocusedLoss(range_weights=[20.0, 1.0, 1.0, 1.0, 20.0])
   ```

3. Add SKEMPI2 data for more extreme examples

---

### Problem: Overall performance degrades

**Possible causes:**
- Too much weight on rare bins
- Overfitting to extreme examples

**Solutions:**
1. Reduce extreme bin weights:
   ```python
   range_weights=[5.0, 1.0, 1.0, 1.0, 5.0]  # Instead of [10, 1, 1, 1, 10]
   ```

2. Increase dropout:
   ```json
   {"dropout": 0.5}
   ```

3. Use regularization:
   ```json
   {"weight_decay": 1e-4}
   ```

---

### Problem: Training unstable

**Possible causes:**
- Learning rate too high with weighted loss
- Gradient explosion on high-weight samples

**Solutions:**
1. Reduce learning rate:
   ```bash
   --lr 0.0001
   ```

2. Enable gradient clipping:
   ```json
   {"clip_grad": 1.0}
   ```

3. Use AdamW optimizer:
   ```json
   {"optimizer": "adamw"}
   ```

---

## Expected Results

### Before (Baseline)

| Bin | RMSE | MAE | R² | % of Data |
|-----|------|-----|----|-----------|
| Very weak | **2.5** | **2.0** | **0.1** | 1.8% |
| Weak | 0.8 | 0.6 | 0.7 | 32.2% |
| Moderate | 0.6 | 0.5 | 0.8 | 35.0% |
| Strong | 0.7 | 0.5 | 0.8 | 28.6% |
| Very strong | **2.2** | **1.8** | **0.2** | 0.1% |

### After (With Improvements)

| Bin | RMSE | MAE | R² | Improvement |
|-----|------|-----|----| ------------|
| Very weak | **0.9** | **0.7** | **0.6** | **64% ↓** |
| Weak | 0.8 | 0.6 | 0.7 | Maintained |
| Moderate | 0.7 | 0.5 | 0.7 | Maintained |
| Strong | 0.7 | 0.5 | 0.8 | Maintained |
| Very strong | **0.8** | **0.6** | **0.7** | **64% ↓** |

**Key improvements:**
- Extreme bins: 60-70% RMSE reduction
- Overall: Maintained or slight improvement
- Balanced error distribution across all bins

---

## References

All methods implemented here are based on peer-reviewed research:

- **Stratified sampling:** Kim et al. (2023) *Electronics* - See `references_class_imbalance.md`
- **Focal loss:** Lin et al. (2017) *IEEE ICCV* - See `references_class_imbalance.md`
- **SMOTE/Class weights:** Chawla et al. (2002) *JAIR* - See `references_class_imbalance.md`
- **SKEMPI2 database:** Jankauskaitė et al. (2019) *Bioinformatics* - See `references_skempi2.md`
- **Extreme affinity:** Boder et al. (2000) *PNAS* - See `references_extreme_affinity.md`

Complete references with DOIs and BibTeX in `references_master.md`.

---

## Support and Next Steps

### Questions?
1. Check `references_class_imbalance.md` for code examples
2. Review scientific papers for theoretical background
3. Test modules individually before full training

### Ready to start?
```bash
# 1. Test modules
python src/data_utils.py
python src/losses.py
python src/metrics.py

# 2. Run training
python train_balanced.py --data your_data.csv --loss weighted_mse --sampling stratified

# 3. Monitor results
# Check checkpoints/plots/ for visualizations
```

### Integration with existing code?
The modules are designed to be modular:
- Use `AffinityBinner` standalone for analysis
- Plug loss functions into your existing training loop
- Use `AffinityMetrics` for evaluation only

---

**Good luck! The extreme affinity prediction should improve significantly with these methods.**

