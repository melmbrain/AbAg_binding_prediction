# Release Notes: v2.6.0-beta

**Release Date**: 2025-11-25
**Status**: Experimental Beta
**Model Size**: 16.3 GB

---

## What's New in v2.6

### Major Architecture Change

- **Dual-Encoder System**: IgT5 (antibody) + ESM-2 3B (antigen)
- **Cross-Attention**: Better antibody-antigen interaction modeling
- **Frozen Encoders**: Only train cross-attention + regressor head
- **6-8× Faster Training**: Compared to v2.5

### Performance

| Metric | Value |
|--------|-------|
| Spearman ρ | 0.390 |
| Pearson r | 0.696 |
| RMSE | 2.10 |
| Recall@pKd≥9 | 100% |
| Precision@pKd≥9 | 35.8% |

---

## Breaking Changes

### Loss Function Changes

Introduced **Soft Spearman Loss** for ranking:

```python
# v2.5 (Old)
loss = huber_loss + classification_loss

# v2.6 (New)
loss = 0.5 * huber_loss + 0.4 * soft_spearman + 0.1 * classification_loss
```

**Impact**: Faster convergence but introduced stability issues.

### Model Architecture

Changed from single encoder to dual encoder:

```python
# v2.5
ESM-2 650M → Regressor → pKd

# v2.6
IgT5 (antibody) ──┐
                  ├─→ Cross-Attention → Regressor → pKd
ESM-2 3B (antigen)┘
```

**Impact**: Requires different tokenizers for antibody and antigen.

---

## Known Issues

### Critical

1. **Recall Instability** (HIGH PRIORITY)
   - Oscillates between 18% and 100%
   - Caused by Soft Spearman loss gradient instability
   - **Fix**: Will be addressed in v2.7 with MSE loss

2. **Invalid Predictions** (MEDIUM PRIORITY)
   - Predicts negative pKd values (impossible)
   - Range: -2.48 to 10.0 (should be 4.0 to 14.0)
   - **Fix**: Will be clamped in v2.7

3. **Prediction Clustering** (LOW PRIORITY)
   - Many predictions at 9.06 and 10.31
   - Not learning continuous values
   - **Investigation**: Ongoing

### Minor

- High memory usage (45GB vs 24GB in v2.5)
- Longer inference time due to dual encoders
- Poor R² score (~0)

---

## Migration from v2.5

### Model Loading

```python
# v2.5
from model import BindingAffinityPredictor
model = BindingAffinityPredictor.load('v2.5_model.pth')

# v2.6
from model import DualEncoderWithCrossAttention
checkpoint = torch.load('checkpoint_latest.pth')
model = DualEncoderWithCrossAttention(dropout=0.3, use_cross_attention=True)
model.load_state_dict(checkpoint['model_state_dict'])
```

### Tokenizers

```python
# v2.5 - Single tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

# v2.6 - Dual tokenizers
ab_tokenizer = AutoTokenizer.from_pretrained("lightonai/RITA_s")  # IgT5
ag_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
```

### Predictions

```python
# v2.5
pKd = model.predict(sequence)

# v2.6
pKd = model(ab_input_ids, ab_mask, ag_input_ids, ag_mask)
# Note: Separate inputs for antibody and antigen
```

---

## Training Details

### Dataset

- **Total Samples**: 134,862
- **Train**: 107,889 (80%)
- **Validation**: 13,486 (10%)
- **Test**: 13,487 (10%)
- **Strong Binders** (pKd ≥ 9): ~15%

### Training Duration

- **Epochs**: 15
- **Time per Epoch**: ~4 hours
- **Total Time**: ~60 hours
- **GPU**: A100 80GB
- **Early Stopping**: Patience 10 (not triggered)

### Convergence

Best model at **epoch 12**:
- Validation Spearman: 0.390
- Validation Loss: 0.413

Training continued to epoch 15 but no improvement.

---

## Files in This Release

### Essential

1. **checkpoint_latest.pth** (16.3 GB)
   - Full model checkpoint
   - Includes optimizer and scheduler states
   - Load with `torch.load()`

2. **README_v2.6.md**
   - Complete model documentation
   - Usage examples
   - Limitations and warnings

3. **RELEASE_NOTES_v2.6.md** (this file)
   - What's new
   - Breaking changes
   - Migration guide

### Training Artifacts

4. **metrics.json**
   - Final test/val metrics
   - Complete hyperparameters
   - Configuration details

5. **training_history_v2.6.csv**
   - Epoch-by-epoch logs
   - Loss, Spearman, Recall, LR

6. **training_history_v2.6.png**
   - Visual training curves
   - Shows recall instability

7. **training_summary_v2.6.json**
   - Summary statistics
   - Best epoch info
   - Stability metrics

8. **test_predictions.csv**
   - Model predictions on test set
   - 5,707 samples
   - Format: `true,pred,error`

---

## Benchmarks

### Training Speed (vs v2.5)

| Metric | v2.5 | v2.6 | Speedup |
|--------|------|------|---------|
| Time/Epoch | 28 hours | 4 hours | **7×** |
| Time/Batch | 45s | 6s | **7.5×** |
| Throughput | 71 samples/s | 533 samples/s | **7.5×** |

### Memory Usage

| Component | v2.5 | v2.6 |
|-----------|------|------|
| Model | 2.5 GB | 14 GB |
| Batch (32) | 8 GB | 18 GB |
| Optimizer | 5 GB | 13 GB |
| **Peak** | **24 GB** | **45 GB** |

### Inference Speed

- **Single prediction**: ~150ms (vs 50ms in v2.5)
- **Batch (32)**: ~1.2s (vs 0.8s in v2.5)
- **Slower due to**: Dual encoders + cross-attention

---

## Experimental Features

### Soft Spearman Loss

```python
def soft_spearman_loss(pred, target):
    """
    Differentiable Spearman correlation loss
    Uses soft ranking via pairwise comparisons
    """
    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    target_diff = target.unsqueeze(1) - target.unsqueeze(0)
    pred_rank = torch.sigmoid(pred_diff / temperature)
    target_rank = torch.sigmoid(target_diff / temperature)
    return F.mse_loss(pred_rank, target_rank)
```

**Status**: Experimental - causes gradient instability
**Recommendation**: Use with caution, disable for production

---

## Reproducibility

### Random Seeds

Training used fixed seeds but **RNG states were not saved** in checkpoints.

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

**Limitation**: Cannot exactly reproduce from checkpoint.
**Fix**: v2.7 will save complete RNG state.

### Environment

```yaml
python: 3.10.12
pytorch: 2.1.0+cu121
transformers: 4.36.0
numpy: 1.23.5
pandas: 1.5.3
cuda: 12.1
cudnn: 8.9.2
```

See [requirements.txt](../requirements.txt) for complete dependencies.

---

## Limitations and Warnings

### Not Suitable For

- **Production Use**: Due to instability issues
- **High-Stakes Decisions**: Model has low precision (35%)
- **Negative pKd Values**: Model doesn't clamp outputs
- **Reproducibility**: RNG state not saved

### Use With Caution For

- **Research**: Understand instability issues first
- **Benchmarking**: Compare with v2.7 when available
- **Method Development**: Good baseline for improvements

### Recommended For

- **Educational Purposes**: Shows importance of loss function choice
- **Ablation Studies**: Compare with different architectures
- **Error Analysis**: Study what causes instability

---

## Next Steps

### Upcoming v2.7 (Expected: 2025-12-01)

**Critical Fixes**:
1. Replace Soft Spearman with MSE loss (stable)
2. Clamp predictions to [4.0, 14.0]
3. Add NaN/Inf detection
4. Save complete RNG state
5. Monitor overfitting ratio

**Expected Improvements**:
- Spearman: 0.39 → 0.45-0.55 (+15-40%)
- Recall: 18-100% (unstable) → 50-70% (stable)
- RMSE: 2.1 → 1.2-1.5 (-30%)
- Pred range: -2.48 to 10.0 → 4.0 to 14.0 (valid)

See [V2.7_IMPROVEMENTS.md](../V2.7_IMPROVEMENTS.md) for details.

---

## Feedback

We welcome feedback on this experimental release:

- **GitHub Issues**: Report bugs and stability issues
- **Discussions**: Share your training results
- **Pull Requests**: Contribute improvements

**Priority**: Feedback on stability issues most valuable.

---

## Acknowledgments

**Research References**:
- ESM-2 protein language model (Meta AI)
- IgT5 antibody language model (LightOn)
- Cross-attention mechanisms (Attention is All You Need)

**Infrastructure**:
- Google Colab Pro+ (A100 80GB access)
- Hugging Face Transformers

---

## Changelog Summary

See [CHANGELOG.md](../CHANGELOG.md) for complete history.

**Added**:
- Dual-encoder architecture (IgT5 + ESM-2 3B)
- Cross-attention layer
- Soft Spearman ranking loss
- 6-8× training speedup

**Changed**:
- Loss function composition
- Model architecture
- Hyperparameters (LR 5e-4, dropout 0.3)

**Deprecated**:
- Single encoder approach from v2.5

**Removed**:
- None

**Fixed**:
- Training speed (7× improvement)
- Memory efficiency (gradient checkpointing)

**Known Issues**:
- Recall instability (HIGH)
- Invalid predictions (MEDIUM)
- Prediction clustering (LOW)

---

## License

MIT License - See [LICENSE](../LICENSE) for details.

---

## Citation

```bibtex
@software{abag_binding_v26,
  title = {Antibody-Antigen Binding Prediction v2.6},
  author = {Your Name},
  year = {2025},
  month = {11},
  version = {2.6.0-beta},
  doi = {10.5281/zenodo.XXXXXXX},  # Add after Zenodo upload
  url = {https://github.com/yourusername/AbAg_binding_prediction}
}
```

---

**WARNING**: This is an experimental beta release. Production deployment is not recommended. Please wait for v2.7 stable release.

---

*Released: 2025-11-25*
*Next release: v2.7.0 (Expected 2025-12-01)*
*Status: Superseded by v2.7 development*
