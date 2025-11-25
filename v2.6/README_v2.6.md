# Antibody-Antigen Binding Prediction v2.6

**Status**: Experimental Beta Release
**Date**: 2025-11-25
**Model Size**: 16.3 GB
**Training Duration**: 15 epochs (~60 hours on A100 80GB)

---

## Overview

This release contains the v2.6 antibody-antigen binding affinity prediction model trained using IgT5 (antibody encoder) + ESM-2 3B (antigen encoder) with cross-attention. The model predicts pKd values from antibody-antigen sequence pairs.

**Important**: This is an experimental model with known instability issues (see Limitations). Version 2.7 addresses these issues with research-validated fixes.

---

## Model Performance

### Best Checkpoint (Epoch 12)

| Metric | Validation | Test |
|--------|------------|------|
| **Spearman ρ** | 0.390 | 0.395 |
| **Pearson r** | 0.698 | 0.696 |
| **RMSE** | 2.10 | 2.10 |
| **MAE** | 1.67 | 1.67 |
| **R²** | 0.004 | -0.002 |
| **Recall@pKd≥9** | 100.0% | 99.96% |
| **Precision@pKd≥9** | 35.8% | 35.8% |

### Training Progression

The model was trained for 15 epochs with the following progression:

```
Epoch  Train Loss  Val Spearman  Val Recall  Learning Rate
    1       0.458         0.329      100.0%      1e-4
    5       0.430         0.390      100.0%      5e-4
   12       0.413         0.390      100.0%      4.5e-5  ← Best
   15       0.409         0.389      99.98%      2.0e-5
```

See [training_history_v2.6.csv](training_history_v2.6.csv) for complete history.

---

## Architecture

### Dual-Encoder with Cross-Attention

```
Input: Antibody sequence + Antigen sequence
  ↓
[IgT5 Encoder]  →  Antibody embeddings (frozen)
  ↓
[ESM-2 3B Encoder] → Antigen embeddings (frozen)
  ↓
[Cross-Attention Layer] → Combined representations (trainable)
  ↓
[Regressor Head] → pKd prediction (trainable)
```

### Model Details

- **Antibody Encoder**: IgT5-base (frozen)
- **Antigen Encoder**: ESM-2 3B (frozen)
- **Trainable Parameters**: ~50M (cross-attention + regressor)
- **Total Parameters**: ~3.5B (mostly frozen)
- **Max Sequence Length**: 512 tokens

---

## Training Configuration

### Hyperparameters

```python
BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 4
EFFECTIVE_BATCH_SIZE = 128
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
DROPOUT = 0.3
WARMUP_EPOCHS = 5
MAX_EPOCHS = 50 (stopped at 15)
```

### Loss Function

Combined loss with three components:

1. **Huber Loss** (0.5 weight): Robust regression
2. **Soft Spearman Loss** (0.4 weight): Ranking correlation
3. **BCE Classification** (0.1 weight): Strong binder prediction

```python
total_loss = 0.5 * huber_loss + 0.4 * soft_spearman + 0.1 * bce_loss
```

### Hardware

- **GPU**: NVIDIA A100 80GB (Google Colab Pro+)
- **Training Time**: ~4 hours per epoch
- **Total Time**: ~60 hours (15 epochs)
- **Memory Usage**: ~45GB peak

---

## Files Included

### Model Files

- **checkpoint_latest.pth** (16.3 GB)
  - Complete checkpoint from epoch 15
  - Contains: model weights, optimizer state, scheduler state
  - Can be loaded with PyTorch

### Training Artifacts

- **metrics.json** - Final test/validation metrics + hyperparameters
- **training_history_v2.6.csv** - Epoch-by-epoch training logs
- **training_history_v2.6.png** - Training curves visualization
- **training_summary_v2.6.json** - Summary statistics
- **test_predictions.csv** - Model predictions on test set (5,707 samples)

---

## Known Limitations

### 1. Recall Instability

The model exhibits severe recall instability:
- **Standard Deviation**: 39.35%
- **Max Jump**: 85.6% between epochs
- **Pattern**: Oscillates between 18% and 100%

**Cause**: Soft Spearman loss with O(n²) pairwise ranking creates unstable gradients.

### 2. Prediction Range Issues

The model produces predictions outside the physically valid pKd range:
- **Predicted Range**: -2.48 to 10.0
- **Valid Range**: 4.0 to 14.0
- **Issue**: Negative pKd values are chemically impossible

**Cause**: No output clamping in the model architecture.

### 3. Prediction Clustering

Test predictions show clustering at discrete values:
- Many predictions at **9.0625** and **10.3125**
- Suggests model is not learning continuous pKd values
- Possibly due to dataset imbalance or loss function

### 4. Poor Generalization

- **R² ≈ 0**: Model explains almost no variance
- **High RMSE**: 2.1 (relative to pKd range 4-14)
- **High recall, low precision**: Overpredicts strong binders

---

## Usage

### Loading the Model

```python
import torch
from model import DualEncoderWithCrossAttention  # Your model class

# Load checkpoint
checkpoint = torch.load('checkpoint_latest.pth', map_location='cuda')

# Initialize model
model = DualEncoderWithCrossAttention(
    num_labels=1,
    dropout=0.3,
    use_cross_attention=True
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Best validation Spearman: {checkpoint.get('best_val_spearman', 'N/A')}")
```

### Making Predictions

```python
from transformers import AutoTokenizer

# Load tokenizers
ab_tokenizer = AutoTokenizer.from_pretrained("lightonai/RITA_s")
ag_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")

# Prepare input
antibody_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS"
antigen_seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

# Tokenize
ab_inputs = ab_tokenizer(antibody_seq, return_tensors='pt', padding=True, truncation=True, max_length=512)
ag_inputs = ag_tokenizer(antigen_seq, return_tensors='pt', padding=True, truncation=True, max_length=512)

# Predict
with torch.no_grad():
    pKd_pred = model(
        ab_inputs['input_ids'].cuda(),
        ab_inputs['attention_mask'].cuda(),
        ag_inputs['input_ids'].cuda(),
        ag_inputs['attention_mask'].cuda()
    )

print(f"Predicted pKd: {pKd_pred.item():.2f}")
```

### Important Notes

- **Input validation**: Check for negative predictions (chemically invalid)
- **Threshold tuning**: Default threshold of pKd ≥ 9 for strong binders may need adjustment
- **Ensemble recommended**: Due to instability, consider using multiple checkpoints

---

## Comparison with v2.5

| Feature | v2.5 | v2.6 |
|---------|------|------|
| Architecture | ESM-2 650M only | IgT5 + ESM-2 3B |
| Spearman | 0.42 | 0.39 (worse) |
| Training Speed | Slow | 6-8× faster |
| Recall Stability | Stable | **Unstable** |
| Memory | 24GB | 45GB |

**Conclusion**: v2.6 achieved faster training but worse stability. See v2.7 for fixes.

---

## Next Steps: v2.7 Improvements

The following issues are addressed in v2.7:

1. **Replace Soft Spearman loss** with stable MSE + BCE (research-validated)
2. **Add prediction clamping** to valid pKd range [4.0, 14.0]
3. **Update hyperparameters** based on Multi-task Bioassay Pre-training 2024
4. **Add NaN detection** to catch numerical issues early
5. **Complete RNG state saving** for reproducibility
6. **Overfitting monitoring** to diagnose stability

See [V2.7_IMPROVEMENTS.md](../V2.7_IMPROVEMENTS.md) for details.

---

## Citation

If you use this model, please cite:

```bibtex
@software{abag_binding_v26,
  title = {Antibody-Antigen Binding Prediction v2.6},
  author = {Your Name},
  year = {2025},
  month = {11},
  version = {2.6.0-beta},
  url = {https://github.com/yourusername/AbAg_binding_prediction}
}
```

---

## Reproducibility

### Training Command

```bash
python train_ultra_speed_v26.py \
  --batch_size 32 \
  --gradient_accumulation 4 \
  --learning_rate 5e-4 \
  --dropout 0.3 \
  --huber_weight 0.5 \
  --spearman_weight 0.4 \
  --class_weight 0.1
```

### Environment

```
Python: 3.10.12
PyTorch: 2.1.0+cu121
Transformers: 4.36.0
CUDA: 12.1
GPU: A100 80GB
```

See [metrics.json](metrics.json) for complete hyperparameters.

---

## License

MIT License - See main repository for details.

---

## Support

For issues or questions:
- **Main Repository**: [https://github.com/yourusername/AbAg_binding_prediction]
- **Documentation**: See main README.md
- **v2.7 Updates**: Check V2.7_IMPROVEMENTS.md

---

**Warning**: This is an experimental release with known stability issues. Production use is not recommended. Consider waiting for v2.7 stable release.

---

*Last updated: 2025-11-25*
*Model version: 2.6.0-beta*
*Status: Superseded by v2.7 (in development)*
