# AbAg Binding Affinity Predictor v2.8 (Stage 2)

## Model Overview

This model predicts antibody-antigen binding affinity (pKd) using a two-stage architecture:

1. **Stage 1**: Protein language model embeddings (IgT5 + ProtT5) with LoRA fine-tuning
2. **Stage 2**: Cross-attention fusion + Residual MLP head

## Performance

| Metric | Validation | Test |
|--------|------------|------|
| R² | 0.7865 | TBD |
| MAE | 0.4254 | TBD |
| RMSE | ~0.71 | TBD |

## Architecture

```
Input: Antibody Embedding (1024-dim) + Antigen Embedding (1024-dim)
    │
    ▼
┌─────────────────────────────────────────┐
│  Cross-Attention Fusion (512-dim)       │
│  - Ab→Ag attention + Ag→Ab attention    │
│  - 8 attention heads                    │
│  - LayerNorm + GELU + Dropout(0.1)      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Residual MLP Head                      │
│  - 512 → 256 → 128 → 1                  │
│  - BatchNorm + GELU + Dropout(0.2)      │
└─────────────────────────────────────────┘
    │
    ▼
Output: pKd (binding affinity)
```

## Training Configuration

```python
CONFIG = {
    'fusion_method': 'cross_attention',
    'head_type': 'residual_mlp',
    'fusion_hidden_dim': 512,
    'fusion_num_heads': 8,
    'fusion_dropout': 0.1,
    'head_hidden_dims': [512, 256, 128],
    'head_dropout': 0.2,
    'batch_size': 256,
    'learning_rate': 1e-3,
    'scheduler': 'cosine',
    'early_stopping_patience': 15,
}
```

## Training Details

- **Dataset**: ~140k antibody-antigen pairs with pKd labels
- **Train/Val/Test Split**: 80/10/10
- **Training Time**: ~85 epochs (early stopped)
- **Hardware**: Google Colab A100 40GB

## Files

- `stage2_experiment_20251210_033341.pth` - Trained model weights
- `stage2_experiment_20251210_033341_curves.png` - Training curves

## Usage

```python
from inference import BindingAffinityPredictor

# Load model
predictor = BindingAffinityPredictor('models/v2.8_stage2/stage2_experiment_20251210_033341.pth')

# Predict from sequences
pKd = predictor.predict_from_sequences(
    antibody_seq="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS...",
    antigen_seq="MKTIIALSYIFCLVFADYKDDDDK..."
)

print(f"Predicted pKd: {pKd:.2f}")
print(f"Predicted Kd: {10**(9-pKd):.2f} nM")
```

## pKd Interpretation

| pKd Range | Binding Strength | Kd (approx) |
|-----------|------------------|-------------|
| ≥ 9 | Very Strong | < 1 nM |
| 7-9 | Strong | 1-100 nM |
| 5-7 | Moderate | 0.1-10 µM |
| < 5 | Weak | > 10 µM |

## Dependencies

```
torch>=2.0
transformers>=4.41.0
pandas
numpy
```

## Citation

If you use this model, please cite:
- IgT5: Exscientia/IgT5
- ProtT5: Rostlab/prot_t5_xl_half_uniref50-enc
