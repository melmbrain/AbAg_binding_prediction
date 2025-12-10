# v2.6.0-beta: Experimental Dual-Encoder Model

⚠️ **Warning**: This is an experimental beta with known stability issues. Not recommended for production use.

## Overview

Experimental release featuring IgT5 + ESM-2 3B dual-encoder architecture with cross-attention. While this version achieves 6-8× training speedup, it exhibits documented stability issues that will be addressed in v2.7.

## Performance

| Metric | Value |
|--------|-------|
| **Spearman ρ** | 0.390 |
| **Pearson r** | 0.696 |
| **RMSE** | 2.10 |
| **Recall@pKd≥9** | 100% |
| **Precision** | 35.8% |

**Training**: 15 epochs on A100 80GB (~60 hours)

## What's New

### Architecture
- 🏗️ **Dual-Encoder**: IgT5 (antibody) + ESM-2 3B (antigen)
- 🔗 **Cross-Attention**: Better Ab-Ag interaction modeling
- ⚡ **6-8× Training Speedup**: 4h/epoch vs 28h in v2.5
- 📊 **Complete Documentation**: All training artifacts included

### Performance vs v2.5
- Training Speed: **7× faster** (4h vs 28h per epoch)
- Memory: 45GB (vs 24GB in v2.5)
- Spearman: 0.390 (vs 0.42 in v2.5 - slight regression)

## Known Issues

### ⚠️ Critical

1. **Recall Instability**
   - Oscillates between 18% and 100%
   - Standard deviation: 39.35%
   - Max jump: 85.6% between epochs
   - **Cause**: Soft Spearman loss with O(n²) gradient instability

2. **Invalid Predictions**
   - Predicts negative pKd values (e.g., -2.48)
   - Range: -2.48 to 10.0 (should be 4.0-14.0)
   - **Cause**: No output clamping

3. **Prediction Clustering**
   - Many predictions at discrete values (9.06, 10.31)
   - Not learning continuous pKd properly

### Why Release With Issues?

This release is valuable for:
- 📚 **Educational**: Demonstrates importance of loss function choice
- 🔬 **Research**: Shows real development process
- 📊 **Baseline**: Benchmark for v2.7 improvements
- 🎯 **Transparency**: Honest documentation of challenges

## Files

### Documentation
- 📖 [README_v2.6.md](https://github.com/melmbrain/AbAg_binding_prediction/blob/v2.6.0-beta/v2.6/README_v2.6.md) - Complete model card with usage examples
- 📋 [RELEASE_NOTES_v2.6.md](https://github.com/melmbrain/AbAg_binding_prediction/blob/v2.6.0-beta/v2.6/RELEASE_NOTES_v2.6.md) - Detailed release notes
- 🔧 [GITHUB_RELEASE_GUIDE.md](https://github.com/melmbrain/AbAg_binding_prediction/blob/v2.6.0-beta/v2.6/GITHUB_RELEASE_GUIDE.md) - Release process documentation
- 📝 [RELEASE_SUMMARY.md](https://github.com/melmbrain/AbAg_binding_prediction/blob/v2.6.0-beta/v2.6/RELEASE_SUMMARY.md) - Project overview

### Training Artifacts
- 📊 `metrics.json` - Final test/val metrics + hyperparameters
- 📈 `training_history_v2.6.csv` - 15 epochs of training logs
- 📉 `training_history_v2.6.png` - Training curves showing oscillation
- 📝 `training_summary_v2.6.json` - Summary statistics
- 🎯 `test_predictions.csv` - Model predictions on 5,707 test samples

### Model File
- 🤖 **best_model_v2.6_beta_esm2_3b.pth** (15.18 GB)
  - [Download from Hugging Face](https://huggingface.co/Kroea/AbAg-binding-prediction/blob/main/best_model_v2.6_beta_esm2_3b.pth) ⭐ Recommended

## Installation

```bash
# Clone repository
git clone https://github.com/melmbrain/AbAg_binding_prediction.git
cd AbAg_binding_prediction

# Checkout v2.6 tag
git checkout v2.6.0-beta

# Install dependencies
pip install -r requirements.txt

# Download model from Hugging Face
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="Kroea/AbAg-binding-prediction",
    filename="best_model_v2.6_beta_esm2_3b.pth"
)
```

## Quick Start

```python
import torch
from model import DualEncoderWithCrossAttention

# Load model
checkpoint = torch.load('best_model_v2.6_beta_esm2_3b.pth')
model = DualEncoderWithCrossAttention(dropout=0.3, use_cross_attention=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# See README_v2.6.md for full usage examples
```

## What's Next: v2.7

**Expected**: December 2025

### Critical Fixes Planned
1. ✅ **Stable MSE + BCE Loss** (no Soft Spearman)
2. ✅ **Prediction Clamping** [4.0, 14.0]
3. ✅ **NaN/Inf Detection**
4. ✅ **Complete RNG State Saving**
5. ✅ **Overfitting Monitoring**

### Expected Improvements
- Spearman: 0.39 → **0.45-0.55** (+15-40%)
- Recall: 18-100% (unstable) → **50-70%** (stable)
- RMSE: 2.1 → **1.2-1.5** (-30%)
- Pred range: -2.48 to 10.0 → **4.0 to 14.0** (valid)

See [V2.7_IMPROVEMENTS.md](https://github.com/melmbrain/AbAg_binding_prediction/blob/main/V2.7_IMPROVEMENTS.md) for complete roadmap with research references.

## Comparison

| Version | Status | Spearman | RMSE | Stability | Use Case |
|---------|--------|----------|------|-----------|----------|
| v2.5 | ✅ Stable | 0.42 | 1.95 | ✅ Stable | Production |
| **v2.6-beta** | ⚠️ Experimental | 0.39 | 2.10 | ❌ Unstable | Educational |
| v2.7 | 🔜 Next | 0.45-0.55 | 1.2-1.5 | ✅ Fixed | Production |

**Recommendation**: Use v2.5 for production. Wait for v2.7 stable release.

## Support

- 📚 [Documentation](https://github.com/melmbrain/AbAg_binding_prediction#readme)
- 🐛 [Report Issues](https://github.com/melmbrain/AbAg_binding_prediction/issues)
- 💬 [Discussions](https://github.com/melmbrain/AbAg_binding_prediction/discussions)
- 📖 [Full Changelog](https://github.com/melmbrain/AbAg_binding_prediction/blob/main/CHANGELOG.md)

## Citation

```bibtex
@software{abag_binding_v26,
  title = {Antibody-Antigen Binding Prediction v2.6},
  author = {Joshua Yoon},
  year = {2025},
  month = {11},
  version = {2.6.0-beta},
  url = {https://github.com/melmbrain/AbAg_binding_prediction},
  note = {Experimental release with documented stability issues}
}
```

---

**⚠️ Important**: This is an experimental release with known limitations. For production use, please use v2.5 or wait for v2.7 stable release.

**Status**: Superseded by v2.7 (in development)
**Released**: 2025-11-25
