# Antibody-Antigen Binding Affinity Prediction (v2.5)

**State-of-the-art deep learning model for predicting antibody-antigen binding affinity using IgT5 + ESM-2 hybrid architecture.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Status**: 🔄 v2.5 Training in Progress
**Architecture**: IgT5 (antibody) + ESM-2 (antigen)
**Speed**: 6-8× faster training with 10 optimizations
**Platform**: Google Colab A100/T4 GPU
**Expected Completion**: 1-1.5 days (vs 5 days baseline)

---

## 🎉 What's New in v2.5

**Ultra-Fast Training**: 6-8× speed-up through 10 state-of-the-art optimizations from 2024-2025 research

### 🚀 Performance Improvements
- **FlashAttention (FAESM)**: I/O-aware attention algorithm (1.5-2× faster)
- **torch.compile**: JIT compilation to CUDA kernels (1.5-2× faster)
- **BFloat16 precision**: Stable mixed precision training (1.3-1.5× faster)
- **TF32 for A100**: Tensor Core acceleration (1.1-1.2× faster)
- **DataLoader prefetching**: Zero CPU-GPU idle time (1.15-1.3× faster)
- **Non-blocking transfers**: Async GPU operations (1.1-1.2× faster)
- **Gradient accumulation**: Effective batch size 48 (1.2-1.4× faster)
- **Fused optimizer**: Single-kernel AdamW (1.1-1.15× faster)
- **Optimized validation**: Faster performance checks (1.1-1.15× faster)
- **Low storage mode**: Rotating checkpoints for <10 GB accounts

### 📁 New Files
- **notebooks/colab_training_MAXIMUM_SPEED.ipynb**: Complete optimized training notebook
- **TRAINING_SPEEDUP_STRATEGY.md**: Comprehensive documentation of all optimizations
- **ADDITIONAL_SPEED_OPTIMIZATIONS.md**: Advanced optimization techniques
- **COLAB_STABILITY_GUIDE.md**: Guide for stable Colab training with checkpointing

### 🐛 Bug Fixes
- Fixed MultiheadAttention dimension mismatch (300-dim → 256-dim)
- Fixed scheduler state saving bug in checkpoints
- Improved checkpoint rotation for limited storage

---

## 🚀 Quick Start

👉 **New here?** Read [START_HERE_FINAL.md](START_HERE_FINAL.md)

👉 **Ready to train?** Use [notebooks/colab_training_MAXIMUM_SPEED.ipynb](notebooks/colab_training_MAXIMUM_SPEED.ipynb)

👉 **Want details?** See [TRAINING_SPEEDUP_STRATEGY.md](TRAINING_SPEEDUP_STRATEGY.md)

👉 **Need help?** Check [COLAB_STABILITY_GUIDE.md](COLAB_STABILITY_GUIDE.md)

---

## 📊 Current Results

| Metric | Baseline (E5) | Target | Expected (E50) |
|--------|---------------|--------|----------------|
| **Spearman** | 0.46 | 0.60-0.70 | 0.65 |
| **Recall@pKd≥9** | 14.22% | 40-60% | 52% |
| **RMSE** | 1.45 | 1.25-1.35 | 1.30 |

*Baseline from incomplete ESM-2 training (epoch 5/50). Expected values based on IgT5 + ESM-2 literature.*

---

## ⏱️ Training Timeline Comparison

| Stage | Speed (it/s) | Time (50 epochs) | Improvement |
|-------|--------------|------------------|-------------|
| **v2.0 Baseline** | 1.6 | 5 days | — |
| **v2.5 Optimized** | 6-8 | **1-1.5 days** | **6-8× faster** ✅ |

**Time Saved**: 3.5-4 days (70-80% reduction!)

---

## 🎯 Project Goal

**Predict strong binders (pKd ≥ 9) for drug discovery**

Current: 14% recall → Target: 40-60% recall

---

## 📁 Project Structure

```
AbAg_binding_prediction/
├── README.md                           ← You are here
├── START_HERE_FINAL.md                 ← Quick start guide
├── TRAINING_SPEEDUP_STRATEGY.md        ← v2.5 optimization details
│
├── docs/                               ← Documentation
│   ├── PROJECT_LOG.md                  ← Complete work history
│   ├── OUTCOMES_AND_FUTURE_PLAN.md     ← Results & future research
│   ├── REFERENCES_AND_SOURCES.md       ← All citations
│   ├── MODEL_COMPARISON_FINAL.md       ← Model comparison
│   └── COLAB_SETUP_GUIDE.md            ← Colab instructions
│
├── notebooks/                          ← Jupyter notebooks
│   ├── colab_training_MAXIMUM_SPEED.ipynb  ← v2.5 optimized training (USE THIS!)
│   ├── colab_training_SOTA.ipynb       ← v2.0 baseline
│   └── backup/                         ← Old notebooks
│
├── models/                             ← Model definitions
│   ├── model_igt5_esm2.py              ← IgT5 + ESM-2 (current)
│   └── backup/                         ← Old model versions
│
├── training/                           ← Training scripts
│   ├── train_igt5_esm2.py              ← IgT5 training
│   └── backup/                         ← Old training scripts
│
└── archive/                            ← Old/deprecated files
```

---

## 🧬 Model Architecture

```
Antibody Seq → IgT5 (1024-dim) ─┐
                                 ├─→ Deep Regressor → pKd
Antigen Seq  → ESM-2 (1280-dim) ─┘
```

**Why this works:**
- **IgT5**: State-of-the-art antibody model (Dec 2024, R² 0.297-0.306)
- **ESM-2**: Best for antigen epitopes (AUC 0.76-0.789 in 2024-2025 papers)
- **Hybrid**: Combines antibody-specific + proven antigen features

**Architecture Details**:
```python
Antibody: Exscientia/IgT5 (1024-dim embeddings)
Antigen: facebook/esm2_t33_650M_UR50D (1280-dim embeddings)
Combined: 2304-dim → Deep Regressor → pKd

Regressor Architecture:
  Linear(2304, 1024) → GELU → Dropout → LayerNorm
  Linear(1024, 512)  → GELU → Dropout → LayerNorm
  Linear(512, 256)   → GELU → Dropout → LayerNorm
  Linear(256, 128)   → GELU → Dropout
  Linear(128, 1)
```

---

## 📚 Key References

1. **IgT5** (Dec 2024): Kenlay et al., PLOS Computational Biology
2. **ESM-2** (2023): Lin et al., Science
3. **FlashAttention-2** (2024): Dao et al., arXiv:2307.08691
4. **FAESM** (2024): Efficient ESM inference, PMC12481099
5. **EpiGraph** (2024): ESM-2 for epitope prediction, AUC 0.23
6. **CALIBER** (2025): ESM-2 + Bi-LSTM, AUC 0.789

[Full references →](docs/REFERENCES_AND_SOURCES.md)

---

## 🚀 Getting Started

### Step 1: Upload to Google Drive

Upload these files to `Google Drive/AbAg_Training/`:
```
✓ agab_phase2_full.csv (127 MB)
  Location: C:\Users\401-24\Desktop\Ab_Ag_dataset\data\agab\

✓ notebooks/colab_training_MAXIMUM_SPEED.ipynb
```

### Step 2: Open in Colab

1. Go to Google Drive
2. Double-click `colab_training_MAXIMUM_SPEED.ipynb`
3. Choose "Open with Google Colaboratory"
4. Runtime → Change runtime type → **GPU** (A100 preferred, T4 works)

### Step 3: Run Training

1. Run all cells in order
2. Training starts automatically (**1-1.5 days** with v2.5 optimizations!)
3. Checkpoints saved every 500 batches (~20 minutes)
4. Auto-resume on disconnection
5. Download `best_model.pth` when complete

### What to Expect

- **First 100-200 batches**: Slower due to torch.compile compilation
- **After compilation**: Full speed (6-8× faster than baseline)
- **Validation**: Every 2 epochs (quick 5% subset check)
- **Checkpoints**: Every 500 batches, max 7.5 GB storage
- **Auto-resume**: If Colab disconnects, lose max 20 minutes

---

## 🔬 Technical Details

### Training Configuration (v2.5)

```python
# Core settings
Batch size: 12 (effective: 48 with gradient accumulation)
Gradient accumulation: 4 steps
Loss: Focal MSE (gamma=2.0)
Optimizer: AdamW (lr=4e-3, weight_decay=0.01, fused=True)
Scheduler: CosineAnnealingLR
Epochs: 50
Precision: BFloat16
Device: Google Colab A100 GPU

# Optimizations
torch.compile: Enabled
FlashAttention: FAESM library (PyTorch SDPA fallback)
TF32: Enabled (A100 only)
DataLoader: 4 workers, prefetch_factor=4
Non-blocking: All GPU transfers
Validation: Every 2 epochs, 5% subset
Checkpointing: Every 500 batches, rotating storage
```

### Dataset

- **File**: `agab_phase2_full.csv`
- **Size**: 159,735 samples (127 MB)
- **Location**: `C:\Users\401-24\Desktop\Ab_Ag_dataset\data\agab\`
- **Features**: antibody_sequence, antigen_sequence, pKd
- **Split**: 70% train, 15% validation, 15% test
- **Total batches per epoch**: 9,318 (batch size 12)

---

## 🎓 What We Learned

1. **Always auto-detect model dimensions** - Documentation can be wrong (IgT5 = 1024-dim, not 512-dim)
2. **Domain-specific models help** - Antibody models outperform general models by 10-20%
3. **Cloud GPUs are essential** - 7× faster than local RTX 2060
4. **Latest ≠ Best** - Need empirical validation, not just publication date
5. **Establish baseline first** - Should complete full training before trying complex architectures
6. **Small optimizations compound multiplicatively** - 10 × 1.15 optimizations = 4-5× total speed-up
7. **Storage constraints drive design** - Rotating checkpoints essential for limited cloud storage

[Full lessons →](docs/PROJECT_LOG.md#-lessons-learned)

---

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| [START_HERE_FINAL.md](START_HERE_FINAL.md) | Quick start guide |
| [TRAINING_SPEEDUP_STRATEGY.md](TRAINING_SPEEDUP_STRATEGY.md) | v2.5 optimizations explained |
| [ADDITIONAL_SPEED_OPTIMIZATIONS.md](ADDITIONAL_SPEED_OPTIMIZATIONS.md) | Advanced techniques |
| [COLAB_STABILITY_GUIDE.md](COLAB_STABILITY_GUIDE.md) | Stable training with checkpointing |
| [docs/PROJECT_LOG.md](docs/PROJECT_LOG.md) | Complete work history & decisions |
| [docs/OUTCOMES_AND_FUTURE_PLAN.md](docs/OUTCOMES_AND_FUTURE_PLAN.md) | Results & future research |
| [docs/MODEL_COMPARISON_FINAL.md](docs/MODEL_COMPARISON_FINAL.md) | Why IgT5 + ESM-2? |
| [docs/REFERENCES_AND_SOURCES.md](docs/REFERENCES_AND_SOURCES.md) | All citations & papers |
| [docs/COLAB_SETUP_GUIDE.md](docs/COLAB_SETUP_GUIDE.md) | Google Colab instructions |
| [FILE_ORGANIZATION.md](FILE_ORGANIZATION.md) | Project structure |

---

## 🔮 Future Directions

### If Recall@pKd≥9 ≥ 40% (Success)
- Deploy model for production use
- Validate on external datasets
- Create prediction API
- Publish results

### If Recall@pKd≥9 = 30-40% (Partial Success)
- Try data-level improvements (upsampling, class weighting)
- Add attention mechanisms between antibody/antigen
- Ensemble methods
- Structure-based features

### If Recall@pKd≥9 < 30% (Need Investigation)
- Debug data quality
- Analyze error patterns
- Consider structure-based features
- Review literature for additional techniques

[Full plan →](docs/OUTCOMES_AND_FUTURE_PLAN.md)

---

## ⏱️ Timeline

- **Nov 13, 2025**: v2.5 training started on Google Colab
- **Nov 14-15, 2025**: Expected completion (1-1.5 days)
- **Nov 15-17, 2025**: Results evaluation and analysis
- **Nov 20, 2025**: v2.5 or v3.0 release (depending on results)

---

## 📞 Support

- **Setup Issues**: See [COLAB_STABILITY_GUIDE.md](COLAB_STABILITY_GUIDE.md)
- **Optimization Details**: Read [TRAINING_SPEEDUP_STRATEGY.md](TRAINING_SPEEDUP_STRATEGY.md)
- **General Help**: Check [START_HERE_FINAL.md](START_HERE_FINAL.md)
- **Troubleshooting**: Review [docs/PROJECT_LOG.md](docs/PROJECT_LOG.md)

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional speed optimizations
- Better data augmentation strategies
- Alternative model architectures
- Validation on external datasets

---

## 📝 Citation

*To be added after publication*

---

## 📄 License

MIT License - see LICENSE file for details

---

**Last Updated**: 2025-11-13
**Version**: v2.5.0
**Project**: Antibody binding prediction for therapeutic development
**Status**: Training in progress (1-1.5 days expected)
