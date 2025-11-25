# Antibody-Antigen Binding Prediction (IgT5 + ESM-2)

**Deep learning model for predicting antibody-antigen binding affinity using dual protein language models.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**Current Version**: v2.6.0-beta (2025-11-25)
**Status**: ⚠️ Experimental Release - v2.7 in development
**Architecture**: IgT5 (antibody) + ESM-2 3B (antigen) with Cross-Attention

---

## 📦 Releases

### v2.6.0-beta (2025-11-25) - Current

**Status**: ⚠️ Experimental (known stability issues - not for production)

- **Model**: IgT5 + ESM-2 3B with cross-attention
- **Performance**:
  - Spearman ρ: 0.390
  - RMSE: 2.10
  - Recall@pKd≥9: 100%
- **Training**: 15 epochs on A100 80GB (~60 hours)
- **Known Issues**:
  - Recall instability (oscillates 18% ↔ 100%)
  - Invalid predictions (negative pKd values)
  - See [v2.6/README_v2.6.md](v2.6/README_v2.6.md) for details
- **Download**: [GitHub Release](https://github.com/melmbrain/AbAg_binding_prediction/releases/tag/v2.6.0-beta) | Model Card: [README_v2.6.md](v2.6/README_v2.6.md)

### v2.5.0 (2025-11-13) - Previous Stable

- **Model**: ESM-2 650M
- **Performance**: Spearman 0.42, RMSE 1.95
- **Status**: ✅ Stable (use for production)
- **Download**: See [CHANGELOG.md](CHANGELOG.md#250---2025-11-13)

### v2.7.0 (In Development) - Next

**Expected**: 2025-12-01

- **Fixes**: Stable MSE loss, prediction clamping, NaN detection, complete RNG state
- **Expected Performance**: Spearman 0.45-0.55, stable recall 50-70%
- **Roadmap**: [V2.7_IMPROVEMENTS.md](V2.7_IMPROVEMENTS.md)

### Pre-trained Models (Hugging Face)

Models hosted at: [Kroea/AbAg-binding-prediction](https://huggingface.co/Kroea/AbAg-binding-prediction)

```python
from huggingface_hub import hf_hub_download

# Download v2.6-beta (experimental - 16GB)
model_path = hf_hub_download(
    repo_id="Kroea/AbAg-binding-prediction",
    filename="best_model_v2.6_beta_esm2_3b.pth"
)

# Or use v2.5 (stable - 4.7GB)
model_path = hf_hub_download(
    repo_id="Kroea/AbAg-binding-prediction",
    filename="best_model_v2.5_esm2_650m.pth"
)
```

> ⚠️ **Important**: v2.6.0-beta has documented stability issues (recall oscillation, invalid predictions). For production, use v2.5 or wait for v2.7 stable release.

---

## 🚀 Quick Start (Google Colab)

### 1. Upload to Google Drive
Place these files in `/MyDrive/AbAg_Training/`:
- `train_ultra_speed_v26.py`
- `agab_phase2_full.csv`

### 2. Run in Colab
```python
# Cell 1: Setup
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/MyDrive/AbAg_Training')

# Cell 2: Install
!pip install -q transformers pandas scipy scikit-learn tqdm sentencepiece faesm bitsandbytes accelerate

# Cell 3: Train
!python train_ultra_speed_v26.py
```

**That's it!** Auto-resumes from checkpoints.

---

## 📊 Performance

- **Speed**: 4.45 iterations/second (confirmed working)
- **Time per epoch**: ~26 minutes
- **50 epochs**: ~21-22 hours total
- **Speedup**: ~5.5× faster than original baseline
- **Memory**: ~12GB GPU (T4/V100/A100)

---

## 🔧 Active Optimizations (17/19)

✅ **Batch embedding generation** - 2-3× faster (biggest win!)
✅ **Sequence bucketing** - 1.3-1.5× faster
✅ **INT8 quantization** - Encoders only
✅ **Activation checkpointing** - Enables batch 16
✅ **BFloat16 mixed precision**
✅ **FlashAttention** - If FAESM available
✅ **Fused AdamW optimizer**
✅ **Gradient accumulation** (×3)
✅ **DataLoader prefetching** (4 workers)
✅ **Async checkpoint saving**
✅ **TF32 precision** (A100)
✅ **Cudnn benchmark mode**
✅ **Fast tokenizers**
✅ **Disk auto-cleanup**
❌ **torch.compile** - DISABLED (prevents CUDA graphs errors)

---

## 📁 Important Files

| File | Purpose |
|------|---------|
| `train_ultra_speed_v26.py` | ✅ **Main training script (WORKING)** |
| `WORKING_CONFIG.md` | Config documentation (4.45 it/s) |
| `notebooks/colab_training_SIMPLE.ipynb` | Simple Colab notebook |
| `archive/experimental_cuda_fix_2025-01-14/` | Old experimental files |

---

## 🔬 Model Architecture

```
Input: Antibody sequence + Antigen sequence
  ↓
IgT5 Encoder (frozen)  →  Mean pooling  →  [1280-dim]
ESM-2 Encoder (frozen) →  CLS token    →  [1280-dim]
  ↓
Concatenate [2560-dim]
  ↓
Regressor: 1024 → 512 → 256 → 128 → 1
  ↓
Output: pKd prediction
```

**Loss**: Focal MSE (gamma=2.0)
**Optimizer**: AdamW (fused, lr=4e-3, weight_decay=0.01)

---

## 💾 Storage & Checkpoints

**Auto-saved files**:
- `checkpoint_latest.pth` - Latest state (auto-resume)
- `checkpoint_backup.pth` - Previous checkpoint
- `best_model.pth` - Best validation Spearman
- `checkpoint_epoch.pth` - End of epoch

**Disk management**:
- Auto cleanup every epoch
- Ultra-aggressive cleanup at 150GB
- Max 4 checkpoint files (~7.5GB total)

---

## ⚙️ Configuration (Working Settings)

**DO NOT CHANGE - This config is proven stable!**

```python
# Training
batch_size = 16
accumulation_steps = 3
epochs = 50
learning_rate = 4e-3

# Optimizations
use_compile = False          # ❌ DISABLED (prevents CUDA graphs errors)
use_checkpointing = True     # ✅ ENABLED (saves memory)
use_quantization = True      # ✅ ENABLED (INT8 for encoders)
use_bucketing = True         # ✅ ENABLED (efficient batching)
use_bfloat16 = True          # ✅ ENABLED (mixed precision)
use_fused_optimizer = True   # ✅ ENABLED (faster optimizer)
```

---

## 🐛 Troubleshooting

### CUDA Graphs Error
**Fixed!** Current `train_ultra_speed_v26.py` has nuclear fix (lines 28-43):
- `torch.compiler.disable()` at import time
- `use_compile=False` in config
- No more crashes ✅

### Out of Memory
Current config uses batch 16 with checkpointing.
If still OOM: Reduce to batch 12, accumulation 4

### Disk Space Full
Auto-cleanup triggers at 150GB.
Manual cleanup: See `WORKING_CONFIG.md`

---

## 📈 Expected Results

- **Validation Spearman**: ~0.7-0.8
- **Recall@pKd≥9**: ~70-80%
- **Training time**: ~21-22 hours
- **Model size**: ~2.6GB

---

## 📚 Documentation

- `WORKING_CONFIG.md` - Detailed working config docs
- `START_HERE.md` - Getting started guide (if exists)
- `archive/experimental_cuda_fix_2025-01-14/` - Old experimental docs

---

## 📁 Project Structure

```
AbAg_binding_prediction/
├── train_ultra_speed_v26.py          # Main script ✅
├── WORKING_CONFIG.md                  # Config docs
├── README.md                          # This file
├── data/
│   └── agab_phase2_full.csv          # Dataset (159k samples)
├── notebooks/
│   ├── colab_training_SIMPLE.ipynb
│   └── colab_training_ULTRA_SPEED_v26.ipynb
├── src/
│   └── model_v3_full_dim.py          # Model source
├── outputs_max_speed/                 # Checkpoints
└── archive/
    └── experimental_cuda_fix_2025-01-14/  # Old experiments
```

---

## 📝 Citation

If you use this code:
- **IgT5**: [Exscientia/IgT5](https://huggingface.co/Exscientia/IgT5)
- **ESM-2**: [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D)

---

## ⚠️ Critical Notes

1. **Current `train_ultra_speed_v26.py` is WORKING** at 4.45 it/s
2. **Do not modify config** unless you understand CUDA graphs issue
3. **All experiments archived** in `archive/experimental_cuda_fix_2025-01-14/`
4. **Nuclear fix applied** - torch.compile disabled globally

---

## 🚀 Version History

- **v2.6** (Current) - ✅ Stable, 17/19 optimizations, 4.45 it/s, no errors
- v2.5 - ❌ CUDA graphs errors with torch.compile + checkpointing
- v2.0 - Initial optimized version
- v1.0 - Baseline implementation

---

**Last Updated**: 2025-01-14
**Status**: ✅ WORKING - 4.45 it/s, no CUDA graphs errors
