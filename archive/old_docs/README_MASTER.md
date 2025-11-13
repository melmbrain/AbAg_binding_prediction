# Antibody-Antigen Binding Prediction

**🎯 Current Status:** Training in progress (Day 1/7)
**📊 Progress:** Epoch 1/50
**⏰ Expected Completion:** 2025-11-17

---

## 🚀 Quick Links

**📖 Complete Documentation:** [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md)
**🔬 Active Training Script:** `train_ultra_optimized.py`
**📝 Training Log:** `training_ultra_optimized.log`
**💾 Outputs:** `outputs_ultra_optimized/`

---

## 📊 Current Training

**Model:** ESM-2 650M + Regression Head (1.4M trainable params)
**Loss:** Focal MSE (gamma=2.0) - focuses on extreme affinities
**Batch Size:** 16 (effective: 64 with gradient accumulation)
**GPU:** RTX 2060 (100% utilization, 74% memory)
**Speed:** 1.75 it/s (~3.4 hours per epoch)

---

## 🎯 Goal

Predict antibody-antigen binding affinity (pKd) with **>60% recall** on strong binders (pKd ≥ 9).

**Previous:** 17% recall ❌
**Target:** 60-80% recall ✅

---

## 📁 File Organization

### ✅ Active (Keep)
- `train_ultra_optimized.py` - **BEST SCRIPT** (in use)
- `training_ultra_optimized.log` - Current training log
- `outputs_ultra_optimized/` - Model checkpoints & results
- `PROJECT_DOCUMENTATION.md` - **MASTER DOCUMENTATION**
- `COMPLETE_COLAB_TRAINING.py` - Colab alternative

### ❌ Deprecated (Can delete after training)
- All `train_fast_v2.py` related files
- All `train_optimized_v1.py` related files
- Old logs: `training_fast_v2.log`, `training_final.log`, etc.
- Old outputs: `outputs_fast_v2*/`, `outputs_optimized_v1*/`
- Old docs: See cleanup script in PROJECT_DOCUMENTATION.md

---

## 🔍 Monitor Progress

**View live training:**
```bash
# WSL/Linux
tail -f training_ultra_optimized.log

# Windows CMD
wsl tail -f /mnt/c/Users/401-24/Desktop/AbAg_binding_prediction/training_ultra_optimized.log

# PowerShell
Get-Content training_ultra_optimized.log -Wait -Tail 50
```

**Check GPU:**
```bash
nvidia-smi
```

**Check process:**
```bash
ps aux | grep train_ultra
```

---

## 📈 Expected Results (After 7 Days)

| Metric | Previous | Target |
|--------|----------|--------|
| Recall @ pKd≥9 | 17% | 60-80% |
| RMSE | ~2.5 | <2.0 |
| Spearman | ~0.4 | >0.6 |
| R² | ~0.3 | >0.5 |

---

## ⚙️ Technical Details

**Hardware:**
- GPU: NVIDIA RTX 2060 6GB
- OS: Windows 11 + WSL2 (Ubuntu)
- CUDA: 12.6

**Optimizations:**
- ✅ Focal Loss (gamma=2.0) - focuses on hard examples
- ✅ Large effective batch (64) - better gradients
- ✅ cuDNN benchmark - 1.3-1.7x speedup
- ✅ TF32 Tensor Cores - faster matmul
- ✅ Mixed precision (AMP) - faster training
- ✅ 4 DataLoader workers - async data loading
- ✅ 100% GPU utilization - fully optimized

**Why No FlashAttention:**
RTX 2060 is Turing architecture. FlashAttention requires Ampere (RTX 3000+) or newer.

---

## 📚 Full Documentation

For complete details, see: **[`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md)**

Includes:
- Detailed training script comparison
- All configuration parameters
- Troubleshooting guide
- Cleanup instructions
- Technical architecture
- Next steps after training

---

## 🧹 Cleanup (After Training Completes)

See cleanup script in [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md#cleanup-script)

**When to clean:**
- After training completes successfully
- After verifying results are good
- After backing up best model

---

## 📞 Quick Troubleshooting

**Training stopped?**
1. Check log: `tail -100 training_ultra_optimized.log`
2. Check GPU: `nvidia-smi`
3. Check process: `ps aux | grep train_ultra`

**OOM errors?**
- Reduce batch size: `--batch_size 12` or `8`
- Reduce workers: `--num_workers 2`

**Slow training?**
- Already optimized! 1.75 it/s is near-maximum for RTX 2060

---

**Last Updated:** 2025-11-10
**Training Started:** 2025-11-10 14:28
**Expected Completion:** 2025-11-17 (~7 days)
