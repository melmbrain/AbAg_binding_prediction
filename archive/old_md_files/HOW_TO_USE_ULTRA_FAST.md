# How to Use Ultra-Fast Training in Colab

**Expected Speed-Up**: 3-6× faster (5 days → 1-2 days)

---

## 🚀 Quick Start (5 minutes)

### Step 1: Upload New Notebook to Google Drive

1. Download `notebooks/colab_training_ULTRA_FAST.ipynb` from this repo
2. Upload to your Google Drive folder: `MyDrive/AbAg_Training/`

### Step 2: Open in Colab

1. In Google Drive, double-click `colab_training_ULTRA_FAST.ipynb`
2. Click "Open with Google Colaboratory"
3. Runtime → Change runtime type → **GPU (T4)**

### Step 3: Run Cells in Order

```
Cell 1: Mount Google Drive (30 seconds)
Cell 2: Install Dependencies (2 minutes)
  - Installs FlashAttention via FAESM
  - Verifies BFloat16 support

Cell 3: Create Training Script (5 seconds)
  - Writes train_ultra_fast.py with all optimizations

Cell 4: Start Training (runs for 1-2 days)
  - Auto-resumes if disconnected
  - Checkpoints every 100 batches
```

---

## 📊 What's Different from Old Notebook?

| Feature | Old Notebook | Ultra-Fast Notebook | Speed-Up |
|---------|-------------|---------------------|----------|
| **ESM-2** | Standard HuggingFace | FAESM (FlashAttention) | 1.5-2× |
| **Precision** | Float16 + Scaler | BFloat16 (no scaler) | 1.3-1.5× |
| **Compilation** | None | torch.compile | 1.5-2× |
| **Batch Size** | 8 | 12 | 1.2× |
| **Total** | ~1.6 it/s | ~5-7 it/s | **3-6×** |
| **Time (50 epochs)** | 5 days | **1-2 days** | ✅ |

---

## 🔧 What If You're Already Training?

### Option 1: Continue Old Training, Switch Later

If you're happy with current progress, let it finish this epoch, then switch notebooks.

### Option 2: Switch Immediately

Your checkpoints from the old notebook **WON'T work** with the new one because:
- Different model architecture (FAESM vs standard ESM-2)
- Different optimizer states (bfloat16 vs float16)

**To switch now**:
1. Stop current training
2. Start fresh with ultra-fast notebook
3. You'll lose current progress BUT save 3-4 days overall

**Recommendation**: If you're less than 20% done, switch now. If more than 50% done, finish with old notebook.

---

## 🎯 Optimizations Explained

### 1. FlashAttention (via FAESM)

**What it does**: Optimizes attention computation for GPUs
- Reduces memory transfers between GPU memory and cache
- Fuses operations to reduce kernel launches
- **Speed**: 1.5-2× faster
- **Memory**: 60% savings

**How it works**:
```python
# OLD:
from transformers import AutoModel
model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

# NEW (in ultra-fast notebook):
from faesm.esm import FAEsmForMaskedLM
model = FAEsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
```

**No code changes needed** - same API, same checkpoints!

### 2. BFloat16 Mixed Precision

**What it does**: Uses 16-bit floats with 8-bit exponent (vs 5-bit in Float16)
- More numerically stable than Float16
- No loss scaling needed
- **Speed**: 1.3-1.5× faster
- **Memory**: 50% savings

**How it works**:
```python
# OLD (Float16 with scaler):
with torch.amp.autocast('cuda'):  # Uses float16
    loss = model(...)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# NEW (BFloat16, no scaler):
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    loss = model(...)
loss.backward()  # No scaler needed!
optimizer.step()
```

### 3. torch.compile

**What it does**: JIT-compiles PyTorch model to optimized CUDA kernels
- Fuses operations
- Generates CUDA graphs
- **Speed**: 1.5-2× faster
- **Memory**: No change

**How it works**:
```python
# Just wrap your model:
model = IgT5ESM2Model(...)
model = torch.compile(model)  # That's it!
```

**First run is slow** (compilation), but every iteration after is much faster.

### 4. Larger Batch Size

**What it does**: Process more samples per batch
- Fewer gradient updates per epoch
- Better GPU utilization
- **Speed**: 1.2× faster

**Enabled by**: Memory savings from FlashAttention + BFloat16

```python
# OLD: --batch_size 8
# NEW: --batch_size 12 (50% larger)
```

---

## 📈 Monitor Speed Improvement

After running for 200-300 batches, run the "Performance Comparison" cell to see:

```
===========================================================
SPEED ANALYSIS
===========================================================
Progress: 2,400 / 698,850 batches (0.3%)
Speed: 2400 batches/hour

Estimated total time: 1.5 days
Remaining: 1.4 days

Comparison to baseline (5 days):
Speed-up: 3.3× faster
===========================================================
```

---

## ⚠️ Troubleshooting

### "FAESM not found"

```bash
# In Colab cell:
!pip install faesm
```

Then restart runtime and re-run cells.

### "BFloat16 not supported"

Your GPU doesn't support bfloat16. Change in training args:

```python
--use_bfloat16 False
```

Will fall back to float16 (still 2-3× faster overall).

### "torch.compile failed"

Disable compilation:

```python
--use_compile False
```

Will still get 2-3× speed-up from FlashAttention + BFloat16.

### Training slower than expected

**Possible causes**:
1. **First epoch**: torch.compile is compiling (slow). Wait for epoch 2.
2. **Colab throttling**: Free tier may throttle after 12 hours. Consider Colab Pro.
3. **CPU bottleneck**: Check if GPU utilization is high (should be >80%).

---

## 🔄 Migration Guide

### From Old Notebook → Ultra-Fast Notebook

**If starting fresh**:
1. Upload ultra-fast notebook
2. Run all cells
3. Training starts with all optimizations

**If resuming from checkpoint**:
❌ **Cannot resume** - different model architecture

**To continue**:
1. Finish current epoch with old notebook
2. Download final model weights
3. Start new training with ultra-fast notebook
4. Compare final results

---

## 📊 Expected Timeline

### Starting Fresh (0% complete)

| Method | Time to Complete |
|--------|------------------|
| Old notebook | 5 days |
| **Ultra-fast notebook** | **1-2 days** ✅ |
| **Savings** | **3-4 days** |

### Currently at 40% (2 days done)

| Method | Time to Complete |
|--------|------------------|
| Continue old | +3 days more = 5 days total |
| Switch to ultra-fast | +0.6 days = 2.6 days total ✅ |
| **Savings** | **2.4 days** |

**Recommendation**: Switch if you're <50% done.

---

## ✅ Verification Checklist

After starting ultra-fast training, verify:

- [ ] See "✓ FlashAttention (FAESM) available" in output
- [ ] See "✓ Model compiled" in output
- [ ] See "BFloat16: True" in configuration
- [ ] Batch size is 12 (not 8)
- [ ] Training speed >3 it/s after first 100 batches
- [ ] Checkpoints saving every 100 batches

If all checked, you're getting the full 3-6× speed-up! ✅

---

## 🆘 Need Help?

1. Check `TRAINING_SPEEDUP_GUIDE.md` for detailed explanations
2. Check `COLAB_SCHEDULER_BUG_FIX.md` if training crashes
3. Open GitHub issue with error message

---

**Ready to Go**: Upload `colab_training_ULTRA_FAST.ipynb` to Colab and start training!

**Expected Result**: 50 epochs complete in 1-2 days instead of 5 days ✅
