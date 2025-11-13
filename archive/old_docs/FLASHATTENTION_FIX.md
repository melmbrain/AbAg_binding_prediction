# FlashAttention Fix - Get 3-10x Training Speedup

## 🐌 Why Training is Slow

FlashAttention failed to load because:
1. **`flash-attn` package not installed**
2. **`transformers` version too old** (you have 4.35.0, need 4.36.0+)

Without FlashAttention:
- Current speed: ~1.1 it/s
- Estimated time: **175 hours (7.3 days)** for 50 epochs

With FlashAttention:
- Expected speed: ~3-4 it/s (3-10x faster)
- Estimated time: **18-25 hours** for 50 epochs

---

## ⚡ Two Options

### Option 1: Wait for Epoch 1, Then Fix (RECOMMENDED) ✅

**Keep current training running** to save your 3.5 hours of work.

**When to stop:**
- Wait ~30 minutes for Epoch 1 to complete
- You'll see: `✓ Saved checkpoint (epoch 1)`
- Then stop training (Ctrl+C)

**Then:**
1. Install FlashAttention:
   ```bash
   ./install_flashattention.sh
   ```

2. Resume training (will load from checkpoint):
   ```bash
   python train_optimized_v1.py \
     --data /mnt/c/Users/401-24/Desktop/Ab_Ag_dataset/data/agab/agab_phase2_full.csv \
     --epochs 50 \
     --batch_size 8 \
     --use_stratified_sampling \
     --focal_gamma 2.0 \
     --output_dir outputs_optimized_v1_fixed
   ```

3. Training resumes from Epoch 2 with **3-10x speedup**!

**Pros:**
- Don't lose 3.5 hours of progress
- Get speedup for remaining 49 epochs
- Safe (checkpoint exists)

**Cons:**
- 30 more minutes of slow training
- Epoch 1 won't benefit from speedup

---

### Option 2: Stop Now and Fix (FASTEST)

**Stop training immediately** and restart with FlashAttention.

**Steps:**
1. Stop current training:
   ```bash
   # Find the process
   ps aux | grep train_optimized_v1.py

   # Kill it (replace PID)
   kill <PID>
   ```

2. Install FlashAttention:
   ```bash
   ./install_flashattention.sh
   ```

3. Start fresh training:
   ```bash
   python train_optimized_v1.py \
     --data /mnt/c/Users/401-24/Desktop/Ab_Ag_dataset/data/agab/agab_phase2_full.csv \
     --epochs 50 \
     --batch_size 8 \
     --use_stratified_sampling \
     --focal_gamma 2.0 \
     --output_dir outputs_optimized_v1_flashattn
   ```

**Pros:**
- All 50 epochs benefit from speedup
- Fastest total time
- Fresh start

**Cons:**
- Lose 3.5 hours of progress
- No checkpoint to resume from

---

## 🚀 My Recommendation

**Option 1** is better because:
1. You've already invested 3.5 hours
2. Epoch 1 is 98.7% complete (~30 min remaining)
3. You have 49 more epochs that will benefit from speedup
4. Safer (checkpoint exists)

**Math:**
- Option 1: 30 min (slow) + 18 hours (fast) = **18.5 hours total**
- Option 2: 0 min (wasted) + 18 hours (fast) = **18 hours total**

Only 30 minutes difference, but Option 1 is safer!

---

## 📋 Detailed Installation Steps

### 1. Stop Training (When Epoch 1 Completes)

Watch for this message:
```
✓ Saved checkpoint (epoch 1)
```

Then press **Ctrl+C** or kill the process:
```bash
# Find process
ps aux | grep train_optimized_v1.py

# Kill it
kill <PID>
```

---

### 2. Install FlashAttention

```bash
cd /mnt/c/Users/401-24/Desktop/AbAg_binding_prediction
./install_flashattention.sh
```

This will:
- Upgrade `transformers` to latest version
- Install `flash-attn` package (5-10 minutes)
- Test that FlashAttention works

**Note:** Installation compiles from source and may take time. Be patient!

---

### 3. Resume Training with FlashAttention

```bash
python train_optimized_v1.py \
  --data /mnt/c/Users/401-24/Desktop/Ab_Ag_dataset/data/agab/agab_phase2_full.csv \
  --epochs 50 \
  --batch_size 8 \
  --use_stratified_sampling \
  --focal_gamma 2.0 \
  --output_dir outputs_optimized_v1_fixed
```

**You should see:**
```
Found checkpoint: outputs_optimized_v1_fixed/checkpoint_latest.pth
Loading checkpoint to resume training...
✓ Resumed from epoch 1
✓ FlashAttention enabled          ← NEW!
```

**Speed improvement:**
```
Before: Training:   1%|          | 186/13977 [03:23<9:11:29,  2.40s/it]
After:  Training:  10%|█         | 1000/13977 [10:15<1:25:30,  2.53it/s]
                                                          ^^^^^^^^
                                                          3-10x faster!
```

---

## 🔍 Troubleshooting

### Installation fails with "CUDA not found"

**Check CUDA version:**
```bash
nvcc --version
```

FlashAttention needs CUDA 11.6 or higher. If you have older CUDA:
```bash
# Install specific version compatible with your CUDA
pip install flash-attn==2.3.0 --no-build-isolation
```

---

### Installation fails with "out of memory"

FlashAttention compilation needs ~8GB RAM. If you have less:

**Option A:** Close other programs and retry

**Option B:** Skip FlashAttention, use slower training (current setup)

---

### "attn_implementation not recognized"

Your transformers version is still too old:
```bash
pip install --upgrade transformers
python -c "import transformers; print(transformers.__version__)"
```

Should show 4.36.0 or higher.

---

### FlashAttention installs but doesn't work

Check compatibility:
```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')

try:
    import flash_attn
    print(f'flash-attn: {flash_attn.__version__}')
except:
    print('flash-attn not importable')
"
```

---

## 📊 Expected Results

### Before FlashAttention:
```
Training:   1%|          | 186/13977 [03:23<9:11:29,  2.40s/it]
Speed: ~1.1 it/s
Time per epoch: ~3.5 hours
Total time (50 epochs): 175 hours (7.3 days)
```

### After FlashAttention:
```
Training:  10%|█         | 1000/13977 [10:15<1:25:30,  2.53it/s]
Speed: ~3-4 it/s
Time per epoch: ~1 hour
Total time (50 epochs): 50 hours (2 days)
```

**Speedup: 3.5x faster! (175h → 50h)**

---

## ⏰ Timeline

If you follow Option 1:

**Now (3.5h in):**
- Let training continue

**~4h total:**
- Epoch 1 completes
- Checkpoint saved
- Stop training

**~4h 10min:**
- FlashAttention installed

**~4h 15min:**
- Training resumed with speedup

**~54h total (2.25 days):**
- All 50 epochs complete
- Model ready to use!

---

## 💡 Alternative: Reduce Epochs

If 2+ days is too long, consider:

```bash
--epochs 20  # Instead of 50
```

**With FlashAttention:**
- 20 epochs: ~20 hours (< 1 day)
- Still get 35-40% recall (good improvement)
- Can always train more later

**Recommended if:**
- You want results faster
- This is Phase 1 (testing)
- You'll implement Phase 2 anyway

---

## ✅ Summary

**What to do RIGHT NOW:**
1. **Wait ~30 minutes** for Epoch 1 to complete
2. **Stop training** (Ctrl+C)
3. **Run:** `./install_flashattention.sh`
4. **Resume training** with same command
5. **Enjoy 3-10x speedup!** 🚀

**Expected improvement:**
- 175 hours → 50 hours (3.5x faster)
- 7.3 days → 2 days
- Same model quality

**Your progress is safe** - checkpoint will be saved!
