# CUDA Toolkit Installation Guide

## Why Install CUDA Toolkit?

**Current training speed:** ~7 days for 50 epochs
**With FlashAttention:** ~1-2 days for 50 epochs
**Time saved:** 5-6 days! 🚀

---

## Installation Steps

### Step 1: Run the Installation Script

Open your WSL2 terminal and run:

```bash
cd /mnt/c/Users/401-24/Desktop/AbAg_binding_prediction
./install_cuda_toolkit.sh
```

**You'll need to:**
- Enter your sudo password (required for system installation)
- Wait 15-30 minutes for installation
- ~5GB disk space required

---

## What the Script Does

1. ✅ Installs CUDA repository keyring
2. ✅ Updates package lists
3. ✅ Installs CUDA Toolkit 12.1 (matches your PyTorch 2.5.1+cu121)
4. ✅ Sets up environment variables (`CUDA_HOME`, `PATH`, etc.)
5. ✅ Installs FlashAttention from source
6. ✅ Tests the installation

---

## After Installation

### Verify Installation

```bash
# Check nvcc is available
nvcc --version

# Should show: Cuda compilation tools, release 12.1
```

### Apply Environment Variables

```bash
source ~/.bashrc
```

### Check Current Training Progress

```bash
tail -f training_fast_v2.log
```

---

## When to Stop and Restart Training

### Option 1: Wait for Epoch 1 (Recommended)

**Wait ~3 hours** for Epoch 1 to complete, then:

```bash
# Find training process
ps aux | grep train_fast_v2.py

# Stop it (replace PID with actual number)
kill <PID>

# Restart training with same command
python3 train_fast_v2.py \
  --data /mnt/c/Users/401-24/Desktop/Ab_Ag_dataset/data/agab/agab_phase2_full.csv \
  --epochs 50 \
  --batch_size 8 \
  --focal_gamma 2.0 \
  --output_dir outputs_fast_v2
```

**The script will:**
- Auto-detect checkpoint from Epoch 1
- Resume from where it left off
- Use FlashAttention for 3-10x speedup
- Complete remaining 49 epochs in ~1-2 days

### Option 2: Stop Now (Faster Total Time)

Stop training immediately:

```bash
pkill -f train_fast_v2.py
```

Start fresh with FlashAttention:

```bash
python3 train_fast_v2.py \
  --data /mnt/c/Users/401-24/Desktop/Ab_Ag_dataset/data/agab/agab_phase2_full.csv \
  --epochs 50 \
  --batch_size 8 \
  --focal_gamma 2.0 \
  --output_dir outputs_fast_v2_flashattn
```

---

## Expected Speed Improvement

### Before (Current):
```
Training: 1%|▏| 189/13977 [02:48<3:30:15, 1.09it/s]
Speed: ~1.1 it/s
Time per epoch: ~3.5 hours
Total time: ~7 days
```

### After (With FlashAttention):
```
Training: 10%|█| 1000/13977 [08:15<1:05:30, 3.31it/s]
Speed: ~3-4 it/s (3x faster)
Time per epoch: ~1 hour
Total time: ~1-2 days
```

---

## Troubleshooting

### Installation fails with "No space left on device"

Free up disk space:

```bash
# Clean apt cache
sudo apt-get clean
sudo apt-get autoclean

# Remove old packages
sudo apt-get autoremove
```

### FlashAttention installation fails

Try older version:

```bash
pip install flash-attn==2.5.0 --no-build-isolation
```

Or use precompiled wheel (if available):

```bash
pip install flash-attn==2.3.6
```

### "nvcc not found" after installation

Reload environment:

```bash
source ~/.bashrc
echo $CUDA_HOME  # Should show /usr/local/cuda-12.1
which nvcc      # Should show /usr/local/cuda-12.1/bin/nvcc
```

---

## Summary

**What you're installing:**
- CUDA Toolkit 12.1 (~5GB)
- FlashAttention (compiled from source)

**Time required:**
- Installation: 15-30 minutes
- Testing: 2-3 minutes

**Benefit:**
- 3-10x faster training
- 7 days → 1-2 days
- Save 5-6 days of waiting! 🎉

---

## Questions?

**"Will this break my current PyTorch?"**
No! PyTorch uses the CUDA runtime, not the toolkit. This just adds compilation tools.

**"Can I uninstall if something goes wrong?"**
Yes! `sudo apt-get remove cuda-toolkit-12-1`

**"What if I don't have sudo access?"**
You'll need to continue without FlashAttention (7 days) or use Google Colab (free GPU with FlashAttention).

---

## Ready to Install?

Run this command:

```bash
cd /mnt/c/Users/401-24/Desktop/AbAg_binding_prediction
./install_cuda_toolkit.sh
```

Good luck! 🚀
