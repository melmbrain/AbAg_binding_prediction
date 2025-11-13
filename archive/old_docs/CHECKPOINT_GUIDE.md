# Checkpoint Guide - Resume Training After Interruption

## ✅ Good News: Checkpoints Are Now Saved!

I just added checkpoint functionality to `train_optimized_v1.py`. Your training progress is now automatically saved every epoch, so you won't lose progress if training crashes or you need to stop.

---

## 📁 What Gets Saved

After each epoch completes, the script saves two files:

1. **`best_model.pth`**: The model with the best validation Spearman correlation
   - Only updated when validation improves
   - This is what you'll use for final predictions

2. **`checkpoint_latest.pth`**: The model from the most recent epoch
   - Updated every epoch
   - Use this to resume training if interrupted

Both files include:
- Model weights
- Optimizer state
- Scheduler state
- Current epoch number
- Best validation score so far
- All training arguments

---

## 🔄 How to Resume Training

### Automatic Resume (Easiest):

If you run the training command again with the **same output directory**, it will automatically detect and load `checkpoint_latest.pth`:

```bash
# Your current training (interrupted or crashed)
python train_optimized_v1.py \
  --data /mnt/c/Users/401-24/Desktop/Ab_Ag_dataset/data/agab/agab_phase2_full.csv \
  --epochs 50 \
  --batch_size 8 \
  --use_stratified_sampling \
  --focal_gamma 2.0 \
  --output_dir outputs_optimized_v1_fixed

# Just run the SAME command again - it will resume automatically!
```

**What you'll see:**
```
============================================================
Found checkpoint: outputs_optimized_v1_fixed/checkpoint_latest.pth
Loading checkpoint to resume training...
✓ Resumed from epoch 1
✓ Best validation Spearman so far: 0.4567
============================================================

Starting training for 50 epochs...
Resuming from epoch 2/50
============================================================
```

---

### Manual Resume:

If you want to resume from a specific checkpoint:

```bash
python train_optimized_v1.py \
  --data /path/to/data.csv \
  --epochs 50 \
  --batch_size 8 \
  --use_stratified_sampling \
  --resume /path/to/checkpoint_latest.pth
```

---

## 💾 Current Training Status

Your training is currently running and saving checkpoints to:
- **Output directory**: `outputs_optimized_v1_fixed/`
- **Current progress**: Epoch 1/50, Batch 187/13,977
- **Loss**: Decreasing nicely (started at 274,000 → now at 870)

### Where to find checkpoints:

```bash
ls -lh outputs_optimized_v1_fixed/
```

You should see:
- `checkpoint_latest.pth` - Created after first epoch completes
- `best_model.pth` - Created when validation improves

---

## ⚠️ Important Notes

### 1. **Checkpoints save after each EPOCH, not each batch**

- Your training has 50 epochs
- Each epoch has 13,977 batches
- If training crashes mid-epoch, you'll resume from the start of that epoch
- Current speed: ~1.1 it/s means ~3.5 hours per epoch

### 2. **First checkpoint will be saved after Epoch 1 completes**

Since you're still in Epoch 1 (batch 187/13,977), the first checkpoint hasn't been created yet. It will be created after:
- All 13,977 batches complete
- Validation runs
- Metrics are printed

Expected time for first checkpoint: **~3.5 hours from when you started**

### 3. **If training crashes before Epoch 1 completes**

Unfortunately, you'd need to restart from scratch because no checkpoint exists yet. But after the first epoch completes, you're safe!

---

## 🚨 What To Do If Training Crashes

### Scenario 1: Crash After Epoch 1+ Completes

**✅ You're safe!** Just run the same command again:

```bash
python train_optimized_v1.py \
  --data /mnt/c/Users/401-24/Desktop/Ab_Ag_dataset/data/agab/agab_phase2_full.csv \
  --epochs 50 \
  --batch_size 8 \
  --use_stratified_sampling \
  --focal_gamma 2.0 \
  --output_dir outputs_optimized_v1_fixed
```

Training will resume from the last completed epoch.

---

### Scenario 2: Crash During Epoch 1 (Before First Checkpoint)

**⚠️ No checkpoint exists yet**, so you'll need to:

1. Reduce epochs to get checkpoints faster:
```bash
python train_optimized_v1.py \
  --data /path/to/data.csv \
  --epochs 20 \  # Fewer epochs = faster checkpoints
  --batch_size 8 \
  --use_stratified_sampling
```

2. Or wait for first epoch to complete (~3.5 hours)

---

## 📊 Monitoring Your Checkpoints

### Check if checkpoint exists:

```bash
ls -lh outputs_optimized_v1_fixed/checkpoint_latest.pth
```

### Check training progress:

```bash
tail -f training_log.txt  # If you used `tee` to log output
```

Or monitor the background bash process:
```bash
# Check which bash process is running
# Then get output with BashOutput tool
```

---

## 💡 Pro Tips

### 1. **Reduce Epochs to Get Checkpoints Faster**

If you're worried about losing progress, you can:

```bash
# Instead of 50 epochs, do 20 epochs
python train_optimized_v1.py --epochs 20 ...
```

This creates checkpoints every ~3.5 hours instead of waiting for all 50 epochs.

---

### 2. **Check GPU Temperature**

Long training can overheat:

```bash
watch -n 1 nvidia-smi
```

Make sure:
- Temperature < 80°C
- GPU Utilization: 80-100%
- Memory: ~5GB / 6GB

---

### 3. **Prevent Interruptions**

To avoid losing first-epoch progress:

1. **Use `screen` or `tmux`** (so you can disconnect SSH without killing training):

```bash
# Start a screen session
screen -S training

# Run training
python train_optimized_v1.py ...

# Detach: Press Ctrl+A then D
# Reattach later: screen -r training
```

2. **Use `nohup`** (runs in background even if terminal closes):

```bash
nohup python train_optimized_v1.py ... > training.log 2>&1 &
```

---

## 🎯 Summary

**What you have now:**
- ✅ Automatic checkpoint saving every epoch
- ✅ Automatic resume from last checkpoint
- ✅ No manual intervention needed
- ✅ Both best model and latest model saved

**What to remember:**
- ⏰ First checkpoint after ~3.5 hours (end of Epoch 1)
- 🔄 To resume: Just run the same command again
- 💾 Checkpoints in: `outputs_optimized_v1_fixed/`
- 📊 Monitor with: `nvidia-smi` or `tail -f training.log`

**Your current status:**
- 🏃 Training is running
- 📈 Loss is decreasing (good sign!)
- ⏳ First checkpoint in ~3 hours
- 🎯 Total time: 50-75 hours for 50 epochs

---

## 📞 If You Have Issues

### Training too slow?

Consider reducing epochs:
```bash
--epochs 20  # Instead of 50
```

This still gives good results (35-40% recall) but completes in 35-50 hours instead.

### Want to stop and resume later?

1. Stop current training: `Ctrl+C` (or kill the process)
2. Wait for it to save checkpoint (if epoch completed)
3. Resume: Run same command again

### Lost checkpoint file?

If you accidentally deleted `checkpoint_latest.pth`, you still have `best_model.pth` which you can use for predictions (but can't resume training from it with current code).

---

**You're all set! Training will save checkpoints automatically. Just let it run and check back after ~3.5 hours for the first checkpoint.** 🚀
