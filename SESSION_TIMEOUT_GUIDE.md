# Google Colab Session Timeout - Complete Guide

**The Problem:** Colab free tier disconnects after ~90 minutes of inactivity or 12 hours maximum runtime.

**The Solution:** Your trained models ARE saved! You just need to reload them.

---

## What Gets Lost vs What's Saved

### ❌ Lost When Session Times Out:
- Variables in memory (`model`, `test_predictions`, etc.)
- Code execution state
- Local files (in `/content/` but NOT in Drive)
- Running processes

### ✅ Saved to Google Drive:
- ✅ **Trained model weights** (`best_model.pth`)
- ✅ **Checkpoints** (every 10 epochs)
- ✅ **Training curves** (if saved before timeout)
- ✅ **All plots and results** (if saved before timeout)

**Key Point:** Your training is NOT lost! You just need to reload the model.

---

## Solution 1: Resume in Colab (Recommended)

Use the **resume notebook** to reload and continue from where you left off.

### Steps:

1. **Upload the resume notebook to Colab:**
   - File: `colab_resume_and_evaluate.ipynb`
   - Upload to: https://colab.research.google.com/

2. **Enable GPU:**
   - Runtime → Change runtime type → GPU

3. **Run all cells:**
   - Runtime → Run all
   - It will:
     - Reconnect to Google Drive
     - Load your saved model
     - Run complete evaluation
     - Generate all plots
     - Save results

**Time:** 5-10 minutes (no retraining needed!)

---

## Solution 2: Use Model Locally

Download the model from Google Drive and use it on your local machine.

### Steps:

1. **Download model from Google Drive:**
   ```
   Google Drive/AbAg_data/models/best_model.pth
   ```
   Save to your local project:
   ```
   /mnt/c/Users/401-24/Desktop/AbAg_binding_prediction/models/
   ```

2. **Run evaluation locally:**
   ```bash
   python use_colab_model_locally.py \
     --model models/best_model.pth \
     --data external_data/merged_with_all_features.csv
   ```

**Benefits:**
- No session timeouts
- Use your own GPU
- Full control

---

## Solution 3: Continue Training from Checkpoint

If training was interrupted, resume from the last checkpoint.

### In Colab Resume Notebook:

Find the cell "Load Model" and change to load a checkpoint:

```python
# Load the latest checkpoint
MODEL_PATH = f"{MODEL_DIR}/checkpoint_epoch_50.pth"  # Change to your last checkpoint
```

Then add a new cell to continue training:

```python
# Continue training from checkpoint
EPOCHS_REMAINING = 50  # If you want 100 total and stopped at 50

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

start_epoch = checkpoint['epoch'] + 1

# Continue training loop...
for epoch in range(start_epoch, start_epoch + EPOCHS_REMAINING):
    # Same training code as before
    ...
```

---

## Preventing Timeouts

### Keep Session Alive (Free Tier)

**Browser trick:**
1. Keep Colab tab open
2. Don't let computer sleep
3. Move mouse occasionally
4. Use a browser extension: "Colab Auto Clicker" (unofficial)

**JavaScript console trick:**
```javascript
// In browser console (F12), run:
function KeepClicking(){
    console.log("Clicking");
    document.querySelector("colab-connect-button").click()
}
setInterval(KeepClicking, 60000)
```

**⚠️ Warning:** This is a workaround and may violate Colab terms of service.

### Upgrade to Colab Pro ($10/month)

**Benefits:**
- ✅ 24-hour sessions (vs 12 hours)
- ✅ Faster GPUs (V100, A100 vs T4)
- ✅ Background execution (can close browser)
- ✅ Priority access
- ✅ More RAM

**Worth it if:**
- Training takes > 12 hours
- You train frequently
- You need faster results

---

## Best Practices

### 1. Save Frequently

Add this to your training loop:

```python
# Save checkpoint every 10 epochs
if (epoch + 1) % 10 == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, f'{OUTPUT_DIR}/checkpoint_epoch_{epoch+1}.pth')
    print(f"  💾 Checkpoint saved!")
```

**Already in the training notebook!**

### 2. Monitor Progress

Check Google Drive periodically:
```
Google Drive/AbAg_data/models/
├── checkpoint_epoch_10.pth   ← Latest checkpoint
├── checkpoint_epoch_20.pth
├── best_model.pth             ← Best so far
```

If you see checkpoints appearing, training is working!

### 3. Set Realistic Epochs

**For 100 epochs (~7-10 hours on T4):**
- ✅ Fits in 12-hour limit
- ✅ Should complete before timeout

**For 200+ epochs:**
- ⚠️ May timeout
- Consider splitting: 100 epochs → save → resume for 100 more

### 4. Use Local Data Copy

Already in the notebook:
```python
# Copies to /content/ for faster I/O
!cp "{DRIVE_DATA_PATH}" "/content/merged_with_all_features.csv"
```

This speeds up training significantly!

---

## Quick Reference

### Session Timed Out - What to Do?

**If training was in progress:**
```
1. Check Google Drive for latest checkpoint
2. Upload colab_resume_and_evaluate.ipynb
3. Modify to load latest checkpoint
4. Continue training OR just evaluate if complete
```

**If training completed but evaluation failed:**
```
1. Upload colab_resume_and_evaluate.ipynb
2. Run all cells
3. All plots and metrics will be generated
```

**If you want to use model locally:**
```
1. Download best_model.pth from Google Drive
2. Run: python use_colab_model_locally.py --model models/best_model.pth
```

---

## Files You Created

### In Your Project:
1. **`colab_training.ipynb`** - Main training notebook
2. **`colab_resume_and_evaluate.ipynb`** - Resume/evaluate after timeout ⭐
3. **`use_colab_model_locally.py`** - Use model locally ⭐
4. **`COLAB_TRAINING_GUIDE.md`** - Complete setup guide
5. **`SESSION_TIMEOUT_GUIDE.md`** - This file

### In Google Drive (after training):
```
Google Drive/AbAg_data/
├── merged_with_all_features.csv   (883 MB - dataset)
└── models/
    ├── best_model.pth              (saved during training)
    ├── final_model.pth             (saved at end)
    ├── checkpoint_epoch_10.pth     (every 10 epochs)
    ├── checkpoint_epoch_20.pth
    ├── ...
    ├── training_curves.png         (if saved before timeout)
    ├── predictions_vs_targets.png
    └── training_results.txt
```

---

## Common Scenarios

### Scenario 1: "Training finished but I lost evaluation results"

**Solution:**
```bash
# Upload colab_resume_and_evaluate.ipynb to Colab
# It will:
# - Load best_model.pth
# - Run full evaluation
# - Generate all plots
# - Save everything to Drive
```

### Scenario 2: "Training stopped at epoch 57"

**Solution:**
```python
# In resume notebook, load checkpoint
MODEL_PATH = f"{MODEL_DIR}/checkpoint_epoch_50.pth"  # Latest saved

# Continue training for remaining epochs
# OR just evaluate the checkpoint as-is
```

### Scenario 3: "I want to use the model locally"

**Solution:**
```bash
# Download from Google Drive
# Models folder → best_model.pth → Download

# Then locally:
python use_colab_model_locally.py \
  --model models/best_model.pth \
  --data external_data/merged_with_all_features.csv
```

### Scenario 4: "I don't know which checkpoint to use"

**Solution:**
```
- best_model.pth → Best validation loss (RECOMMENDED)
- final_model.pth → Last epoch (may be overtrained)
- checkpoint_epoch_XX.pth → Specific epoch
```

Always use **best_model.pth** for inference!

---

## Summary

### Key Points:

✅ **Your model IS saved** - Even if session times out
✅ **Resume notebook** - Reload and continue easily
✅ **Local usage** - Download and use on your machine
✅ **Checkpoints** - Save every 10 epochs automatically
✅ **No retraining needed** - Just reload and evaluate

### What to Do NOW:

1. **If session timed out:** Upload `colab_resume_and_evaluate.ipynb` to Colab
2. **If you want local use:** Download model, run `use_colab_model_locally.py`
3. **If training in progress:** Keep tab open, check Drive for checkpoints

### Remember:

**Training models is expensive (time). Storage is cheap.**

Your trained model is SAFE in Google Drive. Session timeouts are annoying but not catastrophic!

---

## Need Help?

**Check Google Drive first:**
- Do you see checkpoint files?
- Is best_model.pth there?
- Check file sizes (should be ~10-50 MB)

**If model files exist → You're fine! Just reload them.**
**If no files → Training didn't save (rare, but check OUTPUT_DIR path)**
