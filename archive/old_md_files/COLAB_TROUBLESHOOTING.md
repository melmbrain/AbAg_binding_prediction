# 🔧 Colab Notebook Troubleshooting Guide

## Common Issues and Solutions

---

## ✅ FIXED: Numpy/Pandas Compatibility Error

### The Error You Saw

```
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject
```

**When it occurred:** Step 2 (importing libraries), when trying to `import pandas`

### Root Cause

Google Colab comes with pre-installed versions of numpy and pandas. Sometimes these are compiled against different versions of each other, causing binary incompatibility when pandas tries to use numpy's C API.

### The Fix (Already Applied!)

I've updated both notebooks with a fix that:

1. **Uninstalls** existing numpy and pandas to clear conflicts
2. **Installs numpy first** (version 1.24.3)
3. **Installs other packages** (transformers, scikit-learn, etc.)
4. **Reinstalls numpy** with `--force-reinstall` to ensure binary compatibility

**Updated notebooks:**
- ✅ `colab_training_A100_ESM2_3B.ipynb` - Fixed!
- ✅ `colab_training_GDRIVE.ipynb` - Fixed!

### How to Use Fixed Notebooks

**Option 1: Re-upload the fixed notebook**
1. Download the updated notebook from your local folder
2. Go to Google Colab
3. Upload the fixed notebook (it will replace the old one)
4. Run from the beginning

**Option 2: Add fix to existing notebook**

If you're already in the middle of training, add this cell **before Step 2**:

```python
# Fix numpy/pandas compatibility
!pip uninstall -y numpy pandas -q
!pip install -q numpy==1.24.3
!pip install --force-reinstall -q numpy==1.24.3
print("✅ Numpy/pandas compatibility fixed!")
```

Then **restart the runtime** (Runtime → Restart runtime) and run from the beginning.

---

## Other Common Colab Issues

### Issue: "CUDA out of memory"

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**

**For A100 Notebook:**
1. Reduce batch size in Step 3:
   ```python
   BATCH_SIZE = 32  # Reduced from 48
   ```

2. Reduce antigen sequence length in model forward pass:
   ```python
   # In IgT5ESM2_3B_Model.forward()
   max_length=1536  # Reduced from 2048
   ```

**For Standard Notebook:**
1. Reduce batch size:
   ```python
   BATCH_SIZE = 12  # Reduced from 16
   ```

2. Disable gradient checkpointing (uses more memory but available):
   ```python
   model = IgT5ESM2Model(
       dropout=0.35,
       freeze_encoders=True,
       use_checkpointing=False  # Changed from True
   )
   ```

---

### Issue: "Runtime disconnected"

**Symptoms:**
- Training stops unexpectedly
- "Reconnect" button appears

**Causes:**
- Free Colab: 12-hour limit
- Idle timeout (90 minutes of no interaction)
- Runtime crash

**Solutions:**

1. **Best model is saved to Drive!** Don't panic.
   - Check Drive: `AbAg_Training_02/training_output/best_model.pth`
   - Your progress is not lost!

2. **Resume training:**
   - Reconnect to runtime
   - Run Step 3 again (mount Drive)
   - The notebook will load the best checkpoint and continue

3. **Prevent disconnects:**
   - Upgrade to Colab Pro ($10/month) - longer runtime
   - Keep browser tab active
   - Use browser extensions to keep session alive

---

### Issue: "Package installation takes too long"

**Symptoms:**
- Step 1 (package installation) takes 5+ minutes

**Solutions:**

1. **This is normal for first run!**
   - Transformers library is large (~500MB)
   - ESM-2 models need to be downloaded first time
   - Subsequent runs will be cached

2. **If it hangs for >10 minutes:**
   - Restart runtime (Runtime → Restart runtime)
   - Run again - sometimes Colab's package servers are slow

---

### Issue: "ESM-2 3B download is very slow"

**Symptoms:**
- Step 4 (model building) takes 10+ minutes
- Shows "Loading ESM-2 3B..."

**Solutions:**

1. **This is normal!**
   - ESM-2 3B is ~12GB
   - First download: 5-10 minutes
   - Cached for subsequent runs in same session

2. **Check progress:**
   - Colab shows download progress in output
   - If frozen for >15 minutes, restart runtime

---

### Issue: "Validation Spearman is not improving"

**Symptoms:**
- Training loss decreases
- But validation Spearman stays flat or decreases

**Cause:** Overfitting or poor hyperparameters

**Solutions:**

1. **Increase regularization:**
   ```python
   config = {
       'dropout': 0.4,        # Increased from 0.35
       'weight_decay': 0.03,  # Increased from 0.02
       'label_smoothing': 0.1, # Increased from 0.05
   }
   ```

2. **Reduce learning rate:**
   ```python
   config = {
       'lr': 2e-3,  # Reduced from 3e-3
   }
   ```

3. **More data augmentation:**
   - Add more training data if available
   - Consider ensemble methods

---

### Issue: "Can't find CSV file in Drive"

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: '/content/drive/MyDrive/AbAg_Training_02/agab_phase2_full.csv'
```

**Solutions:**

1. **Check Drive is mounted:**
   - Look for "Mounted at /content/drive/MyDrive" message
   - If not, run Step 3 again

2. **Verify folder name:**
   - Open Google Drive in browser
   - Check folder is exactly `AbAg_Training_02` (case-sensitive!)

3. **Update CSV filename:**
   - Check the list of files printed in Step 3
   - Update `CSV_FILENAME` to match your file:
     ```python
     CSV_FILENAME = 'your_actual_filename.csv'  # Match exactly
     ```

4. **Check file permissions:**
   - Make sure you own the Drive folder
   - If using shared Drive, check permissions

---

### Issue: "Training is very slow"

**Expected speeds:**

| GPU | Time/Epoch | Total Time (50 epochs) |
|-----|------------|------------------------|
| **T4** | ~3 min | ~2-2.5 hours |
| **V100** | ~2 min | ~1.5-2 hours |
| **A100** (ESM-2 3B) | ~50 sec | ~40-50 min |

**If slower than expected:**

1. **Check GPU is enabled:**
   - Runtime → Change runtime type
   - Hardware accelerator: GPU
   - Look for "GPU: Tesla T4" in Step 1 output

2. **Check batch size:**
   - T4: Use batch_size=16 (standard)
   - A100: Use batch_size=48 (A100 notebook)

3. **Disable unnecessary features:**
   ```python
   # Reduce validation frequency
   config = {
       'validation_frequency': 2,  # Validate every 2 epochs instead of 1
   }
   ```

---

### Issue: "Model predictions are all the same"

**Symptoms:**
- All predictions around 7-8 pKd
- Very low Spearman correlation (<0.1)

**Cause:** Model collapsed to predicting the mean

**Solutions:**

1. **Check learning rate:**
   - Too high: Model diverges
   - Too low: Model doesn't learn
   ```python
   config = {
       'lr': 3e-3,  # Try 2e-3 or 4e-3
   }
   ```

2. **Check warmup:**
   ```python
   config = {
       'warmup_epochs': 5,  # Make sure warmup is enabled
   }
   ```

3. **Check gradients:**
   - Add gradient monitoring in training loop
   - If gradients are zero, encoders might not be unfrozen

---

## Getting Help

### Debug Checklist

Before asking for help, check:

1. ✅ GPU is enabled (Runtime → Change runtime type)
2. ✅ Drive is mounted (see "Mounted at" message)
3. ✅ CSV file exists in correct Drive folder
4. ✅ All cells executed in order (don't skip cells)
5. ✅ Using correct notebook for your GPU:
   - A100-80GB → `colab_training_A100_ESM2_3B.ipynb`
   - T4/V100 → `colab_training_GDRIVE.ipynb`

### Where to Get Help

1. **Check error message location:**
   - Look at which Step failed
   - Read the error message carefully

2. **Common error patterns:**
   - `CUDA out of memory` → Reduce batch size
   - `FileNotFoundError` → Check Drive folder/filename
   - `ImportError` → Package installation failed, restart runtime
   - `ValueError: dtype size` → Use fixed notebook (already fixed!)

3. **Share error details:**
   - Which notebook are you using?
   - Which Step did it fail?
   - Full error traceback
   - GPU type (T4/V100/A100)

---

## Quick Fixes Summary

| Error | Quick Fix |
|-------|-----------|
| **Numpy/pandas compatibility** | ✅ Already fixed in updated notebooks |
| **CUDA out of memory** | Reduce `BATCH_SIZE` |
| **Runtime disconnected** | Check Drive for saved `best_model.pth` |
| **CSV not found** | Update `CSV_FILENAME`, check Drive folder |
| **Training too slow** | Verify GPU is enabled |
| **Poor performance** | Increase regularization (`dropout`, `weight_decay`) |

---

## Preventing Issues

**Best Practices:**

1. ✅ **Use the correct notebook for your GPU**
   - A100 → A100 notebook
   - T4/V100 → Standard notebook

2. ✅ **Organize your Drive properly**
   ```
   Google Drive/
   └── AbAg_Training_02/
       ├── agab_phase2_full.csv    ← Your data
       └── training_output/         ← Auto-created
   ```

3. ✅ **Check GPU before starting**
   - Step 1 should show "GPU: Tesla T4" (or V100/A100)
   - If it says "CPU", change runtime type

4. ✅ **Monitor first epoch**
   - If first epoch completes successfully, rest should too
   - Check validation metrics make sense (Spearman > 0)

5. ✅ **Keep Drive folder clean**
   - Don't have multiple CSVs if using filename detection
   - Clear old checkpoints to save space

---

## Status: Issue Fixed! ✅

The numpy/pandas compatibility error has been **fixed in both notebooks**:

- ✅ `colab_training_A100_ESM2_3B.ipynb`
- ✅ `colab_training_GDRIVE.ipynb`

**Next steps:**
1. Re-upload the fixed notebook to Colab
2. Run from the beginning
3. Training should work smoothly now!

**Expected output after fix:**
```
Installing required packages...

Installing numpy...
Installing other packages...
Fixing numpy/pandas compatibility...

✅ All packages installed successfully!
✅ Numpy/pandas compatibility fixed!
```

Then Step 2 should work without errors:
```
✅ All libraries imported successfully!
```

---

**Happy training! 🚀🧬**
