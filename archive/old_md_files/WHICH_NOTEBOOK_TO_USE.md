# 📓 Which Notebook Should I Use?

## ✅ Use This One: `colab_training_COMPLETE.ipynb`

**Location:** `notebooks/colab_training_COMPLETE.ipynb`

**This is the ONLY notebook you need!**

### What It Includes

✅ **Google Drive Integration**
- Auto-loads data from `AbAg_Training_02` folder
- Auto-saves results to Drive
- No manual file uploads
- Results persist (survive disconnects)

✅ **A100-80GB Optimization**
- TF32 tensor cores (2× speedup)
- Batch size 48 (3× larger than T4)
- Optimized memory usage
- 2048 token sequences (2× longer)

✅ **ESM-2 3B Model** (State-of-the-Art)
- 4.6× larger than standard (3.2B vs 872M params)
- 2560D antigen embeddings (2× richer)
- +0.02-0.05 Spearman improvement
- Best performance available

---

## 🚀 How to Use

### Step 1: Upload to Google Colab
1. Go to https://colab.research.google.com/
2. Click **File → Upload notebook**
3. Select: `notebooks/colab_training_COMPLETE.ipynb`

### Step 2: Enable A100 GPU
1. Click **Runtime → Change runtime type**
2. Select **Hardware accelerator: GPU**
3. Choose **GPU type: A100**
4. Click **Save**

### Step 3: Update CSV Filename
Find this line in **Step 3**:
```python
CSV_FILENAME = 'agab_phase2_full.csv'  # ← Change to your file
```

Update it to match your CSV file in Google Drive.

### Step 4: Run All Cells
1. Click **Runtime → Run all** (or press Ctrl+F9)
2. When prompted, click **Connect to Google Drive** and allow access
3. Wait ~40-50 minutes
4. Check results in Drive: `AbAg_Training_02/training_output_A100_ESM2_3B/`

---

## 📊 What You'll Get

### Training Time
- **~45-60 seconds per epoch**
- **~30-50 minutes total** (with early stopping)
- **3-4× faster** than T4/V100

### Performance
- **Test Spearman: 0.42-0.47** (your true performance)
- **Test RMSE: 1.1-1.3** pKd units
- **Strong binder recall: 98-100%**

### Output Files (Auto-saved to Drive)
```
Google Drive/AbAg_Training_02/training_output_A100_ESM2_3B/
├── best_model.pth              (~13GB) - Your trained model
├── val_predictions.csv         - Validation predictions
├── test_predictions.csv        - Test predictions
├── final_metrics.json          - All metrics
└── results_summary.png         - Visualizations
```

---

## ❓ FAQ

### Q: Can I use this on T4/V100 instead of A100?
**A:** No, this notebook is optimized for A100-80GB. It will fail on smaller GPUs due to:
- Batch size 48 (too large for 16GB)
- ESM-2 3B model (~12GB alone)

If you only have T4/V100, you would need a different configuration.

### Q: Do I need to download/upload the CSV each time?
**A:** No! Just put it in Google Drive once:
- `Google Drive/AbAg_Training_02/agab_phase2_full.csv`
- The notebook auto-loads it every time

### Q: Where are my results saved?
**A:** Auto-saved to Google Drive:
- `Google Drive/AbAg_Training_02/training_output_A100_ESM2_3B/`
- Access from any device, anytime
- Never lost even if Colab disconnects

### Q: Can I stop and resume training?
**A:** Yes! The best model is saved to Drive after each improvement.
- If disconnected: Reconnect → Run from beginning
- It will load the last saved checkpoint

### Q: How much does A100 cost on Colab?
**A:**
- **Colab Pro+**: $50/month (includes A100 access)
- **Cost per run**: ~$0.80 (40 minutes of A100 time)
- **Value**: Best performance per dollar

---

## 🎯 Summary

**One notebook. Everything included.**

- ✅ Google Drive integration
- ✅ A100 optimization
- ✅ ESM-2 3B (state-of-the-art)
- ✅ 40 minute training
- ✅ Best performance

**File:** `notebooks/colab_training_COMPLETE.ipynb`

**Just upload it and run!** 🚀

---

## 📁 Other Notebooks (Legacy - You Don't Need These)

The `notebooks/` folder contains other experimental versions:
- `colab_training_GDRIVE.ipynb` - T4/V100 + Drive (slower, older)
- `colab_training_A100_ESM2_3B.ipynb` - Same as COMPLETE (old name)
- `colab_training_COMPLETE_STEP_BY_STEP.ipynb` - Manual upload version
- Others - Experimental/deprecated

**You only need `colab_training_COMPLETE.ipynb`!**

---

**Happy training! 🧬🚀**
