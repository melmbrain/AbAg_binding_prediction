# START HERE - Quick Setup Guide

**Your situation:** You have raw data but haven't trained models yet. Let's fix that!

---

## ✅ Good News: You Have Data!

I found your dataset at:
```
/mnt/c/Users/401-24/Desktop/Ab_Ag_dataset/data/agab/
```

**Available files:**
- `agab_phase2_full.csv` - 159,736 samples (127 MB) ⭐ Use this one!
- `agab_phase2_balanced.csv` - Smaller balanced set
- `agab_phase2_sample.csv` - Sample subset
- `agab_full_dataset.csv` - Full 2.5 GB dataset

---

## 🚀 Quick Start - 3 Simple Steps

### Step 1: Upload Data to Google Drive (5 minutes)

1. **Copy your data file:**
   ```bash
   # On your Windows machine, copy this file to Google Drive:
   C:\Users\401-24\Desktop\Ab_Ag_dataset\data\agab\agab_phase2_full.csv
   ```

2. **Upload to Google Drive:**
   - Go to https://drive.google.com
   - Create folder: `AbAg_data`
   - Upload `agab_phase2_full.csv` to this folder

### Step 2: Open in Google Colab (2 minutes)

1. **Convert Python script to notebook:**
   ```bash
   # Install jupytext (if not installed)
   pip install jupytext

   # Convert the script to notebook
   cd /mnt/c/Users/401-24/Desktop/AbAg_binding_prediction
   jupytext --to notebook COMPLETE_COLAB_TRAINING.py
   ```

2. **Upload to Colab:**
   - Go to https://colab.research.google.com
   - Upload `COMPLETE_COLAB_TRAINING.ipynb`
   - OR manually copy the contents of `COMPLETE_COLAB_TRAINING.py`

### Step 3: Run Training (15-20 hours)

1. **Enable GPU:**
   - Runtime → Change runtime type → GPU → T4 (or better)

2. **Update the path in Part 2:**
   ```python
   DRIVE_DATA_PATH = "/content/drive/MyDrive/AbAg_data/agab_phase2_full.csv"
   ```

3. **Run all cells** (Ctrl+F9)

4. **Wait for completion** (~15-20 hours total):
   - Embedding generation: ~10-12 hours
   - Model training: ~3-5 hours

---

## ⚡ Alternative: Quick Test (30 minutes)

Want to test everything works first? Use the sample dataset:

**Use this path instead:**
```python
DRIVE_DATA_PATH = "/content/drive/MyDrive/AbAg_data/agab_phase2_sample.csv"
```

This will:
- Process only ~7,000 samples (vs 159K)
- Take ~30-45 minutes total
- Verify everything works before full training

---

## 📊 What You'll Get

After training completes, you'll have in Google Drive:

```
AbAg_outputs/
├── best_model.pth              # Trained model weights
├── test_predictions.csv        # Predictions on test set
├── results_summary.json        # Performance metrics
├── dataset_with_embeddings.csv # Full dataset with features
└── checkpoint_epoch_*.pth      # Training checkpoints
```

**Expected Performance (based on v2):**
- RMSE: ~1.3-1.5
- MAE: ~1.1-1.3
- R²: ~0.55-0.65
- Spearman ρ: ~0.40-0.50

---

## 🔧 Troubleshooting

**Problem: "Out of memory" error**
- Solution: Reduce batch size in Part 6:
  ```python
  BATCH_SIZE = 64  # or 48, or 32
  ```

**Problem: "Session disconnected"**
- Solution: The script saves checkpoints every 1000 samples
- Just re-run the cell to resume from last checkpoint

**Problem: "Runtime disconnected during embedding generation"**
- Solution: Colab free tier has 12-hour limit
- Use Colab Pro ($9.99/month) for 24-hour sessions
- Or split into multiple runs using checkpoints

**Problem: "Data file not found"**
- Check the path in `DRIVE_DATA_PATH`
- Make sure you mounted Google Drive (Part 2)

---

## 📝 Manual Alternative (No Notebook Conversion)

If you can't convert to notebook, just:

1. Open Google Colab
2. Create a new notebook
3. Copy-paste the code from `COMPLETE_COLAB_TRAINING.py` cell by cell
4. Each `# %%` marks a new cell
5. Run them in order

---

## ⏱️ Time Estimate Breakdown

| Step | Time | What happens |
|------|------|--------------|
| Setup | 5 min | Install packages, mount Drive |
| Load data | 2 min | Read CSV, check format |
| **Generate embeddings** | **10-12 hrs** | ESM-2 processes 159K sequences |
| Prepare data | 5 min | Create train/val/test splits |
| **Train model** | **3-5 hrs** | 100 epochs of training |
| Evaluate | 5 min | Test set evaluation |
| **TOTAL** | **~15-20 hrs** | |

**💡 Tip:** Start the run before going to bed, it will finish by morning!

---

## 🎯 Next Steps After Training

Once training is complete:

1. **Download the model:**
   ```python
   from google.colab import files
   files.download(f'{OUTPUT_DIR}/best_model.pth')
   ```

2. **Use for predictions:**
   - Model is saved with full config
   - Can load and use locally
   - See `examples/basic_usage.py` for inference code

3. **Analyze results:**
   - Check `test_predictions.csv` for detailed results
   - Plot predictions vs true values
   - Analyze errors by affinity range

4. **Iterate:**
   - Try different hyperparameters
   - Experiment with model architecture
   - Add more training data

---

## 📧 Need Help?

If you run into issues:
1. Check the error message
2. Look in Troubleshooting section above
3. Check that your CSV has columns: `antibody_sequence`, `antigen_sequence`, `pKd`
4. Verify GPU is enabled in Colab

---

**Ready to start? Upload your data to Google Drive and run the notebook! 🚀**
