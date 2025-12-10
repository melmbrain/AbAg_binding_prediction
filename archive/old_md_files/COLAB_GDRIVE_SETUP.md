# 📂 Google Drive Colab Notebook - Setup Guide

## Overview

I've created a special version of the Colab notebook that loads data from and saves results to your Google Drive folder `AbAg_Training_02`.

**File:** `notebooks/colab_training_GDRIVE.ipynb`

---

## ✨ What's Different

### Original Notebook vs Google Drive Notebook

| Feature | Original (`colab_training_COMPLETE_STEP_BY_STEP.ipynb`) | Google Drive (`colab_training_GDRIVE.ipynb`) |
|---------|--------------------------------------------------------|---------------------------------------------|
| **Data Loading** | Manual file upload each time | Auto-loads from Drive folder |
| **Model Saving** | Saved to Colab session (lost on disconnect) | Saved to Drive (permanent) |
| **Results** | Need to download manually | Auto-saved to Drive |
| **Setup** | No setup needed | One-time Drive setup |
| **Convenience** | Upload data every run | Data persists in Drive |

**✅ Recommendation:** Use the Google Drive version!

---

## 🚀 Quick Start

### 1. Prepare Your Google Drive

1. Go to your Google Drive
2. You should already have a folder called `AbAg_Training_02`
3. Make sure your CSV dataset is in that folder
4. The notebook will create a `training_output` subfolder automatically

**Your Drive structure:**
```
Google Drive/
└── AbAg_Training_02/
    ├── agab_phase2_full.csv          ← Your data (REQUIRED)
    └── training_output/               ← Created automatically
        ├── best_model.pth
        ├── val_predictions.csv
        ├── test_predictions.csv
        ├── final_metrics.json
        ├── training_curves.png
        ├── predictions_scatter.png
        └── error_distribution.png
```

---

### 2. Upload Notebook to Colab

1. Go to https://colab.research.google.com/
2. Click **File → Upload notebook**
3. Select `colab_training_GDRIVE.ipynb`
4. Click **Upload**

---

### 3. Configure CSV Filename

**IMPORTANT:** Before running, you need to specify your CSV filename!

Find this cell in **Step 3**:
```python
# Load dataset - MODIFY THIS LINE to specify your CSV filename
CSV_FILENAME = 'agab_phase2_full.csv'  # ← CHANGE THIS to your actual CSV filename
```

Change `'agab_phase2_full.csv'` to match your actual CSV filename in Drive.

---

### 4. Enable GPU

1. Click **Runtime → Change runtime type**
2. Select **Hardware accelerator: GPU**
3. Choose **T4** (free tier) or **V100** (Pro)
4. Click **Save**

---

### 5. Run the Notebook

**Option A: Run all at once**
```
Runtime → Run all (Ctrl+F9)
```

**Option B: Run step by step**
- Press `Shift+Enter` to run each cell
- Perfect for learning!

---

## 📊 What Happens

### Step 3: Mount Google Drive

When you run Step 3, you'll see:
```
Mounting Google Drive...
```

A popup will appear asking for permission:
1. Click **Connect to Google Drive**
2. Choose your Google account
3. Click **Allow**

**This is safe!** Colab only accesses the files you specify.

After mounting:
```
✅ Google Drive mounted!

📂 Working directories:
   Data directory: /content/drive/MyDrive/AbAg_Training_02
   Output directory: /content/drive/MyDrive/AbAg_Training_02/training_output

📁 Files in AbAg_Training_02:
   • agab_phase2_full.csv (125.43 MB)
```

---

### During Training

Models and checkpoints are saved directly to Drive:
```
✅ Saved best model to Drive: /content/drive/MyDrive/AbAg_Training_02/training_output/best_model.pth
```

**Benefits:**
- ✅ If Colab disconnects, your progress is saved
- ✅ No need to download files manually
- ✅ Access results from anywhere

---

### After Training

All results automatically appear in your Drive folder:
```
📁 All results saved to: /content/drive/MyDrive/AbAg_Training_02/training_output
```

You can access them immediately from Google Drive on any device!

---

## 📁 Output Files Location

After training, check your Google Drive:

```
Google Drive/
└── AbAg_Training_02/
    └── training_output/
        ├── best_model.pth              ← Trained model (~3.5GB)
        ├── val_predictions.csv         ← Validation predictions
        ├── test_predictions.csv        ← Test predictions
        ├── final_metrics.json          ← All metrics
        ├── training_curves.png         ← Training visualization
        ├── predictions_scatter.png     ← Pred vs actual plots
        └── error_distribution.png      ← Error distribution
```

**No need to download!** Files are already in your Drive.

---

## 💡 Tips & Tricks

### Tip 1: Check Files Before Running
The notebook lists all CSV files in your Drive folder:
```python
📁 Files in AbAg_Training_02:
   • agab_phase2_full.csv (125.43 MB)
   • dataset_v2.csv (98.32 MB)
```

Make sure you see your data file!

### Tip 2: Multiple Training Runs
If you want to keep results from multiple runs, create subfolders:
```python
# Modify in Step 3:
OUTPUT_DIR = f'{DRIVE_DIR}/training_output_run1'  # Run 1
OUTPUT_DIR = f'{DRIVE_DIR}/training_output_run2'  # Run 2
```

### Tip 3: Resume After Disconnect
If Colab disconnects:
1. Reconnect to runtime
2. Remount Drive (Step 3)
3. Load your checkpoint:
```python
checkpoint = torch.load('/content/drive/MyDrive/AbAg_Training_02/training_output/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Tip 4: Share Results
Since results are in Drive, you can:
- Share the folder with collaborators
- Access from phone/tablet
- Create shareable links to specific files

---

## ⚙️ Customization

### Change Data Directory

If your data is in a different Drive folder:

```python
# In Step 3, modify:
DRIVE_DIR = '/content/drive/MyDrive/AbAg_Training_02'  # ← Change this
```

Examples:
```python
DRIVE_DIR = '/content/drive/MyDrive/MyData'
DRIVE_DIR = '/content/drive/MyDrive/Projects/AbAg'
DRIVE_DIR = '/content/drive/Shared drives/TeamData/AbAg'
```

### Use Multiple Datasets

To easily switch between datasets:
```python
# Option 1: Comment/uncomment
# CSV_FILENAME = 'agab_phase2_full.csv'
CSV_FILENAME = 'agab_phase3_full.csv'

# Option 2: User input
CSV_FILENAME = input("Enter CSV filename: ")
```

---

## 🔍 Troubleshooting

### Problem: "No such file or directory"
**Solution:** Check that:
1. Drive is mounted correctly
2. Folder name is exactly `AbAg_Training_02`
3. CSV file exists in that folder
4. `CSV_FILENAME` matches your file name exactly (case-sensitive!)

### Problem: "Permission denied"
**Solution:**
1. Remount Drive (run Step 3 again)
2. Make sure you clicked "Allow" in the permission popup

### Problem: "Out of disk space"
**Solution:**
- Free tier: 15GB Drive storage
- Model checkpoints are ~3.5GB each
- Delete old checkpoints or upgrade Drive storage

### Problem: Training stopped unexpectedly
**Solution:**
- Check if Colab disconnected (free tier has time limits)
- Your best model is saved in Drive!
- Just remount and continue evaluation

---

## 🎯 Complete Example Workflow

### First Time Setup (5 minutes)
1. ✅ Upload notebook to Colab
2. ✅ Enable GPU
3. ✅ Modify `CSV_FILENAME` in Step 3
4. ✅ Run all cells
5. ✅ Grant Drive access when prompted

### Subsequent Runs (0 setup time!)
1. ✅ Open same notebook in Colab
2. ✅ Run all cells
3. ✅ Everything loads automatically from Drive

**That's it!** No more uploading data every time.

---

## 📊 Comparing Results Across Runs

Since all results are in Drive, you can easily compare:

```python
import pandas as pd
import json

# Load results from different runs
with open('training_output_run1/final_metrics.json') as f:
    run1 = json.load(f)

with open('training_output_run2/final_metrics.json') as f:
    run2 = json.load(f)

# Compare
print(f"Run 1 Test Spearman: {run1['test']['spearman']:.4f}")
print(f"Run 2 Test Spearman: {run2['test']['spearman']:.4f}")
```

---

## 🎓 Benefits Summary

### For Convenience
✅ Data persists between sessions
✅ No repeated uploads
✅ Results automatically saved
✅ Access from any device

### For Safety
✅ Models survive disconnects
✅ Automatic backups in Drive
✅ Can resume training
✅ Never lose results

### For Collaboration
✅ Easy to share with team
✅ Version control in Drive
✅ Multiple people can access
✅ Centralized storage

---

## 📝 Summary

**Google Drive Version is Better Because:**
1. No manual file uploads
2. Results auto-saved to Drive
3. Survives Colab disconnects
4. Easy to access and share
5. Supports multiple runs

**One-Time Setup:**
1. Put data in `AbAg_Training_02` folder
2. Update `CSV_FILENAME` in notebook
3. Grant Drive access

**Then forever:**
- Just click "Run all"
- Everything happens automatically!

---

## 🎉 You're All Set!

Your new workflow:
1. Open `colab_training_GDRIVE.ipynb` in Colab
2. Runtime → Run all
3. Wait ~2-3 hours
4. Check Drive for results

**That's it!** 🚀

No more uploading, no more downloading, no more lost results!

---

**Enjoy your streamlined training workflow! 📂🧬**
