# Quick Start: v2.7 Training Checklist

**Goal**: Start training v2.7 with all fixes on Google Colab

---

## ✅ Step-by-Step Checklist

### Step 1: Modify Notebook Locally ⏱️ 15 minutes

**File**: `notebooks/colab_training_v2.7.ipynb` (already created for you!)

**What to do**:
1. Open `notebooks/colab_training_v2.7.ipynb` in Jupyter or VS Code
2. Follow **V2.7_TRAINING_GUIDE.md** to make 8 changes:
   - [ ] Change 1: Replace loss function (cell ~8)
   - [ ] Change 2: Update hyperparameters (cell ~17)
   - [ ] Change 3: Add prediction clamping (cell ~16)
   - [ ] Change 4: Add NaN detection (cell ~22)
   - [ ] Change 5: Save RNG state (cell ~22)
   - [ ] Change 6: Add overfitting monitor (cell ~22)
   - [ ] Change 7: Change scheduler (cell ~20)
   - [ ] Change 8: Update criterion (cell ~20)
3. Save the file

**Tip**: Use Ctrl+F to find the cells mentioned in the guide.

---

### Step 2: Upload to Google Drive ⏱️ 2 minutes

**Upload these 2 files**:

#### File 1: Modified Notebook
- **Local**: `C:\Users\401-24\Desktop\AbAg_binding_prediction\notebooks\colab_training_v2.7.ipynb`
- **Upload to**: `Google Drive → MyDrive/AbAg_Training_02/colab_training_v2.7.ipynb`

#### File 2: Dataset (if not already there)
- **Local**: Your dataset file (wherever it is)
- **Upload to**: `Google Drive → MyDrive/AbAg_Training_02/agab_phase2_full.csv`

**How to upload**:
1. Go to https://drive.google.com
2. Navigate to `MyDrive/AbAg_Training_02/`
3. Drag and drop both files

---

### Step 3: Open in Google Colab ⏱️ 1 minute

1. Go to https://colab.research.google.com
2. Click: **File → Open notebook**
3. Click: **Google Drive** tab
4. Navigate to: `MyDrive/AbAg_Training_02/colab_training_v2.7.ipynb`
5. Click to open

---

### Step 4: Connect to A100 GPU ⏱️ 1 minute

1. In Colab, click: **Runtime → Change runtime type**
2. Select:
   - **Hardware accelerator**: GPU
   - **GPU type**: A100 (if available, otherwise T4/V100)
3. Click: **Save**
4. Click: **Runtime → Run all** (or press Ctrl+F9)

---

### Step 5: Monitor Training 🎯

The training will run for ~40-50 epochs. Watch for:

✅ **Good signs**:
- Spearman increasing steadily
- Recall stable (not jumping)
- Overfit ratio < 3.0
- Predictions in range [4.0, 14.0]

⚠️ **Warning signs**:
- NaN loss (will stop automatically)
- Recall jumping (should be fixed now!)
- Overfit ratio > 3.0

---

## 📊 Expected Results

| Metric | Target |
|--------|--------|
| Training Time | ~40-50 epochs |
| Spearman | 0.45-0.55 |
| Recall | 50-70% (stable!) |
| RMSE | 1.2-1.5 |

---

## 🆘 If Something Goes Wrong

### Problem: "File not found" in Colab
**Solution**: Check that dataset is at `/content/drive/MyDrive/AbAg_Training_02/agab_phase2_full.csv`

### Problem: Out of Memory (OOM)
**Solution**: In the hyperparameters cell, change:
```python
BATCH_SIZE = 8              # Reduce from 16
GRADIENT_ACCUMULATION = 16  # Increase from 8
```

### Problem: Training very slow
**Solution**: Make sure you're using A100, not T4 or V100

### Problem: Recall still unstable
**Solution**: Double-check you made ALL 8 changes, especially Change 1 (loss function)

---

## 📁 File Locations Summary

### Local (Your Computer):
```
C:\Users\401-24\Desktop\AbAg_binding_prediction\
├── notebooks/
│   └── colab_training_v2.7.ipynb  ← Modify this!
├── V2.7_TRAINING_GUIDE.md         ← Follow this!
└── START_V2.7_TRAINING.md         ← You are here!
```

### Google Drive:
```
MyDrive/
└── AbAg_Training_02/
    ├── colab_training_v2.7.ipynb  ← Upload here!
    ├── agab_phase2_full.csv       ← Dataset
    └── training_output_OPTIMIZED_v2/  ← Will be created
        └── (checkpoints saved here during training)
```

---

## ✨ Quick Commands

### Check if files exist:
```python
# Run this in Colab after mounting Drive
import os
print("Dataset:", os.path.exists('/content/drive/MyDrive/AbAg_Training_02/agab_phase2_full.csv'))
print("Notebook:", os.path.exists('/content/drive/MyDrive/AbAg_Training_02/colab_training_v2.7.ipynb'))
```

### Monitor GPU:
```python
# Check you have A100
!nvidia-smi
```

---

## 🎯 Current Status

- [x] ✅ v2.6 training completed
- [x] ✅ v2.6 released to GitHub
- [x] ✅ v2.6 uploaded to Hugging Face
- [x] ✅ v2.7 improvements documented
- [x] ✅ v2.7 training guide created
- [x] ✅ v2.7 notebook template created
- [ ] ⏳ **YOU ARE HERE** → Modify notebook with 8 changes
- [ ] Upload to Google Drive
- [ ] Start training on Colab!

---

## 🚀 Ready?

1. **Open**: `notebooks/colab_training_v2.7.ipynb`
2. **Follow**: `V2.7_TRAINING_GUIDE.md` (make 8 changes)
3. **Upload**: Modified notebook to Google Drive
4. **Open**: In Google Colab
5. **Run**: All cells!

**Good luck with v2.7 training!** 🎉

---

*Last updated: 2025-11-25*
*All files prepared and ready*
