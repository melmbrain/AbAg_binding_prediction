# Simple Colab Training Instructions

## The Problem
The notebook needs you to manually paste the training script into Cell 3b, but that's 932 lines!

## Simple Solution: Copy Files to Colab

### Method 1: Direct Python Execution (EASIEST)

**Step 1: Upload files to Google Drive**
Make sure these files are in `Google Drive/MyDrive/AbAg_Training/`:
- `agab_phase2_full.csv` (your data)
- `train_ultra_speed_v26.py` (the training script)

**Step 2: Run this in a NEW Colab notebook:**

```python
# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/AbAg_Training')
print(f"Current directory: {os.getcwd()}")
print("\nFiles in directory:")
!ls -lh

# Cell 2: Install Dependencies
!pip install -q transformers pandas scipy scikit-learn tqdm sentencepiece faesm bitsandbytes accelerate

import torch
print(f"\n✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA: {torch.cuda.is_available()}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# Cell 3: Run Training
!python train_ultra_speed_v26.py
```

That's it! The script will auto-detect Colab and start training.

---

## Method 2: Copy Script from Local Machine

If `train_ultra_speed_v26.py` is not in your Google Drive yet:

**Step 1: In Colab, create the script:**

```python
# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/AbAg_Training')

# Cell 2: Copy script from your computer
from google.colab import files
uploaded = files.upload()  # Upload train_ultra_speed_v26.py

# Cell 3: Verify
!ls -lh train_ultra_speed_v26.py

# Cell 4: Install Dependencies
!pip install -q transformers pandas scipy scikit-learn tqdm sentencepiece faesm bitsandbytes accelerate

# Cell 5: Run Training
!python train_ultra_speed_v26.py
```

---

## Method 3: Direct Inline Execution (If file upload fails)

I can create a single-cell version that has everything embedded. Would you like that?

---

## What You Should See

After running `!python train_ultra_speed_v26.py`, you'll see:

```
🔧 Detected Jupyter/Colab environment - using default configuration
======================================================================
ULTRA SPEED TRAINING v2.6 - ALL ADVANCED OPTIMIZATIONS
======================================================================

Loading data...
Loaded 159,735 samples

📊 Bucket Distribution:
  ≤256: 21,734 samples (13.6%)
  ≤384: 65,234 samples (40.9%)
  ≤512: 72,767 samples (45.5%)

Found checkpoint: checkpoint_latest.pth
Attempting to load v2.5 checkpoint into v2.6 model...
✓ Loaded regressor weights from checkpoint
Resuming from Epoch 4...

Epoch 4: 100% 6988/6988 [02:30<00:00, 46.52it/s]
```

---

## Current Status

Based on your message showing the warnings, it looks like you already started running the script!

**Is training running now?** You should see progress bars appearing.

If not, which method do you prefer:
1. Upload `train_ultra_speed_v26.py` to Drive and run it
2. I create a single mega-cell with everything embedded
3. Something else?
