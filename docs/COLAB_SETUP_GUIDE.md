# Google Colab Training Setup Guide

**Estimated Training Time**: 3-4 days (vs 36 days locally!)

---

## 📋 Prerequisites

1. Google account (free)
2. Your data file: `agab_phase2_full.csv` (127 MB)

---

## 🚀 Step-by-Step Setup (15 minutes)

### Step 1: Upload Data to Google Drive

1. Go to [Google Drive](https://drive.google.com)
2. Create a new folder called **`AbAg_Training`**
3. Upload these files to that folder:
   - `agab_phase2_full.csv` (from `C:\Users\401-24\Desktop\Ab_Ag_dataset\data\agab\`)
   - `colab_training.ipynb` (created in your project folder)
   - **OPTIONAL**: `checkpoint_latest.pth` (to resume from epoch 5 instead of starting fresh!)

**File locations on your PC:**
```
Data: C:\Users\401-24\Desktop\Ab_Ag_dataset\data\agab\agab_phase2_full.csv
Notebook: C:\Users\401-24\Desktop\AbAg_binding_prediction\colab_training.ipynb
Checkpoint: C:\Users\401-24\Desktop\AbAg_binding_prediction\outputs_cached\checkpoint_latest.pth
```

**⚠️ IMPORTANT: Large File Upload**

The checkpoint file is **2.5 GB**. Google Drive upload can be unreliable for large files:

**Method 1: Direct Upload (Recommended)**
- Drag and drop `checkpoint_latest.pth` to Google Drive folder
- **Wait for "Upload complete" message** (may take 10-30 minutes)
- Verify file size shows ~2.5 GB in Google Drive

**Method 2: Google Drive Desktop App (Most Reliable)**
- Install [Google Drive for Desktop](https://www.google.com/drive/download/)
- Copy file to `G:\My Drive\AbAg_Training\` (or similar path)
- Wait for sync to complete

**Verify Upload Success:**
- File size in Google Drive should be exactly **2,614,433,434 bytes** (2.49 GB)
- If smaller, the upload failed - delete and re-upload
- Run the verification cell in Colab (new cell added before training)

---

### Step 2: Open Notebook in Colab

1. In Google Drive, double-click `colab_training.ipynb`
2. Choose **"Open with Google Colaboratory"**
3. If you don't see this option:
   - Right-click → "Open with" → "Connect more apps"
   - Search for "Colaboratory" and install it

---

### Step 3: Enable GPU

1. In Colab, click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU** (T4 or better)
3. Click **Save**

---

### Step 4: Run the Notebook

**Run cells in order** (click the play button ▶️ on each cell):

#### Cell 1: Mount Google Drive
- Click "Connect to Google Drive" when prompted
- Allow permissions

#### Cell 2: Check GPU
- Verify you have a Tesla T4, V100, or A100 GPU

#### Cell 3: Install Dependencies
- Wait ~2 minutes for packages to install

#### Cell 4-5: Create Tokenization Cache
- **This takes ~10 minutes** (one-time setup)
- Creates SQLite cache for fast data loading

#### Cell 6-7: Start Training
- **This runs for ~3-4 days**
- You can close the browser tab - training continues!
- Just keep the Colab tab open (or use Colab Pro for background execution)

---

## 📊 Monitoring Training

### Check Progress (While Training)

Add a new cell and run:
```python
# Check latest checkpoint
import torch
checkpoint = torch.load('outputs_colab/checkpoint_latest.pth', map_location='cpu')
print(f"Current Epoch: {checkpoint['epoch'] + 1}/50")
print(f"Best Spearman: {checkpoint['best_val_spearman']:.4f}")
print(f"Latest Recall@pKd≥9: {checkpoint['val_metrics']['recall_pkd9']:.2f}%")
```

### Check GPU Usage
```python
!nvidia-smi
```

---

## ⚠️ Important Notes

### Colab Session Limits

**Free Colab:**
- Max runtime: ~12 hours per session
- After 12h, it will disconnect
- **Solution**: Run this code to prevent timeout:

```python
# Add this cell at the top and run it
import IPython
from google.colab import output

def keep_alive():
    display(IPython.display.Javascript('''
        function ClickConnect(){
            console.log("Keeping alive");
            document.querySelector("#connect").click()
        }
        setInterval(ClickConnect, 60000)
    '''))

keep_alive()
```

**Or use Colab Pro** ($10/month):
- 24 hour runtime
- Better GPUs (A100)
- Background execution

### Resume Training After Disconnect

If Colab disconnects, just re-run the training cell. It will automatically resume from the last checkpoint!

The training script saves checkpoints every epoch, so you won't lose progress.

### Using Your Existing Local Progress

**You already completed 5 epochs locally!** If you uploaded `checkpoint_latest.pth`:

1. Use **Option B** in the notebook (Step 6)
2. Training will start from epoch 6 instead of epoch 1
3. **Saves ~1 day of training time!**

Your checkpoint contains:
- Epoch: 5 (will resume at 6)
- Best Spearman: 0.4594
- All optimizer/scheduler state

The script automatically detects and loads it.

---

## 📥 Download Results

After training completes (or anytime):

```python
# Download models
from google.colab import files
files.download('outputs_colab/best_model.pth')
files.download('outputs_colab/checkpoint_latest.pth')
```

Or just copy them from Google Drive folder manually.

---

## 🔍 Expected Results

After 50 epochs (~3-4 days):
- **Spearman correlation**: 0.55-0.65
- **Recall@pKd≥9**: 35-50%
- **RMSE**: 1.30-1.40

Much better than current 17% recall!

---

## 🐛 Troubleshooting

### "UnpicklingError" when loading checkpoint
**Cause**: Checkpoint file upload incomplete or corrupted

**Solution**:
1. Check file size in Google Drive: Should be **2,614,433,434 bytes**
2. If smaller, delete and re-upload using Google Drive Desktop app
3. Run the verification cell in notebook to confirm
4. If still corrupted, just use **Option A** (start from scratch)

### "No GPU available"
- Go to Runtime → Change runtime type → Select GPU

### "Cannot find agab_phase2_full.csv"
- Check the file is in `/content/drive/MyDrive/AbAg_Training/`
- Verify the path in the Mount Drive cell

### Training is slow
- Check GPU: Should show T4, V100, or A100
- If it says CPU, you need to enable GPU

### Session disconnected
- Just re-run the training cell
- It will resume from last checkpoint automatically

---

## 💡 Tips

1. **Use Colab Pro** if you can afford $10/month
   - 2-3x faster GPU (A100)
   - No timeout issues
   - Worth it for 3-4 day training

2. **Check progress daily**
   - Run the checkpoint check cell
   - Monitor Recall@pKd≥9 metric

3. **Download checkpoints regularly**
   - Every few epochs, download the checkpoint
   - Safety backup in case of issues

---

## 📞 Need Help?

Common issues and solutions in the notebook itself.

**Ready to start?** Follow Step 1 above! 🚀
