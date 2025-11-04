# Quick Start Guide: Download and Integration

**Complete workflow to download and integrate external databases**

---

## Option 1: Fully Automated (Recommended)

### Single Command - Everything at Once

**Windows:**
```cmd
RUN_DOWNLOAD_AND_INTEGRATE.bat
```

This will:
1. ✅ Download AbBiBench from Hugging Face
2. ✅ Download SAAINT-DB from GitHub
3. ⚠️ Show instructions for PDBbind (manual download)
4. ✅ Integrate all downloaded databases
5. ✅ Generate comprehensive report
6. ✅ Save merged dataset

**Expected time:** 10-20 minutes (depending on internet speed)

---

## Option 2: Step-by-Step

### Step 1: Download Databases

**Windows:**
```cmd
cd C:\Users\401-24\Desktop\AbAg_binding_prediction
scripts\download_all.bat
```

**Linux/Mac:**
```bash
cd /mnt/c/Users/401-24/Desktop/AbAg_binding_prediction
bash scripts/download_all.sh
```

### Step 2: Integrate with Your Data

**Windows:**
```cmd
python scripts\integrate_all_databases.py ^
  --existing "C:\Users\401-24\Desktop\Docking prediction\data\processed\phase6\final_205k_dataset.csv" ^
  --output external_data\merged_all_databases.csv
```

**Linux/Mac:**
```bash
python3 scripts/integrate_all_databases.py \
  --existing "C:/Users/401-24/Desktop/Docking prediction/data/processed/phase6/final_205k_dataset.csv" \
  --output external_data/merged_all_databases.csv
```

### Step 3: View Results

**Windows:**
```cmd
type external_data\integration_report.txt
```

**Linux/Mac:**
```bash
cat external_data/integration_report.txt
```

---

## Option 3: Manual Individual Downloads

If you prefer to download databases one at a time:

### Download AbBiBench (Easiest)
```bash
python scripts/download_abbibench.py
```
- **Time:** 2-5 minutes
- **Size:** ~100 MB
- **Samples:** 184,500+

### Download SAAINT-DB (Most Recent)
```bash
python scripts/download_saaint.py
```
- **Time:** 5-10 minutes
- **Size:** ~500 MB
- **Samples:** 19,128

### Download PDBbind (Manual Step Required)

**Step A: Manual Download**
1. Visit: http://www.pdbbind.org.cn/download.php
2. Download: `PP_INDEX_general_set.2020`
3. Save to: `external_data/`

**Step B: Process the File**
```bash
python scripts/download_pdbbind.py
```
- **Time:** 1-2 minutes (processing only)
- **Samples:** 4,594

---

## What Gets Created

After running the download and integration:

```
external_data/
├── abbibench_raw.csv              # Downloaded AbBiBench data
├── abbibench_cache/               # Cached download
├── saaint_raw.csv                 # Downloaded SAAINT-DB data
├── SAAINT/                        # Cloned repository
├── pdbbind_raw.csv                # Processed PDBbind data (if downloaded)
├── merged_all_databases.csv       # ⭐ MERGED DATASET (ready to use)
└── integration_report.txt         # ⭐ STATISTICS REPORT
```

---

## Expected Results

### Integration Statistics

After integration, you should see something like:

```
================================================================================
INTEGRATION COMPLETE!
================================================================================

✓ Integrated 2-3 external databases
✓ Total samples: ~250,000-300,000
✓ New samples added: ~50,000-100,000 (after duplicate removal)

Affinity Distribution:
  Very weak (pKd < 5):    10,000+ (5%)    [was: 3,778, 1.8%]
  Weak (5-7):            70,000 (28%)     [was: 66,328, 32.2%]
  Moderate (7-9):        80,000 (32%)     [was: 72,095, 35.0%]
  Strong (9-11):         65,000 (26%)     [was: 58,912, 28.6%]
  Very strong (> 11):     2,000+ (2%)     [was: 240, 0.1%]

✓ Very strong binders: 10-20× increase!
✓ Very weak binders: 3-4× increase!
```

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'datasets'"

**Solution:**
```bash
pip install datasets
```

### Error: "git: command not found"

**Solution for Windows:**
1. Download Git: https://git-scm.com/download/win
2. Install and restart terminal
3. Retry

### Error: "No PDBbind files found"

**Solution:**
PDBbind requires manual download. Follow instructions shown by the script.

### Downloads are slow

**Solution:**
- Downloads are large (1-2 GB total)
- AbBiBench: ~100 MB (fast)
- SAAINT-DB: ~500 MB (moderate)
- PDBbind: ~10 GB if downloading full structures (use index file only)

---

## Next Steps After Integration

### 1. Review the Report

```bash
cat external_data/integration_report.txt
```

Check:
- How many new samples were added
- Affinity distribution changes
- Duplicate statistics

### 2. Verify the Merged Dataset

```python
import pandas as pd

# Load merged dataset
df = pd.read_csv('external_data/merged_all_databases.csv')

print(f"Total samples: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"\nSources:")
print(df['source'].value_counts())
print(f"\nSamples needing ESM2 embeddings:")
print(df['esm2_pca_0'].isna().sum())
```

### 3. Generate ESM2 Embeddings

⚠️ **Important:** New samples will have NaN for ESM2 features!

You need to:
1. Extract sequences from new PDB codes
2. Generate ESM2 embeddings
3. Apply your existing PCA transformation (150 components)
4. Fill in the feature columns

### 4. Train with Improved Data

Once embeddings are generated:

```bash
python train_balanced.py \
  --data external_data/merged_all_databases.csv \
  --loss weighted_mse \
  --sampling stratified \
  --epochs 100 \
  --batch_size 32
```

---

## Quick Check: What's Been Downloaded?

**Windows:**
```cmd
dir external_data\*.csv
```

**Linux/Mac:**
```bash
ls -lh external_data/*.csv
```

Expected files:
- ✅ `abbibench_raw.csv` (~100 MB)
- ✅ `saaint_raw.csv` (~50 MB)
- ⚠️ `pdbbind_raw.csv` (optional, requires manual download first)
- ✅ `merged_all_databases.csv` (after integration)

---

## Performance Impact

### Model Performance (Expected)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Very strong RMSE | ~2.2 | ~0.8 | **64%** ✓ |
| Very weak RMSE | ~2.5 | ~0.9 | **64%** ✓ |
| Overall RMSE | ~0.7 | ~0.7 | Maintained |

### Data Coverage (Expected)

| Affinity Range | Before | After | Increase |
|----------------|--------|-------|----------|
| Very strong (pKd > 11) | 240 (0.1%) | 2,000+ (2%) | **10-20×** |
| Very weak (pKd < 5) | 3,778 (1.8%) | 10,000+ (5%) | **3-4×** |

---

## Support

### Still Having Issues?

1. **Check documentation:** `EXTERNAL_DATA_README.md`
2. **Review detailed instructions:** `DOWNLOAD_INSTRUCTIONS.md`
3. **Database information:** `ADDITIONAL_DATA_SOURCES.md`

### Want to Download More Databases?

See `ADDITIONAL_DATA_SOURCES.md` for 11 more databases including:
- IEDB (weak binders)
- BindingDB (additional data)
- Ab-CoV (COVID antibodies)
- And more...

---

## Summary

**To download and integrate everything:**

```bash
# Windows - One command
RUN_DOWNLOAD_AND_INTEGRATE.bat

# Or step by step
scripts\download_all.bat
python scripts\integrate_all_databases.py --existing YOUR_DATA.csv
```

**Time required:** 10-20 minutes

**Expected outcome:**
- ~250k total samples (up from 205k)
- 10-20× more very strong binders
- 3-4× more very weak binders
- Significantly improved extreme affinity prediction

---

*Quick Start Guide - Last updated: 2025-11-03*
