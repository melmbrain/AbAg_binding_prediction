# File Organization & Cleanup Guide

**Project**: Antibody-Antigen Binding Prediction
**Location**: `C:\Users\401-24\Desktop\AbAg_binding_prediction\`
**Last Updated**: 2025-11-13

---

## 📁 Recommended File Structure

```
AbAg_binding_prediction/
│
├── 📄 README.md                          ← Main entry point (TO CREATE)
├── 📄 START_HERE_FINAL.md                ← Quick start guide ✅
│
├── 📂 docs/                              ← Documentation
│   ├── PROJECT_LOG.md                    ← Work history ✅
│   ├── OUTCOMES_AND_FUTURE_PLAN.md       ← Results & plans ✅
│   ├── REFERENCES_AND_SOURCES.md         ← All citations ✅
│   ├── MODEL_COMPARISON_FINAL.md         ← Model comparison ✅
│   ├── COLAB_SETUP_GUIDE.md              ← Colab instructions ✅
│   └── VALIDATION_CHECKLIST.md           ← Decision framework (TO CREATE)
│
├── 📂 models/                            ← Model definitions
│   ├── model_igt5_esm2.py                ← IgT5 + ESM-2 (current) ✅
│   ├── model_igfold_hybrid.py            ← IgFold + ESM-2 (backup) ✅
│   └── README.md                         ← Model descriptions (TO CREATE)
│
├── 📂 training/                          ← Training scripts
│   ├── train_igt5_esm2.py                ← IgT5 training ✅
│   ├── train_igfold_hybrid.py            ← IgFold training ✅
│   └── README.md                         ← Training guide (TO CREATE)
│
├── 📂 notebooks/                         ← Jupyter notebooks
│   ├── colab_training_SOTA.ipynb         ← Main Colab notebook ✅
│   ├── colab_training.ipynb              ← ESM-2 only (backup) ✅
│   ├── colab_training_igfold.ipynb       ← IgFold version (backup) ✅
│   └── README.md                         ← Notebook guide (TO CREATE)
│
├── 📂 results/                           ← Training outputs
│   ├── outputs_sota/                     ← IgT5 + ESM-2 results (Colab)
│   ├── outputs_cached/                   ← Local training (old)
│   └── analysis/                         ← Analysis notebooks (TO CREATE)
│
├── 📂 data/                              ← Data files
│   └── README.md                         ← Data description (TO CREATE)
│
├── 📂 archive/                           ← Old/deprecated files
│   └── [Move old files here]
│
└── 📂 deprecated/                        ← Explicitly deprecated
    └── [Files marked for deletion]
```

---

## 📋 Current Files Status

### ✅ Keep & Use (Essential)

| File | Purpose | Status |
|------|---------|--------|
| `START_HERE_FINAL.md` | Main entry point | ✅ Complete |
| `colab_training_SOTA.ipynb` | Current training | ✅ Active |
| `model_igt5_esm2.py` | Model definition | ✅ Fixed |
| `train_igt5_esm2.py` | Training script | ✅ Complete |
| `PROJECT_LOG.md` | Work history | ✅ Complete |
| `OUTCOMES_AND_FUTURE_PLAN.md` | Results & plans | ✅ Complete |
| `REFERENCES_AND_SOURCES.md` | Citations | ✅ Complete |
| `MODEL_COMPARISON_FINAL.md` | Model comparison | ✅ Complete |
| `COLAB_SETUP_GUIDE.md` | Colab guide | ✅ Complete |

### 📦 Archive (Backup - Not Active)

| File | Purpose | Action |
|------|---------|--------|
| `colab_training.ipynb` | ESM-2 only | Move to `notebooks/backup/` |
| `colab_training_igfold.ipynb` | IgFold version | Move to `notebooks/backup/` |
| `model_igfold_hybrid.py` | IgFold model | Move to `models/backup/` |
| `train_igfold_hybrid.py` | IgFold training | Move to `training/backup/` |
| `train_ultra_optimized_cached.py` | Old local script | Move to `archive/` |
| `create_tokenization_cache.py` | SQLite cache | Move to `archive/` |

### 🗑️ Delete (Outdated/Redundant)

| File | Reason | Action |
|------|--------|--------|
| `IGFOLD_VS_ESM2_COMPARISON.md` | Superseded by MODEL_COMPARISON_FINAL | Delete |
| `train_fast_v2.py` | Old version | Delete |
| `train_optimized_v1.py` | Old version | Delete |
| `train_ultra_optimized.py` | Old version | Delete |
| `training_*.log` | Old logs | Delete (or archive) |
| `*.txt` (various command files) | Ad-hoc notes | Delete after review |

### ❓ Review (Check Git Status First)

| File | Status | Action |
|------|--------|--------|
| All files with `D` in git status | Deleted in git | Confirm deletion OK |
| All files with `??` in git status | Untracked | Decide: add or delete |
| `tokenization_cache.db` | SQLite cache (777MB) | Keep if reusing local training |

---

## 🗂️ Reorganization Plan

### Step 1: Create Directory Structure

```bash
mkdir -p docs
mkdir -p models/backup
mkdir -p training/backup
mkdir -p notebooks/backup
mkdir -p results/analysis
mkdir -p archive
mkdir -p deprecated
```

### Step 2: Move Documentation

```bash
# Move to docs/
mv PROJECT_LOG.md docs/
mv OUTCOMES_AND_FUTURE_PLAN.md docs/
mv REFERENCES_AND_SOURCES.md docs/
mv MODEL_COMPARISON_FINAL.md docs/
mv COLAB_SETUP_GUIDE.md docs/
```

### Step 3: Organize Models

```bash
# Keep current in models/
mv model_igt5_esm2.py models/

# Move backups
mv model_igfold_hybrid.py models/backup/
```

### Step 4: Organize Training Scripts

```bash
# Keep current in training/
mv train_igt5_esm2.py training/

# Move backups
mv train_igfold_hybrid.py training/backup/
mv train_ultra_optimized_cached.py training/backup/
```

### Step 5: Organize Notebooks

```bash
# Keep current
mv colab_training_SOTA.ipynb notebooks/

# Move backups
mv colab_training.ipynb notebooks/backup/
mv colab_training_igfold.ipynb notebooks/backup/
```

### Step 6: Archive Old Files

```bash
# Move to archive/
mv train_fast_v2.py archive/
mv train_optimized_v1.py archive/
mv train_ultra_optimized.py archive/
mv create_tokenization_cache.py archive/
mv training_*.log archive/
```

### Step 7: Delete Deprecated

```bash
# Review before deleting
rm IGFOLD_VS_ESM2_COMPARISON.md
rm -f *.txt  # After reviewing content
```

---

## 📄 Files to Create

### 1. README.md (Main)

**Location**: Root directory
**Purpose**: Project overview and quick navigation

**Template**:
```markdown
# Antibody-Antigen Binding Prediction

Predicting antibody-antigen binding affinity (pKd) using state-of-the-art protein language models.

## Quick Start
See [START_HERE_FINAL.md](START_HERE_FINAL.md)

## Current Model
IgT5 + ESM-2 (training on Google Colab)

## Documentation
- [Project Log](docs/PROJECT_LOG.md) - Full work history
- [Model Comparison](docs/MODEL_COMPARISON_FINAL.md) - Why IgT5 + ESM-2?
- [References](docs/REFERENCES_AND_SOURCES.md) - All sources

## Results
*To be updated after training completes (Nov 17-18, 2025)*

## Citation
*To be added after publication*
```

### 2. models/README.md

**Purpose**: Describe each model architecture

**Content**:
```markdown
# Model Architectures

## Current: IgT5 + ESM-2
- File: `model_igt5_esm2.py`
- Antibody: IgT5 (1024-dim)
- Antigen: ESM-2 650M (1280-dim)
- Status: Active training

## Backup Models
- `backup/model_igfold_hybrid.py` - IgFold + ESM-2
```

### 3. training/README.md

**Purpose**: Training instructions

**Content**:
```markdown
# Training Scripts

## Current Training
File: `train_igt5_esm2.py`
Platform: Google Colab
Status: Active

## Usage
See notebooks/colab_training_SOTA.ipynb
```

### 4. notebooks/README.md

**Purpose**: Notebook guide

**Content**:
```markdown
# Jupyter Notebooks

## Active
- `colab_training_SOTA.ipynb` - IgT5 + ESM-2 training

## Backup
- `backup/colab_training.ipynb` - ESM-2 only
- `backup/colab_training_igfold.ipynb` - IgFold version
```

### 5. data/README.md

**Purpose**: Data description

**Content**:
```markdown
# Dataset

## Main Data
- File: `agab_phase2_full.csv`
- Location: `C:\Users\401-24\Desktop\Ab_Ag_dataset\data\agab\`
- Size: 159,735 samples (127 MB)
- Features: antibody_sequence, antigen_sequence, pKd

## Split
- Train: 70% (111,814 samples)
- Validation: 15% (23,960 samples)
- Test: 15% (23,961 samples)
- Random seed: 42
```

---

## 🧹 Cleanup Commands (Windows)

### Safe Cleanup (Move to archive)

```cmd
REM Create backup before cleanup
mkdir archive\backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%

REM Move old training scripts
move train_fast_v2.py archive\
move train_optimized_v1.py archive\
move train_ultra_optimized.py archive\

REM Move old logs
move training_*.log archive\

REM Move old txt files (review first!)
move *.txt archive\
```

### Aggressive Cleanup (Delete permanently)

```cmd
REM CAREFUL - This deletes permanently!
REM Review files before running

del IGFOLD_VS_ESM2_COMPARISON.md
del train_fast_v2.py
del train_optimized_v1.py
del train_ultra_optimized.py
```

---

## 📊 File Size Summary

### Large Files (>100MB)

| File | Size | Keep? | Location |
|------|------|-------|----------|
| `tokenization_cache.db` | 777 MB | ✅ Yes | Root (for local training) |
| `agab_phase2_full.csv` | 127 MB | ✅ Yes | External data folder |
| `checkpoint_latest.pth` | ~2.5 GB | ✅ Yes | Google Drive (auto-saved) |
| `best_model.pth` | ~2.5 GB | ✅ Yes | Google Drive (auto-saved) |

### Medium Files (10-100MB)

- Model definitions: <1 MB each
- Training scripts: <100 KB each
- Notebooks: 1-5 MB each
- Documentation: <1 MB total

---

## 🔒 Git Management

### Files to Commit

```bash
# Documentation
git add docs/
git add START_HERE_FINAL.md
git add README.md

# Code
git add models/model_igt5_esm2.py
git add training/train_igt5_esm2.py
git add notebooks/colab_training_SOTA.ipynb

# Commit
git commit -m "Add IgT5 + ESM-2 model and comprehensive documentation"
```

### Files to Ignore (.gitignore)

```
# Large data files
*.csv
*.db
*.pth
*.pkl

# Logs
*.log
training_*.log

# Cache
__pycache__/
*.pyc
.ipynb_checkpoints/

# OS files
.DS_Store
Thumbs.db

# Temporary
*.tmp
*.bak
~*

# Results
outputs_*/
results/
checkpoints/
```

### Files Already Deleted (in git status)

All files marked with `D` in your git status have been intentionally deleted. Confirm this is correct before committing.

---

## 📝 Maintenance Schedule

### Daily (While Training)
- [ ] Check Colab training progress
- [ ] Update PROJECT_LOG.md with observations
- [ ] Download checkpoint backups (optional)

### After Training Completes
- [ ] Update OUTCOMES_AND_FUTURE_PLAN.md with actual results
- [ ] Create analysis notebook
- [ ] Archive old checkpoints
- [ ] Clean up temporary files

### Before Publication
- [ ] Verify all references in REFERENCES_AND_SOURCES.md
- [ ] Create DOI/citation for dataset
- [ ] Write comprehensive README.md
- [ ] Archive all old versions

---

## ✅ Cleanup Checklist

### Phase 1: Organization
- [ ] Create directory structure (docs/, models/, training/, notebooks/)
- [ ] Move files to appropriate directories
- [ ] Create README.md for each directory
- [ ] Update main README.md

### Phase 2: Archive
- [ ] Move old training scripts to archive/
- [ ] Move old notebooks to notebooks/backup/
- [ ] Move old logs to archive/
- [ ] Keep tokenization_cache.db (might reuse)

### Phase 3: Delete
- [ ] Review all *.txt files, delete if not needed
- [ ] Delete superseded documentation
- [ ] Delete old training script versions
- [ ] Clean up git status (D files)

### Phase 4: Document
- [ ] Update FILE_ORGANIZATION.md (this file)
- [ ] Update PROJECT_LOG.md with file changes
- [ ] Create comprehensive README.md
- [ ] Add .gitignore if using git

---

## 🎯 Final Structure Goal

```
AbAg_binding_prediction/
├── README.md (main entry)
├── START_HERE_FINAL.md (quick start)
├── docs/ (all documentation)
├── models/ (model definitions)
├── training/ (training scripts)
├── notebooks/ (Jupyter notebooks)
├── results/ (training outputs)
├── data/ (dataset info, actual data elsewhere)
└── archive/ (old files for reference)
```

**Clean, organized, professional structure** ✅

---

**Status**: Reorganization plan ready
**Next Steps**: Execute cleanup after training completes
**Estimated Time**: 30 minutes for full reorganization
