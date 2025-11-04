# Repository Cleanup Guide for GitHub Distribution

This guide helps you clean up the repository before pushing to GitHub.

---

## 📁 Files to KEEP (Include in GitHub)

### Core Code
- ✅ `abag_affinity/` - Main package
- ✅ `src/` - Training utilities
- ✅ `scripts/` - All scripts
- ✅ `examples/` - Usage examples
- ✅ `tests/` - Unit tests

### Documentation (Keep Organized)
- ✅ `README_GITHUB.md` → Rename to `README.md` (main readme)
- ✅ `CHANGELOG.md` - Version history
- ✅ `PROJECT_STRUCTURE.md` - Repo organization
- ✅ `NAVIGATION.md` - Navigation guide
- ✅ `LICENSE` - MIT license
- ✅ `CONTRIBUTING.md` - Contribution guidelines
- ✅ `references.bib` - Citations

### Training Notebooks
- ✅ `colab_training_v2_improved.ipynb` - Improved training (v2)
- ✅ `colab_resume_and_evaluate.ipynb` - Resume/evaluate
- ✅ `train_balanced.py` - Local training script
- ✅ `use_colab_model_locally.py` - Model inference

### Configuration
- ✅ `setup.py` - Package setup
- ✅ `requirements.txt` - Dependencies
- ✅ `.gitignore` - Git ignore rules

### Guides (Essential)
- ✅ `COLAB_TRAINING_GUIDE.md` - How to train on Colab
- ✅ `QUICK_START_V2.md` - Quick start guide
- ✅ `V2_IMPROVEMENTS.md` - Technical improvements
- ✅ `SESSION_TIMEOUT_GUIDE.md` - Handling timeouts
- ✅ `LAUNCH_CHECKLIST.md` - Training checklist

### Documentation Folders
- ✅ `docs/guides/` - User guides
- ✅ `docs/references/` - Citations and sources
- ✅ `docs/reports/` - Analysis reports

---

## 📦 Files to CONSOLIDATE or MOVE

### Redundant/Old Documentation
These have overlapping content - consolidate or remove:

- ⚠️ `README.md` → Replace with `README_GITHUB.md`
- ⚠️ `README_COMPLETE.md` → Archive (covered in new README)
- ⚠️ `docs/guides/QUICK_START.md` → Merge with `QUICK_START_V2.md`
- ⚠️ `docs/guides/QUICK_START_GUIDE.md` → Merge with above
- ⚠️ `docs/guides/DOCUMENTATION_INDEX.md` → Update or remove

### Old Training Files
- ⚠️ `colab_training.ipynb` → Move to `archive/old_versions/` (v1)
- ⚠️ `STATUS_AND_NEXT_STEPS.md` → Archive (outdated)
- ⚠️ `TRAINING_RESULTS_ANALYSIS.md` → Move to `docs/reports/v1_analysis.md`

### Temporary Analysis Scripts
- ⚠️ `analyze_affinity.sh` → Keep or move to `scripts/analysis/`
- ⚠️ `analyze_affinity_distribution.py` → Move to `scripts/analysis/`
- ⚠️ `create_antibody_antigen_summary.sh` → Move to `scripts/analysis/`
- ⚠️ `extract_extreme_affinity.sh` → Move to `scripts/analysis/`

---

## 🗑️ Files to EXCLUDE (Don't push to GitHub)

### Large Data Files (Already in .gitignore)
- ❌ `external_data/` - Too large (800+ MB)
- ❌ `*.csv` - Data files
- ❌ `*.npy` - Embeddings
- ❌ `*.pkl` - Checkpoints
- ❌ `*.pth` - Model files (except maybe one small example)

### Results and Outputs
- ❌ `results/` - Generated results
- ❌ `colab result/` - Colab outputs
- ❌ `*.log` - Log files

### Build Artifacts
- ❌ `__pycache__/` - Python cache
- ❌ `*.pyc` - Compiled Python
- ❌ `dist/` - Distribution files
- ❌ `build/` - Build files
- ❌ `*.egg-info/` - Package info

---

## 🔧 Recommended Cleanup Actions

### Step 1: Create Archive Folder

```bash
mkdir -p archive/old_versions
mkdir -p archive/old_docs
mkdir -p scripts/analysis
```

### Step 2: Move Old Versions

```bash
# Move v1 training notebook
mv colab_training.ipynb archive/old_versions/

# Move old documentation
mv README_COMPLETE.md archive/old_docs/
mv STATUS_AND_NEXT_STEPS.md archive/old_docs/
mv TRAINING_RESULTS_ANALYSIS.md docs/reports/v1_analysis.md
```

### Step 3: Consolidate Documentation

```bash
# Replace main README with GitHub version
mv README.md README_OLD.md
mv README_GITHUB.md README.md

# Merge quick start guides
# (manually combine QUICK_START.md and QUICK_START_V2.md)
```

### Step 4: Organize Analysis Scripts

```bash
# Move analysis scripts to subfolder
mv analyze_affinity*.* scripts/analysis/
mv create_antibody_antigen_summary.sh scripts/analysis/
mv extract_extreme_affinity.sh scripts/analysis/
```

### Step 5: Clean Up Docs

```bash
# Remove redundant quick start guides (after merging)
# rm docs/guides/QUICK_START.md
# rm docs/guides/QUICK_START_GUIDE.md

# Update documentation index
# Edit docs/guides/DOCUMENTATION_INDEX.md
```

---

## 📋 Pre-Push Checklist

Before pushing to GitHub, verify:

### Documentation
- [ ] README.md is comprehensive and clear
- [ ] All links in README work
- [ ] CHANGELOG.md is up to date
- [ ] LICENSE file exists
- [ ] CONTRIBUTING.md exists

### Code
- [ ] All import paths work
- [ ] No hardcoded paths to your local machine
- [ ] No sensitive information (API keys, passwords)
- [ ] Examples run without errors
- [ ] Tests pass: `pytest tests/`

### Files
- [ ] .gitignore includes all large files
- [ ] No data files in repo (check with `git status`)
- [ ] No model files except small examples
- [ ] No personal information

### Structure
- [ ] Folders are organized logically
- [ ] Documentation is easy to navigate
- [ ] No redundant files
- [ ] Archive folder for old versions

---

## 🚀 Automated Cleanup Script

Run this script to automatically organize files:

```bash
bash cleanup_for_github.sh
```

(Script is created in next step)

---

## 📝 Notes

### What Users Will Download

When users clone your repo, they'll get:
- ✅ Code and scripts (small)
- ✅ Documentation (small)
- ✅ Training notebooks (small)
- ❌ NOT data files (they download separately)
- ❌ NOT model files (they download separately)

### Recommended README Instructions

Add to README:
```markdown
## Downloading Data

Large files are not included in the repository. Download them separately:

```bash
# Download pre-trained model (optional)
wget https://yourserver.com/models/best_model_v2.pth -P models/

# Download dataset (for training)
bash scripts/download_all.sh
```
```

---

## ✅ Final Check

Repository should be:
- 📦 **Small** (<10 MB without data)
- 📖 **Well-documented** (clear README, guides)
- 🧹 **Clean** (no redundant files)
- 🔒 **Safe** (no sensitive data)
- 🎯 **Organized** (logical structure)
- 🚀 **Ready** (users can run immediately)

---

## 🎯 Target Repository Size

**Goal:** ~5-10 MB (excluding data/models)

**Breakdown:**
- Code: ~1 MB
- Documentation: ~2-3 MB
- Notebooks: ~2-3 MB
- Scripts: ~1 MB
- Other: ~1 MB

**What's excluded (users download separately):**
- Data: ~900 MB
- Models: ~1-10 MB each
- Results: varies

---

Ready to clean up? Follow the steps above or run the automated script!
