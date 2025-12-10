# Release Guide for v2.5.0 and v3.0.0

**Created**: 2025-11-13
**Purpose**: Step-by-step instructions for releasing versions

---

## 🚀 Release v2.5.0 (Do This Now)

### Step 1: Stage All Changes

```bash
cd /mnt/c/Users/401-24/Desktop/AbAg_binding_prediction

# Add all new and modified files
git add .

# Or selectively add important files:
git add models/model_igt5_esm2.py
git add training/train_igt5_esm2.py
git add notebooks/colab_training_SOTA.ipynb
git add docs/
git add README.md
git add START_HERE_FINAL.md
git add FILE_ORGANIZATION.md
git add PROJECT_STATUS.md
git add ESSENTIAL_FILES.md
git add CHANGELOG.md
git add VERSION_PLAN.md
git add RELEASE_GUIDE.md
```

### Step 2: Verify Staged Changes

```bash
# Check what will be committed
git status

# Review changes
git diff --staged
```

### Step 3: Commit Changes

```bash
git commit -m "Release v2.5.0: IgT5 + ESM-2 hybrid architecture

Major Changes:
- Add IgT5 encoder for antibody-specific features
- Implement hybrid IgT5 + ESM-2 architecture
- Add Google Colab training pipeline with auto-checkpointing
- Reorganize project structure (docs/, models/, training/, notebooks/)
- Add comprehensive documentation suite
- Remove old training outputs (~10.6 GB cleanup)

Technical Details:
- Model: IgT5 (1024-dim) + ESM-2 (1280-dim) → 2304-dim combined
- Training: Google Colab T4 GPU, 50 epochs, Focal MSE loss
- Expected: Spearman 0.60-0.70, Recall@pKd≥9 40-60%

Documentation:
- docs/PROJECT_LOG.md - Complete work history
- docs/OUTCOMES_AND_FUTURE_PLAN.md - Research plan
- docs/REFERENCES_AND_SOURCES.md - All citations
- docs/MODEL_COMPARISON_FINAL.md - Model selection rationale
- docs/COLAB_SETUP_GUIDE.md - Training instructions

See CHANGELOG.md for complete details.
"
```

### Step 4: Create Git Tag

```bash
# Create annotated tag
git tag -a v2.5.0 -m "Release v2.5.0: IgT5 + ESM-2 hybrid architecture

This release introduces a state-of-the-art hybrid architecture combining:
- IgT5 (Exscientia, Dec 2024) for antibody encoding
- ESM-2 650M for antigen encoding

Expected improvements:
- Spearman: 0.46 → 0.60-0.70 (+30-52%)
- Recall@pKd≥9: 14.22% → 40-60% (+181-322%)
- RMSE: 1.45 → 1.25-1.35 (-7-14%)

Key Features:
- Google Colab training pipeline (4-5 days vs 36 days local)
- Comprehensive documentation suite
- Auto-dimension detection
- Organized project structure

Training in progress, actual results in v3.0.0 (Nov 18-20, 2025).
"

# Verify tag
git tag -n9 v2.5.0
```

### Step 5: Push to GitHub

```bash
# Push commits
git push origin main

# Push tags
git push origin v2.5.0

# Or push everything at once
git push origin main --tags
```

### Step 6: Create GitHub Release

**Go to**: https://github.com/melmbrain/AbAg_binding_prediction/releases/new

**Tag version**: `v2.5.0`

**Release title**: `v2.5.0 - IgT5 + ESM-2 Hybrid Architecture`

**Description**: (Copy this)

```markdown
# AbAg Binding Prediction v2.5.0 - Architecture Upgrade

## 🎯 Highlights

This release introduces a **hybrid architecture** combining state-of-the-art antibody-specific (IgT5) and general protein (ESM-2) language models for improved binding affinity prediction.

## 🧬 What's New

### IgT5 + ESM-2 Hybrid Architecture
- **Antibody encoder**: IgT5 (Exscientia/IgT5, Dec 2024)
  - 1024-dim embeddings
  - Trained on 2B antibody sequences
  - State-of-the-art for binding affinity (R² 0.297-0.306)
- **Antigen encoder**: ESM-2 650M
  - 1280-dim embeddings
  - Proven for epitope prediction (AUC 0.76-0.789)

### Google Colab Training Pipeline
- Complete end-to-end notebook (`notebooks/colab_training_SOTA.ipynb`)
- Auto-checkpointing to Google Drive
- 7x faster than local training (4-5 days vs 36 days)
- Automatic resume from checkpoints

### Comprehensive Documentation
- [Project Log](docs/PROJECT_LOG.md) - Complete work history (402 lines)
- [Outcomes & Future Plan](docs/OUTCOMES_AND_FUTURE_PLAN.md) - Research roadmap (480 lines)
- [References](docs/REFERENCES_AND_SOURCES.md) - All citations (300+ lines)
- [Model Comparison](docs/MODEL_COMPARISON_FINAL.md) - Why IgT5 + ESM-2?
- [Colab Setup Guide](docs/COLAB_SETUP_GUIDE.md) - Training instructions

### Project Organization
- Reorganized from 20+ scattered files to clean structure
- `docs/` - All documentation
- `models/` - Model definitions
- `training/` - Training scripts
- `notebooks/` - Jupyter notebooks
- Removed ~10.6 GB of old training outputs

## 📊 Expected Improvements

| Metric | v2.0 (Baseline) | v2.5 (Target) | Improvement |
|--------|-----------------|---------------|-------------|
| **Spearman** | 0.46 | 0.60-0.70 | +30-52% |
| **Recall@pKd≥9** | 14.22% | 40-60% | +181-322% |
| **RMSE** | 1.45 | 1.25-1.35 | -7-14% |

**Note**: Model currently training on Google Colab. Actual results will be released in **v3.0.0** (Nov 18-20, 2025).

## 🚀 Quick Start

See [START_HERE_FINAL.md](START_HERE_FINAL.md) for complete instructions.

### Training on Colab

1. Upload files to Google Drive:
   - `notebooks/colab_training_SOTA.ipynb`
   - `agab_phase2_full.csv` (dataset)

2. Open notebook in Google Colab:
   - Double-click in Drive → "Open with Google Colaboratory"
   - Runtime → Change runtime type → GPU (T4)

3. Run all cells:
   - Training starts automatically
   - Checkpoints saved every epoch
   - ~4-5 days to complete

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Project overview |
| [START_HERE_FINAL.md](START_HERE_FINAL.md) | Quick start guide |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [docs/PROJECT_LOG.md](docs/PROJECT_LOG.md) | Complete work history |
| [docs/MODEL_COMPARISON_FINAL.md](docs/MODEL_COMPARISON_FINAL.md) | Model selection |
| [docs/REFERENCES_AND_SOURCES.md](docs/REFERENCES_AND_SOURCES.md) | Citations |
| [VERSION_PLAN.md](VERSION_PLAN.md) | Release roadmap |

## 🔧 Technical Details

### Model Architecture
```
Antibody Seq → IgT5 (1024-dim) ─┐
                                 ├─→ Deep Regressor → pKd
Antigen Seq  → ESM-2 (1280-dim) ─┘
```

### Training Configuration
- **Batch size**: 8
- **Loss**: Focal MSE (gamma=2.0)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.01)
- **Scheduler**: CosineAnnealingLR
- **Epochs**: 50
- **Device**: Google Colab T4 GPU

### Dataset
- **File**: agab_phase2_full.csv
- **Size**: 159,735 antibody-antigen pairs
- **Split**: 70% train, 15% validation, 15% test

## 🙏 Acknowledgments

- **IgT5**: Kenlay et al., "Large scale paired antibody language models", PLOS Computational Biology, Dec 2024
- **ESM-2**: Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model", Science, 2023
- Google Colab for providing free GPU resources

## 📅 What's Next?

**v3.0.0** (Nov 18-20, 2025) will include:
- ✅ Trained model weights (~2.5 GB)
- ✅ Complete performance benchmarks
- ✅ Inference API for production use
- ✅ Actual vs expected performance comparison

## 🐛 Known Issues

None currently. Report issues at: https://github.com/melmbrain/AbAg_binding_prediction/issues

## 📝 Full Changelog

See [CHANGELOG.md](CHANGELOG.md) for complete version history.
```

**Attachments**: None (code only release, model weights come in v3.0)

**Check**: ✅ Set as latest release

**Click**: Publish release

---

## 📋 Post-Release v2.5.0 Checklist

After releasing v2.5.0:

- [ ] Verify release appears on GitHub
- [ ] Test cloning fresh repository
- [ ] Verify documentation links work
- [ ] Check README.md renders correctly
- [ ] Announce release (if desired)
- [ ] Update project website/documentation (if exists)

---

## 🎯 Release v3.0.0 (After Training Completes)

**Timeline**: November 18-20, 2025 (after Colab training finishes)

### Prerequisites

1. **Training Completes**:
   - [ ] 50 epochs complete on Google Colab
   - [ ] Download `best_model.pth` from Google Drive (~2.5 GB)
   - [ ] Download `checkpoint_latest.pth` (optional)
   - [ ] Download training logs

2. **Evaluate on Test Set**:
   ```python
   # Run evaluation
   from training.train_igt5_esm2 import evaluate

   model = IgT5ESM2Model()
   model.load_state_dict(torch.load('best_model.pth'))

   test_metrics = evaluate(model, test_loader, device)

   # Save results
   import json
   with open('results/v3/test_metrics.json', 'w') as f:
       json.dump(test_metrics, f, indent=2)
   ```

3. **Verify Performance**:
   ```python
   recall_pkd9 = test_metrics['recall_pkd9']

   if recall_pkd9 >= 40:
       print("✅ Target met! Release as v3.0.0")
   elif recall_pkd9 >= 30:
       print("⚠️ Partial success, release as v2.6.0")
   else:
       print("❌ Target not met, continue optimization")
   ```

### Step 1: Prepare Files

```bash
# Create results directory
mkdir -p results/v3

# Add model checkpoint (use Git LFS for large files)
git lfs install
git lfs track "*.pth"
git add .gitattributes

# Add model
cp /path/to/downloaded/best_model.pth models/checkpoints/best_model_v3.pth
git add models/checkpoints/best_model_v3.pth

# Add results
git add results/v3/test_metrics.json
git add results/v3/predictions.csv
git add results/v3/analysis.ipynb
```

### Step 2: Update Documentation

```bash
# Update CHANGELOG.md
# Replace [Unreleased] section with actual v3.0.0 results

# Update OUTCOMES_AND_FUTURE_PLAN.md
# Fill in actual results vs expected

# Update README.md
# Update performance table with actual metrics
```

### Step 3: Create Inference API

```python
# Create abag_affinity/predictor_v3.py
# See VERSION_PLAN.md for template
```

### Step 4: Commit and Tag

```bash
# Commit
git add .
git commit -m "Release v3.0.0: Trained IgT5 + ESM-2 model

Results on Test Set:
- Spearman: [ACTUAL]
- Recall@pKd≥9: [ACTUAL]%
- RMSE: [ACTUAL]
- MAE: [ACTUAL]

Files Added:
- models/checkpoints/best_model_v3.pth - Trained weights
- results/v3/test_metrics.json - Evaluation results
- results/v3/analysis.ipynb - Result analysis
- abag_affinity/predictor_v3.py - Inference API

Training Details:
- 50 epochs on Google Colab T4 GPU
- Dataset: 159,735 antibody-antigen pairs
- Training time: ~4-5 days

See CHANGELOG.md for complete details.
"

# Tag
git tag -a v3.0.0 -m "Release v3.0.0: Trained IgT5 + ESM-2 Model

Final performance on test set:
- Spearman: [ACTUAL]
- Recall@pKd≥9: [ACTUAL]%
- RMSE: [ACTUAL]

This release includes:
- Trained model weights (2.5 GB)
- Complete evaluation results
- Production-ready inference API
- Performance benchmarks

Training: 50 epochs, Google Colab T4, 4-5 days
Dataset: 159,735 antibody-antigen pairs
"

# Push
git push origin main --tags
```

### Step 5: Create GitHub Release with Model

**Go to**: https://github.com/melmbrain/AbAg_binding_prediction/releases/new

**Tag version**: `v3.0.0`

**Release title**: `v3.0.0 - Trained IgT5 + ESM-2 Model`

**Description**: See VERSION_PLAN.md for template (update with actual results)

**Attachments**:
- If model <2GB: Attach `best_model_v3.pth` directly
- If model >2GB: Use Git LFS or provide external download link

**Example external link note**:
```markdown
## 📦 Download Model Weights

Due to file size, model weights are hosted externally:

**Download**: [best_model_v3.pth (2.5 GB)](https://drive.google.com/XXX)

Or use command:
```bash
wget https://github.com/melmbrain/AbAg_binding_prediction/releases/download/v3.0.0/best_model_v3.pth
# Or
python -m abag_affinity.download_model --version v3
```

Place in: `models/checkpoints/best_model_v3.pth`
```

---

## 🔧 Git LFS Setup (For Large Files)

If model is >100MB, use Git LFS:

```bash
# Install Git LFS
git lfs install

# Track .pth files
git lfs track "*.pth"
git add .gitattributes

# Add and commit large file
git add models/checkpoints/best_model_v3.pth
git commit -m "Add trained model weights (Git LFS)"
git push origin main

# Verify LFS
git lfs ls-files
```

**Note**: GitHub has 2GB file size limit. For larger files, host externally (Google Drive, Hugging Face, etc.)

---

## 📊 Version Comparison Table

Update this in README.md after v3.0.0:

```markdown
| Version | Architecture | Spearman | Recall@pKd≥9 | RMSE | Status |
|---------|-------------|----------|--------------|------|--------|
| v1.0.0 | ESM-2 only | ~0.40 | ~10% | ~1.50 | Released |
| v2.0.0 | ESM-2 + GELU | 0.46 | 14.22% | 1.45 | Released |
| v2.5.0 | IgT5 + ESM-2 | - | - | - | Training |
| v3.0.0 | IgT5 + ESM-2 (trained) | [ACTUAL] | [ACTUAL]% | [ACTUAL] | Planned |
```

---

## ✅ Final Checklist

### v2.5.0 (Now)
- [ ] All changes staged
- [ ] CHANGELOG.md created
- [ ] Commit message written
- [ ] Tag v2.5.0 created
- [ ] Pushed to GitHub
- [ ] GitHub release created
- [ ] Documentation verified

### v3.0.0 (Later)
- [ ] Training completed
- [ ] Model downloaded
- [ ] Test evaluation done
- [ ] Performance verified
- [ ] Results documented
- [ ] Inference API created
- [ ] Files committed
- [ ] Tag v3.0.0 created
- [ ] GitHub release created
- [ ] Model weights uploaded

---

## 🎯 Quick Reference

### Check Current Version
```bash
git describe --tags --abbrev=0
```

### List All Tags
```bash
git tag -l
```

### View Tag Details
```bash
git show v2.5.0
```

### Delete Tag (If Needed)
```bash
# Local
git tag -d v2.5.0

# Remote
git push origin :refs/tags/v2.5.0
```

### Re-tag (If Needed)
```bash
# Delete old tag
git tag -d v2.5.0
git push origin :refs/tags/v2.5.0

# Create new tag
git tag -a v2.5.0 -m "Updated message"
git push origin v2.5.0
```

---

**Ready to Release**: v2.5.0 ✅
**Next Release**: v3.0.0 (pending training completion)
**Last Updated**: 2025-11-13
