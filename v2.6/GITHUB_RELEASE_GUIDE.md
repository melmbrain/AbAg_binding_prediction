# GitHub Release Guide: v2.6.0-beta

This guide walks you through releasing v2.6 to GitHub with proper documentation.

---

## Prerequisites

- [x] Git repository initialized
- [x] GitHub repository created
- [x] Model files ready in `v2.6/` directory
- [x] Documentation complete

---

## Step 1: Prepare Large Files for Upload

The model file (16.3 GB) is too large for regular Git. Use one of these options:

### Option A: Git LFS (Recommended)

```bash
# Install Git LFS (if not already)
git lfs install

# Track large files
git lfs track "*.pth"
git lfs track "v2.6/*.pth"

# Verify tracking
git lfs ls-files
```

### Option B: Upload to External Storage

Upload `checkpoint_latest.pth` to:
- Hugging Face Hub (recommended)
- Google Drive
- Zenodo
- AWS S3

Then link to it in the release notes.

---

## Step 2: Stage and Commit Files

### Add v2.6 Documentation (Small Files)

```bash
# Navigate to repo
cd C:\Users\401-24\Desktop\AbAg_binding_prediction

# Check current status
git status

# Add v2.6 documentation and artifacts (excluding large .pth file)
git add v2.6/README_v2.6.md
git add v2.6/RELEASE_NOTES_v2.6.md
git add v2.6/GITHUB_RELEASE_GUIDE.md
git add v2.6/metrics.json
git add v2.6/training_history_v2.6.csv
git add v2.6/training_history_v2.6.png
git add v2.6/training_summary_v2.6.json
git add v2.6/test_predictions.csv

# Commit
git commit -m "Release v2.6.0-beta: Dual-encoder with known stability issues

- Add IgT5 + ESM-2 3B dual-encoder architecture
- Add cross-attention layer for Ab-Ag interactions
- Achieve 6-8× training speedup
- Document known stability issues (recall oscillation)
- Include complete training artifacts and metrics

Performance:
- Spearman ρ: 0.390
- RMSE: 2.10
- Recall@pKd≥9: 100%

Known Issues:
- Recall instability (18% ↔ 100%)
- Invalid predictions (negative pKd)
- See v2.7 roadmap for fixes

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Add Model File (If Using Git LFS)

```bash
# Only if using Git LFS
git add v2.6/checkpoint_latest.pth
git commit -m "Add v2.6 model checkpoint (16.3GB)

Training: 15 epochs on A100 80GB
Best epoch: 12
Validation Spearman: 0.390

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Step 3: Create Git Tag

```bash
# Create annotated tag for v2.6
git tag -a v2.6.0-beta -m "Release v2.6.0-beta

Experimental dual-encoder model with known stability issues.

Model: IgT5 + ESM-2 3B + Cross-Attention
Performance: Spearman 0.390, RMSE 2.10
Status: Superseded by v2.7 (in development)

Full release notes: v2.6/RELEASE_NOTES_v2.6.md"

# Verify tag
git tag -l -n9 v2.6.0-beta
```

---

## Step 4: Push to GitHub

### Push Commits

```bash
# Push main branch
git push origin main

# Push tag
git push origin v2.6.0-beta
```

### Push LFS Files (If Applicable)

```bash
# Git LFS will automatically push large files
# You may see progress like:
# Uploading LFS objects: 100% (1/1), 16 GB | 5 MB/s
```

**Note**: LFS push can take 30-60 minutes for 16GB file depending on connection speed.

---

## Step 5: Create GitHub Release

### Via Web Interface

1. **Go to your repository** on GitHub
2. **Click "Releases"** (right sidebar)
3. **Click "Create a new release"**
4. **Fill in the form**:

   - **Tag**: `v2.6.0-beta` (select from dropdown)
   - **Release title**: `v2.6.0-beta: Dual-Encoder with Cross-Attention`
   - **Description**: Copy from below

5. **Upload binaries** (if NOT using LFS):
   - Drag `checkpoint_latest.pth` to upload area
   - Or link to external storage

6. **Check "This is a pre-release"** ✓
7. **Click "Publish release"**

### Release Description Template

```markdown
# v2.6.0-beta: Experimental Dual-Encoder Release

⚠️ **Warning**: This is an experimental beta with known stability issues. Production use not recommended.

## What's New

- 🏗️ **Dual-Encoder Architecture**: IgT5 (antibody) + ESM-2 3B (antigen)
- 🔗 **Cross-Attention**: Better Ab-Ag interaction modeling
- ⚡ **6-8× Training Speedup**: 4 hours/epoch vs 28 hours in v2.5
- 📊 **Complete Training Logs**: Full 15-epoch history included

## Performance

| Metric | Value |
|--------|-------|
| **Spearman ρ** | 0.390 |
| **Pearson r** | 0.696 |
| **RMSE** | 2.10 |
| **Recall@pKd≥9** | 100% |
| **Precision** | 35.8% |

## Known Issues

### ⚠️ Critical

1. **Recall Instability**: Oscillates between 18% and 100% (σ = 39.35%)
   - Cause: Soft Spearman loss gradient instability
   - Fix: v2.7 will use MSE loss

2. **Invalid Predictions**: Negative pKd values (impossible)
   - Range: -2.48 to 10.0 (should be 4.0-14.0)
   - Fix: v2.7 will clamp outputs

3. **Prediction Clustering**: Many predictions at 9.06 and 10.31

## Files

### Documentation
- 📖 [README_v2.6.md](https://github.com/yourusername/AbAg_binding_prediction/blob/main/v2.6/README_v2.6.md) - Complete model documentation
- 📋 [RELEASE_NOTES_v2.6.md](https://github.com/yourusername/AbAg_binding_prediction/blob/main/v2.6/RELEASE_NOTES_v2.6.md) - Detailed release notes
- 🔧 Usage examples and API reference

### Training Artifacts
- 📊 `metrics.json` - Final test/val metrics + hyperparameters
- 📈 `training_history_v2.6.csv` - Epoch-by-epoch logs (15 epochs)
- 📉 `training_history_v2.6.png` - Training curves visualization
- 📝 `training_summary_v2.6.json` - Summary statistics
- 🎯 `test_predictions.csv` - Model predictions on test set (5,707 samples)

### Model File
- 🤖 `checkpoint_latest.pth` (16.3 GB) - Full model checkpoint
  - [Download from GitHub LFS](#) OR
  - [Download from Hugging Face](#) OR
  - [Download from Google Drive](#)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/AbAg_binding_prediction.git
cd AbAg_binding_prediction

# Checkout v2.6 tag
git checkout v2.6.0-beta

# Install dependencies
pip install -r requirements.txt

# Download model (if not using LFS)
# [Instructions for your chosen storage method]
```

## Quick Start

```python
import torch
from model import DualEncoderWithCrossAttention

# Load model
checkpoint = torch.load('v2.6/checkpoint_latest.pth')
model = DualEncoderWithCrossAttention(dropout=0.3, use_cross_attention=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
# See README_v2.6.md for full examples
```

## What's Next: v2.7

Expected: **2025-12-01**

**Fixes**:
- ✅ Stable MSE + BCE loss (no Soft Spearman)
- ✅ Prediction clamping [4.0, 14.0]
- ✅ NaN/Inf detection
- ✅ Complete RNG state saving
- ✅ Overfitting monitoring

**Expected Improvements**:
- Spearman: 0.39 → 0.45-0.55 (+15-40%)
- Recall: 18-100% (unstable) → 50-70% (stable)
- RMSE: 2.1 → 1.2-1.5 (-30%)

See [V2.7_IMPROVEMENTS.md](https://github.com/yourusername/AbAg_binding_prediction/blob/main/V2.7_IMPROVEMENTS.md) for full roadmap.

## Support

- 📚 [Full Documentation](https://github.com/yourusername/AbAg_binding_prediction#readme)
- 🐛 [Report Issues](https://github.com/yourusername/AbAg_binding_prediction/issues)
- 💬 [Discussions](https://github.com/yourusername/AbAg_binding_prediction/discussions)

## Citation

```bibtex
@software{abag_binding_v26,
  title = {Antibody-Antigen Binding Prediction v2.6},
  author = {Your Name},
  year = {2025},
  month = {11},
  version = {2.6.0-beta},
  url = {https://github.com/yourusername/AbAg_binding_prediction}
}
```

---

**⚠️ Experimental Release**: This version has known stability issues. Wait for v2.7 stable release for production use.

**Status**: Superseded by v2.7 (in development)
```

---

## Step 6: Alternative - Upload to Hugging Face Hub

If GitHub LFS is slow or you want better visibility:

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create model repo
huggingface-cli repo create AbAg-binding-v2.6 --type model

# Upload model
huggingface-cli upload yourusername/AbAg-binding-v2.6 \
  v2.6/checkpoint_latest.pth \
  --repo-type model

# Upload artifacts
huggingface-cli upload yourusername/AbAg-binding-v2.6 \
  v2.6/ \
  --repo-type model
```

Then link to Hugging Face in your GitHub release.

---

## Step 7: Update Main README

Add v2.6 release info to main README:

```bash
# Edit main README
# Add section:
```

```markdown
## 📦 Releases

### v2.6.0-beta (2025-11-25) - Current
- **Model**: IgT5 + ESM-2 3B with cross-attention
- **Performance**: Spearman 0.390, RMSE 2.10
- **Status**: ⚠️ Experimental (known stability issues)
- **Download**: [GitHub Release](https://github.com/yourusername/AbAg_binding_prediction/releases/tag/v2.6.0-beta)
- **Model Card**: [README_v2.6.md](v2.6/README_v2.6.md)

### v2.5.0 (2025-11-13) - Stable
- **Model**: ESM-2 650M
- **Performance**: Spearman 0.42, RMSE 1.95
- **Status**: ✅ Stable
```

```bash
# Commit README update
git add README.md
git commit -m "Update README with v2.6 release info"
git push origin main
```

---

## Step 8: Announce Release

### Create Discussion Post

1. Go to your repo's **Discussions** tab
2. Create new discussion: **"v2.6.0-beta Released: Experimental Dual-Encoder Model"**
3. Include:
   - Release highlights
   - Known issues and warnings
   - Link to full release notes
   - Call for feedback

### Share on Social Media (Optional)

```text
🚀 Just released v2.6.0-beta of our Ab-Ag binding predictor!

✨ Dual-encoder (IgT5 + ESM-2 3B)
⚡ 6-8× training speedup
📊 Spearman 0.39, RMSE 2.10

⚠️ Experimental: Has stability issues
📖 Full details: [GitHub link]

#MachineLearning #Antibodies #ProteinDesign
```

---

## Verification Checklist

Before announcing:

- [ ] All files committed and pushed
- [ ] Tag created and pushed
- [ ] GitHub release published
- [ ] Model file accessible (LFS or external)
- [ ] README updated
- [ ] Release marked as "pre-release"
- [ ] Documentation complete (README, RELEASE_NOTES)
- [ ] Known issues clearly documented
- [ ] v2.7 roadmap mentioned

---

## Troubleshooting

### Git LFS Slow Upload

```bash
# Check LFS status
git lfs status

# If stuck, try pushing in smaller batches
# Or upload to Hugging Face instead
```

### File Too Large Error

```bash
# Error: file exceeds GitHub's 100 MB limit
# Solution: Use Git LFS or external storage
git lfs track "*.pth"
git add .gitattributes
git commit -m "Add LFS tracking"
```

### Release Not Showing

- Check tag pushed: `git ls-remote --tags origin`
- Verify release published (not draft)
- Wait a few minutes for processing

---

## After Release

1. **Monitor Issues**: Watch for bug reports
2. **Respond to Feedback**: Acknowledge stability issues
3. **Start v2.7**: Begin work on fixes
4. **Track Downloads**: Monitor release metrics

---

## Need Help?

- **Git LFS**: https://git-lfs.github.com/
- **GitHub Releases**: https://docs.github.com/en/repositories/releasing-projects-on-github
- **Hugging Face Hub**: https://huggingface.co/docs/hub/

---

*Last updated: 2025-11-25*
*Guide version: 1.0*
