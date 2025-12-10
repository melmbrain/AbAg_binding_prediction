# Version Release Plan

**Current Version**: v2.0.0 (ESM-2 model with GELU activation)
**Next Versions**: v2.5.0 (Architecture upgrade) → v3.0.0 (Trained model release)

---

## 📊 Version History

### v1.0.0 (Initial Release)
- Basic ESM-2 model
- PCA-reduced features (150-dim)
- Standard MSE loss
- Performance: Spearman ~0.40, Recall@pKd≥9 ~10%

### v2.0.0 (Current - Nov 2025)
- ✅ GELU activation (improved from ReLU)
- ✅ Deeper architecture (4 hidden layers)
- ✅ Focal MSE loss
- ✅ 10x class weighting for extreme affinities
- **Performance**: Spearman 0.46 (epoch 5/50, incomplete)
- **Status**: Incomplete training (stopped at epoch 6)

---

## 🎯 v2.5.0 - Architecture Upgrade (Submit Now)

**Release Date**: November 13-14, 2025
**Status**: Ready to commit
**Type**: Major architectural change (feature update)

### What's New

#### 1. **Hybrid Architecture: IgT5 + ESM-2**
```python
# v2.0: ESM-2 only
antibody → ESM-2 (1280-dim) ─┐
antigen  → ESM-2 (1280-dim) ─┴→ Regressor → pKd

# v2.5: IgT5 + ESM-2 hybrid
antibody → IgT5 (1024-dim) ─┐
antigen  → ESM-2 (1280-dim) ─┴→ Regressor → pKd
```

**Why**:
- IgT5 is state-of-the-art for antibodies (Dec 2024, R² 0.297-0.306)
- Antibody-specific features vs general protein model
- Expected +10-20% improvement in Recall@pKd≥9

#### 2. **Auto-Dimension Detection**
```python
# Fixed hardcoded dimensions bug
igt5_dim = self.igt5_model.config.d_model  # Auto-detect (1024)
esm2_dim = self.esm2_model.config.hidden_size  # Auto-detect (1280)
```

#### 3. **Google Colab Training Pipeline**
- Complete end-to-end notebook
- Auto-checkpointing to Google Drive
- T4 GPU optimization (4-5 days vs 36 days local)

#### 4. **Project Reorganization**
```
Before: ~20 markdown files scattered in root
After:  Organized docs/, models/, training/, notebooks/
```

#### 5. **Comprehensive Documentation**
- `docs/PROJECT_LOG.md` - Complete work history (402 lines)
- `docs/OUTCOMES_AND_FUTURE_PLAN.md` - Expected results & research plan (480 lines)
- `docs/REFERENCES_AND_SOURCES.md` - All citations (300+ lines)
- `docs/MODEL_COMPARISON_FINAL.md` - Model selection rationale
- `docs/COLAB_SETUP_GUIDE.md` - Training instructions

### Changes Summary

**Added**:
- IgT5 antibody encoder (Exscientia/IgT5)
- Hybrid model architecture
- Colab training notebook
- Complete documentation suite
- Organized directory structure

**Changed**:
- Model architecture (ESM-2 only → IgT5 + ESM-2)
- Embedding dimensions (2560-dim → 2304-dim)
- Training platform (local → Colab)

**Removed**:
- Old training outputs (~10.6 GB)
- Redundant documentation
- Failed training attempts

### Expected Performance (After Training)

| Metric | v2.0 (E5) | v2.5 Target | Improvement |
|--------|-----------|-------------|-------------|
| Spearman | 0.46 | 0.60-0.70 | +30-52% |
| Recall@pKd≥9 | 14.22% | 40-60% | +181-322% |
| RMSE | 1.45 | 1.25-1.35 | -7-14% |

**Note**: Actual performance to be measured when training completes (Nov 17-18)

### Files to Commit

```bash
# New core files
models/model_igt5_esm2.py
training/train_igt5_esm2.py
notebooks/colab_training_SOTA.ipynb

# Documentation
docs/PROJECT_LOG.md
docs/OUTCOMES_AND_FUTURE_PLAN.md
docs/REFERENCES_AND_SOURCES.md
docs/MODEL_COMPARISON_FINAL.md
docs/COLAB_SETUP_GUIDE.md

# Updated
README.md
START_HERE_FINAL.md

# New guides
FILE_ORGANIZATION.md
PROJECT_STATUS.md
ESSENTIAL_FILES.md
VERSION_PLAN.md
```

---

## 🚀 v3.0.0 - Trained Model Release (After Training)

**Release Date**: November 18-20, 2025 (after training completes)
**Status**: Pending training completion
**Type**: Major version (trained model + evaluation)

### Prerequisites

- [🔄] Training completes on Colab (Nov 17-18)
- [ ] Evaluate on test set
- [ ] Verify performance meets targets
- [ ] Download best_model.pth from Google Drive

### What's New (Planned)

#### 1. **Trained Model Weights**
```
models/
├── model_igt5_esm2.py          ← Architecture definition
└── checkpoints/
    └── best_model_v3.pth       ← Trained weights (~2.5 GB)
```

**If Recall@pKd≥9 ≥ 40%**: Release as v3.0.0 ✅
**If Recall@pKd≥9 = 30-40%**: Release as v2.6.0 (partial success)
**If Recall@pKd≥9 < 30%**: Don't release, continue optimization

#### 2. **Performance Benchmarks**
```
results/v3/
├── test_metrics.json           ← Final evaluation results
├── predictions.csv             ← Test set predictions
└── analysis.ipynb              ← Result analysis notebook
```

#### 3. **Inference API**
```python
# New file: abag_affinity/predictor_v3.py
from abag_affinity import AffinityPredictorV3

predictor = AffinityPredictorV3.from_pretrained('models/checkpoints/best_model_v3.pth')

result = predictor.predict(
    antibody_sequence="EVQLQQSGPGLVKPSQTLSLTCAISGDSVSSNSAAWN...",
    antigen_sequence="KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNF..."
)

print(f"Predicted pKd: {result['pKd']:.2f}")
print(f"Predicted Kd: {result['Kd_nM']:.1f} nM")
print(f"Confidence: {result['confidence']:.2f}")
```

#### 4. **Updated Documentation**
- `CHANGELOG.md` - Complete version history
- `docs/OUTCOMES_AND_FUTURE_PLAN.md` - Actual results (vs expected)
- `docs/PERFORMANCE_ANALYSIS.md` - Detailed evaluation
- `README.md` - Updated with v3.0 results

#### 5. **Release Assets**

**GitHub Release** includes:
- Source code (automatic)
- `best_model_v3.pth` (2.5 GB) - via Git LFS or external link
- `test_metrics.json` - Performance benchmarks
- `requirements.txt` - Dependencies
- `INSTALL.md` - Installation guide

### Conditional Release Strategy

#### Scenario A: High Performance (Recall ≥ 40%)
**Action**: Release as **v3.0.0** 🎉
**Tag**: `v3.0.0` - "IgT5 + ESM-2 trained model"
**Highlights**:
- ✅ Achieved 3x improvement in Recall@pKd≥9
- ✅ State-of-the-art antibody-antigen binding prediction
- ✅ Production-ready model

#### Scenario B: Medium Performance (Recall 30-40%)
**Action**: Release as **v2.6.0** (incremental)
**Tag**: `v2.6.0` - "IgT5 + ESM-2 initial training"
**Notes**:
- Partial success, needs optimization
- Document limitations
- Plan v3.0 with improvements (upsampling, attention, etc.)

#### Scenario C: Low Performance (Recall < 30%)
**Action**: Do NOT release trained model
**Next Steps**:
- Debug and analyze errors
- Implement Phase 3 optimizations (see OUTCOMES_AND_FUTURE_PLAN.md)
- Re-train with improvements
- Release as v3.0 when targets met

---

## 📋 Release Checklist

### For v2.5.0 (Now)

- [ ] Add all modified files to staging
- [ ] Create CHANGELOG.md for v2.5.0
- [ ] Update README.md version badge
- [ ] Commit with message: "Release v2.5.0: IgT5 + ESM-2 architecture"
- [ ] Tag commit: `git tag -a v2.5.0 -m "Version 2.5.0"`
- [ ] Push to GitHub: `git push origin main --tags`
- [ ] Create GitHub release with notes

### For v3.0.0 (After Training)

**Preparation** (Before Training Completes):
- [ ] Prepare inference script template
- [ ] Prepare evaluation notebook
- [ ] Draft CHANGELOG.md for v3.0

**After Training Completes** (Nov 17-18):
- [ ] Download best_model.pth from Colab
- [ ] Run evaluation on test set
- [ ] Generate test_metrics.json
- [ ] Analyze results vs targets

**If Results Good** (Recall ≥ 40%):
- [ ] Add model checkpoint to repo (Git LFS)
- [ ] Update CHANGELOG.md with actual results
- [ ] Update README.md with benchmarks
- [ ] Update OUTCOMES_AND_FUTURE_PLAN.md with actual vs expected
- [ ] Create inference API (predictor_v3.py)
- [ ] Commit: "Release v3.0.0: Trained IgT5 + ESM-2 model"
- [ ] Tag: `git tag -a v3.0.0 -m "Version 3.0.0 - Trained model"`
- [ ] Push: `git push origin main --tags`
- [ ] Create GitHub release with model weights

---

## 📝 CHANGELOG Template

### For v2.5.0 (Create Now)

```markdown
# Changelog

## [2.5.0] - 2025-11-13

### Added
- **IgT5 encoder** for antibody-specific features (Exscientia/IgT5)
- **Hybrid architecture** combining IgT5 (antibody) + ESM-2 (antigen)
- **Google Colab training pipeline** with auto-checkpointing
- Comprehensive documentation suite (PROJECT_LOG, OUTCOMES, REFERENCES)
- Auto-dimension detection for model configs
- Organized project structure (docs/, models/, training/, notebooks/)

### Changed
- Model architecture from ESM-2 only → IgT5 + ESM-2 hybrid
- Embedding dimensions from 2560-dim → 2304-dim (1024 + 1280)
- Training platform from local RTX 2060 → Google Colab T4
- Documentation structure (consolidated into docs/)

### Removed
- Old training outputs and logs (~10.6 GB)
- Redundant documentation files
- Incomplete ESM-2 checkpoints

### Expected Improvements
- Spearman correlation: 0.46 → 0.60-0.70 (+30-52%)
- Recall@pKd≥9: 14.22% → 40-60% (+181-322%)
- RMSE: 1.45 → 1.25-1.35 (-7-14%)

**Note**: Training in progress, actual results TBD (Nov 17-18, 2025)

## [2.0.0] - 2025-11-XX

### Added
- GELU activation for smoother gradients
- Deeper architecture (4 hidden layers: 512→256→128→64)
- Focal MSE loss for hard example mining
- 10x stronger class weights for extreme affinities

### Improvements
- Overall performance: 6-14% improvement
- Moderate affinities: 26% improvement

## [1.0.0] - 2025-XX-XX

### Initial Release
- ESM-2 based antibody-antigen binding prediction
- PCA-reduced features (150-dim)
- Basic architecture with weighted MSE loss
```

### For v3.0.0 (Create After Training)

```markdown
## [3.0.0] - 2025-11-18

### Added
- **Trained model weights** (best_model_v3.pth, 2.5 GB)
- **Inference API** for production use (predictor_v3.py)
- **Performance benchmarks** on test set
- Complete evaluation results and analysis

### Performance (Actual Results)
- Spearman correlation: [ACTUAL] (target: 0.60-0.70)
- Recall@pKd≥9: [ACTUAL]% (target: 40-60%)
- RMSE: [ACTUAL] (target: 1.25-1.35)
- MAE: [ACTUAL]
- Pearson correlation: [ACTUAL]

### Model Details
- Architecture: IgT5 (antibody) + ESM-2 (antigen)
- Training: 50 epochs on Google Colab T4 GPU
- Dataset: 159,735 antibody-antigen pairs
- Loss: Focal MSE (gamma=2.0)
- Optimizer: AdamW with cosine annealing

### Files
- `models/checkpoints/best_model_v3.pth` - Trained weights
- `results/v3/test_metrics.json` - Evaluation results
- `abag_affinity/predictor_v3.py` - Inference API

### Installation
```bash
pip install -r requirements.txt
python -m abag_affinity.download_model  # Downloads v3 weights
```

### Usage
```python
from abag_affinity import AffinityPredictorV3
predictor = AffinityPredictorV3.from_pretrained()
result = predictor.predict(antibody_seq, antigen_seq)
```
```

---

## 🔖 Git Tag Strategy

### Version Naming
- **Major.Minor.Patch** (Semantic Versioning)
- **Major**: Breaking changes, new architecture
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes

### Tags
- `v1.0.0` - Initial ESM-2 model
- `v2.0.0` - GELU + Focal loss improvements
- `v2.5.0` - IgT5 + ESM-2 architecture (current)
- `v3.0.0` - Trained model release (pending)

### Tag Commands
```bash
# For v2.5.0 (now)
git tag -a v2.5.0 -m "Release v2.5.0: IgT5 + ESM-2 hybrid architecture"
git push origin v2.5.0

# For v3.0.0 (after training)
git tag -a v3.0.0 -m "Release v3.0.0: Trained IgT5 + ESM-2 model"
git push origin v3.0.0
```

---

## 📦 GitHub Release Notes

### v2.5.0 Release Notes Template

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
- Complete end-to-end notebook
- Auto-checkpointing to Google Drive
- 7x faster than local training (4-5 days vs 36 days)

### Comprehensive Documentation
- Complete work history and decision rationale
- Future research plan with optimization strategies
- All citations and references
- Model comparison and selection process

## 📊 Expected Improvements

| Metric | v2.0 | v2.5 (Target) | Improvement |
|--------|------|---------------|-------------|
| Spearman | 0.46 | 0.60-0.70 | +30-52% |
| Recall@pKd≥9 | 14.22% | 40-60% | +181-322% |
| RMSE | 1.45 | 1.25-1.35 | -7-14% |

**Note**: Model currently training on Google Colab. Actual results will be released in v3.0.0 (Nov 18-20, 2025).

## 🚀 Quick Start

See [START_HERE_FINAL.md](START_HERE_FINAL.md) for complete instructions.

## 📖 Documentation

- [Project Log](docs/PROJECT_LOG.md) - Complete work history
- [Model Comparison](docs/MODEL_COMPARISON_FINAL.md) - Why IgT5 + ESM-2?
- [References](docs/REFERENCES_AND_SOURCES.md) - All citations
- [Future Plan](docs/OUTCOMES_AND_FUTURE_PLAN.md) - Research roadmap

## 🙏 Acknowledgments

- **IgT5**: Kenlay et al., PLOS Computational Biology, Dec 2024
- **ESM-2**: Lin et al., Science, 2023
- Google Colab for providing free GPU resources

## 📅 What's Next?

**v3.0.0** (Nov 18-20, 2025) will include:
- Trained model weights
- Complete performance benchmarks
- Inference API for production use
```

---

## 🎯 Summary

### Immediate Actions (v2.5.0)
1. Create CHANGELOG.md
2. Stage all changes
3. Commit with proper message
4. Tag as v2.5.0
5. Push to GitHub
6. Create GitHub release

### Future Actions (v3.0.0)
1. Wait for training to complete (Nov 17-18)
2. Evaluate results
3. If good: prepare v3.0.0 release
4. If not: iterate and improve
5. Release trained model when ready

---

**Current Status**: Ready to release v2.5.0
**Next Milestone**: v3.0.0 (pending training completion)
**Last Updated**: 2025-11-13
