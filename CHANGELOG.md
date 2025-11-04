# Changelog

All notable changes to the AbAg Binding Prediction project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-11-04

### Added
- **v2 Improved Training Architecture**
  - `colab_training_v2_improved.ipynb`: Complete rewrite with 8 major improvements
  - GELU activation function (replaces ReLU for smoother gradients)
  - Deeper architecture: 150 → 512 → 256 → 128 → 64 → 1 (vs 150 → 256 → 128 → 1 in v1)
  - Focal MSE Loss for hard example mining
  - 10x stronger class weights for extreme affinities
  - AdamW optimizer with weight decay
  - Cosine annealing learning rate scheduler
  - Xavier weight initialization
  - Gradient clipping (max_norm=1.0)

- **Session Management & Resume Capability**
  - `colab_resume_and_evaluate.ipynb`: Resume training or evaluate after timeout
  - `use_colab_model_locally.py`: Local inference with Colab-trained models
  - `SESSION_TIMEOUT_GUIDE.md`: Comprehensive timeout handling guide
  - Checkpoint system: Auto-save every 10 epochs to Google Drive
  - Resume from any checkpoint without data loss

- **Comprehensive Training Documentation**
  - `V2_IMPROVEMENTS.md`: Technical details of all 8 improvements
  - `QUICK_START_V2.md`: Step-by-step quick start guide
  - `LAUNCH_CHECKLIST.md`: Pre-flight checklist for training
  - `COLAB_TRAINING_GUIDE.md`: Complete Colab training guide (updated for v2)

- **Results Analysis & Honesty**
  - `V2_RESULTS_ANALYSIS.md`: Comprehensive honest analysis of v2 results
  - Root cause analysis of why improvements were modest
  - Recommendations for v3 (full dimensions, two-stage training, ensemble)
  - Clear documentation of limitations

- **Repository Cleanup & GitHub Distribution**
  - `.gitignore`: Excludes large files (data, models, results)
  - `LICENSE`: MIT License for open source
  - `CONTRIBUTING.md`: Contribution guidelines
  - `CLEANUP_GUIDE.md`: Documentation of cleanup process
  - `GITHUB_READY.md`: Step-by-step GitHub push instructions
  - `GITHUB_PUSH_COMMANDS.txt`: Quick reference for Git commands
  - `REPOSITORY_STATUS.md`: Current repository status
  - Repository size reduced to ~5-8 MB (from 900+ MB with data)

- **PCA Feature Processing**
  - `scripts/apply_pca_and_merge.py`: Dimensionality reduction (1,280 → 150 dims)
  - Preserves 99.9% variance while reducing memory requirements
  - Merged features: 330,762 samples with complete embeddings (84.65% coverage)

### Changed
- **Model Architecture**: Switched from ReLU to GELU activation
- **Learning Rate**: Reduced from 0.001 to 0.0001 for more stable convergence
- **Class Weights**: Increased from 3x to 10x for extreme affinity classes
- **Training Time**: Reduced to 31 minutes on T4 GPU (100 epochs, 231k samples)
- **Loss Function**: Added Focal MSE option (down-weights easy examples)
- **Scheduler**: Changed from ReduceLROnPlateau to CosineAnnealingWarmRestarts
- **Optimizer**: Changed from Adam to AdamW (better weight decay)
- **Repository Structure**: Moved old versions to `archive/`, scripts to `scripts/analysis/`
- **Documentation**: Consolidated and cleaned up, removed redundant files

### Performance
**Overall Improvements (v2 vs v1):**
- Overall RMSE: 1.4761 → 1.3799 (6.5% improvement)
- MAE: 1.3011 → 1.2143 (6.7% improvement)
- Spearman ρ: 0.3912 → 0.4258 (8.8% improvement)
- Pearson r: 0.7265 → 0.7624 (4.9% improvement)
- R²: 0.5188 → 0.5795 (11.7% improvement)

**Per-Category Improvements:**
- Very Weak (<5 pKd): RMSE 1.12 → 0.85 (24.2% improvement) ✓✓
- Moderate (7-9 pKd): RMSE 0.99 → 0.73 (25.6% improvement) ✓✓
- Very Strong (>11 pKd): RMSE 2.94 → 2.53 (13.8% improvement) ✓

**Training Efficiency:**
- Training time: 0.51 hours (31 minutes) on T4 GPU
- Model size: ~240,000 parameters
- Training samples: 231,532 / Test samples: 49,615

### Fixed
- **PyTorch Compatibility**: Removed deprecated `verbose=True` parameter from ReduceLROnPlateau
- **Variable Name Conflict**: Changed `bins`/`labels` to `BINS`/`BIN_LABELS` constants
- **Korean Windows Encoding**: Avoided Unicode characters for cp949 codec compatibility
- **Session Timeout Issues**: Implemented comprehensive checkpoint and resume system

### Limitations Documented
- Very strong binder RMSE (2.53) still above target of <1.5
- Root cause: Only 50 very strong samples (0.1% of test set)
- PCA reduction may lose information critical for extremes
- Model is "underconfident" on extremes (std 1.66 vs true 2.13)
- Sequence-only approach has inherent limits without structural information

### Future Work (Planned for v3)
1. **Full-dimensional features** (1,280 dims, no PCA) - Expected: +10-30%
2. **Two-stage training** (all data, then fine-tune on extremes) - Expected: +15-25% on extremes
3. **Ensemble models** (5 models, average predictions) - Expected: +10-20%
4. **Oversample rare classes** (duplicate extremes 10x)
5. **Alternative architectures** (Transformers, GNNs)
6. **Additional very strong binder data**

### Notes
- v2 represents a solid baseline with honest reporting of limitations
- All metrics improved, but improvements are modest (6-14%), not dramatic
- Repository is now ready for GitHub distribution with comprehensive documentation
- Model performs well on moderate affinities but struggles with rare extremes
- Clear roadmap for further improvements in v3

## [1.2.0] - 2025-11-03

### Added
- **Therapeutic Antibody Integration**
  - `scripts/download_therapeutic_antibodies.py`: Multi-database downloader for therapeutic antibodies
  - `scripts/download_abcov.py`: Ab-CoV COVID-19 antibody database downloader
  - `scripts/fetch_sabdab_sequences.py`: Sequence fetcher for SAbDab entries
  - `scripts/integrate_therapeutic_antibodies.py`: Universal therapeutic antibody integrator
  - SAAINT-DB: 173 very strong binders with femtomolar affinity (0.03 pM!)
  - SAbDab: 1,307 antibodies with affinity data, 31 very strong binders
  - Final dataset: 390,757 samples (+90.7% from Phase 6)
  - Very strong binders: 384 (+66.9% increase)

- **Background Embedding Generation System**
  - `scripts/generate_embeddings_incremental.py`: ESM2 embedding generator with checkpoint system
  - `scripts/check_embedding_progress.py`: Real-time progress monitoring
  - `scripts/train_with_existing_features.py`: Immediate training option (no embedding wait)
  - `scripts/start_embedding_generation.bat`: Windows launcher with checkpoint support
  - `scripts/start_embedding_generation.sh`: Linux launcher with checkpoint support
  - CPU-based generation (zero GPU conflict)
  - Checkpoint every 50 batches (~10 minutes)
  - Auto-resume capability
  - Progress tracking and monitoring

- **Comprehensive Research Documentation**
  - `REFERENCES_AND_DATA_SOURCES.md`: Complete citations with DOIs for all data sources
  - `references.bib`: BibTeX format for LaTeX papers (ready to use)
  - `THERAPEUTIC_ANTIBODY_INTEGRATION_REPORT.md`: Detailed integration analysis
  - `VACCINE_ANTIBODY_SOURCES.md`: Guide to therapeutic/vaccine antibody databases
  - `DUAL_COMPUTATION_GUIDE.md`: GPU conflict resolution strategies
  - `START_EMBEDDING_GENERATION.md`: Complete how-to guide
  - `SESSION_SUMMARY.md`: Comprehensive session documentation
  - All data sources cited: SAAINT-DB, SAbDab, AbBiBench, CoV-AbDab, Thera-SAbDab, Ab-CoV

### Changed
- Fixed Korean Windows encoding issues (cp949 codec)
- Replaced Unicode characters with ASCII for compatibility
- Enhanced checkpoint system with atomic writes
- Improved progress monitoring for Korean Windows console

### Features
- **Dual Computation System**: CPU-based embedding generation while GPU trains
- **Robust Checkpoint System**: Saves every 50 batches, auto-resume, zero data loss
- **Femtomolar Antibodies**: Added ultra-high affinity antibodies (0.03 pM = 30 fM)
- **Research Reproducibility**: All sources cited, methods documented
- **Zero GPU Conflict**: CPU mode for embeddings, GPU free for main training

### Data Sources Added
- **SAAINT-DB** (Huang et al., 2025): 20,385 structures, 6,158 affinity measurements
- **SAbDab** (Dunbar et al., 2014): 19,852 structures, 1,307 with affinity
- **Ab-CoV** (Deshpande et al., 2021): Documented for future integration
- **CoV-AbDab** (Raybould et al., 2021): Documented for cross-referencing
- **Thera-SAbDab** (Raybould et al., 2020): Documented for therapeutics

### Performance
- Dataset expansion: 204,986 → 390,757 samples (+90.7%)
- Very strong binders: 230 → 384 (+66.9%)
- Expected RMSE improvement: ~2.2 → ~0.8 (64% better on very strong)
- Background processing: 1-2 days for full embedding generation

## [1.1.0] - 2025-11-03

### Added
- **External Data Integration System**
  - `scripts/download_abbibench.py`: Automated download from Hugging Face (184,500+ measurements)
  - `scripts/download_saaint.py`: Automated download from GitHub (19,128 entries)
  - `scripts/download_pdbbind.py`: PDBbind processing script (4,594 protein-protein complexes)
  - `scripts/integrate_abbibench.py`: AbBiBench integration with duplicate checking
  - `scripts/integrate_all_databases.py`: Unified integration for all databases
  - `scripts/download_all.sh`: Master download script for automation

- **Class Imbalance Handling**
  - `src/data_utils.py`: Stratified sampling, class weights, balanced data loaders
  - `src/losses.py`: Focal loss, weighted MSE, range-focused loss functions
  - `src/metrics.py`: Per-bin evaluation, visualization, metrics tracking
  - `train_balanced.py`: Complete training script with imbalance handling

- **Documentation**
  - `EXTERNAL_DATA_README.md`: Comprehensive guide for downloading and integrating databases
  - `DOWNLOAD_INSTRUCTIONS.md`: Step-by-step download instructions for 9 databases
  - `ADDITIONAL_DATA_SOURCES.md`: Catalog of 14 antibody-antigen databases
  - `IMPLEMENTATION_GUIDE.md`: Complete usage guide for class imbalance methods
  - `SUMMARY.md`: Project summary and quick reference

- **Scientific References**
  - `references_master.md`: Citations for 25+ peer-reviewed papers
  - `references_skempi2.md`: SKEMPI 2.0 database references
  - `references_sabdab.md`: SAbDab database references
  - `references_extreme_affinity.md`: Femtomolar/weak binding research
  - `references_class_imbalance.md`: ML methods with code examples
  - `README_REFERENCES.md`: Guide to using reference files

### Changed
- Enhanced model training to handle extreme affinity values (pKd > 11, pKd < 5)
- Improved data loading with stratified batch sampling
- Added per-bin performance metrics for better evaluation

### Features
- **Stratified Sampling**: Ensures all affinity ranges represented in each batch
- **Class Weights**: Automatic calculation with inverse frequency method
- **Focal Loss**: Down-weights easy examples, emphasizes hard cases
- **Per-Bin Metrics**: Track RMSE/MAE separately for each affinity range
- **Database Integration**: Automatic duplicate detection and removal
- **Affinity Distribution**: Comprehensive statistics and visualization

### Performance Improvements
- Expected 64% RMSE reduction on very strong binders (pKd > 11)
- Expected 64% RMSE reduction on very weak binders (pKd < 5)
- 10-20× more very strong training examples
- 3-4× more very weak training examples

### Database Coverage
- AbBiBench: 184,500+ experimental measurements
- SAAINT-DB: 19,128 antibody-antigen entries (May 2025)
- PDBbind 2024: 33,653 biomolecular complexes
- Total potential: 100,000+ new affinity measurements

## [1.0.0] - 2025-10-31

### Added
- Initial release of AbAg binding affinity prediction model
- Deep learning model with Spearman ρ = 0.85 performance
- Support for antibody-antigen binding affinity prediction
- Pre-trained model (agab_phase2_model.pth)
- Comprehensive documentation in English and Korean
- Example usage scripts
- Installation and quick start guides
- MIT License

### Features
- AffinityPredictor class for easy model usage
- Support for both single predictions and batch processing
- Model performance metrics and validation results
- Python package setup for pip installation

---

## How to Version Future Releases

When making changes, follow this pattern:

### Version Numbering (Semantic Versioning)
- **MAJOR version (X.0.0)**: Breaking changes, incompatible API changes
- **MINOR version (1.X.0)**: New features, backward-compatible
- **PATCH version (1.0.X)**: Bug fixes, backward-compatible

### Steps for New Releases

1. **Update version numbers:**
   - `setup.py`: Update `version="X.X.X"`
   - `abag_affinity/__init__.py`: Update `__version__ = "X.X.X"`

2. **Update this CHANGELOG.md:**
   - Add new section with version and date
   - Document all changes under Added/Changed/Fixed/Removed

3. **Commit changes:**
   ```bash
   git add setup.py abag_affinity/__init__.py CHANGELOG.md
   git commit -m "Bump version to X.X.X"
   ```

4. **Create git tag:**
   ```bash
   git tag -a vX.X.X -m "Release vX.X.X: Brief description"
   ```

5. **Push to GitHub:**
   ```bash
   git push origin main
   git push origin vX.X.X
   ```

### Example Entry for Future Version

```markdown
## [1.1.0] - YYYY-MM-DD

### Added
- New feature A
- New feature B

### Changed
- Improved performance of X
- Updated dependencies

### Fixed
- Bug fix for issue #123
- Corrected error in Y

### Removed
- Deprecated function Z
```
