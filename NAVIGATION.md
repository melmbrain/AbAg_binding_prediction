# Project Navigation Guide

**Version:** 1.2.0
**Last Updated:** 2025-11-03
**Status:** ✅ Clean and Organized (686 MB saved)

---

## Start Here

| **If you want to...** | **Go to...** |
|-----------------------|--------------|
| Use the API for predictions | [README.md](README.md) |
| Understand the complete project | [README_COMPLETE.md](README_COMPLETE.md) |
| See project organization | [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) |
| Check version history | [CHANGELOG.md](CHANGELOG.md) |
| Cite in research paper | [references.bib](references.bib) |

---

## Quick Navigation Matrix

### By Task

| Task | Document | Location |
|------|----------|----------|
| **Get Started Quickly** | Quick Start | [docs/guides/QUICK_START.md](docs/guides/QUICK_START.md) |
| **Monitor Embeddings** | Embedding Guide | [docs/guides/START_EMBEDDING_GENERATION.md](docs/guides/START_EMBEDDING_GENERATION.md) |
| **Handle Class Imbalance** | Implementation Guide | [docs/guides/IMPLEMENTATION_GUIDE.md](docs/guides/IMPLEMENTATION_GUIDE.md) |
| **Download More Data** | Download Instructions | [docs/guides/DOWNLOAD_INSTRUCTIONS.md](docs/guides/DOWNLOAD_INSTRUCTIONS.md) |
| **Resolve GPU Conflicts** | Dual Computation Guide | [docs/guides/DUAL_COMPUTATION_GUIDE.md](docs/guides/DUAL_COMPUTATION_GUIDE.md) |
| **Understand Data Sources** | Data Sources | [docs/references/REFERENCES_AND_DATA_SOURCES.md](docs/references/REFERENCES_AND_DATA_SOURCES.md) |
| **Find Citations** | All References | [docs/references/references_master.md](docs/references/references_master.md) |
| **Review Session Work** | Session Summary | [docs/reports/SESSION_SUMMARY.md](docs/reports/SESSION_SUMMARY.md) |

### By Role

#### Researcher (Writing Paper)
1. [references.bib](references.bib) - BibTeX citations
2. [docs/references/REFERENCES_AND_DATA_SOURCES.md](docs/references/REFERENCES_AND_DATA_SOURCES.md) - All sources with DOIs
3. [docs/reports/SESSION_SUMMARY.md](docs/reports/SESSION_SUMMARY.md) - Methods and results
4. [CHANGELOG.md](CHANGELOG.md) - What was done

#### Data Scientist (Training Models)
1. [docs/guides/IMPLEMENTATION_GUIDE.md](docs/guides/IMPLEMENTATION_GUIDE.md) - Class imbalance methods
2. [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Data file locations
3. [docs/guides/DUAL_COMPUTATION_GUIDE.md](docs/guides/DUAL_COMPUTATION_GUIDE.md) - Resource management
4. [docs/reports/DISTRIBUTION_SUMMARY.md](docs/reports/DISTRIBUTION_SUMMARY.md) - Data statistics

#### Developer (Using API)
1. [README.md](README.md) - API documentation
2. [requirements.txt](requirements.txt) - Dependencies
3. [docs/guides/QUICK_START.md](docs/guides/QUICK_START.md) - Quick start

---

## Directory Structure

```
AbAg_binding_prediction/
│
├── README.md                          # API Documentation (START HERE for API users)
├── README_COMPLETE.md                 # Complete Project Overview
├── NAVIGATION.md                      # This file - Navigation Guide
├── PROJECT_STRUCTURE.md               # Directory Layout
├── CHANGELOG.md                       # Version History
├── references.bib                     # BibTeX Citations
├── requirements.txt                   # Dependencies
│
├── docs/                              # All Documentation (Organized)
│   ├── guides/                        # User Guides (8 files)
│   │   ├── QUICK_START.md
│   │   ├── IMPLEMENTATION_GUIDE.md
│   │   ├── DOWNLOAD_INSTRUCTIONS.md
│   │   ├── START_EMBEDDING_GENERATION.md
│   │   ├── DUAL_COMPUTATION_GUIDE.md
│   │   └── ...
│   │
│   ├── reports/                       # Analysis Reports (5 files)
│   │   ├── SESSION_SUMMARY.md
│   │   ├── THERAPEUTIC_ANTIBODY_INTEGRATION_REPORT.md
│   │   ├── EXTREME_AFFINITY_REPORT.md
│   │   └── ...
│   │
│   ├── references/                    # Research Citations (9 files)
│   │   ├── REFERENCES_AND_DATA_SOURCES.md
│   │   ├── references_master.md
│   │   ├── references_class_imbalance.md
│   │   └── ...
│   │
│   └── archived/                      # Old Documentation (14 files)
│       └── (Preserved for history)
│
├── external_data/                     # Data Files
│   ├── merged_with_therapeutics.csv   # MASTER Dataset (390,757 samples)
│   ├── train_ready_with_features.csv  # Ready for Training (204,986 samples)
│   ├── embedding_checkpoint.pkl       # Active Checkpoint
│   ├── SAAINT/                        # Source Database (read-only)
│   └── therapeutic/                   # Therapeutic Antibody Data
│
├── scripts/                           # Python Scripts
│   ├── check_embedding_progress.py    # Monitor Progress
│   ├── generate_embeddings_incremental.py  # ESM2 Generation (RUNNING)
│   ├── download_*.py                  # Database Downloaders
│   └── integrate_*.py                 # Integration Scripts
│
├── src/                               # Source Code
│   ├── data_utils.py                  # Data Loading
│   ├── losses.py                      # Loss Functions
│   └── metrics.py                     # Evaluation Metrics
│
├── abag_affinity/                     # Main Package
│   └── predictor.py                   # AffinityPredictor Class
│
└── archive/                           # Safety Backup
    └── cleanup_20251103_123344/       # Old Files (686 MB archived)
```

---

## Documentation Categories

### 📚 Guides (docs/guides/)

Learn how to use the system:

| Guide | Purpose | When to Read |
|-------|---------|--------------|
| [QUICK_START.md](docs/guides/QUICK_START.md) | Get started fast | First time using project |
| [QUICK_START_GUIDE.md](docs/guides/QUICK_START_GUIDE.md) | Extended quick start | Need more details |
| [IMPLEMENTATION_GUIDE.md](docs/guides/IMPLEMENTATION_GUIDE.md) | Class imbalance methods | Training with imbalanced data |
| [DOWNLOAD_INSTRUCTIONS.md](docs/guides/DOWNLOAD_INSTRUCTIONS.md) | Download databases | Need more data |
| [EXTERNAL_DATA_README.md](docs/guides/EXTERNAL_DATA_README.md) | External data integration | Integrating new sources |
| [START_EMBEDDING_GENERATION.md](docs/guides/START_EMBEDDING_GENERATION.md) | Generate embeddings | Creating features |
| [DUAL_COMPUTATION_GUIDE.md](docs/guides/DUAL_COMPUTATION_GUIDE.md) | GPU conflict solutions | Running multiple processes |
| [DOCUMENTATION_INDEX.md](docs/guides/DOCUMENTATION_INDEX.md) | Documentation map | Finding specific docs |

### 📊 Reports (docs/reports/)

Analysis and session summaries:

| Report | Content | Use For |
|--------|---------|---------|
| [SESSION_SUMMARY.md](docs/reports/SESSION_SUMMARY.md) | Latest session work | Understanding what was done |
| [THERAPEUTIC_ANTIBODY_INTEGRATION_REPORT.md](docs/reports/THERAPEUTIC_ANTIBODY_INTEGRATION_REPORT.md) | Therapeutic Ab integration | Dataset enhancement details |
| [EXTREME_AFFINITY_REPORT.md](docs/reports/EXTREME_AFFINITY_REPORT.md) | Femtomolar antibodies | Extreme affinity analysis |
| [EXTREME_AFFINITY_ANALYSIS_REPORT.md](docs/reports/EXTREME_AFFINITY_ANALYSIS_REPORT.md) | Detailed affinity analysis | Research insights |
| [DISTRIBUTION_SUMMARY.md](docs/reports/DISTRIBUTION_SUMMARY.md) | Affinity distribution stats | Dataset statistics |

### 📖 References (docs/references/)

Research citations and sources:

| Reference | Content | Use For |
|-----------|---------|---------|
| [REFERENCES_AND_DATA_SOURCES.md](docs/references/REFERENCES_AND_DATA_SOURCES.md) | Complete citations with DOIs | Paper writing (PRIMARY) |
| [references_master.md](docs/references/references_master.md) | All paper citations | Literature review |
| [references_class_imbalance.md](docs/references/references_class_imbalance.md) | ML methods | Implementation details |
| [references_extreme_affinity.md](docs/references/references_extreme_affinity.md) | Affinity research | Domain knowledge |
| [references_sabdab.md](docs/references/references_sabdab.md) | SAbDab database | Data source citation |
| [references_skempi2.md](docs/references/references_skempi2.md) | SKEMPI 2.0 | Mutation data |
| [ADDITIONAL_DATA_SOURCES.md](docs/references/ADDITIONAL_DATA_SOURCES.md) | 14+ databases | Finding more data |
| [VACCINE_ANTIBODY_SOURCES.md](docs/references/VACCINE_ANTIBODY_SOURCES.md) | Therapeutic antibodies | High-affinity sources |
| [README_REFERENCES.md](docs/references/README_REFERENCES.md) | Reference guide | Using citations |

---

## Data Files

### Active Data (Keep These)

| File | Size | Samples | Description |
|------|------|---------|-------------|
| `merged_with_therapeutics.csv` | 500 MB | 390,757 | **MASTER** - All data integrated |
| `train_ready_with_features.csv` | 421 MB | 204,986 | Subset with ESM2 features (ready to train) |
| `embedding_checkpoint.pkl` | 20 MB | Active | Embedding generation progress |

### Source Data (Read-Only)

| Directory | Size | Content |
|-----------|------|---------|
| `external_data/SAAINT/` | 614 MB | SAAINT-DB source files |
| `external_data/therapeutic/` | 8.8 MB | Therapeutic antibody data |

### Archived Data (Preserved)

| Location | Size | Content |
|----------|------|---------|
| `archive/cleanup_20251103_123344/` | 644 MB | Old/redundant files (safe backup) |

---

## Current Status

### Embedding Generation (Active)

```bash
# Check progress
python.exe scripts/check_embedding_progress.py

# Current: ~2.15% complete (4,000 / 185,771 samples)
# Estimated: ~2 days remaining
# PID: 12835 (CPU mode, zero GPU conflict)
```

### Dataset Summary

| Metric | Value |
|--------|-------|
| Total Samples | 390,757 |
| Very Strong Binders (pKd > 11) | 384 |
| Best Affinity | 0.03 pM (PDB: 7rew) |
| Data Sources | 4 (AbBiBench, SAAINT-DB, SAbDab, Phase 6) |

---

## Common Tasks

### Check Embedding Progress

```bash
python.exe scripts/check_embedding_progress.py
```

### Train Model (Immediate - Existing Features)

```bash
python train_balanced.py \
  --data external_data/train_ready_with_features.csv \
  --loss weighted_mse \
  --sampling stratified
```

### Train Model (After Embeddings Complete)

```bash
python train_balanced.py \
  --data external_data/merged_with_all_features.csv \
  --loss weighted_mse \
  --sampling stratified
```

### Find Specific Information

| Looking for... | Check file... |
|----------------|---------------|
| How to cite data sources | [docs/references/REFERENCES_AND_DATA_SOURCES.md](docs/references/REFERENCES_AND_DATA_SOURCES.md) |
| BibTeX citations | [references.bib](references.bib) |
| What was done this session | [docs/reports/SESSION_SUMMARY.md](docs/reports/SESSION_SUMMARY.md) |
| Dataset statistics | [docs/reports/DISTRIBUTION_SUMMARY.md](docs/reports/DISTRIBUTION_SUMMARY.md) |
| Class imbalance methods | [docs/guides/IMPLEMENTATION_GUIDE.md](docs/guides/IMPLEMENTATION_GUIDE.md) |
| GPU usage strategies | [docs/guides/DUAL_COMPUTATION_GUIDE.md](docs/guides/DUAL_COMPUTATION_GUIDE.md) |

---

## For Research Paper

### Quick Citations (LaTeX)

All citations ready in [references.bib](references.bib):

```latex
\cite{abbibench2024}     % AbBiBench dataset
\cite{saaintdb2025}      % SAAINT-DB
\cite{sabdab2014}        % SAbDab
\cite{esm2_2023}         % ESM2 embeddings
\cite{focal_loss_2017}   % Focal loss
\cite{pytorch_2019}      % PyTorch
```

### Dataset Statistics

See [docs/reports/SESSION_SUMMARY.md](docs/reports/SESSION_SUMMARY.md) for complete statistics.

---

## Cleanup History

**Latest:** 2025-11-03 12:33:44

### Removed
- 686 MB redundant files
- 35 duplicate documentation files
- Old intermediate data files

### Preserved
- All files archived in `archive/cleanup_20251103_123344/`
- Zero data loss
- Complete history maintained

---

## Version

**Current:** v1.2.0 (2025-11-03)

See [CHANGELOG.md](CHANGELOG.md) for complete history.

---

**This navigation guide helps you find exactly what you need, when you need it.**
