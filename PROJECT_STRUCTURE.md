# Project Structure

## Directory Layout

```
AbAg_binding_prediction/
├── README.md                          # Main project overview
├── README_COMPLETE.md                 # Complete documentation
├── CHANGELOG.md                       # Version history
├── references.bib                     # BibTeX citations
├── requirements.txt                   # Python dependencies
│
├── docs/                              # All documentation
│   ├── guides/                        # User guides and tutorials
│   ├── reports/                       # Analysis and session reports
│   ├── references/                    # Research citations and sources
│   └── archived/                      # Old/redundant documentation
│
├── scripts/                           # Python scripts
│   ├── generate_embeddings_incremental.py  # ESM2 embedding generation
│   ├── check_embedding_progress.py         # Progress monitoring
│   ├── download_*.py                       # Database downloaders
│   ├── integrate_*.py                      # Data integration scripts
│   └── train_*.py                          # Training scripts
│
├── external_data/                     # Data files
│   ├── merged_with_therapeutics.csv   # MASTER dataset (390,757 samples)
│   ├── train_ready_with_features.csv  # Subset with ESM2 features
│   ├── embedding_checkpoint.pkl       # Embedding generation checkpoint
│   ├── SAAINT/                        # SAAINT-DB source (read-only)
│   └── therapeutic/                   # Therapeutic antibody data
│
├── src/                               # Source code modules
│   ├── data_utils.py                  # Data loading utilities
│   ├── losses.py                      # Loss functions
│   └── metrics.py                     # Evaluation metrics
│
├── abag_affinity/                     # Main package
│   ├── __init__.py
│   └── predictor.py                   # AffinityPredictor class
│
└── archive/                           # Archived old files
    └── cleanup_YYYYMMDD_HHMMSS/       # Timestamped cleanup sessions

```

## Key Files

### Data Files (Keep Only These)
- `external_data/merged_with_therapeutics.csv` - Master dataset (390,757 samples)
- `external_data/train_ready_with_features.csv` - Ready for immediate training

### Documentation
- `README_COMPLETE.md` - Start here for complete overview
- `docs/guides/` - How-to guides and tutorials
- `docs/reports/` - Analysis reports and session summaries
- `docs/references/` - All research citations

### Scripts
- `scripts/check_embedding_progress.py` - Monitor embedding generation
- `scripts/train_balanced.py` - Train with class imbalance handling

## Removed Files

The following redundant files have been archived:

### Data Files (Archived)
- `abbibench_raw.csv`, `abbibench_processed.csv` - Intermediate files
- `merged_with_abbibench.csv` - Old version (superseded)
- `extreme_affinity_data/` - Analysis files (data now in main dataset)

### Documentation (Consolidated)
- Multiple README variants → Single `README_COMPLETE.md`
- Multiple summary files → `docs/reports/SESSION_SUMMARY.md`
- Reference files → `docs/references/`

### Space Saved
- Data cleanup: ~650 MB
- Documentation consolidation: Improved organization

## Version Control

This cleanup was performed on: 20251103_123344

All archived files are preserved in: `archive/cleanup_20251103_123344/`
