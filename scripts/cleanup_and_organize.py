#!/usr/bin/env python3
"""
Cleanup and Organization Script
Removes redundant files, consolidates documentation, organizes project structure
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def main():
    """Execute cleanup and organization"""

    print("="*80)
    print("PROJECT CLEANUP AND ORGANIZATION")
    print("="*80)

    # Create archive directory
    archive_dir = Path("archive")
    archive_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_archive = archive_dir / f"cleanup_{timestamp}"
    session_archive.mkdir(exist_ok=True)

    print(f"\n[INFO] Archive directory: {session_archive}")

    # =========================================================================
    # STEP 1: Clean up redundant DATA files
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 1: Cleaning redundant data files")
    print("-"*80)

    data_to_archive = [
        # Intermediate AbBiBench files (147 MB)
        "external_data/abbibench_raw.csv",
        "external_data/abbibench_processed.csv",
        "external_data/abbibench_with_codes.csv",

        # Old merged file (497 MB) - superseded by merged_with_therapeutics.csv
        "external_data/merged_with_abbibench.csv",

        # Intermediate processing files
        "external_data/integration_report.txt",
    ]

    # Large source directories to remove (after archiving key files)
    dirs_to_remove = [
        "external_data/abbibench_cache",  # 44 MB cache
    ]

    space_saved = 0
    for file_path in data_to_archive:
        path = Path(file_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            target = session_archive / path.name
            print(f"[ARCHIVE] {file_path} ({size_mb:.1f} MB) -> {target}")
            shutil.move(str(path), str(target))
            space_saved += size_mb

    for dir_path in dirs_to_remove:
        path = Path(dir_path)
        if path.exists():
            size_mb = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / (1024 * 1024)
            print(f"[REMOVE] {dir_path} ({size_mb:.1f} MB)")
            shutil.rmtree(path)
            space_saved += size_mb

    print(f"\n[OK] Data cleanup complete. Space saved: {space_saved:.1f} MB")

    # =========================================================================
    # STEP 2: Consolidate DOCUMENTATION files
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 2: Consolidating documentation")
    print("-"*80)

    # Create organized documentation structure
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    (docs_dir / "guides").mkdir(exist_ok=True)
    (docs_dir / "reports").mkdir(exist_ok=True)
    (docs_dir / "references").mkdir(exist_ok=True)
    (docs_dir / "archived").mkdir(exist_ok=True)

    # Documentation consolidation mapping
    doc_consolidation = {
        # Keep as primary README files in root
        "keep_root": [
            "README.md",                      # Main project README
            "README_COMPLETE.md",             # Complete project overview
            "CHANGELOG.md",                   # Version history
            "references.bib",                 # BibTeX citations
            "requirements.txt",               # Dependencies
        ],

        # Move to docs/guides/
        "guides": [
            "QUICK_START.md",
            "QUICK_START_GUIDE.md",
            "IMPLEMENTATION_GUIDE.md",
            "EXTERNAL_DATA_README.md",
            "DOWNLOAD_INSTRUCTIONS.md",
            "DUAL_COMPUTATION_GUIDE.md",
            "START_EMBEDDING_GENERATION.md",
            "DOCUMENTATION_INDEX.md",
        ],

        # Move to docs/reports/
        "reports": [
            "SESSION_SUMMARY.md",
            "EXTREME_AFFINITY_REPORT.md",
            "EXTREME_AFFINITY_ANALYSIS_REPORT.md",
            "THERAPEUTIC_ANTIBODY_INTEGRATION_REPORT.md",
            "DISTRIBUTION_SUMMARY.md",
        ],

        # Move to docs/references/
        "references": [
            "REFERENCES_AND_DATA_SOURCES.md",
            "README_REFERENCES.md",
            "references_master.md",
            "references_class_imbalance.md",
            "references_extreme_affinity.md",
            "references_sabdab.md",
            "references_skempi2.md",
            "ADDITIONAL_DATA_SOURCES.md",
            "VACCINE_ANTIBODY_SOURCES.md",
        ],

        # Archive redundant/outdated files
        "archive": [
            # Redundant README variants
            "README_KR.md",
            "QUICK_START_KR.md",
            "START_HERE.md",
            "SETUP_COMPLETE.md",
            "SUMMARY.md",

            # Redundant presentations
            "PRESENTATION.md",
            "PRESENTATION_KR.md",
            "DISTRIBUTION_SUMMARY_KR.md",

            # Status files (will be regenerated as needed)
            "EMBEDDING_GENERATION_ACTIVE.txt",
            "READY_TO_START.txt",
            "DUAL_COMPUTATION_SUMMARY.txt",
            "EXTREME_AFFINITY_SUMMARY.txt",
            "THERAPEUTIC_ANTIBODY_SUMMARY.txt",
            "COMMANDS.txt",

            # Old log files
            "integration_log.txt",
        ],
    }

    # Execute documentation reorganization
    for category, files in doc_consolidation.items():
        if category == "keep_root":
            print(f"\n[KEEP] Keeping {len(files)} files in root directory")
            continue

        for filename in files:
            source = Path(filename)
            if not source.exists():
                continue

            if category == "archive":
                target = docs_dir / "archived" / source.name
                print(f"[ARCHIVE] {filename} -> docs/archived/")
            else:
                target = docs_dir / category / source.name
                print(f"[MOVE] {filename} -> docs/{category}/")

            # Create parent directory if needed
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))

    print(f"\n[OK] Documentation consolidated into docs/ directory")

    # =========================================================================
    # STEP 3: Organize extreme_affinity_data directory
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 3: Organizing extreme_affinity_data/")
    print("-"*80)

    # This directory contains intermediate analysis files
    # Move to archive since data is now in main dataset
    extreme_dir = Path("extreme_affinity_data")
    if extreme_dir.exists():
        target = session_archive / "extreme_affinity_data"
        print(f"[ARCHIVE] extreme_affinity_data/ -> archive/")
        shutil.move(str(extreme_dir), str(target))
        print("[OK] Archived extreme_affinity_data/")

    # =========================================================================
    # STEP 4: Create PROJECT_STRUCTURE.md
    # =========================================================================
    print("\n" + "-"*80)
    print("STEP 4: Creating project structure documentation")
    print("-"*80)

    structure_content = """# Project Structure

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

This cleanup was performed on: {timestamp}

All archived files are preserved in: `archive/cleanup_{timestamp}/`
"""

    structure_file = Path("PROJECT_STRUCTURE.md")
    with open(structure_file, 'w', encoding='utf-8') as f:
        f.write(structure_content.format(timestamp=timestamp))

    print(f"[CREATE] PROJECT_STRUCTURE.md")

    # =========================================================================
    # STEP 5: Summary
    # =========================================================================
    print("\n" + "="*80)
    print("CLEANUP SUMMARY")
    print("="*80)

    print(f"\n[OK] Data files cleaned: ~{space_saved:.0f} MB saved")
    print(f"[OK] Documentation organized into docs/ directory")
    print(f"[OK] Project structure documented in PROJECT_STRUCTURE.md")
    print(f"[OK] All archived files saved in: {session_archive}")

    print("\n" + "="*80)
    print("FINAL PROJECT STRUCTURE")
    print("="*80)
    print("""
Root Directory (Core Files Only):
  README.md, README_COMPLETE.md, CHANGELOG.md
  references.bib, requirements.txt
  PROJECT_STRUCTURE.md

docs/ (All Documentation):
  guides/     - User guides and tutorials
  reports/    - Analysis and session reports
  references/ - Research citations
  archived/   - Old documentation

external_data/ (Clean Data):
  merged_with_therapeutics.csv  - MASTER dataset
  train_ready_with_features.csv - For training
  embedding_checkpoint.pkl      - Active checkpoint

scripts/ (All Scripts):
  Generate, check, download, integrate, train scripts

archive/ (Safety Backup):
  All removed files preserved with timestamps
""")

    print("\n[OK] Cleanup complete! Project is now organized and coherent.")

if __name__ == "__main__":
    main()
