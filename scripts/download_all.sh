#!/bin/bash

# Master script to download all priority databases
# This script automates the download of AbBiBench, SAAINT-DB, and PDBbind

set -e  # Exit on error

echo "================================================================================"
echo "DOWNLOADING ALL PRIORITY DATABASES"
echo "================================================================================"
echo ""
echo "This script will download:"
echo "  1. AbBiBench (from Hugging Face)"
echo "  2. SAAINT-DB (from GitHub)"
echo "  3. PDBbind (requires manual download - instructions will be shown)"
echo ""
echo "================================================================================"
echo ""

# Create external_data directory
mkdir -p external_data

# 1. Download AbBiBench
echo ""
echo "================================================================================"
echo "1. DOWNLOADING ABBIBENCH"
echo "================================================================================"
echo ""

if [ -f "external_data/abbibench_raw.csv" ]; then
    echo "✓ AbBiBench already downloaded (abbibench_raw.csv exists)"
    echo "  Skipping download..."
else
    echo "Running: python3 scripts/download_abbibench.py"
    python3 scripts/download_abbibench.py
fi

# 2. Download SAAINT-DB
echo ""
echo "================================================================================"
echo "2. DOWNLOADING SAAINT-DB"
echo "================================================================================"
echo ""

if [ -f "external_data/saaint_raw.csv" ]; then
    echo "✓ SAAINT-DB already downloaded (saaint_raw.csv exists)"
    echo "  Skipping download..."
else
    echo "Running: python3 scripts/download_saaint.py"
    python3 scripts/download_saaint.py
fi

# 3. PDBbind (manual download required)
echo ""
echo "================================================================================"
echo "3. PDBBIND (MANUAL DOWNLOAD REQUIRED)"
echo "================================================================================"
echo ""

if [ -f "external_data/pdbbind_raw.csv" ]; then
    echo "✓ PDBbind already downloaded (pdbbind_raw.csv exists)"
    echo "  Skipping download..."
else
    echo "⚠ PDBbind requires manual download from website"
    echo ""
    echo "Please follow these steps:"
    echo "  1. Visit: http://www.pdbbind.org.cn/download.php"
    echo "  2. Download: PP_INDEX_general_set.2020"
    echo "  3. Save to: external_data/"
    echo "  4. Run: python3 scripts/download_pdbbind.py"
    echo ""
    echo "Or register for 2024 version:"
    echo "  1. Visit: https://www.pdbbind-plus.org.cn/"
    echo "  2. Register (free for academics)"
    echo "  3. Download protein-protein index"
    echo "  4. Save to: external_data/"
    echo "  5. Run: python3 scripts/download_pdbbind.py"
    echo ""
    echo "You can continue with AbBiBench and SAAINT-DB for now."
fi

# Summary
echo ""
echo "================================================================================"
echo "DOWNLOAD SUMMARY"
echo "================================================================================"
echo ""

ls -lh external_data/*.csv 2>/dev/null || echo "No CSV files found yet"

echo ""
echo "Next steps:"
echo "  1. Download PDBbind (if not done)"
echo "  2. Review downloaded data files"
echo "  3. Run integration:"
echo "     python3 scripts/integrate_all_databases.py --existing YOUR_EXISTING_DATA.csv"
echo ""
echo "================================================================================"
