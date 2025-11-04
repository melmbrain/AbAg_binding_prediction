# Step-by-Step Download Instructions for Additional Affinity Data

**Quick Reference Guide**

---

## 🚀 Quick Start: Download Top 3 Databases

### 1. SAAINT-DB (Recommended First)

**Time:** 5-10 minutes | **Size:** ~500 MB | **Best for:** Most comprehensive recent data

```bash
# Step 1: Go to GitHub
# Open browser: https://github.com/tommyhuangthu/SAAINT

# Step 2: Download repository
git clone https://github.com/tommyhuangthu/SAAINT.git
cd SAAINT

# Step 3: Download database files
# Look for links to Zenodo in README.md
# Or check releases section

# Step 4: Extract affinity data
# The summary file contains all affinity measurements
# Usually named something like: SAAINT_summary.csv or database.csv
```

**What you get:**
- CSV file with ~19,000 antibody-antigen entries
- PDB codes, sequences, affinity values (Kd, pKd)
- Experimental methods and conditions

---

### 2. AbBiBench (Easiest Download)

**Time:** 2 minutes | **Size:** Varies | **Best for:** Ready-to-use ML data

**Option A: Hugging Face Website (No Code)**
```
1. Open: https://huggingface.co/datasets/AbBibench/Antibody_Binding_Benchmark_Dataset
2. Click "Files and versions" tab
3. Download the CSV files you need
4. Save to your data directory
```

**Option B: Python (Recommended)**
```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("AbBibench/Antibody_Binding_Benchmark_Dataset")

# Save to CSV
dataset['train'].to_csv('abbibench_data.csv')

# Or access directly
print(f"Total samples: {len(dataset['train'])}")
print(dataset['train'][0])  # First sample
```

**What you get:**
- 184,500+ measurements
- Antibody sequences (heavy + light chains)
- Antigen sequences
- Affinity values
- Structure information

---

### 3. PDBbind 2020 (Free Version)

**Time:** 10-15 minutes | **Size:** ~10 GB | **Best for:** Gold standard data

```bash
# Step 1: Visit website
# Open: http://www.pdbbind.org.cn/download.php

# Step 2: No registration needed for 2020 version
# Find "PDBbind v2020" section

# Step 3: Download options:
# - General set: Largest collection
# - Refined set: Higher quality
# - Core set: Benchmark set

# Step 4: For antibody-antigen data specifically
# Download: "Protein-protein index"
# File: PP_INDEX_general_set.2020

# Step 5: Extract
tar -xzf [downloaded_file].tar.gz
```

**For 2024 version (requires free registration):**
```
1. Go to: https://www.pdbbind-plus.org.cn/
2. Click "Register" (free for academics)
3. Verify email
4. Login and download
```

**What you get:**
- 4,594 protein-protein complexes
- Kd, Ki, IC50 values
- PDB structures
- Standardized format

---

## 📥 Additional Databases (Step-by-Step)

### 4. AACDB Download

```bash
# Step 1: Open website
# http://i.uestc.edu.cn/AACDB

# Step 2: Use search interface
# - Search for all entries OR
# - Filter by specific antibody/antigen

# Step 3: Export results
# - Click "Export" or "Download" button
# - Choose CSV or TXT format
# - Save file

# Step 4: Multiple downloads if needed
# The database may limit export size
# Download in batches if necessary
```

---

### 5. BindingDB Download

```bash
# Step 1: Go to download page
# https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp

# Step 2: Choose dataset
# Options:
# - Complete database (large)
# - Specific targets
# - Antibodies only (if available)

# Step 3: Download TSV file
# Format: Tab-separated values
# Can be opened in Excel or loaded with pandas

# Step 4: Filter for antibodies
# After download, filter for antibody entries
```

**Python script to filter:**
```python
import pandas as pd

# Load BindingDB
df = pd.read_csv('bindingdb_all.tsv', sep='\t')

# Filter for antibodies
antibody_keywords = ['antibody', 'immunoglobulin', 'IgG', 'Fab', 'scFv']
mask = df['Target Name'].str.contains('|'.join(antibody_keywords),
                                      case=False, na=False)
antibody_data = df[mask]

# Save
antibody_data.to_csv('bindingdb_antibodies.csv', index=False)
print(f"Found {len(antibody_data)} antibody entries")
```

---

### 6. IEDB Download

```bash
# Step 1: Go to IEDB
# https://www.iedb.org/

# Step 2: Use search interface
# Advanced search options:
# - Epitope: Any
# - Host: Human (or any)
# - Assay: B Cell Assays
# - Measurement: Kd, IC50, EC50

# Step 3: Run search

# Step 4: Export results
# - Click "Export" button
# - Choose CSV format
# - Select fields to include:
#   - Epitope
#   - Antibody
#   - Affinity (Kd)
#   - Method
#   - Reference

# Step 5: Download file
```

**API approach (for larger datasets):**
```python
import requests
import pandas as pd

# IEDB API endpoint
url = "http://tools-api.iedb.org/main/bcell_search/"

# Parameters
params = {
    'assay_type': 'binding',
    'output_format': 'json',
    # Add more filters as needed
}

# Request
response = requests.get(url, params=params)
data = response.json()

# Convert to DataFrame
df = pd.DataFrame(data)
df.to_csv('iedb_antibody_affinity.csv', index=False)
```

---

### 7. Ab-CoV Download

```bash
# Note: Ab-CoV data may need to be compiled from paper
# Check supplementary materials

# Step 1: Search for paper
# "Ab-CoV: a curated database for binding affinity"
# Bioinformatics (2022)

# Step 2: Access supplementary materials
# Usually available from journal website

# Step 3: Download Excel/CSV files
# Contains:
# - Antibody sequences
# - IC50, EC50, Kd values
# - SARS-CoV-2, SARS-CoV, MERS-CoV data

# Alternative: Check if online database exists
# Search for "Ab-CoV database online"
```

---

### 8. BioLiP2 Download

```bash
# Step 1: Go to download page
# https://zhanggroup.org/BioLiP/download.html

# Step 2: Choose version
# - Redundant: All interactions
# - Non-redundant: 95% sequence identity cutoff

# Step 3: Download files
# Three sets available:
# 1. Receptor structures
# 2. Ligand structures
# 3. Interaction details (THIS ONE for affinity)

# Step 4: Download interaction file
wget https://zhanggroup.org/BioLiP/download/[filename].tar.bz2

# Step 5: Extract
tar -xjf [filename].tar.bz2
```

---

### 9. CoV-AbDab Download

```bash
# Step 1: Visit website
# http://opig.stats.ox.ac.uk/webapps/coronavirus

# Step 2: Search/Browse
# All entries or filter by:
# - Virus (SARS-CoV-2, SARS-CoV-1, MERS)
# - Antibody type
# - Structure availability

# Step 3: Download
# Click "Download" button for CSV
# Contains sequences and metadata

# Note: For affinity data, use Ab-CoV database instead
```

---

## 🔧 Automated Download Script

Save this as `download_databases.sh`:

```bash
#!/bin/bash

echo "Downloading antibody affinity databases..."
mkdir -p external_data
cd external_data

# 1. SAAINT-DB
echo "Cloning SAAINT-DB..."
git clone https://github.com/tommyhuangthu/SAAINT.git

# 2. PDBbind 2020 (you'll need to manually download from website)
echo "For PDBbind, please visit: http://www.pdbbind.org.cn/download.php"

# 3. Download example datasets from papers
# (URLs would need to be updated with actual links)

echo "Downloads complete! Check external_data/ directory"
```

---

## 🐍 Python Integration Script

Save this as `download_and_integrate.py`:

```python
"""
Download and integrate external antibody affinity databases
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
from datasets import load_dataset

def download_abbibench():
    """Download AbBiBench from HuggingFace"""
    print("Downloading AbBiBench...")
    dataset = load_dataset("AbBibench/Antibody_Binding_Benchmark_Dataset")

    # Convert to pandas DataFrame
    df = pd.DataFrame(dataset['train'])

    # Save
    df.to_csv('external_data/abbibench_raw.csv', index=False)
    print(f"✓ Downloaded {len(df)} samples from AbBiBench")

    return df

def download_saaint_db():
    """Download SAAINT-DB"""
    print("Downloading SAAINT-DB...")
    print("Please clone repository manually:")
    print("  git clone https://github.com/tommyhuangthu/SAAINT.git")
    print("Then run this script again.")

    # Check if already downloaded
    saaint_path = Path('external_data/SAAINT')
    if saaint_path.exists():
        # Look for summary file
        summary_files = list(saaint_path.glob('*summary*.csv'))
        if summary_files:
            df = pd.read_csv(summary_files[0])
            print(f"✓ Loaded {len(df)} samples from SAAINT-DB")
            return df

    return None

def process_abbibench(df):
    """Process AbBiBench data to standard format"""
    print("Processing AbBiBench data...")

    # Extract relevant columns
    processed = pd.DataFrame({
        'pdb_code': df['pdb_code'],
        'antibody_sequence': df['antibody_heavy'] + df['antibody_light'],
        'antigen_sequence': df['antigen'],
        'affinity': df['binding_affinity'],
        'affinity_unit': df['affinity_unit'],
        'source': 'AbBiBench'
    })

    # Convert to pKd if needed
    # (Add conversion logic based on unit)

    return processed

def integrate_with_existing(external_df, existing_path):
    """Integrate external data with existing dataset"""
    print(f"Integrating with existing dataset at {existing_path}...")

    # Load existing
    existing_df = pd.read_csv(existing_path)

    # Check for duplicates
    if 'pdb_code' in existing_df.columns and 'pdb_code' in external_df.columns:
        existing_pdbs = set(existing_df['pdb_code'].str.lower())
        external_pdbs = set(external_df['pdb_code'].str.lower())
        duplicates = existing_pdbs.intersection(external_pdbs)

        print(f"  Found {len(duplicates)} duplicate PDB codes")

        # Remove duplicates
        external_df = external_df[
            ~external_df['pdb_code'].str.lower().isin(duplicates)
        ]
        print(f"  {len(external_df)} new samples after removing duplicates")

    # Merge
    merged_df = pd.concat([existing_df, external_df], ignore_index=True)

    return merged_df

def main():
    """Main download and integration workflow"""
    print("="*80)
    print("DOWNLOADING EXTERNAL ANTIBODY AFFINITY DATA")
    print("="*80)

    # Create output directory
    Path('external_data').mkdir(exist_ok=True)

    # Download datasets
    abbibench_df = download_abbibench()
    saaint_df = download_saaint_db()

    # Process
    if abbibench_df is not None:
        processed_abbibench = process_abbibench(abbibench_df)
        processed_abbibench.to_csv('external_data/abbibench_processed.csv',
                                   index=False)

    print("\n✓ Downloads complete!")
    print("Next steps:")
    print("1. Generate ESM2 embeddings for new sequences")
    print("2. Apply PCA transformation")
    print("3. Use integrate_external_data.py to merge with existing dataset")

if __name__ == "__main__":
    main()
```

Run with:
```bash
python download_and_integrate.py
```

---

## ✅ Download Checklist

Use this checklist to track your downloads:

### Priority Downloads:
- [ ] SAAINT-DB (GitHub)
- [ ] AbBiBench (Hugging Face)
- [ ] PDBbind 2020 (or register for 2024)

### Additional Downloads:
- [ ] AACDB (if needed)
- [ ] BindingDB (if needed)
- [ ] IEDB (for weak binders)
- [ ] Ab-CoV (for COVID antibodies)
- [ ] BioLiP2 (if needed)

### Post-Download:
- [ ] Check file integrity
- [ ] Verify affinity data format
- [ ] Count total samples
- [ ] Check for antibody-antigen specific entries
- [ ] Identify PDB code overlaps

---

## 🚨 Common Issues and Solutions

### Issue 1: Large file size
**Solution:** Download in parts or use wget/curl with resume capability
```bash
wget -c [URL]  # -c flag resumes interrupted downloads
```

### Issue 2: Registration required
**Solution:** Most registrations are instant and free for academics
- Use institutional email
- Fill out simple form
- Check email for confirmation

### Issue 3: File format unknown
**Solution:** Check first few lines
```bash
head -20 downloaded_file.txt
```
Most are CSV, TSV, or structured text

### Issue 4: No direct download link
**Solution:**
1. Check paper supplementary materials
2. Contact authors via email
3. Check Zenodo/GitHub for datasets
4. Search "[database_name] download" on Google Scholar

### Issue 5: API limits
**Solution:** Add delays between requests
```python
import time
time.sleep(1)  # Wait 1 second between requests
```

---

## 📞 Need Help?

### Can't find download link?
1. Check database documentation/FAQ
2. Look for "Downloads" or "Access" in menu
3. Search Google: "[database name] download"

### File won't open?
1. Check file extension
2. Try different tools (Excel, pandas, text editor)
3. Look for README or documentation

### Data format unclear?
1. Check database schema/data dictionary
2. Read associated paper
3. Contact database maintainers

---

## 🎉 After Downloading

You should now have:
- [ ] 3-5 external datasets downloaded
- [ ] Files saved in `external_data/` directory
- [ ] Total of 50,000-100,000 new affinity measurements
- [ ] Mix of extreme and moderate affinity values

**Next step:** Proceed to data integration using provided scripts!

---

*Guide created: 2025-11-03*
*Total databases covered: 9*
*Estimated total download time: 1-2 hours*
*Expected data size: 1-5 GB*

