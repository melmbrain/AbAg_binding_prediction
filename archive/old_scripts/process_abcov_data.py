#!/usr/bin/env python3
"""
Process Ab-CoV data and merge with existing dataset.

Steps:
1. Load Ab-CoV files (KD values)
2. Cross-reference with CoV-AbDab for antibody sequences
3. Add SARS-CoV-2 Spike RBD as antigen sequence
4. Convert KD (nM) to pKd
5. Merge with existing v5 dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import re

print("="*70)
print("PROCESSING Ab-CoV DATA")
print("="*70)
print()

# Paths
ABCOV_DIR = Path("C:/Users/401-24/Desktop/Ab_Ag_dataset/data/Ab_CoV")
COVABDAB_FILE = Path("C:/Users/401-24/Desktop/Ab_Ag_dataset/data/new_sources/CoV-AbDab.csv")
V5_DATASET = Path("C:/Users/401-24/Desktop/Ab_Ag_dataset/data/agab/agab_balanced_v5_final.csv")
OUTPUT_DIR = Path("C:/Users/401-24/Desktop/Ab_Ag_dataset/data/agab")

# SARS-CoV-2 Spike RBD sequence (residues 319-541)
# From UniProt P0DTC2
SARS_COV2_RBD = """RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"""

print("Step 1: Loading Ab-CoV files...")
print("-"*50)

# Load all Ab-CoV CSV files
abcov_dfs = []

for kd_range in ['Kd_0-100_nM', 'Kd_100-1000_nM']:
    range_dir = ABCOV_DIR / kd_range
    if range_dir.exists():
        files = list(range_dir.glob("*.csv"))
        print(f"  {kd_range}: {len(files)} files")
        for f in files:
            try:
                df = pd.read_csv(f)
                df['kd_range'] = kd_range
                abcov_dfs.append(df)
            except Exception as e:
                print(f"    Error reading {f.name}: {e}")

if not abcov_dfs:
    print("ERROR: No Ab-CoV files found!")
    exit(1)

abcov = pd.concat(abcov_dfs, ignore_index=True)
print(f"\n  Total Ab-CoV entries: {len(abcov):,}")

# Check columns
print(f"  Columns: {list(abcov.columns)}")

# =============================================================================
# Step 2: Parse KD values
# =============================================================================
print("\nStep 2: Parsing KD values...")
print("-"*50)

# The KD column is "Binding Affinity (KD)"
kd_col = 'Binding Affinity (KD)'
if kd_col not in abcov.columns:
    # Try alternative names
    kd_cols = [c for c in abcov.columns if 'kd' in c.lower() or 'affinity' in c.lower()]
    if kd_cols:
        kd_col = kd_cols[0]
    else:
        print("ERROR: No KD column found!")
        print(f"Available columns: {list(abcov.columns)}")
        exit(1)

print(f"  Using KD column: '{kd_col}'")

# Parse KD values (handle ">500" type entries)
def parse_kd(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if value.startswith('>') or value.startswith('<'):
        return np.nan  # Skip uncertain values
    try:
        return float(value)
    except:
        return np.nan

abcov['KD_nM'] = abcov[kd_col].apply(parse_kd)

# Filter valid KD values
valid_kd = abcov['KD_nM'].notna()
print(f"  Valid KD values: {valid_kd.sum():,} / {len(abcov):,}")

# Convert KD (nM) to pKd
# pKd = -log10(Kd in Molar) = -log10(Kd_nM * 1e-9) = 9 - log10(Kd_nM)
abcov['pKd'] = 9 - np.log10(abcov['KD_nM'])

# Show pKd distribution
print("\n  pKd distribution in Ab-CoV:")
bins = [(4, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 14)]
for low, high in bins:
    count = ((abcov['pKd'] >= low) & (abcov['pKd'] < high)).sum()
    print(f"    [{low}-{high}): {count:,}")

# =============================================================================
# Step 3: Load CoV-AbDab for sequences
# =============================================================================
print("\nStep 3: Loading CoV-AbDab for sequences...")
print("-"*50)

if not COVABDAB_FILE.exists():
    print(f"ERROR: CoV-AbDab file not found at {COVABDAB_FILE}")
    print("Please download from: https://opig.stats.ox.ac.uk/webapps/covabdab/")
    exit(1)

covabdab = pd.read_csv(COVABDAB_FILE)
print(f"  CoV-AbDab entries: {len(covabdab):,}")

# Check sequence columns
seq_cols = [c for c in covabdab.columns if 'vh' in c.lower() or 'vl' in c.lower() or 'seq' in c.lower()]
print(f"  Sequence columns: {seq_cols[:5]}")

# Create lookup by antibody name
# CoV-AbDab uses 'Name' column
name_col = 'Name' if 'Name' in covabdab.columns else covabdab.columns[0]
print(f"  Name column: '{name_col}'")

# Create sequence lookup
vh_col = 'VHorVHH' if 'VHorVHH' in covabdab.columns else 'VH'
vl_col = 'VL' if 'VL' in covabdab.columns else None

print(f"  VH column: '{vh_col}'")
print(f"  VL column: '{vl_col}'")

# Build name to sequence mapping
name_to_seq = {}
for _, row in covabdab.iterrows():
    name = str(row[name_col]).strip()
    vh = str(row[vh_col]) if pd.notna(row[vh_col]) else ''
    vl = str(row[vl_col]) if vl_col and pd.notna(row[vl_col]) else ''

    # Combine VH + VL
    ab_seq = vh + vl
    if len(ab_seq) >= 50:  # Minimum length
        name_to_seq[name] = ab_seq
        # Also try without spaces and lowercase
        name_to_seq[name.replace(' ', '')] = ab_seq
        name_to_seq[name.lower()] = ab_seq
        name_to_seq[name.lower().replace(' ', '')] = ab_seq

print(f"  Antibodies with sequences: {len(name_to_seq):,}")

# =============================================================================
# Step 4: Match Ab-CoV entries with CoV-AbDab sequences
# =============================================================================
print("\nStep 4: Matching antibody names to sequences...")
print("-"*50)

# Ab-CoV uses 'Antibody name' column
ab_name_col = 'Antibody name'

def find_sequence(name):
    """Try to find sequence for antibody name"""
    if pd.isna(name):
        return None
    name = str(name).strip()

    # Try exact match
    if name in name_to_seq:
        return name_to_seq[name]

    # Try without spaces
    if name.replace(' ', '') in name_to_seq:
        return name_to_seq[name.replace(' ', '')]

    # Try lowercase
    if name.lower() in name_to_seq:
        return name_to_seq[name.lower()]

    # Try lowercase without spaces
    if name.lower().replace(' ', '') in name_to_seq:
        return name_to_seq[name.lower().replace(' ', '')]

    return None

abcov['antibody_sequence'] = abcov[ab_name_col].apply(find_sequence)

matched = abcov['antibody_sequence'].notna()
print(f"  Matched: {matched.sum():,} / {len(abcov):,} ({100*matched.sum()/len(abcov):.1f}%)")

# Filter to only matched entries with valid pKd
usable = abcov[matched & abcov['pKd'].notna()].copy()
print(f"  Usable entries (with sequence + pKd): {len(usable):,}")

if len(usable) == 0:
    print("\nWARNING: No usable entries found!")
    print("Antibody names in Ab-CoV don't match CoV-AbDab names.")
    print("\nSample Ab-CoV names:")
    print(abcov[ab_name_col].head(20).tolist())
    print("\nSample CoV-AbDab names:")
    print(list(name_to_seq.keys())[:20])

    # Try fuzzy matching
    print("\nAttempting partial name matching...")

    def fuzzy_find_sequence(name):
        if pd.isna(name):
            return None
        name = str(name).strip().upper()
        for cov_name, seq in name_to_seq.items():
            if name in cov_name.upper() or cov_name.upper() in name:
                return seq
        return None

    abcov['antibody_sequence'] = abcov[ab_name_col].apply(fuzzy_find_sequence)
    matched = abcov['antibody_sequence'].notna()
    print(f"  Fuzzy matched: {matched.sum():,}")

    usable = abcov[matched & abcov['pKd'].notna()].copy()
    print(f"  Usable after fuzzy matching: {len(usable):,}")

# =============================================================================
# Step 5: Add antigen sequence (SARS-CoV-2 RBD)
# =============================================================================
print("\nStep 5: Adding antigen sequence...")
print("-"*50)

usable['antigen_sequence'] = SARS_COV2_RBD
usable['source'] = 'Ab-CoV'

# Keep only needed columns
usable = usable[['antibody_sequence', 'antigen_sequence', 'pKd', 'source']].copy()

print(f"  Entries ready for merging: {len(usable):,}")

# =============================================================================
# Step 6: Check pKd distribution of new data
# =============================================================================
print("\nStep 6: New data pKd distribution:")
print("-"*50)

for low, high in bins:
    count = ((usable['pKd'] >= low) & (usable['pKd'] < high)).sum()
    if count > 0:
        print(f"  [{low}-{high}): {count:,}")

# =============================================================================
# Step 7: Load existing v5 dataset and merge
# =============================================================================
print("\nStep 7: Merging with existing dataset...")
print("-"*50)

if V5_DATASET.exists():
    v5 = pd.read_csv(V5_DATASET)
    print(f"  Existing v5 samples: {len(v5):,}")

    # Check for duplicates
    v5['pair_key'] = v5['antibody_sequence'] + '|||' + v5['antigen_sequence']
    usable['pair_key'] = usable['antibody_sequence'] + '|||' + usable['antigen_sequence']

    existing_pairs = set(v5['pair_key'])
    new_pairs = ~usable['pair_key'].isin(existing_pairs)

    new_data = usable[new_pairs].drop(columns=['pair_key'])
    print(f"  New unique pairs: {len(new_data):,}")

    if len(new_data) > 0:
        # Combine
        v5 = v5.drop(columns=['pair_key'])
        combined = pd.concat([v5, new_data], ignore_index=True)

        print(f"\n  Combined dataset: {len(combined):,} samples")

        # New distribution
        print("\n  Updated pKd distribution:")
        for low, high in bins:
            count = ((combined['pKd'] >= low) & (combined['pKd'] < high)).sum()
            pct = 100 * count / len(combined)
            print(f"    [{low}-{high}): {count:>6,} ({pct:>5.1f}%)")

        # Save
        output_file = OUTPUT_DIR / "agab_balanced_v6_with_abcov.csv"
        combined.to_csv(output_file, index=False)
        print(f"\n  Saved to: {output_file}")
        print(f"  File size: {output_file.stat().st_size / (1024*1024):.1f} MB")
    else:
        print("  No new unique pairs to add.")
else:
    print(f"  V5 dataset not found at {V5_DATASET}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
Ab-CoV Data Processing Results:
- Total Ab-CoV entries: {len(abcov):,}
- Entries with valid KD: {(abcov['KD_nM'].notna()).sum():,}
- Matched with sequences: {matched.sum():,}
- Usable for training: {len(usable):,}

Note: Ab-CoV data is for SARS-CoV-2 antibodies only.
This adds diversity in the pKd [6-8) range but limited antigen diversity.
""")
