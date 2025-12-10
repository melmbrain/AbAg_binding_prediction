#!/usr/bin/env python3
"""
Analyze downloaded data sources for antibody-antigen binding affinity.
Check what data is usable for our prediction task.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("C:/Users/401-24/Desktop/Ab_Ag_dataset/data/new_sources")

print("="*70)
print("ANALYZING NEW DATA SOURCES FOR Ab-Ag BINDING AFFINITY")
print("="*70)
print()

usable_data = []

# =============================================================================
# 1. CoV-AbDab
# =============================================================================
covabdab_file = DATA_DIR / "CoV-AbDab.csv"
if covabdab_file.exists():
    print("1. CoV-AbDab (Coronavirus Antibody Database)")
    print("-"*60)

    df = pd.read_csv(covabdab_file)
    print(f"   Total entries: {len(df):,}")
    print(f"   Columns: {list(df.columns)[:10]}...")

    # Check for affinity data
    affinity_cols = [c for c in df.columns if any(x in c.lower() for x in ['kd', 'affinity', 'ic50', 'binding'])]
    print(f"   Affinity-related columns: {affinity_cols}")

    # Check for sequence data
    seq_cols = [c for c in df.columns if any(x in c.lower() for x in ['seq', 'heavy', 'light', 'vh', 'vl', 'cdr'])]
    print(f"   Sequence columns: {seq_cols[:5]}...")

    # Check how many have VH/VL sequences
    if 'VH' in df.columns:
        has_vh = df['VH'].notna().sum()
        print(f"   Entries with VH sequence: {has_vh:,}")
    if 'VL' in df.columns:
        has_vl = df['VL'].notna().sum()
        print(f"   Entries with VL sequence: {has_vl:,}")

    # The issue: CoV-AbDab doesn't have direct pKd values
    # It has antibody sequences but not quantitative binding affinity
    print()
    print("   VERDICT: CoV-AbDab has sequences but NO quantitative pKd values")
    print("            Need to cross-reference with Ab-CoV for affinity data")
    print()
else:
    print("1. CoV-AbDab - NOT FOUND")
    print(f"   Expected at: {covabdab_file}")
    print()

# =============================================================================
# 2. TDC AntibodyAff
# =============================================================================
tdc_file = DATA_DIR / "tdc_antibody_sabdab.csv"
if tdc_file.exists():
    print("2. TDC AntibodyAff (Therapeutics Data Commons)")
    print("-"*60)

    df = pd.read_csv(tdc_file)
    print(f"   Total entries: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")

    # Check for Y (affinity) column
    if 'Y' in df.columns:
        print(f"   Affinity (Y) range: {df['Y'].min():.2f} - {df['Y'].max():.2f}")
        print(f"   Affinity (Y) mean: {df['Y'].mean():.2f}")

    # Check sequences
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].iloc[0] if len(df) > 0 else None
            if sample and len(str(sample)) > 50:
                print(f"   {col}: sequence-like (len={len(str(sample))})")

    print()
    print("   VERDICT: TDC data is from SAbDab - likely overlaps with our existing data")
    print("            But good to check for any new entries")

    usable_data.append(('TDC', df, 'Y'))
    print()
else:
    print("2. TDC AntibodyAff - NOT FOUND")
    print(f"   Expected at: {tdc_file}")
    print()

# =============================================================================
# 3. Zenodo Antibody Kd
# =============================================================================
zenodo_file = DATA_DIR / "zenodo_antibody_kd.csv"
if zenodo_file.exists():
    print("3. Zenodo Antibody Kd Dataset")
    print("-"*60)

    df = pd.read_csv(zenodo_file)
    print(f"   Total entries: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")

    # Check affinity column
    affinity_cols = [c for c in df.columns if any(x in c.lower() for x in ['kd', 'affinity', 'y'])]
    if affinity_cols:
        for col in affinity_cols:
            if df[col].dtype in [np.float64, np.int64]:
                print(f"   {col} range: {df[col].min():.2f} - {df[col].max():.2f}")

    print()
    print("   VERDICT: From SAbDab via TDC - same source, check for overlap")

    usable_data.append(('Zenodo', df, affinity_cols[0] if affinity_cols else None))
    print()
else:
    print("3. Zenodo Antibody Kd - NOT FOUND")
    print(f"   Expected at: {zenodo_file}")
    print()

# =============================================================================
# Check existing SAbDab data for comparison
# =============================================================================
print("="*70)
print("COMPARISON WITH EXISTING DATA")
print("="*70)
print()

existing_sabdab = Path("C:/Users/401-24/Desktop/Ab_Ag_dataset/data/sabdab/sabdab_sequences_with_affinity.csv")
if existing_sabdab.exists():
    df_existing = pd.read_csv(existing_sabdab)
    print(f"Existing SAbDab data: {len(df_existing):,} samples")
    print(f"  pKd range: {df_existing['pkd'].min():.2f} - {df_existing['pkd'].max():.2f}")

    # Check for potential new entries in TDC
    if usable_data:
        print()
        print("Checking for new entries not in existing data...")

        for source_name, df_new, affinity_col in usable_data:
            # Try to find matching columns
            if 'Antibody' in df_new.columns:
                ab_col = 'Antibody'
            elif 'antibody_sequence' in df_new.columns:
                ab_col = 'antibody_sequence'
            else:
                continue

            # Create identifier
            existing_ab = set(df_existing['heavy_seq'].fillna('') + df_existing['light_seq'].fillna(''))
            new_ab = set(df_new[ab_col].fillna(''))

            new_entries = new_ab - existing_ab
            print(f"  {source_name}: {len(new_entries):,} potentially new antibody sequences")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("="*70)
print("SUMMARY: NEW DATA SOURCES FOR YOUR DATASET")
print("="*70)
print()
print("PROBLEM: Most 'new' sources are derived from SAbDab which you already have!")
print()
print("Actually NEW sources to explore:")
print("  1. Ab-CoV (https://web.iitm.ac.in/bioinfo2/ab-cov/)")
print("     - 568 KD values for coronavirus antibodies")
print("     - Download from Figshare: https://figshare.com/s/6e38ad3c2e130a066d19")
print()
print("  2. AB-Bind (mutational data)")
print("     - 1,101 ΔΔG values across 32 complexes")
print("     - Useful for augmentation, not direct pKd")
print()
print("  3. Updated SAbDab (weekly updates)")
print("     - Your current SAbDab might be outdated")
print("     - Re-download latest from: https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab")
print()
print("RECOMMENDATION:")
print("  - Download fresh SAbDab data (it updates weekly)")
print("  - Get Ab-CoV KD data and convert to pKd")
print("  - These are the only truly NEW affinity data sources found")
