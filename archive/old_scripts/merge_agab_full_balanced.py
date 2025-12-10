#!/usr/bin/env python3
"""
Merge AgAb Full + SAbDab for Balanced, Scaled Dataset

This script creates a balanced, scaled dataset from:
1. AgAb Full (1.2M) filtered to pKd [4.0, 14.0] → 187k samples
2. SAbDab (954) for high-affinity boost

Expected output: ~188k balanced samples with good weak binder coverage
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("CREATING BALANCED SCALED DATASET")
print("="*80)
print()

# Paths
agab_full_path = Path("C:/Users/401-24/Desktop/Ab_Ag_dataset/data/agab/agab_full_dataset.csv")
agab_current_path = Path("C:/Users/401-24/Desktop/Ab_Ag_dataset/data/agab/agab_phase2_full.csv")
sabdab_path = Path("C:/Users/401-24/Desktop/Ab_Ag_dataset/data/sabdab/sabdab_sequences_with_affinity.csv")
output_path = Path("C:/Users/401-24/Desktop/Ab_Ag_dataset/data/agab/agab_phase2_full_v2_balanced.csv")

# =============================================================================
# Step 1: Load and Filter AgAb Full
# =============================================================================
print("Step 1: Loading AgAb Full dataset...")
df_full = pd.read_csv(agab_full_path)
print(f"  Loaded: {len(df_full):,} samples")
print(f"  Columns: {list(df_full.columns)}")
print()

# Filter to valid pKd range [4.0, 14.0]
print("Step 2: Filtering to pKd [4.0, 14.0]...")

# Convert affinity to numeric, coerce errors to NaN
df_full['affinity'] = pd.to_numeric(df_full['affinity'], errors='coerce')

# Remove rows with NaN affinity
df_full = df_full.dropna(subset=['affinity'])
print(f"  After removing non-numeric affinity: {len(df_full):,} samples")

print(f"  Original pKd range: [{df_full['affinity'].min():.2f}, {df_full['affinity'].max():.2f}]")

df_full_filtered = df_full[
    (df_full['affinity'] >= 4.0) &
    (df_full['affinity'] <= 14.0)
].copy()

print(f"  Filtered: {len(df_full_filtered):,} samples ({100*len(df_full_filtered)/len(df_full):.1f}%)")
print(f"  Removed: {len(df_full) - len(df_full_filtered):,} samples")
print()

# Rename 'affinity' to 'pKd' for consistency
df_full_filtered = df_full_filtered.rename(columns={'affinity': 'pKd'})

# Create antibody_sequence by combining heavy + light chains (if separate)
if 'antibody_sequence' not in df_full_filtered.columns:
    if 'heavy_sequence' in df_full_filtered.columns and 'light_sequence' in df_full_filtered.columns:
        print("  Creating antibody_sequence from heavy + light chains...")
        df_full_filtered['antibody_sequence'] = df_full_filtered['heavy_sequence'].fillna('') + df_full_filtered['light_sequence'].fillna('')
    elif 'scfv' in df_full_filtered.columns:
        print("  Using scfv as antibody_sequence...")
        df_full_filtered['antibody_sequence'] = df_full_filtered['scfv']
    else:
        print("  WARNING: No antibody sequence column found!")

# Remove rows with empty antibody or antigen sequences
if 'antibody_sequence' in df_full_filtered.columns and 'antigen_sequence' in df_full_filtered.columns:
    df_full_filtered = df_full_filtered[
        (df_full_filtered['antibody_sequence'].notna()) &
        (df_full_filtered['antibody_sequence'] != '') &
        (df_full_filtered['antigen_sequence'].notna()) &
        (df_full_filtered['antigen_sequence'] != '')
    ]
    print(f"  After removing empty sequences: {len(df_full_filtered):,} samples")

# Distribution analysis
print("  Distribution after filtering:")
bins = [4, 6, 8, 10, 12, 14]
hist, _ = np.histogram(df_full_filtered['pKd'], bins=bins)
print(f"    [4-6]:   {hist[0]:>6,} samples ({100*hist[0]/len(df_full_filtered):5.1f}%)")
print(f"    [6-8]:   {hist[1]:>6,} samples ({100*hist[1]/len(df_full_filtered):5.1f}%)")
print(f"    [8-10]:  {hist[2]:>6,} samples ({100*hist[2]/len(df_full_filtered):5.1f}%)")
print(f"    [10-12]: {hist[3]:>6,} samples ({100*hist[3]/len(df_full_filtered):5.1f}%)")
print(f"    [12-14]: {hist[4]:>6,} samples ({100*hist[4]/len(df_full_filtered):5.1f}%)")
print()

# =============================================================================
# Step 3: Compare with Current Dataset
# =============================================================================
print("Step 3: Comparing with current dataset...")
df_current = pd.read_csv(agab_current_path)
print(f"  Current dataset: {len(df_current):,} samples")

# Check column compatibility
print(f"  Current columns: {list(df_current.columns)}")
print(f"  New columns: {list(df_full_filtered.columns)}")

# Find common columns
common_cols = set(df_current.columns) & set(df_full_filtered.columns)
print(f"  Common columns: {list(common_cols)}")

# Check for duplicates (if antibody_sequence and antigen_sequence are present)
if 'antibody_sequence' in common_cols and 'antigen_sequence' in common_cols:
    current_pairs = set(zip(df_current['antibody_sequence'], df_current['antigen_sequence']))
    full_pairs = set(zip(df_full_filtered['antibody_sequence'], df_full_filtered['antigen_sequence']))

    overlap = len(current_pairs & full_pairs)
    new_samples = len(full_pairs - current_pairs)

    print(f"  Overlap: {overlap:,} samples ({100*overlap/len(df_current):.1f}% of current)")
    print(f"  New samples: {new_samples:,}")
    print(f"  Total unique pairs: {len(full_pairs):,}")
else:
    print("  WARNING: Cannot check for duplicates (missing sequence columns)")
print()

# =============================================================================
# Step 4: Load and Merge SAbDab
# =============================================================================
print("Step 4: Loading SAbDab dataset...")
df_sabdab = pd.read_csv(sabdab_path)
print(f"  Loaded: {len(df_sabdab):,} samples")
print(f"  Columns: {list(df_sabdab.columns)}")
print(f"  pKd range: [{df_sabdab['pkd'].min():.2f}, {df_sabdab['pkd'].max():.2f}]")
print()

# Rename 'pkd' to 'pKd' for consistency
df_sabdab = df_sabdab.rename(columns={'pkd': 'pKd'})

# Create antibody_sequence and antigen_sequence for SAbDab
if 'antibody_sequence' not in df_sabdab.columns:
    if 'heavy_seq' in df_sabdab.columns and 'light_seq' in df_sabdab.columns:
        print("  Creating antibody_sequence from heavy_seq + light_seq...")
        df_sabdab['antibody_sequence'] = df_sabdab['heavy_seq'].fillna('') + df_sabdab['light_seq'].fillna('')

if 'antigen_sequence' not in df_sabdab.columns and 'antigen_seq' in df_sabdab.columns:
    print("  Renaming antigen_seq to antigen_sequence...")
    df_sabdab = df_sabdab.rename(columns={'antigen_seq': 'antigen_sequence'})

# Remove rows with empty sequences
df_sabdab = df_sabdab[
    (df_sabdab['antibody_sequence'].notna()) &
    (df_sabdab['antibody_sequence'] != '') &
    (df_sabdab['antigen_sequence'].notna()) &
    (df_sabdab['antigen_sequence'] != '')
]
print(f"  After removing empty sequences: {len(df_sabdab):,} samples")

# Find common columns between AgAb and SAbDab
common_cols_sabdab = set(df_full_filtered.columns) & set(df_sabdab.columns)
print(f"  Common columns with AgAb: {list(common_cols_sabdab)}")

# Check if we can merge
if 'antibody_sequence' in common_cols_sabdab and 'antigen_sequence' in common_cols_sabdab and 'pKd' in common_cols_sabdab:
    print("  OK: Can merge! Common key columns found.")

    # Align columns
    merge_cols = ['antibody_sequence', 'antigen_sequence', 'pKd']
    df_sabdab_aligned = df_sabdab[merge_cols].copy()
    df_sabdab_aligned['source'] = 'SAbDab'

    # Check for overlap
    sabdab_pairs = set(zip(df_sabdab_aligned['antibody_sequence'], df_sabdab_aligned['antigen_sequence']))
    overlap_sabdab = len(sabdab_pairs & full_pairs)
    new_sabdab = len(sabdab_pairs - full_pairs)

    print(f"  Overlap with AgAb: {overlap_sabdab} samples")
    print(f"  New unique SAbDab samples: {new_sabdab}")
    print()

    # Merge: Concatenate and deduplicate
    print("Step 5: Merging datasets...")

    # Prepare AgAb data with same columns
    df_agab_aligned = df_full_filtered[merge_cols].copy()
    df_agab_aligned['source'] = 'AgAb'

    # Concatenate
    df_merged = pd.concat([df_agab_aligned, df_sabdab_aligned], ignore_index=True)
    print(f"  After concatenation: {len(df_merged):,} samples")

    # Deduplicate by antibody_sequence + antigen_sequence
    df_merged = df_merged.drop_duplicates(subset=['antibody_sequence', 'antigen_sequence'], keep='first')
    print(f"  After deduplication: {len(df_merged):,} samples")
    print(f"  Removed duplicates: {len(pd.concat([df_agab_aligned, df_sabdab_aligned])) - len(df_merged):,}")

else:
    print("  WARNING: Cannot merge SAbDab - missing common columns")
    print("  Proceeding with AgAb Full only...")
    df_merged = df_full_filtered.copy()
    df_merged['source'] = 'AgAb'
print()

# =============================================================================
# Step 6: Final Statistics and Distribution
# =============================================================================
print("="*80)
print("FINAL MERGED DATASET STATISTICS")
print("="*80)
print(f"Total samples: {len(df_merged):,}")
print(f"pKd range: [{df_merged['pKd'].min():.2f}, {df_merged['pKd'].max():.2f}]")
print(f"pKd mean: {df_merged['pKd'].mean():.2f} ± {df_merged['pKd'].std():.2f}")
print()

print("Distribution by pKd bins:")
bins = [4, 6, 8, 10, 12, 14]
hist, _ = np.histogram(df_merged['pKd'], bins=bins)
print(f"  [4-6]:   {hist[0]:>6,} samples ({100*hist[0]/len(df_merged):5.1f}%) ← Weak binders")
print(f"  [6-8]:   {hist[1]:>6,} samples ({100*hist[1]/len(df_merged):5.1f}%)")
print(f"  [8-10]:  {hist[2]:>6,} samples ({100*hist[2]/len(df_merged):5.1f}%)")
print(f"  [10-12]: {hist[3]:>6,} samples ({100*hist[3]/len(df_merged):5.1f}%) ← Strong binders")
print(f"  [12-14]: {hist[4]:>6,} samples ({100*hist[4]/len(df_merged):5.1f}%)")
print()

if 'source' in df_merged.columns:
    print("Samples by source:")
    print(df_merged['source'].value_counts())
    print()

# =============================================================================
# Step 7: Save Merged Dataset
# =============================================================================
print("Step 6: Saving merged dataset...")
df_merged.to_csv(output_path, index=False)
print(f"  SUCCESS: Saved to: {output_path}")
print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
print()

# =============================================================================
# Step 8: Comparison Summary
# =============================================================================
print("="*80)
print("IMPROVEMENT SUMMARY")
print("="*80)
print(f"Current dataset (agab_phase2_full.csv):")
print(f"  Total: {len(df_current):,} samples")
current_valid = df_current[(df_current['pKd'] >= 4.0) & (df_current['pKd'] <= 14.0)]
print(f"  Valid [4-14]: {len(current_valid):,} samples")
current_weak = current_valid[(current_valid['pKd'] >= 4.0) & (current_valid['pKd'] < 6.0)]
print(f"  Weak binders [4-6]: {len(current_weak):,} samples")
print()

print(f"New merged dataset (agab_phase2_full_v2_balanced.csv):")
print(f"  Total: {len(df_merged):,} samples (+{len(df_merged) - len(current_valid):,}, +{100*(len(df_merged) - len(current_valid))/len(current_valid):.1f}%)")
new_weak = df_merged[(df_merged['pKd'] >= 4.0) & (df_merged['pKd'] < 6.0)]
print(f"  Weak binders [4-6]: {len(new_weak):,} samples (+{len(new_weak) - len(current_weak):,}, {len(new_weak)/len(current_weak):.0f}x more!)")
print()

print("SUCCESS: Dataset scaling complete!")
print()
print("Next steps:")
print("1. Upload to Google Drive: MyDrive/AbAg_Training_02/agab_phase2_full_v2_balanced.csv")
print("2. Update colab_training_v2.7.ipynb Cell 11:")
print("   CSV_FILENAME = 'agab_phase2_full_v2_balanced.csv'")
print("3. Restart training and monitor weak binder predictions!")
print()
print("Expected improvements:")
print(f"  - {len(new_weak)/len(current_weak):.0f}x more weak binder training data")
print("  - Better balanced distribution across all pKd ranges")
print("  - Expected Spearman: 0.55-0.65 (vs current target 0.45-0.55)")
