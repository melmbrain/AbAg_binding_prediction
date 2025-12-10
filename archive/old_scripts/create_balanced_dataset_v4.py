#!/usr/bin/env python3
"""
Create properly balanced dataset using ALL available data from AgAb Full.

Goal: Fill gaps in [6-7), [7-8), and [10+) ranges.
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("="*60)
print("CREATING BALANCED DATASET V4")
print("="*60)
print()

DATA_DIR = Path("C:/Users/401-24/Desktop/Ab_Ag_dataset/data")

# =============================================================================
# STEP 1: Load and process AgAb Full Dataset
# =============================================================================
print("STEP 1: Loading AgAb Full Dataset...")
agab_full = pd.read_csv(DATA_DIR / "agab" / "agab_full_dataset.csv")
print(f"  Total samples: {len(agab_full):,}")

# Convert affinity to numeric
agab_full['affinity'] = pd.to_numeric(agab_full['affinity'], errors='coerce')

# Filter valid pKd range [4, 14]
agab_full = agab_full[(agab_full['affinity'] >= 4) & (agab_full['affinity'] <= 14)]
print(f"  Valid pKd [4-14]: {len(agab_full):,}")

# Create antibody sequence (heavy + light)
agab_full['antibody_sequence'] = agab_full['heavy_sequence'].fillna('') + agab_full['light_sequence'].fillna('')

# Rename columns
agab_full = agab_full.rename(columns={'affinity': 'pKd', 'antigen_sequence': 'antigen_sequence'})

# Remove rows with missing sequences
agab_full = agab_full.dropna(subset=['antibody_sequence', 'antigen_sequence'])

# Filter by sequence length
agab_full = agab_full[agab_full['antibody_sequence'].str.len() >= 100]
agab_full = agab_full[agab_full['antigen_sequence'].str.len() >= 20]

# Remove suspicious pKd = 6.0 (placeholder values)
agab_full = agab_full[agab_full['pKd'] != 6.0]

# Check for valid amino acids
valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
def has_valid_aa(seq):
    if pd.isna(seq) or len(str(seq)) == 0:
        return False
    return set(str(seq).upper()).issubset(valid_aa)

agab_full = agab_full[agab_full['antibody_sequence'].apply(has_valid_aa)]
agab_full = agab_full[agab_full['antigen_sequence'].apply(has_valid_aa)]

# Keep only needed columns
agab_full = agab_full[['antibody_sequence', 'antigen_sequence', 'pKd']].copy()
agab_full['source'] = 'AgAb'

print(f"  After cleaning: {len(agab_full):,}")
print()

# =============================================================================
# STEP 2: Load SAbDab high-quality data
# =============================================================================
print("STEP 2: Loading SAbDab high-quality data...")
sabdab = pd.read_csv(DATA_DIR / "sabdab" / "sabdab_sequences_with_affinity.csv")

# Filter valid range and high resolution
sabdab = sabdab[(sabdab['pkd'] >= 4) & (sabdab['pkd'] <= 14)]
sabdab = sabdab[sabdab['resolution'] <= 3.0]

# Create antibody sequence
sabdab['antibody_sequence'] = sabdab['heavy_seq'].fillna('') + sabdab['light_seq'].fillna('')

# Filter by length
sabdab = sabdab[sabdab['antibody_sequence'].str.len() >= 100]
sabdab = sabdab[sabdab['antigen_seq'].str.len() >= 20]

# Format
sabdab_formatted = pd.DataFrame({
    'antibody_sequence': sabdab['antibody_sequence'],
    'antigen_sequence': sabdab['antigen_seq'],
    'pKd': sabdab['pkd'],
    'source': 'SAbDab'
})

print(f"  SAbDab samples: {len(sabdab_formatted):,}")
print()

# =============================================================================
# STEP 3: Combine and deduplicate
# =============================================================================
print("STEP 3: Combining datasets...")
df = pd.concat([agab_full, sabdab_formatted], ignore_index=True)
print(f"  Combined: {len(df):,}")

# Deduplicate by sequence pair
df = df.drop_duplicates(subset=['antibody_sequence', 'antigen_sequence'], keep='first')
print(f"  After deduplication: {len(df):,}")
print()

# =============================================================================
# STEP 4: Check distribution before balancing
# =============================================================================
print("STEP 4: Distribution before balancing:")
bins = [(4, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 14)]
for low, high in bins:
    count = ((df['pKd'] >= low) & (df['pKd'] < high)).sum()
    print(f"  [{low}-{high}): {count:,}")
print()

# =============================================================================
# STEP 5: Balanced sampling
# =============================================================================
print("STEP 5: Creating balanced dataset...")

# Target: Use all available data, but cap very large bins
# We want good representation in ALL ranges

balanced_dfs = []

# Define target sizes based on what's available
# We'll use all data from small bins, and sample from large bins
bin_targets = {
    (4, 6): 20000,   # Cap weak binders
    (6, 7): None,    # Use ALL (this was the gap!)
    (7, 8): None,    # Use ALL
    (8, 9): 20000,   # Cap
    (9, 10): 20000,  # Cap
    (10, 14): None,  # Use ALL (this was small)
}

for (low, high), target in bin_targets.items():
    bin_df = df[(df['pKd'] >= low) & (df['pKd'] < high)]

    if target is None or len(bin_df) <= target:
        balanced_dfs.append(bin_df)
        print(f"  [{low}-{high}): Using ALL {len(bin_df):,} samples")
    else:
        sampled = bin_df.sample(n=target, random_state=42)
        balanced_dfs.append(sampled)
        print(f"  [{low}-{high}): Sampled {target:,} from {len(bin_df):,}")

df_balanced = pd.concat(balanced_dfs, ignore_index=True)
print()

# =============================================================================
# STEP 6: Final statistics
# =============================================================================
print("="*60)
print("FINAL DATASET STATISTICS")
print("="*60)

print(f"\nTotal samples: {len(df_balanced):,}")
print()

print("pKd distribution:")
total = len(df_balanced)
for low, high in bins:
    count = ((df_balanced['pKd'] >= low) & (df_balanced['pKd'] < high)).sum()
    bar = '#' * int(count / total * 50)
    print(f"  [{low:2}-{high:2}): {count:>6,} ({100*count/total:>5.1f}%) {bar}")

print()
print("Source distribution:")
print(df_balanced['source'].value_counts().to_string())

print()
print("For your goal (pKd >= 9):")
high_aff = (df_balanced['pKd'] >= 9).sum()
print(f"  High affinity (>=9):     {high_aff:,} ({100*high_aff/total:.1f}%)")
print(f"  Not high affinity (<9):  {total - high_aff:,} ({100*(1 - high_aff/total):.1f}%)")

print()
print("Sequence statistics:")
print(f"  Antibody length: {df_balanced['antibody_sequence'].str.len().min()} - {df_balanced['antibody_sequence'].str.len().max()} (mean: {df_balanced['antibody_sequence'].str.len().mean():.0f})")
print(f"  Antigen length:  {df_balanced['antigen_sequence'].str.len().min()} - {df_balanced['antigen_sequence'].str.len().max()} (mean: {df_balanced['antigen_sequence'].str.len().mean():.0f})")

print()
print("pKd statistics:")
print(f"  Range: {df_balanced['pKd'].min():.2f} - {df_balanced['pKd'].max():.2f}")
print(f"  Mean:  {df_balanced['pKd'].mean():.2f} +/- {df_balanced['pKd'].std():.2f}")

# =============================================================================
# STEP 7: Save
# =============================================================================
print()
print("="*60)
print("SAVING DATASET")
print("="*60)

output_path = DATA_DIR / "agab" / "agab_balanced_v4.csv"
df_balanced.to_csv(output_path, index=False)

file_size = output_path.stat().st_size / (1024 * 1024)
print(f"\nSaved to: {output_path}")
print(f"File size: {file_size:.1f} MB")

print()
print("="*60)
print("COMPARISON: Old vs New")
print("="*60)
print()
print("              Old (v3)    New (v4)    Improvement")
print("  [4-6):      20,000      20,000      Same")
print("  [6-7):         824      FILLED      +++")
print("  [7-8):       6,000      FILLED      +++")
print("  [8-9):      16,518      20,000      +")
print("  [9-10):     20,000      20,000      Same")
print("  [10+):         215      FILLED      +++")
print()
print("Next steps:")
print("  1. Upload agab_balanced_v4.csv to Google Drive")
print("  2. Update notebook: CSV_FILENAME = 'agab_balanced_v4.csv'")
print("  3. Train!")
