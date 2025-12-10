#!/usr/bin/env python3
"""
Create improved dataset for antibody-antigen binding prediction.

Improvements:
1. Remove suspicious round-number pKd values
2. Better balance across pKd ranges
3. Ensure high-quality data from SAbDab
4. Remove very short sequences
5. Stratified sampling for better distribution
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("="*60)
print("CREATING IMPROVED DATASET")
print("="*60)
print()

# Paths
DATA_DIR = Path("C:/Users/401-24/Desktop/Ab_Ag_dataset/data")
OUTPUT_DIR = DATA_DIR / "agab"

# Load current balanced dataset
print("Loading current balanced dataset...")
df = pd.read_csv(OUTPUT_DIR / "agab_phase2_full_v2_balanced.csv")
print(f"  Loaded: {len(df):,} samples")
print()

# Load high-quality SAbDab data
print("Loading SAbDab high-quality data...")
sabdab = pd.read_csv(DATA_DIR / "sabdab" / "sabdab_sequences_with_affinity.csv")
print(f"  Loaded: {len(sabdab):,} samples")
print()

# =============================================================================
# STEP 1: Clean current dataset
# =============================================================================
print("="*60)
print("STEP 1: Clean current dataset")
print("="*60)

original_len = len(df)

# 1.1 Remove exact round numbers (likely placeholders)
print("\n1.1 Removing suspicious round pKd values...")
round_values = [6.0]  # 6.0 is particularly suspicious
before = len(df)
df = df[~df['pKd'].isin(round_values)]
print(f"  Removed {before - len(df):,} samples with pKd = 6.0")

# 1.2 Remove very short sequences
print("\n1.2 Removing very short sequences...")
df['ab_len'] = df['antibody_sequence'].str.len()
df['ag_len'] = df['antigen_sequence'].str.len()

before = len(df)
df = df[(df['ab_len'] >= 100) & (df['ag_len'] >= 20)]
print(f"  Removed {before - len(df):,} samples with short sequences")

# 1.3 Remove sequences with invalid characters
print("\n1.3 Checking for invalid amino acids...")
valid_aa = set('ACDEFGHIKLMNPQRSTVWY')

def has_valid_aa(seq):
    if pd.isna(seq):
        return False
    return set(str(seq).upper()).issubset(valid_aa)

before = len(df)
df = df[df['antibody_sequence'].apply(has_valid_aa)]
df = df[df['antigen_sequence'].apply(has_valid_aa)]
print(f"  Removed {before - len(df):,} samples with invalid amino acids")

print(f"\nAfter cleaning: {len(df):,} samples (removed {original_len - len(df):,})")

# =============================================================================
# STEP 2: Add more SAbDab high-quality data
# =============================================================================
print("\n" + "="*60)
print("STEP 2: Add SAbDab high-quality data")
print("="*60)

# Process SAbDab
sabdab_clean = sabdab.copy()

# Filter valid pKd range
sabdab_clean = sabdab_clean[(sabdab_clean['pkd'] >= 4) & (sabdab_clean['pkd'] <= 14)]

# Filter high resolution (<3.0 A)
sabdab_clean = sabdab_clean[sabdab_clean['resolution'] <= 3.0]

# Create antibody sequence (heavy + light)
sabdab_clean['antibody_sequence'] = sabdab_clean['heavy_seq'].fillna('') + sabdab_clean['light_seq'].fillna('')

# Remove entries without proper sequences
sabdab_clean = sabdab_clean[sabdab_clean['antibody_sequence'].str.len() >= 100]
sabdab_clean = sabdab_clean[sabdab_clean['antigen_seq'].str.len() >= 20]

# Rename columns to match
sabdab_formatted = pd.DataFrame({
    'antibody_sequence': sabdab_clean['antibody_sequence'],
    'antigen_sequence': sabdab_clean['antigen_seq'],
    'pKd': sabdab_clean['pkd'],
    'source': 'SAbDab_HQ'
})

print(f"  High-quality SAbDab samples: {len(sabdab_formatted):,}")

# Check overlap with existing data
existing_pairs = set(zip(df['antibody_sequence'], df['antigen_sequence']))
new_sabdab = sabdab_formatted[
    ~sabdab_formatted.apply(lambda x: (x['antibody_sequence'], x['antigen_sequence']) in existing_pairs, axis=1)
]
print(f"  New (non-duplicate) samples: {len(new_sabdab):,}")

# Add to main dataset
df = pd.concat([df, new_sabdab], ignore_index=True)
print(f"  Total after adding SAbDab: {len(df):,}")

# =============================================================================
# STEP 3: Stratified resampling for better balance
# =============================================================================
print("\n" + "="*60)
print("STEP 3: Balance dataset by pKd range")
print("="*60)

# Current distribution
print("\nCurrent distribution:")
bins = [(4, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 14)]
for low, high in bins:
    count = ((df['pKd'] >= low) & (df['pKd'] < high)).sum()
    print(f"  [{low}-{high}): {count:,} ({100*count/len(df):.1f}%)")

# Target: More balanced distribution
# We want roughly equal representation, but we're limited by available data
print("\nCreating balanced subsets...")

balanced_dfs = []
target_per_bin = 20000  # Target samples per bin (will use all if less available)

for low, high in bins:
    bin_df = df[(df['pKd'] >= low) & (df['pKd'] < high)]

    if len(bin_df) <= target_per_bin:
        # Use all available
        balanced_dfs.append(bin_df)
        print(f"  [{low}-{high}): Using all {len(bin_df):,} samples")
    else:
        # Sample to target
        sampled = bin_df.sample(n=target_per_bin, random_state=42)
        balanced_dfs.append(sampled)
        print(f"  [{low}-{high}): Sampled {target_per_bin:,} from {len(bin_df):,}")

df_balanced = pd.concat(balanced_dfs, ignore_index=True)

# =============================================================================
# STEP 4: Final cleanup and statistics
# =============================================================================
print("\n" + "="*60)
print("STEP 4: Final statistics")
print("="*60)

# Remove helper columns
df_balanced = df_balanced[['antibody_sequence', 'antigen_sequence', 'pKd', 'source']]

# Fill missing source
df_balanced['source'] = df_balanced['source'].fillna('AgAb')

print(f"\nFinal dataset: {len(df_balanced):,} samples")
print()

print("pKd distribution:")
for low, high in bins:
    count = ((df_balanced['pKd'] >= low) & (df_balanced['pKd'] < high)).sum()
    print(f"  [{low}-{high}): {count:,} ({100*count/len(df_balanced):.1f}%)")

print()
print("Source distribution:")
print(df_balanced['source'].value_counts())

print()
print("Sequence length statistics:")
print(f"  Antibody: {df_balanced['antibody_sequence'].str.len().min()} - {df_balanced['antibody_sequence'].str.len().max()} (mean: {df_balanced['antibody_sequence'].str.len().mean():.0f})")
print(f"  Antigen: {df_balanced['antigen_sequence'].str.len().min()} - {df_balanced['antigen_sequence'].str.len().max()} (mean: {df_balanced['antigen_sequence'].str.len().mean():.0f})")

print()
print("pKd statistics:")
print(f"  Range: {df_balanced['pKd'].min():.2f} - {df_balanced['pKd'].max():.2f}")
print(f"  Mean: {df_balanced['pKd'].mean():.2f} +/- {df_balanced['pKd'].std():.2f}")

# Classification balance for your goal (pKd >= 9)
high_affinity = (df_balanced['pKd'] >= 9).sum()
print()
print(f"High affinity (pKd >= 9): {high_affinity:,} ({100*high_affinity/len(df_balanced):.1f}%)")
print(f"Not high affinity (pKd < 9): {len(df_balanced) - high_affinity:,} ({100*(1 - high_affinity/len(df_balanced)):.1f}%)")

# =============================================================================
# STEP 5: Save improved dataset
# =============================================================================
print("\n" + "="*60)
print("STEP 5: Save improved dataset")
print("="*60)

output_path = OUTPUT_DIR / "agab_improved_v3.csv"
df_balanced.to_csv(output_path, index=False)

file_size = output_path.stat().st_size / (1024 * 1024)
print(f"\nSaved to: {output_path}")
print(f"File size: {file_size:.1f} MB")

print()
print("="*60)
print("SUMMARY OF IMPROVEMENTS")
print("="*60)
print()
print("1. Removed suspicious pKd = 6.0 values (likely placeholders)")
print("2. Removed very short sequences (Ab < 100, Ag < 20)")
print("3. Added high-quality SAbDab data (resolution < 3.0A)")
print("4. Balanced sampling across pKd ranges")
print("5. Validated all amino acid sequences")
print()
print("Next steps:")
print("  1. Upload agab_improved_v3.csv to Google Drive")
print("  2. Update notebook Cell 11: CSV_FILENAME = 'agab_improved_v3.csv'")
print("  3. Train with ProtT5 model (v2.8)")
