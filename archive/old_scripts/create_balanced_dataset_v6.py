#!/usr/bin/env python3
"""
Create balanced dataset v6 - Handle duplicate pairs properly.

KEY FIX: When same Ab-Ag pair has multiple pKd values, use MEDIAN.
This preserves more data in underrepresented ranges.
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("="*60)
print("CREATING BALANCED DATASET V6")
print("="*60)
print()

DATA_DIR = Path("C:/Users/401-24/Desktop/Ab_Ag_dataset/data")

# =============================================================================
# STEP 1: Load and clean AgAb
# =============================================================================
print("STEP 1: Loading AgAb Full Dataset...")
agab = pd.read_csv(DATA_DIR / "agab" / "agab_full_dataset.csv")
print(f"  Total samples: {len(agab):,}")

agab['affinity'] = pd.to_numeric(agab['affinity'], errors='coerce')
agab = agab[(agab['affinity'] >= 4) & (agab['affinity'] <= 14)]
agab = agab[agab['affinity'] != 6.0]  # Remove placeholder
print(f"  After basic filtering: {len(agab):,}")

# Create sequences
agab['antibody_sequence'] = agab['heavy_sequence'].fillna('') + agab['light_sequence'].fillna('')
agab = agab.rename(columns={'affinity': 'pKd'})
agab = agab.dropna(subset=['antibody_sequence', 'antigen_sequence'])
agab = agab[agab['antibody_sequence'].str.len() >= 50]
agab = agab[agab['antigen_sequence'].str.len() >= 10]
agab = agab[['antibody_sequence', 'antigen_sequence', 'pKd']].copy()
agab['source'] = 'AgAb'
print(f"  After sequence filtering: {len(agab):,}")
print()

# =============================================================================
# STEP 2: Load SAbDab
# =============================================================================
print("STEP 2: Loading SAbDab...")
sabdab = pd.read_csv(DATA_DIR / "sabdab" / "sabdab_sequences_with_affinity.csv")
sabdab = sabdab[(sabdab['pkd'] >= 4) & (sabdab['pkd'] <= 14)]
sabdab['antibody_sequence'] = sabdab['heavy_seq'].fillna('') + sabdab['light_seq'].fillna('')
sabdab = sabdab[sabdab['antibody_sequence'].str.len() >= 50]
sabdab = sabdab[sabdab['antigen_seq'].str.len() >= 10]
sabdab_formatted = pd.DataFrame({
    'antibody_sequence': sabdab['antibody_sequence'],
    'antigen_sequence': sabdab['antigen_seq'],
    'pKd': sabdab['pkd'],
    'source': 'SAbDab'
})
print(f"  SAbDab samples: {len(sabdab_formatted):,}")
print()

# =============================================================================
# STEP 3: Combine
# =============================================================================
print("STEP 3: Combining datasets...")
df = pd.concat([agab, sabdab_formatted], ignore_index=True)
print(f"  Combined: {len(df):,}")

# =============================================================================
# STEP 4: Handle duplicates with MEDIAN pKd
# =============================================================================
print()
print("STEP 4: Handling duplicate pairs...")
print("-"*50)

# Find duplicates
df['pair_key'] = df['antibody_sequence'] + '|||' + df['antigen_sequence']
dup_counts = df.groupby('pair_key').size()
duplicates = dup_counts[dup_counts > 1]
print(f"  Unique pairs: {len(dup_counts):,}")
print(f"  Pairs with multiple pKd values: {len(duplicates):,}")

# For duplicates, check pKd variation
dup_pairs = df[df['pair_key'].isin(duplicates.index)]
dup_stats = dup_pairs.groupby('pair_key')['pKd'].agg(['count', 'min', 'max', 'mean', 'std'])
high_var = dup_stats[dup_stats['std'] > 1.0]
print(f"  High variance pairs (std > 1.0): {len(high_var):,}")

# Use MEDIAN pKd for each unique pair
print()
print("  Using MEDIAN pKd for duplicate pairs...")
df_grouped = df.groupby('pair_key').agg({
    'antibody_sequence': 'first',
    'antigen_sequence': 'first',
    'pKd': 'median',  # MEDIAN instead of keeping first!
    'source': 'first'
}).reset_index(drop=True)

print(f"  After aggregation: {len(df_grouped):,}")
print()

# =============================================================================
# STEP 5: Distribution after deduplication
# =============================================================================
print("STEP 5: Distribution after median aggregation:")
bins = [(4, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 14)]
for low, high in bins:
    count = ((df_grouped['pKd'] >= low) & (df_grouped['pKd'] < high)).sum()
    print(f"  [{low}-{high}): {count:,}")
print()

# =============================================================================
# STEP 6: Balanced sampling
# =============================================================================
print("STEP 6: Creating balanced dataset...")

balanced_dfs = []
bin_targets = {
    (4, 6): 20000,
    (6, 7): None,   # Use ALL
    (7, 8): None,   # Use ALL
    (8, 9): 20000,
    (9, 10): 20000,
    (10, 14): None, # Use ALL
}

for (low, high), target in bin_targets.items():
    bin_df = df_grouped[(df_grouped['pKd'] >= low) & (df_grouped['pKd'] < high)]

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
# FINAL STATISTICS
# =============================================================================
print("="*60)
print("FINAL DATASET STATISTICS")
print("="*60)

total = len(df_balanced)
print(f"\nTotal samples: {total:,}")
print()

print("pKd distribution:")
print("-"*50)
for low, high in bins:
    count = ((df_balanced['pKd'] >= low) & (df_balanced['pKd'] < high)).sum()
    pct = 100*count/total
    bar = '#' * int(pct * 0.5)
    print(f"  [{low:2}-{high:2}): {count:>6,} ({pct:>5.1f}%) {bar}")

print()
print("For your goal (pKd >= 9):")
print("-"*50)
high_aff = (df_balanced['pKd'] >= 9).sum()
print(f"  High affinity (pKd >= 9):  {high_aff:,} ({100*high_aff/total:.1f}%)")
print(f"  NOT high affinity (< 9):   {total - high_aff:,} ({100*(1 - high_aff/total):.1f}%)")

# =============================================================================
# SAVE
# =============================================================================
print()
print("="*60)
output_path = DATA_DIR / "agab" / "agab_balanced_v6_median.csv"
df_balanced.to_csv(output_path, index=False)
file_size = output_path.stat().st_size / (1024 * 1024)
print(f"Saved to: {output_path}")
print(f"File size: {file_size:.1f} MB")

print()
print("="*60)
print("V6 IMPROVEMENTS:")
print("-"*60)
print("  1. Used MEDIAN pKd for duplicate Ab-Ag pairs")
print("  2. This recovers data lost during deduplication")
print("  3. More samples in underrepresented ranges")
print()
print("NEXT: Upload agab_balanced_v6_median.csv to Google Drive")
