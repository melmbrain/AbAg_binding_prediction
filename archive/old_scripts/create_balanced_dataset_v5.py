#!/usr/bin/env python3
"""
Create FINAL balanced dataset v5 - Maximum data from all ranges.
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("="*60)
print("CREATING BALANCED DATASET V5 (FINAL)")
print("="*60)
print()

DATA_DIR = Path("C:/Users/401-24/Desktop/Ab_Ag_dataset/data")

# =============================================================================
# STEP 1: Load AgAb Full with minimal cleaning
# =============================================================================
print("STEP 1: Loading AgAb Full Dataset...")
agab = pd.read_csv(DATA_DIR / "agab" / "agab_full_dataset.csv")
print(f"  Total samples: {len(agab):,}")

# Convert affinity
agab['affinity'] = pd.to_numeric(agab['affinity'], errors='coerce')

# Filter valid range
agab = agab[(agab['affinity'] >= 4) & (agab['affinity'] <= 14)]
print(f"  Valid pKd [4-14]: {len(agab):,}")

# ONLY remove pKd = 6.0 (clear placeholder - 58k of them)
before = len(agab)
agab = agab[agab['affinity'] != 6.0]
print(f"  Removed pKd=6.0: {before - len(agab):,} samples")

# Create antibody sequence
agab['antibody_sequence'] = agab['heavy_sequence'].fillna('') + agab['light_sequence'].fillna('')

# Rename
agab = agab.rename(columns={'affinity': 'pKd'})

# Filter missing sequences (but keep shorter ones)
agab = agab.dropna(subset=['antibody_sequence', 'antigen_sequence'])
agab = agab[agab['antibody_sequence'].str.len() > 0]
agab = agab[agab['antigen_sequence'].str.len() > 0]

# Minimal length filter (less strict)
agab = agab[agab['antibody_sequence'].str.len() >= 50]  # Relaxed from 100
agab = agab[agab['antigen_sequence'].str.len() >= 10]   # Relaxed from 20

# Check valid amino acids
valid_aa = set('ACDEFGHIKLMNPQRSTVWYX')  # Added X for unknown
def has_mostly_valid_aa(seq):
    if pd.isna(seq) or len(str(seq)) == 0:
        return False
    valid_count = sum(1 for c in str(seq).upper() if c in valid_aa)
    return valid_count / len(seq) > 0.95  # Allow 5% unknown

agab = agab[agab['antibody_sequence'].apply(has_mostly_valid_aa)]
agab = agab[agab['antigen_sequence'].apply(has_mostly_valid_aa)]

# Keep only needed columns
agab = agab[['antibody_sequence', 'antigen_sequence', 'pKd']].copy()
agab['source'] = 'AgAb'

print(f"  After cleaning: {len(agab):,}")
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
# STEP 3: Combine and deduplicate
# =============================================================================
print("STEP 3: Combining datasets...")
df = pd.concat([agab, sabdab_formatted], ignore_index=True)
print(f"  Combined: {len(df):,}")

df = df.drop_duplicates(subset=['antibody_sequence', 'antigen_sequence'], keep='first')
print(f"  After deduplication: {len(df):,}")
print()

# =============================================================================
# STEP 4: Distribution before balancing
# =============================================================================
print("STEP 4: Distribution before balancing:")
bins = [(4, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 14)]
for low, high in bins:
    count = ((df['pKd'] >= low) & (df['pKd'] < high)).sum()
    print(f"  [{low}-{high}): {count:,}")
print()

# =============================================================================
# STEP 5: Balanced sampling - USE ALL from small bins!
# =============================================================================
print("STEP 5: Creating balanced dataset...")

balanced_dfs = []

# Target: ~20k per major bin, ALL from small bins
bin_configs = [
    ((4, 6), 20000),    # Cap
    ((6, 7), None),     # Use ALL
    ((7, 8), None),     # Use ALL
    ((8, 9), 20000),    # Cap
    ((9, 10), 20000),   # Cap
    ((10, 14), None),   # Use ALL
]

for (low, high), target in bin_configs:
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
    bar = '#' * int(count / total * 40)
    print(f"  [{low:2}-{high:2}): {count:>6,} ({100*count/total:>5.1f}%) {bar}")

print()
print("For your goal (pKd >= 9):")
print("-"*50)
high_aff = (df_balanced['pKd'] >= 9).sum()
print(f"  High affinity (pKd >= 9):  {high_aff:,} ({100*high_aff/total:.1f}%)")
print(f"  NOT high affinity (< 9):   {total - high_aff:,} ({100*(1 - high_aff/total):.1f}%)")

print()
print("Source distribution:")
print("-"*50)
for src, count in df_balanced['source'].value_counts().items():
    print(f"  {src}: {count:,}")

print()
print("Sequence length:")
print("-"*50)
print(f"  Antibody: min={df_balanced['antibody_sequence'].str.len().min()}, max={df_balanced['antibody_sequence'].str.len().max()}, mean={df_balanced['antibody_sequence'].str.len().mean():.0f}")
print(f"  Antigen:  min={df_balanced['antigen_sequence'].str.len().min()}, max={df_balanced['antigen_sequence'].str.len().max()}, mean={df_balanced['antigen_sequence'].str.len().mean():.0f}")

# =============================================================================
# SAVE
# =============================================================================
print()
print("="*60)
print("SAVING")
print("="*60)

output_path = DATA_DIR / "agab" / "agab_balanced_v5_final.csv"
df_balanced.to_csv(output_path, index=False)

file_size = output_path.stat().st_size / (1024 * 1024)
print(f"\nSaved to: {output_path}")
print(f"File size: {file_size:.1f} MB")

print()
print("="*60)
print("SUMMARY")
print("="*60)
print("""
IMPROVEMENTS IN V5:
1. Removed ONLY pKd=6.0 (58k placeholder values)
2. Relaxed sequence length filters (Ab>=50, Ag>=10)
3. Used ALL available data from underrepresented ranges
4. Balanced sampling for overrepresented ranges

RESULT:
- Better coverage in [6-7), [7-8), [10+) ranges
- 68% negative examples for pKd>=9 classification
- Higher quality data (no placeholder values)

NEXT STEPS:
1. Upload agab_balanced_v5_final.csv to Google Drive
2. Update notebook: CSV_FILENAME = 'agab_balanced_v5_final.csv'
3. Train with ProtT5!
""")
