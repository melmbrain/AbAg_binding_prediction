#!/usr/bin/env python3
"""
Process Pierce Lab Antibody Benchmark data.
Extract sequences from PDB files and match with affinity data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import re

print("="*70)
print("PROCESSING PIERCE LAB ANTIBODY BENCHMARK")
print("="*70)
print()

BENCHMARK_DIR = Path("C:/Users/401-24/Desktop/Ab_Ag_dataset/data/new_sources/antibody_benchmark")
OUTPUT_DIR = Path("C:/Users/401-24/Desktop/Ab_Ag_dataset/data/agab")

# =============================================================================
# Step 1: Load affinity data
# =============================================================================
print("Step 1: Loading affinity data...")
affinity_file = BENCHMARK_DIR / "antibody_antigen_affinities.txt"
affinity_df = pd.read_csv(affinity_file, sep='\t')
print(f"  Entries with affinity: {len(affinity_df)}")
print(f"  Columns: {list(affinity_df.columns)}")

# Convert Kd to pKd
# Kd is in nM, pKd = 9 - log10(Kd_nM)
affinity_df['pKd'] = 9 - np.log10(affinity_df['Kd (nM)'])
print(f"  pKd range: {affinity_df['pKd'].min():.2f} - {affinity_df['pKd'].max():.2f}")

# Count by pKd range
print("\n  pKd distribution:")
bins = [(4, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 14)]
for low, high in bins:
    count = ((affinity_df['pKd'] >= low) & (affinity_df['pKd'] < high)).sum()
    if count > 0:
        print(f"    [{low}-{high}): {count}")

# =============================================================================
# Step 2: Load case information (antibody/antigen info)
# =============================================================================
print("\nStep 2: Loading case information...")
cases_file = BENCHMARK_DIR / "antibody_antigen_cases.txt"
cases_df = pd.read_csv(cases_file, sep='\t')
print(f"  Total cases: {len(cases_df)}")

# =============================================================================
# Step 3: Extract sequences from PDB files
# =============================================================================
print("\nStep 3: Extracting sequences from PDB files...")

def extract_sequence_from_pdb(pdb_file):
    """Extract amino acid sequence from PDB file."""
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    }

    sequences = {}
    current_chain = None
    last_resnum = None

    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    chain = line[21]
                    resname = line[17:20].strip()
                    resnum = int(line[22:26].strip())

                    if chain not in sequences:
                        sequences[chain] = []
                        last_resnum = None

                    if resnum != last_resnum:
                        if resname in three_to_one:
                            sequences[chain].append(three_to_one[resname])
                        last_resnum = resnum
    except Exception as e:
        return {}

    return {k: ''.join(v) for k, v in sequences.items()}

pdbs_dir = BENCHMARK_DIR / "pdbs"
processed_data = []

for _, case in cases_df.iterrows():
    pdb_code = case['Complex PDB'].split('_')[0]

    # Find matching affinity
    affinity_match = affinity_df[affinity_df['Case'] == pdb_code]
    if len(affinity_match) == 0:
        continue

    kd = affinity_match['Kd (nM)'].values[0]
    pkd = affinity_match['pKd'].values[0]

    # Get antibody bound structure (contains both Ab and Ag)
    complex_pdb = pdbs_dir / f"{pdb_code}_l_b.pdb"
    antigen_pdb = pdbs_dir / f"{pdb_code}_r_b.pdb"

    if not complex_pdb.exists():
        continue

    # Extract sequences
    ab_seqs = extract_sequence_from_pdb(complex_pdb)
    ag_seqs = extract_sequence_from_pdb(antigen_pdb) if antigen_pdb.exists() else {}

    if not ab_seqs or not ag_seqs:
        continue

    # Combine antibody chains (usually H and L, or A and B)
    ab_sequence = ''.join(ab_seqs.values())
    ag_sequence = ''.join(ag_seqs.values())

    if len(ab_sequence) >= 50 and len(ag_sequence) >= 10:
        processed_data.append({
            'antibody_sequence': ab_sequence,
            'antigen_sequence': ag_sequence,
            'pKd': pkd,
            'source': 'PierceLab',
            'pdb_code': pdb_code,
            'antigen_name': case['Antigen']
        })

print(f"  Successfully processed: {len(processed_data)} entries")

# =============================================================================
# Step 4: Create DataFrame and analyze
# =============================================================================
print("\nStep 4: Analyzing processed data...")

if processed_data:
    pierce_df = pd.DataFrame(processed_data)

    print(f"  Total entries: {len(pierce_df)}")
    print(f"  pKd range: {pierce_df['pKd'].min():.2f} - {pierce_df['pKd'].max():.2f}")

    print("\n  pKd distribution:")
    for low, high in bins:
        count = ((pierce_df['pKd'] >= low) & (pierce_df['pKd'] < high)).sum()
        if count > 0:
            print(f"    [{low}-{high}): {count}")

    # High affinity count
    high_aff = (pierce_df['pKd'] >= 9).sum()
    ultra_high = (pierce_df['pKd'] >= 10).sum()
    print(f"\n  High affinity (pKd >= 9): {high_aff}")
    print(f"  Ultra high affinity (pKd >= 10): {ultra_high}")

    # Unique antigens
    print(f"\n  Unique antigens: {pierce_df['antigen_name'].nunique()}")
    print("  Antigen names:")
    for ag in pierce_df['antigen_name'].unique()[:10]:
        print(f"    - {ag}")
    if len(pierce_df['antigen_name'].unique()) > 10:
        print(f"    ... and {len(pierce_df['antigen_name'].unique()) - 10} more")

    # Save
    output_file = OUTPUT_DIR / "pierce_lab_processed.csv"
    pierce_df[['antibody_sequence', 'antigen_sequence', 'pKd', 'source']].to_csv(output_file, index=False)
    print(f"\n  Saved to: {output_file}")
else:
    print("  WARNING: No data processed!")

print("\n" + "="*70)
print("DONE")
print("="*70)
