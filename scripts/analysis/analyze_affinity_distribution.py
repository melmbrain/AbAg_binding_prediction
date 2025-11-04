import pandas as pd
import numpy as np
import os
from pathlib import Path

# Base paths
docking_pred_path = Path("/mnt/c/Users/401-24/Desktop/Docking prediction")
data_path = docking_pred_path / "data"

print("="*80)
print("ANALYZING AFFINITY DISTRIBUTION IN CURRENT DATASETS")
print("="*80)

# Key datasets to analyze
datasets = {
    "Phase6 Final 205k": data_path / "processed/phase6/final_205k_dataset.csv",
    "Phase6 Final 19k": data_path / "processed/phase6/final_19k_dataset.csv",
    "AgAb Phase2 Full": data_path / "raw/agab/agab_phase2_full.csv",
    "SAbDab with Affinity": data_path / "raw/sabdab/sabdab_with_affinity.csv",
}

def analyze_affinity(df, name):
    """Analyze affinity distribution in a dataset"""
    affinity_cols = ['affinity', 'Affinity', 'affinity_value', 'delta_g', 'binding_affinity', 'kd', 'Kd']
    affinity_col = None

    for col in affinity_cols:
        if col in df.columns:
            affinity_col = col
            break

    if affinity_col is None:
        print(f"\n{name}: No affinity column found")
        print(f"  Available columns: {', '.join(df.columns[:10])}")
        return None

    # Filter out NaN values
    affinity_data = df[affinity_col].dropna()

    if len(affinity_data) == 0:
        print(f"\n{name}: No non-NaN affinity values")
        return None

    print(f"\n{name}:")
    print(f"  Affinity column: {affinity_col}")
    print(f"  Total samples: {len(df)}")
    print(f"  Samples with affinity: {len(affinity_data)}")
    print(f"  Affinity range: [{affinity_data.min():.2f}, {affinity_data.max():.2f}]")
    print(f"  Mean: {affinity_data.mean():.2f}")
    print(f"  Median: {affinity_data.median():.2f}")
    print(f"  Std: {affinity_data.std():.2f}")

    # Distribution by ranges
    print(f"\n  Distribution:")
    if affinity_data.min() < 0:  # Likely log-scale or ΔG
        ranges = [(-np.inf, -12), (-12, -10), (-10, -8), (-8, -6), (-6, -4), (-4, 0), (0, np.inf)]
        labels = ["< -12 (very strong)", "-12 to -10 (strong)", "-10 to -8 (moderate-strong)",
                  "-8 to -6 (moderate)", "-6 to -4 (weak-moderate)", "-4 to 0 (weak)", "> 0 (very weak)"]
    else:  # Likely Kd in nM or μM
        ranges = [(0, 1), (1, 10), (10, 100), (100, 1000), (1000, 10000), (10000, np.inf)]
        labels = ["0-1 nM (very strong)", "1-10 nM (strong)", "10-100 nM (moderate-strong)",
                  "100-1000 nM (moderate)", "1-10 μM (weak)", "> 10 μM (very weak)"]

    for (low, high), label in zip(ranges, labels):
        count = ((affinity_data > low) & (affinity_data <= high)).sum()
        pct = 100 * count / len(affinity_data)
        print(f"    {label}: {count} ({pct:.1f}%)")

    return affinity_data

print("\n" + "="*80)
print("CURRENT DATASETS")
print("="*80)

for name, path in datasets.items():
    if path.exists():
        try:
            df = pd.read_csv(path, nrows=100000)  # Limit rows for speed
            analyze_affinity(df, name)
        except Exception as e:
            print(f"\n{name}: Error reading file - {e}")
    else:
        print(f"\n{name}: File not found")

print("\n" + "="*80)
print("AFFINITY DATABASES (POTENTIAL ADDITIONAL DATA)")
print("="*80)

# Check affinity databases
affinity_databases = {
    "PDBbind": data_path / "raw/affinity_databases/pdbind/pdbind_affinity_data.csv",
    "PPB-Affinity": data_path / "raw/affinity_databases/ppb_affinity/PPB-Affinity.csv",
    "SAAINT": data_path / "raw/affinity_databases/saaint/saaintdb_affinity.tsv",
    "SAbDab Affinity": data_path / "raw/affinity_databases/sabdab/sabdab_affinity_data.csv",
    "SKEMPI2": data_path / "raw/affinity_databases/skempi2.csv",
}

for name, path in affinity_databases.items():
    if path.exists():
        try:
            # Read with appropriate separator
            sep = '\t' if path.suffix == '.tsv' else ','
            df = pd.read_csv(path, sep=sep)
            analyze_affinity(df, name)
        except Exception as e:
            print(f"\n{name}: Error reading file - {e}")
    else:
        print(f"\n{name}: File not found at {path}")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("\nBased on the analysis above, I'll identify databases with:")
print("1. Strong/very strong binders (to improve model on high affinity)")
print("2. Weak/very weak binders (to improve model on low affinity)")
print("3. Antibody-antigen specific data (if not already included)")
