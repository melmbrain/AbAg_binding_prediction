#!/usr/bin/env python3
"""
Integrate Therapeutic Antibody Data with Existing Dataset

Merges high-affinity therapeutic/vaccine antibody data from:
- Ab-CoV
- Thera-SAbDab
- SAbDab (with affinity filter)
- CoV-AbDab

Focus: Boosting very strong binders (pKd > 11)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def load_existing_data(path):
    """Load existing merged dataset"""
    print("="*80)
    print("LOADING EXISTING DATASET")
    print("="*80)

    df = pd.read_csv(path)
    print(f"[OK] Loaded {len(df)} existing samples")

    # Check affinity distribution
    bins = [0, 5, 7, 9, 11, 16]
    labels = ['very_weak', 'weak', 'moderate', 'strong', 'very_strong']
    df['affinity_bin'] = pd.cut(df['pKd'], bins=bins, labels=labels, include_lowest=True)

    print("\nExisting Affinity Distribution:")
    for label in labels:
        count = (df['affinity_bin'] == label).sum()
        pct = count / len(df) * 100
        print(f"  {label:12s}: {count:6d} ({pct:5.2f}%)")

    return df

def load_therapeutic_data():
    """Load all therapeutic antibody datasets"""
    print("\n" + "="*80)
    print("LOADING THERAPEUTIC ANTIBODY DATA")
    print("="*80)

    therapeutic_dir = Path("external_data/therapeutic")
    datasets = {}

    # Ab-CoV
    abcov_file = therapeutic_dir / "abcov_processed.csv"
    if abcov_file.exists():
        try:
            df = pd.read_csv(abcov_file)
            print(f"\n[OK] Ab-CoV: {len(df)} entries")
            if 'pKd' in df.columns:
                very_strong = (df['pKd'] > 11).sum()
                print(f"     Very strong (pKd > 11): {very_strong}")
            datasets['abcov'] = df
        except Exception as e:
            print(f"[ERROR] Failed to load Ab-CoV: {e}")

    # Thera-SAbDab
    thera_file = therapeutic_dir / "therasabdab_processed.csv"
    if thera_file.exists():
        try:
            df = pd.read_csv(thera_file)
            print(f"\n[OK] Thera-SAbDab: {len(df)} entries")
            datasets['therasabdab'] = df
        except Exception as e:
            print(f"[ERROR] Failed to load Thera-SAbDab: {e}")

    # SAbDab with affinity
    sabdab_file = therapeutic_dir / "sabdab_affinity.tsv"
    if sabdab_file.exists():
        try:
            df = pd.read_csv(sabdab_file, sep='\t')
            print(f"\n[OK] SAbDab: {len(df)} entries")
            datasets['sabdab'] = df
        except Exception as e:
            print(f"[ERROR] Failed to load SAbDab: {e}")

    # CoV-AbDab
    covabdab_file = therapeutic_dir / "covabdab_processed.csv"
    if covabdab_file.exists():
        try:
            df = pd.read_csv(covabdab_file)
            print(f"\n[OK] CoV-AbDab: {len(df)} entries")
            datasets['covabdab'] = df
        except Exception as e:
            print(f"[ERROR] Failed to load CoV-AbDab: {e}")

    if not datasets:
        print("\n[WARNING] No therapeutic antibody datasets found")
        print("\n[INFO] Please run download scripts first:")
        print("       python scripts/download_abcov.py")
        print("       python scripts/download_therapeutic_antibodies.py")
        return None

    return datasets

def standardize_columns(df, source_name):
    """Standardize column names across different databases"""
    print(f"\n[INFO] Standardizing columns for {source_name}...")

    # Create mapping for common variations
    column_map = {
        # Sequences
        'VH': 'heavy_chain_seq',
        'VL': 'light_chain_seq',
        'heavy': 'heavy_chain_seq',
        'light': 'light_chain_seq',
        'H_sequence': 'heavy_chain_seq',
        'L_sequence': 'light_chain_seq',
        'Hchain': 'heavy_chain_seq',
        'Lchain': 'light_chain_seq',

        # Affinity
        'Kd': 'kd_value',
        'KD': 'kd_value',
        'affinity': 'kd_value',
        'binding_affinity': 'kd_value',

        # Names
        'antibody': 'antibody_name',
        'Ab_name': 'antibody_name',
        'antigen': 'antigen_name',
        'Ag_name': 'antigen_name',
        'target': 'antigen_name',

        # PDB
        'pdb': 'pdb_code',
        'PDB': 'pdb_code',
        'pdb_id': 'pdb_code',
    }

    # Apply mapping
    df_std = df.rename(columns=column_map)

    # Add source
    df_std['source'] = source_name

    return df_std

def extract_very_strong_binders(datasets):
    """Extract only very strong binders (pKd > 11) from all sources"""
    print("\n" + "="*80)
    print("EXTRACTING VERY STRONG BINDERS (pKd > 11)")
    print("="*80)

    all_strong = []

    for source_name, df in datasets.items():
        df_std = standardize_columns(df, source_name)

        # Check if pKd exists
        if 'pKd' not in df_std.columns:
            print(f"\n[WARNING] {source_name}: No pKd column, skipping")
            continue

        # Filter for very strong
        df_strong = df_std[df_std['pKd'] > 11].copy()

        if len(df_strong) > 0:
            print(f"\n[OK] {source_name}: {len(df_strong)} very strong binders")
            print(f"     pKd range: {df_strong['pKd'].min():.2f} - {df_strong['pKd'].max():.2f}")

            all_strong.append(df_strong)
        else:
            print(f"\n[INFO] {source_name}: No very strong binders found")

    if not all_strong:
        print("\n[WARNING] No very strong binders found in any dataset")
        return None

    # Combine all
    df_combined = pd.concat(all_strong, ignore_index=True)
    print(f"\n[OK] Combined: {len(df_combined)} very strong binders from all sources")

    return df_combined

def remove_duplicates(df_new, df_existing):
    """Remove duplicates based on PDB code and sequences"""
    print("\n" + "="*80)
    print("REMOVING DUPLICATES")
    print("="*80)

    initial_count = len(df_new)

    # Check PDB code duplicates
    if 'pdb_code' in df_new.columns and 'pdb_code' in df_existing.columns:
        existing_pdbs = set(df_existing['pdb_code'].dropna().str.lower())
        pdb_duplicates = df_new['pdb_code'].notna() & df_new['pdb_code'].str.lower().isin(existing_pdbs)
        pdb_dup_count = pdb_duplicates.sum()

        if pdb_dup_count > 0:
            print(f"\n[INFO] Found {pdb_dup_count} PDB code duplicates")
            df_new = df_new[~pdb_duplicates].copy()

    # Check sequence duplicates (for entries without PDB codes)
    if 'heavy_chain_seq' in df_new.columns and 'heavy_chain_seq' in df_existing.columns:
        # Create sequence fingerprints
        df_new['seq_fp'] = df_new['heavy_chain_seq'].fillna('') + '||' + df_new.get('light_chain_seq', '').fillna('')
        df_existing['seq_fp'] = df_existing['heavy_chain_seq'].fillna('') + '||' + df_existing.get('light_chain_seq', '').fillna('')

        existing_seqs = set(df_existing['seq_fp'])
        seq_duplicates = df_new['seq_fp'].isin(existing_seqs)
        seq_dup_count = seq_duplicates.sum()

        if seq_dup_count > 0:
            print(f"[INFO] Found {seq_dup_count} sequence duplicates")
            df_new = df_new[~seq_duplicates].copy()

        # Clean up temporary column
        df_new = df_new.drop(columns=['seq_fp'])

    final_count = len(df_new)
    removed = initial_count - final_count

    print(f"\n[OK] Removed {removed} duplicates")
    print(f"[OK] Remaining unique samples: {final_count}")

    return df_new

def align_features(df_new, df_existing):
    """Align feature columns with existing dataset"""
    print("\n" + "="*80)
    print("ALIGNING FEATURES")
    print("="*80)

    # Get ESM2 PCA columns from existing
    esm2_cols = [col for col in df_existing.columns if col.startswith('esm2_pca_')]
    print(f"\n[INFO] Found {len(esm2_cols)} ESM2 PCA feature columns")

    # Add missing ESM2 columns with NaN
    for col in esm2_cols:
        if col not in df_new.columns:
            df_new[col] = np.nan

    print(f"[OK] Added {len(esm2_cols)} ESM2 feature columns (NaN values)")
    print("[INFO] ESM2 embeddings will need to be generated for these new samples")

    # Ensure all required columns exist
    required_cols = ['pdb_code', 'pKd', 'heavy_chain_seq', 'light_chain_seq', 'source']
    for col in required_cols:
        if col not in df_new.columns:
            df_new[col] = np.nan if col != 'source' else 'therapeutic'

    return df_new

def merge_datasets(df_existing, df_new):
    """Merge new therapeutic data with existing dataset"""
    print("\n" + "="*80)
    print("MERGING DATASETS")
    print("="*80)

    # Get common columns
    common_cols = list(set(df_existing.columns) & set(df_new.columns))
    print(f"\n[INFO] Common columns: {len(common_cols)}")

    # Merge on common columns
    df_merged = pd.concat([df_existing, df_new[common_cols]], ignore_index=True)

    print(f"\n[OK] Merged dataset: {len(df_merged)} total samples")
    print(f"    Existing: {len(df_existing)}")
    print(f"    New:      {len(df_new)}")

    return df_merged

def analyze_improvement(df_before, df_after):
    """Analyze improvement in extreme affinity coverage"""
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)

    bins = [0, 5, 7, 9, 11, 16]
    labels = ['very_weak', 'weak', 'moderate', 'strong', 'very_strong']

    df_before['affinity_bin'] = pd.cut(df_before['pKd'], bins=bins, labels=labels, include_lowest=True)
    df_after['affinity_bin'] = pd.cut(df_after['pKd'], bins=bins, labels=labels, include_lowest=True)

    print("\n" + "-"*80)
    print("AFFINITY DISTRIBUTION COMPARISON")
    print("-"*80)
    print(f"\n{'Bin':<15} | {'Before':<15} | {'After':<15} | {'Change':<15}")
    print("-"*80)

    for label in labels:
        count_before = (df_before['affinity_bin'] == label).sum()
        count_after = (df_after['affinity_bin'] == label).sum()
        change = count_after - count_before
        pct_change = (change / count_before * 100) if count_before > 0 else 0

        print(f"{label:<15} | {count_before:6d} ({count_before/len(df_before)*100:5.2f}%) | "
              f"{count_after:6d} ({count_after/len(df_after)*100:5.2f}%) | "
              f"{change:+6d} ({pct_change:+6.1f}%)")

    print("-"*80)

    # Highlight very strong improvement
    vs_before = (df_before['affinity_bin'] == 'very_strong').sum()
    vs_after = (df_after['affinity_bin'] == 'very_strong').sum()
    vs_increase = vs_after - vs_before
    vs_pct = (vs_increase / vs_before * 100) if vs_before > 0 else 0

    print(f"\n[OK] VERY STRONG BINDERS (pKd > 11):")
    print(f"     Before: {vs_before}")
    print(f"     After:  {vs_after}")
    print(f"     Added:  +{vs_increase} ({vs_pct:+.1f}%)")

def main():
    """Main integration workflow"""
    print("="*80)
    print("THERAPEUTIC ANTIBODY INTEGRATION")
    print("="*80)
    print("\nIntegrating high-affinity therapeutic/vaccine antibodies")
    print("Target: Boost very strong binders (pKd > 11)")
    print("")

    # Load existing data
    existing_file = Path("external_data/merged_with_abbibench.csv")
    if not existing_file.exists():
        print(f"[ERROR] Existing dataset not found: {existing_file}")
        print("[INFO] Please ensure merged_with_abbibench.csv exists")
        return None

    df_existing = load_existing_data(existing_file)

    # Load therapeutic data
    datasets = load_therapeutic_data()
    if not datasets:
        return None

    # Extract very strong binders
    df_strong = extract_very_strong_binders(datasets)
    if df_strong is None or len(df_strong) == 0:
        print("\n[WARNING] No very strong binders to integrate")
        return None

    # Remove duplicates
    df_unique = remove_duplicates(df_strong, df_existing)
    if len(df_unique) == 0:
        print("\n[WARNING] All therapeutic antibodies are duplicates")
        return None

    # Align features
    df_aligned = align_features(df_unique, df_existing)

    # Merge
    df_merged = merge_datasets(df_existing, df_aligned)

    # Analyze improvement
    analyze_improvement(df_existing, df_merged)

    # Save
    output_file = Path("external_data/merged_with_therapeutics.csv")
    df_merged.to_csv(output_file, index=False)

    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"\n[OK] Saved merged dataset to: {output_file}")
    print(f"     Size: {size_mb:.2f} MB")
    print(f"     Samples: {len(df_merged)}")

    # Save summary
    summary_file = Path("external_data/therapeutic_integration_report.txt")
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("THERAPEUTIC ANTIBODY INTEGRATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {pd.Timestamp.now()}\n\n")
        f.write(f"Existing dataset: {len(df_existing)} samples\n")
        f.write(f"Therapeutic data added: {len(df_aligned)} samples\n")
        f.write(f"Final dataset: {len(df_merged)} samples\n\n")

        bins = [0, 5, 7, 9, 11, 16]
        labels = ['very_weak', 'weak', 'moderate', 'strong', 'very_strong']
        df_merged['affinity_bin'] = pd.cut(df_merged['pKd'], bins=bins, labels=labels, include_lowest=True)

        f.write("Affinity Distribution:\n")
        for label in labels:
            count = (df_merged['affinity_bin'] == label).sum()
            pct = count / len(df_merged) * 100
            f.write(f"  {label:<15}: {count:6d} ({pct:5.2f}%)\n")

    print(f"[OK] Saved summary to: {summary_file}")

    return df_merged

if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0 if result is not None else 1)
    except KeyboardInterrupt:
        print("\n\n[INFO] Integration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
