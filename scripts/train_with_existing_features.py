#!/usr/bin/env python3
"""
Train Model with Existing Features Only

This script trains ONLY on samples that already have ESM2 embeddings,
avoiding the need to generate new embeddings.

This allows immediate training while your GPU is busy with other tasks.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def filter_samples_with_features(input_file, output_file):
    """Filter dataset to only samples with ESM2 features"""

    print("="*80)
    print("FILTERING SAMPLES WITH EXISTING ESM2 FEATURES")
    print("="*80)

    # Load full dataset
    print(f"\n[INFO] Loading full dataset: {input_file}")
    df = pd.read_csv(input_file)
    print(f"[OK] Loaded {len(df)} samples")

    # Check for ESM2 features
    esm2_cols = [col for col in df.columns if col.startswith('esm2_pca_')]
    print(f"\n[INFO] Found {len(esm2_cols)} ESM2 PCA feature columns")

    # Filter for samples with features (non-NaN in first ESM2 column)
    if esm2_cols:
        first_esm2_col = esm2_cols[0]
        df_with_features = df[df[first_esm2_col].notna()].copy()
        df_without_features = df[df[first_esm2_col].isna()].copy()

        print(f"\n[INFO] Samples WITH features: {len(df_with_features)}")
        print(f"[INFO] Samples WITHOUT features: {len(df_without_features)}")

        # Show distribution for both
        bins = [0, 5, 7, 9, 11, 16]
        labels = ['very_weak', 'weak', 'moderate', 'strong', 'very_strong']

        df_with_features['affinity_bin'] = pd.cut(
            df_with_features['pKd'], bins=bins, labels=labels, include_lowest=True
        )
        df_without_features['affinity_bin'] = pd.cut(
            df_without_features['pKd'], bins=bins, labels=labels, include_lowest=True
        )

        print("\n" + "-"*80)
        print("AFFINITY DISTRIBUTION COMPARISON")
        print("-"*80)
        print(f"\n{'Bin':<15} | {'With Features':<20} | {'Without Features':<20}")
        print("-"*80)

        for label in labels:
            count_with = (df_with_features['affinity_bin'] == label).sum()
            count_without = (df_without_features['affinity_bin'] == label).sum()
            pct_with = count_with / len(df_with_features) * 100 if len(df_with_features) > 0 else 0
            pct_without = count_without / len(df_without_features) * 100 if len(df_without_features) > 0 else 0

            print(f"{label:<15} | {count_with:6d} ({pct_with:5.2f}%) | {count_without:6d} ({pct_without:5.2f}%)")

        print("-"*80)

        # Highlight very strong
        vs_with = (df_with_features['affinity_bin'] == 'very_strong').sum()
        vs_without = (df_without_features['affinity_bin'] == 'very_strong').sum()

        print(f"\n[INFO] Very strong binders:")
        print(f"       With features:    {vs_with}")
        print(f"       Without features: {vs_without}")
        print(f"       Total:            {vs_with + vs_without}")

        # Save filtered dataset
        print(f"\n[INFO] Saving filtered dataset with features only...")
        df_with_features.to_csv(output_file, index=False)

        import os
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"[OK] Saved to: {output_file}")
        print(f"     Size: {size_mb:.2f} MB")
        print(f"     Samples: {len(df_with_features):,}")

        return df_with_features

    else:
        print("[ERROR] No ESM2 feature columns found!")
        return None

def main():
    """Main execution"""

    # Input/output files
    input_file = Path("external_data/merged_with_therapeutics.csv")
    output_file = Path("external_data/train_ready_with_features.csv")

    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_file}")
        return None

    # Filter samples
    df_filtered = filter_samples_with_features(input_file, output_file)

    if df_filtered is not None:
        print("\n" + "="*80)
        print("READY FOR TRAINING")
        print("="*80)
        print(f"\nYou can now train with existing features using:")
        print(f"\n  python train_balanced.py \\")
        print(f"    --data {output_file} \\")
        print(f"    --loss weighted_mse \\")
        print(f"    --sampling stratified \\")
        print(f"    --epochs 100 \\")
        print(f"    --batch_size 32")
        print(f"\nThis will train on {len(df_filtered):,} samples with complete features.")
        print(f"GPU resources will not be used for embedding generation.")

        # Summary
        print("\n" + "="*80)
        print("WHAT THIS APPROACH DOES")
        print("="*80)
        print("\n[+] Trains immediately without waiting for embeddings")
        print("[+] No GPU resource conflict with your other training")
        print("[+] Uses your original 205k dataset with improved methods")
        print("[+] Stratified sampling ensures extremes in every batch")
        print("[+] Class weights prioritize rare very weak/strong cases")
        print("\n[-] Doesn't use the 185k AbBiBench samples (no features yet)")
        print("[-] Doesn't use the 53 therapeutic antibodies (no features yet)")
        print("\n[INFO] You can generate embeddings later and retrain with full dataset")

    return df_filtered

if __name__ == "__main__":
    result = main()
    sys.exit(0 if result is not None else 1)
