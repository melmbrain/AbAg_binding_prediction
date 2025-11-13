#!/usr/bin/env python3
"""
Prepare Full-Dimensional Features (NO PCA) for Colab Pro Training

This script:
1. Loads ESM2 embeddings (1,280 dimensions)
2. SKIPS PCA - keeps all 1,280 dimensions
3. Merges with existing dataset
4. Creates training-ready dataset for full-dimensional training

Requirements:
- ~4-6 GB RAM for loading embeddings
- Output file will be ~800-1000 MB
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
import argparse

def prepare_full_dimensional_features(embeddings_file, indices_file, dataset_file, output_file):
    """Prepare full-dimensional features without PCA"""

    print("="*80)
    print("PREPARING FULL-DIMENSIONAL FEATURES (1,280 DIMS - NO PCA)")
    print("="*80)

    # Check files exist
    for file in [embeddings_file, indices_file, dataset_file]:
        if not Path(file).exists():
            print(f"[ERROR] Required file not found: {file}")
            return False

    # Step 1: Load embeddings
    print(f"\n[1/4] Loading ESM2 embeddings...")
    start_time = time.time()
    embeddings = np.load(embeddings_file)
    indices = np.load(indices_file)
    print(f"[OK] Loaded embeddings: {embeddings.shape}")
    print(f"[OK] Loaded indices: {len(indices)}")
    print(f"      Memory: ~{embeddings.nbytes / (1024**3):.2f} GB")
    print(f"      Time: {time.time() - start_time:.1f}s")

    # Verify dimensions
    if embeddings.shape[1] != 1280:
        print(f"[ERROR] Expected 1,280 dimensions, got {embeddings.shape[1]}")
        return False

    print(f"\n[INFO] Using FULL {embeddings.shape[1]} dimensions (NO PCA)")
    print(f"       This will preserve 100% of variance!")

    # Step 2: Load dataset
    print(f"\n[2/4] Loading dataset...")
    start_time = time.time()
    df = pd.read_csv(dataset_file, low_memory=False)
    print(f"[OK] Loaded dataset: {len(df):,} rows")
    print(f"      Time: {time.time() - start_time:.1f}s")

    # Step 3: Merge embeddings with dataset
    print(f"\n[3/4] Merging full-dimensional embeddings with dataset...")
    start_time = time.time()

    # Get column names for 1,280 dimensions
    full_dim_cols = [f'esm2_dim_{i}' for i in range(1280)]

    # Create empty columns if they don't exist
    for col in full_dim_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Count existing features
    samples_with_features_before = df[full_dim_cols[0]].notna().sum()
    print(f"      Samples with features before: {samples_with_features_before:,}")
    print(f"      Samples without features: {len(df) - samples_with_features_before:,}")

    # Fill in embeddings using indices
    print(f"      Filling in {len(indices):,} samples...")

    # Efficient batch update
    for i, idx in enumerate(indices):
        if i % 10000 == 0 and i > 0:
            print(f"      Progress: {i:,}/{len(indices):,} ({i/len(indices)*100:.1f}%)")

        # Update all 1,280 features for this row
        df.loc[idx, full_dim_cols] = embeddings[i]

    print(f"      Progress: {len(indices):,}/{len(indices):,} (100.0%)")

    # Count features after merge
    samples_with_features_after = df[full_dim_cols[0]].notna().sum()
    print(f"      Samples with features after: {samples_with_features_after:,}")
    print(f"      New features added: {samples_with_features_after - samples_with_features_before:,}")
    print(f"      Time: {time.time() - start_time:.1f}s")

    # Step 4: Save dataset with full features
    print(f"\n[4/4] Saving dataset with full-dimensional features...")
    start_time = time.time()

    # Save with compression to reduce file size
    df.to_csv(output_file, index=False)

    import os
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"[OK] Saved to: {output_file}")
    print(f"      Size: {size_mb:.1f} MB")
    print(f"      Time: {time.time() - start_time:.1f}s")

    # Final statistics
    print("\n" + "="*80)
    print("FINAL DATASET STATISTICS")
    print("="*80)

    # Affinity distribution
    bins = [0, 5, 7, 9, 11, 16]
    labels = ['very_weak', 'weak', 'moderate', 'strong', 'very_strong']
    df['affinity_bin'] = pd.cut(df['pKd'], bins=bins, labels=labels, include_lowest=True)

    print("\nAffinity Distribution:")
    print(f"{'Bin':<15} | {'Total':<15} | {'With Features':<15}")
    print("-"*50)

    for label in labels:
        total = (df['affinity_bin'] == label).sum()
        with_features = df[(df['affinity_bin'] == label) & (df[full_dim_cols[0]].notna())].shape[0]
        print(f"{label:<15} | {total:6,} ({total/len(df)*100:5.2f}%) | {with_features:6,}")

    print("-"*50)
    print(f"{'Total':<15} | {len(df):6,} ({100.0:5.2f}%) | {samples_with_features_after:6,}")

    # Very strong binders
    very_strong_total = (df['affinity_bin'] == 'very_strong').sum()
    very_strong_with_features = df[(df['affinity_bin'] == 'very_strong') & (df[full_dim_cols[0]].notna())].shape[0]

    print(f"\nVery Strong Binders (pKd > 11):")
    print(f"  Total: {very_strong_total}")
    print(f"  With features: {very_strong_with_features}")

    # Dataset completeness
    completeness = samples_with_features_after / len(df) * 100
    print(f"\nDataset Completeness: {completeness:.2f}%")

    # Feature dimensionality comparison
    print(f"\nFeature Comparison:")
    print(f"  v1/v2: 150 dimensions (PCA, 99.9% variance)")
    print(f"  v3:    1,280 dimensions (FULL, 100% variance)")
    print(f"  Improvement: {1280/150:.1f}x more features!")

    print("\n" + "="*80)
    print("READY FOR COLAB PRO TRAINING!")
    print("="*80)
    print(f"\nUpload this file to Google Drive:")
    print(f"  {output_file}")
    print(f"\nThen use: colab_training_v3_full_dimensions.ipynb")
    print(f"\nRequired GPU: T4 (16GB) or better")
    print(f"Estimated training time: ~12-15 hours (100 epochs)")

    return True

def main():
    """Main execution"""

    parser = argparse.ArgumentParser(description='Prepare full-dimensional features for training')
    parser.add_argument('--embeddings',
                       default='external_data/new_embeddings.npy',
                       help='Path to embeddings file')
    parser.add_argument('--indices',
                       default='external_data/new_embedding_indices.npy',
                       help='Path to indices file')
    parser.add_argument('--dataset',
                       default='external_data/merged_with_therapeutics.csv',
                       help='Path to dataset file')
    parser.add_argument('--output',
                       default='external_data/merged_with_full_features.csv',
                       help='Output file path')

    args = parser.parse_args()

    start_total = time.time()

    success = prepare_full_dimensional_features(
        args.embeddings,
        args.indices,
        args.dataset,
        args.output
    )

    total_time = time.time() - start_total

    if success:
        print(f"\n{'='*80}")
        print(f"TOTAL TIME: {total_time/60:.1f} minutes")
        print(f"{'='*80}\n")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
