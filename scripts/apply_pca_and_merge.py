#!/usr/bin/env python3
"""
Apply PCA to New Embeddings and Merge with Dataset

This script:
1. Loads new ESM2 embeddings (1,280 dimensions)
2. Applies PCA to reduce to 150 dimensions
3. Merges with existing dataset using embedding indices
4. Creates final training-ready dataset
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import sys
import time

def apply_pca_and_merge():
    """Apply PCA to new embeddings and merge with dataset"""

    print("="*80)
    print("APPLYING PCA AND MERGING EMBEDDINGS")
    print("="*80)

    # File paths
    embeddings_file = Path("external_data/new_embeddings.npy")
    indices_file = Path("external_data/new_embedding_indices.npy")
    dataset_file = Path("external_data/merged_with_therapeutics.csv")
    output_file = Path("external_data/merged_with_all_features.csv")

    # Check files exist
    for file in [embeddings_file, indices_file, dataset_file]:
        if not file.exists():
            print(f"[ERROR] Required file not found: {file}")
            return False

    # Step 1: Load embeddings
    print(f"\n[1/5] Loading new embeddings...")
    start_time = time.time()
    embeddings = np.load(embeddings_file)
    indices = np.load(indices_file)
    print(f"[OK] Loaded embeddings: {embeddings.shape}")
    print(f"[OK] Loaded indices: {len(indices)}")
    print(f"      Time: {time.time() - start_time:.1f}s")

    # Step 2: Apply PCA
    print(f"\n[2/5] Applying PCA (1,280 -> 150 dimensions)...")
    start_time = time.time()
    pca = PCA(n_components=150, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    print(f"[OK] PCA complete: {embeddings_pca.shape}")
    print(f"      Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"      Time: {time.time() - start_time:.1f}s")

    # Step 3: Load dataset
    print(f"\n[3/5] Loading dataset...")
    start_time = time.time()
    df = pd.read_csv(dataset_file, low_memory=False)
    print(f"[OK] Loaded dataset: {len(df)} rows")
    print(f"      Time: {time.time() - start_time:.1f}s")

    # Step 4: Merge embeddings with dataset
    print(f"\n[4/5] Merging embeddings with dataset...")
    start_time = time.time()

    # Get PCA column names
    pca_cols = [f'esm2_pca_{i}' for i in range(150)]

    # Count existing features
    samples_with_features_before = df[pca_cols[0]].notna().sum()
    print(f"      Samples with features before: {samples_with_features_before:,}")
    print(f"      Samples without features: {len(df) - samples_with_features_before:,}")

    # Fill in new embeddings using indices
    for i, idx in enumerate(indices):
        if i % 10000 == 0:
            print(f"      Progress: {i:,}/{len(indices):,} ({i/len(indices)*100:.1f}%)")

        # Update PCA features for this row
        for j, col in enumerate(pca_cols):
            df.at[idx, col] = embeddings_pca[i, j]

    print(f"      Progress: {len(indices):,}/{len(indices):,} (100.0%)")

    # Count features after merge
    samples_with_features_after = df[pca_cols[0]].notna().sum()
    print(f"      Samples with features after: {samples_with_features_after:,}")
    print(f"      New features added: {samples_with_features_after - samples_with_features_before:,}")
    print(f"      Time: {time.time() - start_time:.1f}s")

    # Step 5: Save merged dataset
    print(f"\n[5/5] Saving merged dataset...")
    start_time = time.time()
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
        with_features = df[(df['affinity_bin'] == label) & (df[pca_cols[0]].notna())].shape[0]
        print(f"{label:<15} | {total:6,} ({total/len(df)*100:5.2f}%) | {with_features:6,}")

    print("-"*50)
    print(f"{'Total':<15} | {len(df):6,} ({100.0:5.2f}%) | {samples_with_features_after:6,}")

    # Very strong binders
    very_strong_total = (df['affinity_bin'] == 'very_strong').sum()
    very_strong_with_features = df[(df['affinity_bin'] == 'very_strong') & (df[pca_cols[0]].notna())].shape[0]

    print(f"\nVery Strong Binders (pKd > 11):")
    print(f"  Total: {very_strong_total}")
    print(f"  With features: {very_strong_with_features}")

    # Dataset completeness
    completeness = samples_with_features_after / len(df) * 100
    print(f"\nDataset Completeness: {completeness:.2f}%")

    print("\n" + "="*80)
    print("READY FOR TRAINING!")
    print("="*80)
    print(f"\nNext step - Train the model:")
    print(f"\n  python train_balanced.py \\")
    print(f"    --data {output_file} \\")
    print(f"    --loss weighted_mse \\")
    print(f"    --sampling stratified \\")
    print(f"    --epochs 20 \\")
    print(f"    --batch_size 64")

    print(f"\nEstimated training time: 3-3.5 hours (20 epochs)")
    print(f"For full training (100 epochs): ~14 hours")

    return True

def main():
    """Main execution"""

    start_total = time.time()

    success = apply_pca_and_merge()

    total_time = time.time() - start_total

    if success:
        print(f"\n{'='*80}")
        print(f"TOTAL TIME: {total_time/60:.1f} minutes")
        print(f"{'='*80}\n")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
