"""
Integrate AbBiBench data with existing dataset
Handles duplicate checking, feature alignment, and merging
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_utils import AffinityBinner, print_dataset_statistics


class AbBiBenchIntegrator:
    """
    Integrates AbBiBench data with existing dataset
    """
    def __init__(self, abbibench_path: str, existing_data_path: str):
        """
        Args:
            abbibench_path: Path to downloaded AbBiBench CSV
            existing_data_path: Path to existing dataset CSV
        """
        self.abbibench_path = abbibench_path
        self.existing_data_path = existing_data_path
        self.binner = AffinityBinner()

    def load_abbibench(self):
        """Load AbBiBench data"""
        print("\n" + "="*80)
        print("LOADING ABBIBENCH DATA")
        print("="*80)

        print(f"\nLoading from: {self.abbibench_path}")
        df = pd.read_csv(self.abbibench_path)

        print(f"  Loaded {len(df)} samples")
        print(f"  Columns: {list(df.columns)}")

        # Check for pKd column
        if 'pKd' in df.columns:
            valid_pkd = df['pKd'].notna().sum()
            print(f"  Samples with pKd: {valid_pkd}")
            print_dataset_statistics(df['pKd'].dropna().values,
                                   "AbBiBench", self.binner)
        else:
            print("  WARNING No pKd column found")

        return df

    def load_existing(self):
        """Load existing dataset"""
        print("\n" + "="*80)
        print("LOADING EXISTING DATASET")
        print("="*80)

        print(f"\nLoading from: {self.existing_data_path}")
        df = pd.read_csv(self.existing_data_path)

        print(f"  Loaded {len(df)} samples")
        print(f"  Columns: {list(df.columns)[:10]}...")

        if 'pKd' in df.columns:
            print_dataset_statistics(df['pKd'].values,
                                   "Existing Dataset", self.binner)

        return df

    def extract_pdb_codes(self, df):
        """
        Extract PDB codes from AbBiBench

        Args:
            df: AbBiBench DataFrame

        Returns:
            DataFrame with pdb_code column
        """
        print("\n" + "="*80)
        print("EXTRACTING PDB CODES")
        print("="*80)

        # Common PDB code column names
        possible_pdb_cols = [
            'pdb_code',
            'pdb',
            'PDB',
            'pdb_id',
            'complex_id',
            'structure_id'
        ]

        pdb_col = None
        for col in possible_pdb_cols:
            if col in df.columns:
                pdb_col = col
                print(f"\nOK Found PDB column: '{pdb_col}'")
                break

        if pdb_col is None:
            print("\nWARNING No PDB code column found")
            print("Available columns:", df.columns.tolist())

            # Try to infer from other columns
            # Sometimes PDB codes are in complex_name or similar
            for col in df.columns:
                if 'complex' in col.lower() or 'name' in col.lower():
                    print(f"\nTrying to extract from: {col}")
                    # Extract 4-character codes (standard PDB format)
                    df['pdb_code'] = df[col].str.extract(r'([0-9][A-Za-z0-9]{3})',
                                                         expand=False)
                    if df['pdb_code'].notna().any():
                        print(f"OK Extracted {df['pdb_code'].notna().sum()} PDB codes")
                        pdb_col = 'pdb_code'
                        break

        if pdb_col and pdb_col != 'pdb_code':
            df['pdb_code'] = df[pdb_col]

        # Standardize PDB codes (lowercase, remove whitespace)
        if 'pdb_code' in df.columns:
            df['pdb_code'] = df['pdb_code'].str.strip().str.lower()
            unique_pdbs = df['pdb_code'].nunique()
            print(f"\n  Total samples: {len(df)}")
            print(f"  Unique PDB codes: {unique_pdbs}")
            print(f"  Samples per PDB (avg): {len(df)/unique_pdbs:.1f}")

        return df

    def check_duplicates(self, abbibench_df, existing_df):
        """
        Check for duplicate PDB codes between datasets

        Args:
            abbibench_df: AbBiBench DataFrame
            existing_df: Existing dataset DataFrame

        Returns:
            Filtered AbBiBench DataFrame without duplicates
        """
        print("\n" + "="*80)
        print("CHECKING FOR DUPLICATES")
        print("="*80)

        if 'pdb_code' not in abbibench_df.columns:
            print("\nWARNING No pdb_code column in AbBiBench")
            print("Cannot check for duplicates")
            return abbibench_df

        if 'pdb_code' not in existing_df.columns:
            print("\nWARNING No pdb_code column in existing dataset")
            print("Cannot check for duplicates")
            return abbibench_df

        # Get PDB codes from both datasets
        existing_pdbs = set(existing_df['pdb_code'].dropna().str.lower())
        abbibench_pdbs = set(abbibench_df['pdb_code'].dropna().str.lower())

        # Find overlaps
        duplicates = existing_pdbs.intersection(abbibench_pdbs)

        print(f"\n  Existing dataset PDBs: {len(existing_pdbs)}")
        print(f"  AbBiBench PDBs: {len(abbibench_pdbs)}")
        print(f"  Duplicate PDBs: {len(duplicates)}")

        if duplicates:
            print(f"\n  Duplicate PDB codes ({len(duplicates)}):")
            dup_list = sorted(list(duplicates))[:20]
            print(f"    {', '.join(dup_list)}")
            if len(duplicates) > 20:
                print(f"    ... and {len(duplicates) - 20} more")

            # Count samples that will be removed
            dup_mask = abbibench_df['pdb_code'].str.lower().isin(duplicates)
            n_dup_samples = dup_mask.sum()

            print(f"\n  Samples to remove: {n_dup_samples} ({n_dup_samples/len(abbibench_df)*100:.1f}%)")

            # Remove duplicates
            abbibench_filtered = abbibench_df[~dup_mask].copy()

            print(f"  Remaining samples: {len(abbibench_filtered)}")

            return abbibench_filtered

        else:
            print("\n  OK No duplicates found! All samples are new.")
            return abbibench_df

    def filter_by_affinity(self, df, min_pkd=None, max_pkd=None,
                          focus_extremes=True):
        """
        Filter data by affinity range

        Args:
            df: DataFrame
            min_pkd: Minimum pKd (optional)
            max_pkd: Maximum pKd (optional)
            focus_extremes: If True, prioritize extreme affinities

        Returns:
            Filtered DataFrame
        """
        print("\n" + "="*80)
        print("FILTERING BY AFFINITY")
        print("="*80)

        if 'pKd' not in df.columns:
            print("\nWARNING No pKd column, skipping filtering")
            return df

        initial_count = len(df)

        # Apply min/max filters
        if min_pkd is not None:
            df = df[df['pKd'] >= min_pkd]
            print(f"\n  Applied min_pkd >= {min_pkd}: {len(df)} samples")

        if max_pkd is not None:
            df = df[df['pKd'] <= max_pkd]
            print(f"  Applied max_pkd <= {max_pkd}: {len(df)} samples")

        # Focus on extremes if requested
        if focus_extremes:
            print("\n  Focusing on extreme affinities...")

            # Categorize by affinity
            bins = self.binner.bin_array(df['pKd'].values)

            # Count by bin
            print("\n  Current distribution:")
            for i, label in enumerate(self.binner.bin_labels):
                count = (bins == i).sum()
                pct = 100 * count / len(df) if len(df) > 0 else 0
                print(f"    {label}: {count} ({pct:.1f}%)")

            # Optionally: increase sampling of extremes
            # (keep all, but could add oversampling here)

        print(f"\n  Filtered: {initial_count} → {len(df)} samples")

        return df

    def prepare_for_integration(self, abbibench_df, existing_df):
        """
        Prepare AbBiBench data for integration with existing dataset

        Args:
            abbibench_df: Filtered AbBiBench DataFrame
            existing_df: Existing dataset DataFrame

        Returns:
            AbBiBench DataFrame ready for merging
        """
        print("\n" + "="*80)
        print("PREPARING FOR INTEGRATION")
        print("="*80)

        # Add source label
        abbibench_df['source'] = 'AbBiBench'

        # Check what columns we need
        print("\nExisting dataset columns:")
        print(f"  {list(existing_df.columns[:10])}...")

        # Essential columns
        essential_cols = ['pdb_code', 'pKd', 'source']

        # Feature columns (ESM2 PCA)
        feature_cols = [col for col in existing_df.columns
                       if col.startswith('esm2_pca_')]

        print(f"\nFeature columns in existing dataset: {len(feature_cols)}")

        if feature_cols:
            print(f"  {feature_cols[:5]}...")

            # Add placeholder feature columns to AbBiBench
            print("\nWARNING AbBiBench data needs ESM2 embeddings!")
            print("  Setting feature columns to NaN for now")
            print("  You'll need to:")
            print("    1. Extract sequences from AbBiBench")
            print("    2. Generate ESM2 embeddings")
            print("    3. Apply PCA transformation")

            for col in feature_cols:
                abbibench_df[col] = np.nan

        # Ensure column compatibility
        for col in essential_cols:
            if col not in abbibench_df.columns:
                print(f"  WARNING Missing column: {col}")

        # Reorder columns to match existing dataset
        common_cols = list(set(existing_df.columns).intersection(
                          set(abbibench_df.columns)))
        abbibench_only_cols = list(set(abbibench_df.columns) -
                                   set(existing_df.columns))

        # Start with common columns in same order as existing
        ordered_cols = [col for col in existing_df.columns if col in common_cols]
        # Add AbBiBench-specific columns
        ordered_cols.extend(abbibench_only_cols)

        abbibench_df = abbibench_df[ordered_cols]

        print(f"\n  Prepared {len(abbibench_df)} samples")
        print(f"  Columns: {len(abbibench_df.columns)}")

        return abbibench_df

    def merge_datasets(self, existing_df, abbibench_df, output_path):
        """
        Merge datasets and save

        Args:
            existing_df: Existing dataset
            abbibench_df: Prepared AbBiBench data
            output_path: Output file path

        Returns:
            Merged DataFrame
        """
        print("\n" + "="*80)
        print("MERGING DATASETS")
        print("="*80)

        print(f"\n  Existing samples: {len(existing_df)}")
        print(f"  New samples: {len(abbibench_df)}")

        # Merge
        merged_df = pd.concat([existing_df, abbibench_df],
                             ignore_index=True, sort=False)

        print(f"  Merged samples: {len(merged_df)}")

        # Show distribution
        if 'pKd' in merged_df.columns:
            print_dataset_statistics(merged_df['pKd'].dropna().values,
                                   "Merged Dataset", self.binner)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        merged_df.to_csv(output_path, index=False)

        print(f"\nOK Saved merged dataset to: {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

        return merged_df

    def run(self, output_path='external_data/merged_with_abbibench.csv',
           filter_extremes=True):
        """
        Complete integration pipeline

        Args:
            output_path: Output path for merged dataset
            filter_extremes: Focus on extreme affinity values

        Returns:
            Merged DataFrame
        """
        print("\n" + "="*80)
        print("ABBIBENCH INTEGRATION PIPELINE")
        print("="*80)

        # 1. Load data
        abbibench_df = self.load_abbibench()
        existing_df = self.load_existing()

        # 2. Extract PDB codes
        abbibench_df = self.extract_pdb_codes(abbibench_df)

        # 3. Check duplicates
        abbibench_df = self.check_duplicates(abbibench_df, existing_df)

        if len(abbibench_df) == 0:
            print("\nWARNING No new samples to add after duplicate removal!")
            return existing_df

        # 4. Filter by affinity (optional)
        if filter_extremes:
            abbibench_df = self.filter_by_affinity(abbibench_df,
                                                   focus_extremes=True)

        # 5. Prepare for integration
        abbibench_df = self.prepare_for_integration(abbibench_df, existing_df)

        # 6. Merge
        merged_df = self.merge_datasets(existing_df, abbibench_df, output_path)

        # Summary
        print("\n" + "="*80)
        print("INTEGRATION COMPLETE!")
        print("="*80)

        print(f"\nOK Added {len(abbibench_df)} new samples")
        print(f"OK Total dataset size: {len(merged_df)}")
        print(f"OK Saved to: {output_path}")

        print("\nWARNING IMPORTANT NEXT STEPS:")
        print("1. The AbBiBench samples have NaN for feature columns")
        print("2. You need to:")
        print("   a. Extract sequences from AbBiBench")
        print("   b. Generate ESM2 embeddings")
        print("   c. Apply your existing PCA transformation")
        print("   d. Fill in the feature columns")
        print("\n3. Until then, you can:")
        print("   a. Use this for affinity distribution analysis")
        print("   b. Filter out NaN features for training")

        return merged_df


def main():
    """Main integration workflow"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Integrate AbBiBench with existing dataset'
    )
    parser.add_argument('--abbibench', type=str,
                       default='external_data/abbibench_raw.csv',
                       help='Path to AbBiBench CSV')
    parser.add_argument('--existing', type=str, required=True,
                       help='Path to existing dataset CSV')
    parser.add_argument('--output', type=str,
                       default='external_data/merged_with_abbibench.csv',
                       help='Output path for merged dataset')
    parser.add_argument('--no_filter', action='store_true',
                       help='Do not filter for extreme affinities')

    args = parser.parse_args()

    # Run integration
    integrator = AbBiBenchIntegrator(args.abbibench, args.existing)
    merged_df = integrator.run(
        output_path=args.output,
        filter_extremes=not args.no_filter
    )

    print("\nOK Done!")


if __name__ == "__main__":
    main()
