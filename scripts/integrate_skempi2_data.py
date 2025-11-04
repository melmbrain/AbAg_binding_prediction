"""
Script to integrate SKEMPI2 antibody-antigen extreme affinity data
with the existing dataset

This script:
1. Loads the extracted SKEMPI2 antibody-antigen complexes
2. Processes and formats them to match your existing data format
3. Adds sequence features (if needed)
4. Merges with existing dataset
5. Saves the enhanced dataset
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_utils import AffinityBinner, print_dataset_statistics


class SKEMPI2Integrator:
    """
    Integrates SKEMPI2 extreme affinity data with existing dataset
    """
    def __init__(self, existing_data_path: str, skempi2_dir: str):
        """
        Args:
            existing_data_path: Path to existing dataset CSV
            skempi2_dir: Directory containing extracted SKEMPI2 data
        """
        self.existing_data_path = existing_data_path
        self.skempi2_dir = skempi2_dir
        self.binner = AffinityBinner()

    def load_existing_data(self) -> pd.DataFrame:
        """Load existing dataset"""
        print(f"Loading existing data from: {self.existing_data_path}")
        df = pd.read_csv(self.existing_data_path)
        print(f"  Loaded {len(df)} samples")

        return df

    def load_skempi2_weak(self) -> pd.DataFrame:
        """Load SKEMPI2 weak antibody-antigen binders"""
        weak_path = os.path.join(self.skempi2_dir, 'skempi2_antibody_weak.csv')
        very_weak_path = os.path.join(self.skempi2_dir, 'skempi2_antibody_very_weak.csv')

        dfs = []

        if os.path.exists(weak_path):
            df_weak = pd.read_csv(weak_path)
            print(f"  Loaded {len(df_weak)} weak binders from SKEMPI2")
            dfs.append(df_weak)

        if os.path.exists(very_weak_path):
            df_very_weak = pd.read_csv(very_weak_path)
            print(f"  Loaded {len(df_very_weak)} very weak binders from SKEMPI2")
            dfs.append(df_very_weak)

        if not dfs:
            raise FileNotFoundError("No SKEMPI2 weak binder files found")

        df = pd.concat(dfs, ignore_index=True)
        return df

    def convert_skempi2_to_pkd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert SKEMPI2 format to pKd

        SKEMPI2 has Affinity_wt in Molar units
        Convert to pKd = -log10(Kd_M)
        """
        df = df.copy()

        # Check if Affinity_wt column exists
        if 'Affinity_wt (M)' in df.columns:
            affinity_col = 'Affinity_wt (M)'
        elif 'Affinity_wt' in df.columns:
            affinity_col = 'Affinity_wt'
        else:
            raise ValueError("No affinity column found in SKEMPI2 data")

        # Convert to numeric, handle errors
        df[affinity_col] = pd.to_numeric(df[affinity_col], errors='coerce')

        # Calculate pKd
        mask = (df[affinity_col].notna()) & (df[affinity_col] > 0)
        df.loc[mask, 'pKd'] = -np.log10(df.loc[mask, affinity_col])

        # Filter valid pKd values
        df = df[df['pKd'].notna()].copy()

        print(f"  Converted {len(df)} entries to pKd")
        print(f"  pKd range: [{df['pKd'].min():.2f}, {df['pKd'].max():.2f}]")

        return df

    def extract_pdb_and_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract PDB code and protein information from SKEMPI2

        SKEMPI2 format: PDB code is in '#Pdb' column (e.g., '1FCC_A_C')
        """
        df = df.copy()

        # Extract PDB code (first part before underscore)
        if '#Pdb' in df.columns:
            df['pdb_code'] = df['#Pdb'].str.split('_').str[0].str.lower()
        elif 'Pdb' in df.columns:
            df['pdb_code'] = df['Pdb'].str.split('_').str[0].str.lower()

        # Add source label
        df['source'] = 'SKEMPI2'

        # Keep relevant columns
        cols_to_keep = ['pdb_code', 'pKd', 'source']

        # Add protein names if available
        if 'Protein 1' in df.columns:
            cols_to_keep.append('Protein 1')
        if 'Protein 2' in df.columns:
            cols_to_keep.append('Protein 2')

        df = df[cols_to_keep].copy()

        return df

    def check_for_duplicates(self, existing_df: pd.DataFrame,
                            new_df: pd.DataFrame) -> pd.DataFrame:
        """
        Check for PDB codes already in existing dataset

        Args:
            existing_df: Existing dataset
            new_df: New SKEMPI2 data

        Returns:
            Filtered new_df without duplicates
        """
        if 'pdb_code' not in existing_df.columns:
            print("  Warning: No pdb_code column in existing data, cannot check for duplicates")
            return new_df

        existing_pdbs = set(existing_df['pdb_code'].str.lower())
        new_pdbs = set(new_df['pdb_code'].str.lower())

        duplicates = existing_pdbs.intersection(new_pdbs)

        if duplicates:
            print(f"  Found {len(duplicates)} duplicate PDB codes:")
            print(f"    {', '.join(sorted(list(duplicates)[:10]))}" +
                  ("..." if len(duplicates) > 10 else ""))

            # Remove duplicates
            new_df = new_df[~new_df['pdb_code'].str.lower().isin(duplicates)].copy()
            print(f"  Removed duplicates, {len(new_df)} new entries remaining")

        else:
            print(f"  No duplicates found, all {len(new_df)} entries are new")

        return new_df

    def align_feature_columns(self, existing_df: pd.DataFrame,
                             new_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align feature columns between existing and new data

        This is a placeholder - you'll need to adapt based on your actual feature columns
        """
        # Get feature columns from existing data (assuming they start with esm2_pca_)
        feature_cols = [col for col in existing_df.columns if col.startswith('esm2_pca_')]

        if not feature_cols:
            print("  Warning: No ESM2 PCA features found in existing data")
            print("  You'll need to generate features for SKEMPI2 entries")
            return new_df

        print(f"  Existing data has {len(feature_cols)} feature columns")

        # For now, set features to NaN - they need to be generated separately
        for col in feature_cols:
            new_df[col] = np.nan

        print("  Note: Feature columns added but set to NaN")
        print("  You need to:")
        print("    1. Fetch sequences for SKEMPI2 PDB codes")
        print("    2. Generate ESM2 embeddings")
        print("    3. Apply PCA transformation")

        return new_df

    def merge_datasets(self, existing_df: pd.DataFrame,
                      new_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge existing and new datasets

        Args:
            existing_df: Existing dataset
            new_df: New SKEMPI2 data (preprocessed)

        Returns:
            Merged dataset
        """
        print("\nMerging datasets...")

        # Ensure column compatibility
        common_cols = list(set(existing_df.columns).intersection(set(new_df.columns)))
        print(f"  Common columns: {len(common_cols)}")

        # Merge
        merged_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)

        print(f"  Merged dataset size: {len(merged_df)}")

        return merged_df

    def run_integration(self, output_path: Optional[str] = None,
                       check_duplicates: bool = True) -> pd.DataFrame:
        """
        Run the complete integration pipeline

        Args:
            output_path: Path to save merged dataset (optional)
            check_duplicates: Whether to check and remove duplicates

        Returns:
            Merged dataset
        """
        print("="*80)
        print("INTEGRATING SKEMPI2 EXTREME AFFINITY DATA")
        print("="*80)

        # 1. Load existing data
        print("\n1. Loading existing data...")
        existing_df = self.load_existing_data()
        print_dataset_statistics(existing_df['pKd'].values, "Existing Dataset", self.binner)

        # 2. Load SKEMPI2 data
        print("\n2. Loading SKEMPI2 weak binders...")
        skempi2_df = self.load_skempi2_weak()

        # 3. Convert to pKd
        print("\n3. Converting SKEMPI2 to pKd format...")
        skempi2_df = self.convert_skempi2_to_pkd(skempi2_df)

        # 4. Extract PDB and info
        print("\n4. Extracting PDB codes and metadata...")
        skempi2_df = self.extract_pdb_and_info(skempi2_df)

        # 5. Check for duplicates
        if check_duplicates:
            print("\n5. Checking for duplicate PDB codes...")
            skempi2_df = self.check_for_duplicates(existing_df, skempi2_df)
        else:
            print("\n5. Skipping duplicate check (as requested)")

        if len(skempi2_df) == 0:
            print("\n  No new entries to add after duplicate removal!")
            return existing_df

        print_dataset_statistics(skempi2_df['pKd'].values, "New SKEMPI2 Data", self.binner)

        # 6. Align feature columns (placeholder)
        print("\n6. Aligning feature columns...")
        skempi2_df = self.align_feature_columns(existing_df, skempi2_df)

        # 7. Merge datasets
        merged_df = self.merge_datasets(existing_df, skempi2_df)

        # 8. Final statistics
        print("\n" + "="*80)
        print("INTEGRATION COMPLETE")
        print("="*80)
        print_dataset_statistics(merged_df['pKd'].values, "Merged Dataset", self.binner)

        # Save if requested
        if output_path:
            print(f"\nSaving merged dataset to: {output_path}")
            merged_df.to_csv(output_path, index=False)
            print(f"  Saved {len(merged_df)} samples")

        return merged_df


def main():
    """Main function for standalone execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Integrate SKEMPI2 extreme affinity data with existing dataset'
    )
    parser.add_argument('--existing_data', type=str, required=True,
                       help='Path to existing dataset CSV')
    parser.add_argument('--skempi2_dir', type=str, required=True,
                       help='Directory containing extracted SKEMPI2 data')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for merged dataset')
    parser.add_argument('--no_duplicate_check', action='store_true',
                       help='Skip duplicate checking')

    args = parser.parse_args()

    # Run integration
    integrator = SKEMPI2Integrator(args.existing_data, args.skempi2_dir)
    merged_df = integrator.run_integration(
        output_path=args.output,
        check_duplicates=not args.no_duplicate_check
    )

    print("\n✓ Integration complete!")
    print(f"  Merged dataset saved to: {args.output}")
    print(f"  Total samples: {len(merged_df)}")


if __name__ == "__main__":
    # Example usage for testing
    print("SKEMPI2 Data Integration Script")
    print("="*80)
    print("\nUsage:")
    print("  python integrate_skempi2_data.py \\")
    print("    --existing_data /path/to/existing_data.csv \\")
    print("    --skempi2_dir /path/to/extreme_affinity_data \\")
    print("    --output /path/to/merged_data.csv")
    print("\n" + "="*80)

    # Check if command line args provided
    if len(sys.argv) > 1:
        main()
    else:
        print("\nNo arguments provided. Run with --help for usage information.")
