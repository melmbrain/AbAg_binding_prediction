"""
Unified integration script for all external databases
Combines AbBiBench, SAAINT-DB, and PDBbind with existing dataset

This script will:
1. Detect which databases have been downloaded
2. Load and standardize each dataset
3. Check for duplicates across all sources
4. Merge with existing dataset
5. Generate comprehensive statistics
6. Prepare for ESM2 embedding generation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_utils import AffinityBinner, print_dataset_statistics


class UnifiedDatabaseIntegrator:
    """
    Integrates multiple external databases with existing dataset
    """
    def __init__(self, existing_data_path: str, external_data_dir: str = './external_data'):
        """
        Args:
            existing_data_path: Path to existing dataset CSV
            external_data_dir: Directory containing downloaded databases
        """
        self.existing_data_path = existing_data_path
        self.external_data_dir = Path(external_data_dir)
        self.binner = AffinityBinner()

        self.existing_df = None
        self.external_datasets = {}

    def load_existing(self):
        """Load existing dataset"""
        print("\n" + "="*80)
        print("LOADING EXISTING DATASET")
        print("="*80)

        print(f"\nLoading from: {self.existing_data_path}")
        self.existing_df = pd.read_csv(self.existing_data_path)

        print(f"  Loaded {len(self.existing_df)} samples")
        print(f"  Columns: {len(self.existing_df.columns)}")

        if 'pKd' in self.existing_df.columns:
            print_dataset_statistics(self.existing_df['pKd'].values,
                                   "Existing Dataset", self.binner)

        return self.existing_df

    def detect_downloaded_databases(self):
        """
        Detect which databases have been downloaded

        Returns:
            Dict of database_name -> file_path
        """
        print("\n" + "="*80)
        print("DETECTING DOWNLOADED DATABASES")
        print("="*80)

        detected = {}

        # AbBiBench
        abbibench_files = list(self.external_data_dir.glob('abbibench*.csv'))
        if abbibench_files:
            detected['abbibench'] = abbibench_files[0]
            print(f"\nOK AbBiBench: {abbibench_files[0].name}")

        # SAAINT-DB
        saaint_files = list(self.external_data_dir.glob('saaint*.csv'))
        if saaint_files:
            detected['saaint'] = saaint_files[0]
            print(f"OK SAAINT-DB: {saaint_files[0].name}")

        # PDBbind
        pdbbind_files = list(self.external_data_dir.glob('pdbbind*.csv'))
        if pdbbind_files:
            detected['pdbbind'] = pdbbind_files[0]
            print(f"OK PDBbind: {pdbbind_files[0].name}")

        if not detected:
            print("\nWARNING No downloaded databases found")
            print(f"Searched in: {self.external_data_dir}")
            print("\nExpected files:")
            print("  - abbibench_raw.csv")
            print("  - saaint_raw.csv")
            print("  - pdbbind_raw.csv")
            print("\nRun download scripts first:")
            print("  python scripts/download_abbibench.py")
            print("  python scripts/download_saaint.py")
            print("  python scripts/download_pdbbind.py")

        return detected

    def load_external_database(self, db_name: str, file_path: Path):
        """
        Load and standardize external database

        Args:
            db_name: Name of database (abbibench, saaint, pdbbind)
            file_path: Path to CSV file

        Returns:
            Standardized DataFrame
        """
        print(f"\nLoading {db_name}...")

        df = pd.read_csv(file_path)
        print(f"  Raw data: {len(df)} samples")

        # Standardize column names
        df = self._standardize_columns(df, db_name)

        # Add source label
        df['source'] = db_name

        # Filter for valid pKd
        if 'pKd' in df.columns:
            valid_mask = df['pKd'].notna() & np.isfinite(df['pKd'])
            n_valid = valid_mask.sum()
            if n_valid < len(df):
                print(f"  Filtering: {len(df)} → {n_valid} (valid pKd)")
                df = df[valid_mask].copy()

        print(f"  Processed: {len(df)} samples")

        return df

    def _standardize_columns(self, df: pd.DataFrame, db_name: str):
        """
        Standardize column names across databases

        Args:
            df: DataFrame to standardize
            db_name: Name of database

        Returns:
            Standardized DataFrame
        """
        # Map common column names
        column_mapping = {}

        # PDB code
        for old_col in ['pdb', 'PDB', 'pdb_id', 'structure_id', 'complex_id']:
            if old_col in df.columns and 'pdb_code' not in df.columns:
                column_mapping[old_col] = 'pdb_code'
                break

        # Apply mapping
        if column_mapping:
            df = df.rename(columns=column_mapping)
            print(f"  Standardized columns: {column_mapping}")

        # Ensure pdb_code is lowercase
        if 'pdb_code' in df.columns:
            df['pdb_code'] = df['pdb_code'].astype(str).str.strip().str.lower()

        return df

    def check_all_duplicates(self):
        """
        Check for duplicates across all datasets

        Returns:
            Dict of duplicate statistics
        """
        print("\n" + "="*80)
        print("CHECKING FOR DUPLICATES")
        print("="*80)

        if self.existing_df is None:
            print("\nWARNING Existing dataset not loaded")
            return {}

        # Get PDB codes from existing dataset
        if 'pdb_code' not in self.existing_df.columns:
            print("\nWARNING No pdb_code column in existing dataset")
            return {}

        existing_pdbs = set(self.existing_df['pdb_code'].dropna().str.lower())
        print(f"\nExisting dataset: {len(existing_pdbs)} unique PDB codes")

        duplicate_stats = {}

        # Check each external database
        for db_name, df in self.external_datasets.items():
            if 'pdb_code' not in df.columns:
                print(f"\nWARNING {db_name}: No pdb_code column")
                continue

            db_pdbs = set(df['pdb_code'].dropna().str.lower())
            duplicates = existing_pdbs.intersection(db_pdbs)
            new_pdbs = db_pdbs - existing_pdbs

            print(f"\n{db_name}:")
            print(f"  Total PDB codes: {len(db_pdbs)}")
            print(f"  Duplicates with existing: {len(duplicates)} ({len(duplicates)/len(db_pdbs)*100:.1f}%)")
            print(f"  New PDB codes: {len(new_pdbs)} ({len(new_pdbs)/len(db_pdbs)*100:.1f}%)")

            if duplicates:
                dup_list = sorted(list(duplicates))[:10]
                print(f"  Examples: {', '.join(dup_list)}...")

            duplicate_stats[db_name] = {
                'total': len(db_pdbs),
                'duplicates': len(duplicates),
                'new': len(new_pdbs)
            }

        # Check for duplicates between external databases
        print("\n" + "-"*80)
        print("Cross-database duplicates:")
        print("-"*80)

        db_names = list(self.external_datasets.keys())
        for i, db1 in enumerate(db_names):
            for db2 in db_names[i+1:]:
                if 'pdb_code' not in self.external_datasets[db1].columns:
                    continue
                if 'pdb_code' not in self.external_datasets[db2].columns:
                    continue

                pdbs1 = set(self.external_datasets[db1]['pdb_code'].dropna())
                pdbs2 = set(self.external_datasets[db2]['pdb_code'].dropna())
                overlap = pdbs1.intersection(pdbs2)

                if overlap:
                    print(f"\n{db1} ∩ {db2}: {len(overlap)} common PDB codes")

        return duplicate_stats

    def remove_duplicates(self):
        """
        Remove duplicates from external datasets

        Returns:
            Dict of filtered datasets
        """
        print("\n" + "="*80)
        print("REMOVING DUPLICATES")
        print("="*80)

        if self.existing_df is None or 'pdb_code' not in self.existing_df.columns:
            print("\nWARNING Cannot remove duplicates without existing dataset")
            return self.external_datasets

        existing_pdbs = set(self.existing_df['pdb_code'].dropna().str.lower())

        filtered_datasets = {}

        for db_name, df in self.external_datasets.items():
            if 'pdb_code' not in df.columns:
                print(f"\nWARNING {db_name}: No pdb_code column, keeping all")
                filtered_datasets[db_name] = df
                continue

            # Remove duplicates with existing dataset
            dup_mask = df['pdb_code'].str.lower().isin(existing_pdbs)
            n_dup = dup_mask.sum()

            df_filtered = df[~dup_mask].copy()

            print(f"\n{db_name}:")
            print(f"  Original: {len(df)} samples")
            print(f"  Removed: {n_dup} duplicates")
            print(f"  Remaining: {len(df_filtered)} samples")

            filtered_datasets[db_name] = df_filtered

        self.external_datasets = filtered_datasets
        return filtered_datasets

    def analyze_affinity_distribution(self):
        """
        Analyze affinity distribution in each dataset

        Returns:
            Dict of distribution statistics
        """
        print("\n" + "="*80)
        print("AFFINITY DISTRIBUTION ANALYSIS")
        print("="*80)

        distributions = {}

        # Existing dataset
        if self.existing_df is not None and 'pKd' in self.existing_df.columns:
            print("\nExisting Dataset:")
            print_dataset_statistics(self.existing_df['pKd'].values,
                                   "Existing", self.binner)
            distributions['existing'] = self._get_distribution_dict(
                self.existing_df['pKd'].values)

        # External datasets
        for db_name, df in self.external_datasets.items():
            if 'pKd' not in df.columns:
                continue

            valid_pkd = df['pKd'].dropna()
            if len(valid_pkd) == 0:
                continue

            print(f"\n{db_name}:")
            print_dataset_statistics(valid_pkd.values, db_name, self.binner)
            distributions[db_name] = self._get_distribution_dict(valid_pkd.values)

        return distributions

    def _get_distribution_dict(self, pkd_values):
        """Get distribution as dict"""
        bins = self.binner.bin_array(pkd_values)
        dist = {}
        for i, label in enumerate(self.binner.bin_labels):
            count = (bins == i).sum()
            pct = 100 * count / len(pkd_values)
            dist[label] = {'count': int(count), 'percent': float(pct)}
        return dist

    def prepare_for_integration(self):
        """
        Prepare external datasets for integration

        Returns:
            List of prepared DataFrames
        """
        print("\n" + "="*80)
        print("PREPARING FOR INTEGRATION")
        print("="*80)

        if self.existing_df is None:
            print("\nWARNING Existing dataset not loaded")
            return []

        # Get feature columns from existing dataset
        feature_cols = [col for col in self.existing_df.columns
                       if col.startswith('esm2_pca_')]

        print(f"\nFeature columns in existing dataset: {len(feature_cols)}")

        prepared_datasets = []

        for db_name, df in self.external_datasets.items():
            print(f"\nPreparing {db_name}...")

            # Add missing feature columns as NaN
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = np.nan

            # Ensure essential columns
            essential_cols = ['pdb_code', 'pKd', 'source']
            missing = [col for col in essential_cols if col not in df.columns]

            if missing:
                print(f"  WARNING Missing columns: {missing}")
                continue

            # Reorder columns to match existing dataset
            common_cols = [col for col in self.existing_df.columns if col in df.columns]
            new_cols = [col for col in df.columns if col not in self.existing_df.columns]

            ordered_cols = common_cols + new_cols
            df = df[ordered_cols]

            print(f"  Prepared {len(df)} samples with {len(df.columns)} columns")

            prepared_datasets.append(df)

        print(f"\nOK Prepared {len(prepared_datasets)} datasets for integration")

        return prepared_datasets

    def merge_all(self, output_path: str):
        """
        Merge all datasets and save

        Args:
            output_path: Output file path

        Returns:
            Merged DataFrame
        """
        print("\n" + "="*80)
        print("MERGING ALL DATASETS")
        print("="*80)

        if self.existing_df is None:
            print("\nWARNING Existing dataset not loaded")
            return None

        # Prepare external datasets
        prepared = self.prepare_for_integration()

        if not prepared:
            print("\nWARNING No external datasets to merge")
            return self.existing_df

        # Combine all
        all_dfs = [self.existing_df] + prepared

        print(f"\nMerging {len(all_dfs)} datasets...")
        for i, df in enumerate(all_dfs):
            source = df['source'].iloc[0] if 'source' in df.columns else 'existing'
            print(f"  {i+1}. {source}: {len(df)} samples")

        merged_df = pd.concat(all_dfs, ignore_index=True, sort=False)

        print(f"\nOK Merged dataset: {len(merged_df)} samples")

        # Show final distribution
        if 'pKd' in merged_df.columns:
            print_dataset_statistics(merged_df['pKd'].dropna().values,
                                   "Merged Dataset", self.binner)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        merged_df.to_csv(output_path, index=False)

        print(f"\nOK Saved to: {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

        return merged_df

    def generate_summary_report(self, output_path: str = None):
        """
        Generate comprehensive summary report

        Args:
            output_path: Optional path to save report
        """
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DATABASE INTEGRATION SUMMARY")
        report_lines.append("=" * 80)

        # Dataset counts
        report_lines.append("\n## Datasets Loaded:\n")
        if self.existing_df is not None:
            report_lines.append(f"- Existing: {len(self.existing_df)} samples")

        for db_name, df in self.external_datasets.items():
            report_lines.append(f"- {db_name}: {len(df)} samples")

        # Affinity distribution comparison
        report_lines.append("\n## Affinity Distribution (after removing duplicates):\n")

        distributions = self.analyze_affinity_distribution()

        # Create comparison table
        report_lines.append("| Bin | Existing | AbBiBench | SAAINT | PDBbind |")
        report_lines.append("|-----|----------|-----------|--------|---------|")

        for bin_label in self.binner.bin_labels:
            row = f"| {bin_label:12s} |"

            for db_name in ['existing', 'abbibench', 'saaint', 'pdbbind']:
                if db_name in distributions:
                    dist = distributions[db_name]
                    if bin_label in dist:
                        count = dist[bin_label]['count']
                        pct = dist[bin_label]['percent']
                        row += f" {count:6d} ({pct:4.1f}%) |"
                    else:
                        row += " - |"
                else:
                    row += " - |"

            report_lines.append(row)

        # Next steps
        report_lines.append("\n## Next Steps:\n")
        report_lines.append("1. OK Data downloaded and integrated")
        report_lines.append("2. WARNING ESM2 embeddings needed for new sequences")
        report_lines.append("3. WARNING PCA transformation required")
        report_lines.append("4. Re-train model with augmented dataset")

        report_lines.append("\n## ESM2 Embedding Generation:\n")
        report_lines.append("The new samples have NaN for ESM2 PCA features.")
        report_lines.append("You need to:")
        report_lines.append("  a. Extract antibody/antigen sequences")
        report_lines.append("  b. Generate ESM2 embeddings")
        report_lines.append("  c. Apply existing PCA transformation")
        report_lines.append("  d. Fill in feature columns")

        # Print report
        report_text = "\n".join(report_lines)
        print(report_text)

        # Save if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_text)
            print(f"\nOK Report saved to: {output_path}")

        return report_text

    def run(self, output_path: str = 'external_data/merged_all_databases.csv',
           report_path: str = 'external_data/integration_report.txt'):
        """
        Complete integration pipeline

        Args:
            output_path: Output path for merged dataset
            report_path: Output path for summary report

        Returns:
            Merged DataFrame
        """
        print("\n" + "="*80)
        print("UNIFIED DATABASE INTEGRATION PIPELINE")
        print("="*80)

        # 1. Load existing dataset
        self.load_existing()

        # 2. Detect downloaded databases
        detected = self.detect_downloaded_databases()

        if not detected:
            print("\nWARNING No external databases found to integrate")
            return self.existing_df

        # 3. Load external databases
        print("\n" + "="*80)
        print("LOADING EXTERNAL DATABASES")
        print("="*80)

        for db_name, file_path in detected.items():
            try:
                df = self.load_external_database(db_name, file_path)
                self.external_datasets[db_name] = df
            except Exception as e:
                print(f"\nERROR Error loading {db_name}: {e}")
                continue

        if not self.external_datasets:
            print("\nWARNING No external databases loaded successfully")
            return self.existing_df

        # 4. Check duplicates
        self.check_all_duplicates()

        # 5. Remove duplicates
        self.remove_duplicates()

        # 6. Analyze distributions
        self.analyze_affinity_distribution()

        # 7. Merge all
        merged_df = self.merge_all(output_path)

        # 8. Generate report
        self.generate_summary_report(report_path)

        # Final summary
        print("\n" + "="*80)
        print("INTEGRATION COMPLETE!")
        print("="*80)

        print(f"\nOK Integrated {len(self.external_datasets)} external databases")
        print(f"OK Total samples: {len(merged_df)}")
        print(f"OK Saved to: {output_path}")
        print(f"OK Report: {report_path}")

        print("\nWARNING IMPORTANT:")
        print("New samples need ESM2 embeddings!")
        print("See report for next steps.")

        return merged_df


def main():
    """Main integration workflow"""
    parser = argparse.ArgumentParser(
        description='Integrate all external databases with existing dataset'
    )
    parser.add_argument('--existing', type=str, required=True,
                       help='Path to existing dataset CSV')
    parser.add_argument('--external_dir', type=str,
                       default='external_data',
                       help='Directory containing downloaded databases')
    parser.add_argument('--output', type=str,
                       default='external_data/merged_all_databases.csv',
                       help='Output path for merged dataset')
    parser.add_argument('--report', type=str,
                       default='external_data/integration_report.txt',
                       help='Output path for summary report')

    args = parser.parse_args()

    # Run integration
    integrator = UnifiedDatabaseIntegrator(args.existing, args.external_dir)
    merged_df = integrator.run(
        output_path=args.output,
        report_path=args.report
    )

    print("\nOK Done!")


if __name__ == "__main__":
    main()
