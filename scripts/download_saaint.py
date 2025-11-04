"""
Download SAAINT-DB dataset from GitHub
SAAINT is the most comprehensive and recent antibody-antigen affinity database

SAAINT-DB contains:
- 19,128 data entries from 9,757 PDB structures
- Nearly 2× more affinity data than SAbDab
- Manual curation for high quality
- Last updated: May 1, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import subprocess
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_utils import AffinityBinner, print_dataset_statistics


def clone_saaint_repository(target_dir='./external_data/SAAINT'):
    """
    Clone SAAINT repository from GitHub

    Args:
        target_dir: Directory to clone into

    Returns:
        Path to cloned repository
    """
    print("\n" + "="*80)
    print("CLONING SAAINT-DB REPOSITORY")
    print("="*80)

    target_path = Path(target_dir)

    if target_path.exists():
        print(f"\nOK Repository already exists at: {target_path}")
        print("  Skipping clone...")
        return target_path

    print(f"\nCloning to: {target_path}")
    print("This may take a few minutes...")

    try:
        # Clone repository
        subprocess.run([
            'git', 'clone',
            'https://github.com/tommyhuangthu/SAAINT.git',
            str(target_path)
        ], check=True, capture_output=True, text=True)

        print(f"\nOK Clone complete!")
        return target_path

    except subprocess.CalledProcessError as e:
        print(f"\nERROR Error cloning repository: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Check if git is installed: git --version")
        print("3. Try manual download from: https://github.com/tommyhuangthu/SAAINT")
        return None


def find_affinity_files(repo_path):
    """
    Find affinity data files in SAAINT repository

    Args:
        repo_path: Path to SAAINT repository

    Returns:
        List of paths to affinity data files
    """
    print("\n" + "="*80)
    print("SEARCHING FOR AFFINITY DATA FILES")
    print("="*80)

    repo_path = Path(repo_path)

    # Common patterns for affinity data files
    patterns = [
        '*summary*.csv',
        '*affinity*.csv',
        '*database*.csv',
        '*binding*.csv',
        '*.csv'
    ]

    found_files = []

    for pattern in patterns:
        files = list(repo_path.rglob(pattern))
        found_files.extend(files)

    # Remove duplicates
    found_files = list(set(found_files))

    print(f"\nFound {len(found_files)} CSV files:")
    for f in found_files[:20]:  # Show first 20
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  - {f.name} ({size_mb:.2f} MB)")

    if len(found_files) > 20:
        print(f"  ... and {len(found_files) - 20} more")

    return found_files


def load_saaint_data(file_path):
    """
    Load SAAINT affinity data file

    Args:
        file_path: Path to CSV file

    Returns:
        DataFrame with affinity data
    """
    print("\n" + "="*80)
    print("LOADING SAAINT DATA")
    print("="*80)

    print(f"\nLoading from: {file_path}")

    try:
        df = pd.read_csv(file_path)

        print(f"  OK Loaded {len(df)} samples")
        print(f"  Columns ({len(df.columns)}): {list(df.columns[:10])}...")

        return df

    except Exception as e:
        print(f"  ERROR Error loading file: {e}")
        return None


def explore_saaint_structure(df):
    """
    Explore the structure of SAAINT data

    Args:
        df: SAAINT DataFrame
    """
    print("\n" + "="*80)
    print("DATASET STRUCTURE")
    print("="*80)

    print(f"\nTotal samples: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\nAll columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

    # Check for PDB codes
    pdb_cols = [col for col in df.columns if
                'pdb' in col.lower() or 'code' in col.lower() or 'id' in col.lower()]

    if pdb_cols:
        print(f"\nPDB-related columns: {pdb_cols}")
        for col in pdb_cols[:3]:
            unique = df[col].nunique()
            print(f"  {col}: {unique} unique values")

    # Check for affinity columns
    affinity_cols = [col for col in df.columns if
                     any(term in col.lower() for term in
                         ['kd', 'ki', 'affinity', 'pkd', 'bind', 'ic50'])]

    if affinity_cols:
        print(f"\nAffinity-related columns: {affinity_cols}")
        for col in affinity_cols:
            non_null = df[col].notna().sum()
            print(f"  {col}: {non_null} non-null values")

    # Show first sample
    print("\nFirst sample:")
    first = df.iloc[0]
    for key, value in first.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}... (truncated)")
        else:
            print(f"  {key}: {value}")


def extract_affinity_data(df):
    """
    Extract and process affinity data from SAAINT

    Args:
        df: SAAINT DataFrame

    Returns:
        Processed DataFrame with affinity values
    """
    print("\n" + "="*80)
    print("EXTRACTING AFFINITY DATA")
    print("="*80)

    # Common affinity column names
    possible_affinity_cols = [
        'Kd', 'KD', 'kd',
        'pKd', 'pkd',
        'Ki', 'KI', 'ki',
        'IC50', 'ic50',
        'binding_affinity',
        'affinity',
        'delta_g',
        'ddG'
    ]

    affinity_col = None
    for col in possible_affinity_cols:
        if col in df.columns:
            affinity_col = col
            print(f"\nOK Found affinity column: '{affinity_col}'")
            break

    if affinity_col is None:
        print("\nWARNING No standard affinity column found")
        print("Checking for related columns...")
        affinity_related = [col for col in df.columns if
                          any(term in col.lower() for term in
                              ['affinity', 'kd', 'ki', 'bind', 'ic50'])]
        if affinity_related:
            print(f"Found related columns: {affinity_related}")
            affinity_col = affinity_related[0]
            print(f"Using: {affinity_col}")
        else:
            print("ERROR Could not find affinity data")
            return df

    # Statistics
    print(f"\nAffinity column: {affinity_col}")
    print(f"  Non-null values: {df[affinity_col].notna().sum()}")
    print(f"  Null values: {df[affinity_col].isna().sum()}")

    if df[affinity_col].notna().any():
        valid_data = df[affinity_col].dropna()
        print(f"  Min: {valid_data.min()}")
        print(f"  Max: {valid_data.max()}")
        print(f"  Mean: {valid_data.mean():.4f}")
        print(f"  Median: {valid_data.median():.4f}")

    return df


def convert_to_pkd(df, affinity_col=None):
    """
    Convert SAAINT affinity values to pKd

    Args:
        df: DataFrame
        affinity_col: Column containing affinity values (auto-detect if None)

    Returns:
        DataFrame with pKd column added
    """
    print("\n" + "="*80)
    print("CONVERTING TO pKd")
    print("="*80)

    df = df.copy()

    # Auto-detect affinity column
    if affinity_col is None:
        for col in ['pKd', 'pkd', 'Kd', 'KD', 'kd', 'Ki', 'IC50']:
            if col in df.columns:
                affinity_col = col
                break

    if affinity_col is None:
        print("ERROR No affinity column specified or found")
        return df

    print(f"\nUsing column: {affinity_col}")

    # Check if already pKd
    if 'pkd' in affinity_col.lower():
        print("  Data appears to be already in pKd format")
        df['pKd'] = df[affinity_col]

    else:
        # Assume Kd in Molar (typical for SAAINT)
        print("  Converting Kd (M) to pKd...")

        # Get sample values to check range
        sample_values = df[affinity_col].dropna().head(100)
        median_val = sample_values.median()

        # Determine unit based on range
        if median_val < 1e-6:
            print(f"  Values appear to be in Molar (median: {median_val:.2e})")
            df['pKd'] = -np.log10(df[affinity_col])
        elif 1 <= median_val <= 1000:
            print(f"  Values appear to be in nM (median: {median_val:.2f})")
            df['pKd'] = -np.log10(df[affinity_col] / 1e9)
        elif 0.001 <= median_val <= 1:
            print(f"  Values appear to be in μM (median: {median_val:.4f})")
            df['pKd'] = -np.log10(df[affinity_col] / 1e6)
        else:
            print(f"  WARNING Cannot determine units (median: {median_val:.2e})")
            print("  Assuming Molar...")
            df['pKd'] = -np.log10(df[affinity_col])

    # Remove invalid values
    invalid_mask = np.isinf(df['pKd']) | np.isnan(df['pKd'])
    if invalid_mask.any():
        n_invalid = invalid_mask.sum()
        print(f"\n  WARNING Removing {n_invalid} invalid pKd values")
        df = df[~invalid_mask].copy()

    # Statistics
    valid_pkd = df['pKd'].dropna()
    if len(valid_pkd) > 0:
        print(f"\npKd conversion results:")
        print(f"  Converted: {len(valid_pkd)} values")
        print(f"  Range: [{valid_pkd.min():.2f}, {valid_pkd.max():.2f}]")
        print(f"  Mean: {valid_pkd.mean():.2f}")
        print(f"  Median: {valid_pkd.median():.2f}")

        # Show distribution
        binner = AffinityBinner()
        print_dataset_statistics(valid_pkd.values, "SAAINT-DB", binner)

    return df


def filter_antibody_antigen(df):
    """
    Filter for antibody-antigen complexes

    Args:
        df: SAAINT DataFrame

    Returns:
        Filtered DataFrame with only antibody-antigen complexes
    """
    print("\n" + "="*80)
    print("FILTERING FOR ANTIBODY-ANTIGEN COMPLEXES")
    print("="*80)

    initial_count = len(df)

    # Look for antibody/antigen indicator columns
    indicator_cols = [col for col in df.columns if
                     any(term in col.lower() for term in
                         ['type', 'class', 'category', 'antibody', 'antigen'])]

    if indicator_cols:
        print(f"\nFound indicator columns: {indicator_cols}")

        # Try to filter
        for col in indicator_cols:
            unique_vals = df[col].unique()
            print(f"\n{col} values: {unique_vals[:10]}")

            # Check for antibody indicators
            antibody_mask = df[col].astype(str).str.contains(
                'antibody|immunoglobulin|IgG|Fab|scFv',
                case=False, na=False
            )

            if antibody_mask.any():
                df = df[antibody_mask].copy()
                print(f"  Filtered to {len(df)} antibody-related entries")
                break

    else:
        print("\nWARNING No type indicator columns found")
        print("Attempting to filter by PDB codes...")

        # Look for PDB column
        pdb_cols = [col for col in df.columns if 'pdb' in col.lower()]
        if pdb_cols:
            print(f"Using PDB column: {pdb_cols[0]}")
            print("Manual filtering may be required")

    print(f"\nFiltered: {initial_count} → {len(df)} samples")

    return df


def save_dataset(df, output_path='external_data/saaint_raw.csv'):
    """
    Save dataset to CSV

    Args:
        df: DataFrame to save
        output_path: Output file path
    """
    print("\n" + "="*80)
    print("SAVING DATASET")
    print("="*80)

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    df.to_csv(output_path, index=False)

    print(f"\nOK Saved to: {output_path}")
    print(f"  Samples: {len(df)}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return output_path


def main():
    """Main download workflow"""
    print("\n" + "="*80)
    print("SAAINT-DB DOWNLOAD SCRIPT")
    print("="*80)
    print("\nThis script will:")
    print("1. Clone SAAINT repository from GitHub")
    print("2. Find affinity data files")
    print("3. Extract and convert to pKd format")
    print("4. Filter for antibody-antigen complexes")
    print("5. Save to CSV")
    print("\n" + "="*80)

    # 1. Clone repository
    repo_path = clone_saaint_repository()
    if repo_path is None:
        return

    # 2. Find affinity files
    affinity_files = find_affinity_files(repo_path)

    if not affinity_files:
        print("\nWARNING No affinity data files found")
        print("Please check the repository structure manually")
        return

    # 3. Try to load each file
    print("\n" + "="*80)
    print("TRYING TO LOAD FILES")
    print("="*80)

    df = None
    for file_path in affinity_files:
        print(f"\nTrying: {file_path.name}")
        temp_df = load_saaint_data(file_path)

        if temp_df is not None and len(temp_df) > 0:
            # Check if it has affinity data
            affinity_cols = [col for col in temp_df.columns if
                           any(term in col.lower() for term in
                               ['kd', 'ki', 'affinity', 'pkd', 'bind'])]

            if affinity_cols:
                print(f"  OK Found affinity columns: {affinity_cols}")
                df = temp_df
                break

    if df is None:
        print("\nERROR Could not find suitable affinity data file")
        print("Please manually inspect the repository files")
        return

    # 4. Explore structure
    explore_saaint_structure(df)

    # 5. Extract affinity data
    df = extract_affinity_data(df)

    # 6. Convert to pKd
    df = convert_to_pkd(df)

    # 7. Filter for antibody-antigen (optional)
    # Uncomment if you want to filter
    # df = filter_antibody_antigen(df)

    # 8. Save
    output_path = save_dataset(df)

    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"\nOK Dataset saved to: {output_path}")
    print(f"OK Total samples: {len(df)}")

    if 'pKd' in df.columns:
        valid_pkd = df['pKd'].notna().sum()
        print(f"OK Samples with pKd: {valid_pkd}")

    print("\nNext steps:")
    print("1. Review the data: check the CSV file")
    print("2. Filter for antibody-antigen complexes if needed")
    print("3. Run integration script to merge with existing dataset")
    print("4. Generate ESM2 embeddings for new sequences")

    print("\nFiles created:")
    print(f"  - {output_path}")
    print(f"  - {repo_path}/ (cloned repository)")


if __name__ == "__main__":
    main()
