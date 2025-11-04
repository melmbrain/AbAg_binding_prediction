"""
Download AbBiBench dataset from Hugging Face
This is the easiest database to download and has high-quality affinity data

AbBiBench contains 184,500+ experimental measurements across:
- 14 antibodies
- 9 antigens (HER2, VEGF, influenza, SARS-CoV-2, lysozyme, etc.)
- Both heavy and light chain mutations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_utils import AffinityBinner, print_dataset_statistics


def install_datasets():
    """Install Hugging Face datasets library if not present"""
    try:
        import datasets
        print("OK datasets library already installed")
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        print("OK datasets library installed")


def download_abbibench(cache_dir='./external_data/abbibench_cache'):
    """
    Download AbBiBench dataset from Hugging Face

    Args:
        cache_dir: Directory to cache downloaded data

    Returns:
        dataset: Hugging Face Dataset object
    """
    print("\n" + "="*80)
    print("DOWNLOADING ABBIBENCH FROM HUGGING FACE")
    print("="*80)

    # Install datasets library if needed
    install_datasets()

    from datasets import load_dataset

    print("\nDownloading dataset...")
    print("This may take a few minutes on first download...")

    try:
        # Load dataset
        dataset = load_dataset(
            "AbBibench/Antibody_Binding_Benchmark_Dataset",
            cache_dir=cache_dir
        )

        print(f"\nOK Download complete!")
        print(f"  Dataset splits: {list(dataset.keys())}")

        if 'train' in dataset:
            print(f"  Train samples: {len(dataset['train'])}")
        if 'test' in dataset:
            print(f"  Test samples: {len(dataset['test'])}")
        if 'validation' in dataset:
            print(f"  Validation samples: {len(dataset['validation'])}")

        return dataset

    except Exception as e:
        print(f"\nERROR Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Try: pip install --upgrade datasets")
        print("3. Visit: https://huggingface.co/datasets/AbBibench/Antibody_Binding_Benchmark_Dataset")
        return None


def explore_dataset_structure(dataset):
    """
    Explore the structure of the dataset

    Args:
        dataset: Hugging Face Dataset object
    """
    print("\n" + "="*80)
    print("DATASET STRUCTURE")
    print("="*80)

    # Get first split (usually 'train')
    split_name = list(dataset.keys())[0]
    data = dataset[split_name]

    print(f"\nExamining '{split_name}' split...")
    print(f"Total samples: {len(data)}")

    # Show column names
    print(f"\nColumns ({len(data.column_names)}):")
    for col in data.column_names:
        print(f"  - {col}")

    # Show first example
    print("\nFirst sample:")
    first_sample = data[0]
    for key, value in first_sample.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}... (truncated)")
        else:
            print(f"  {key}: {value}")

    return data


def convert_to_dataframe(dataset):
    """
    Convert Hugging Face dataset to pandas DataFrame

    Args:
        dataset: Hugging Face Dataset object

    Returns:
        DataFrame with all data
    """
    print("\n" + "="*80)
    print("CONVERTING TO PANDAS DATAFRAME")
    print("="*80)

    dfs = []

    for split_name, data in dataset.items():
        print(f"\nConverting '{split_name}' split...")
        df = pd.DataFrame(data)
        df['split'] = split_name
        dfs.append(df)

    # Combine all splits
    combined_df = pd.concat(dfs, ignore_index=True)

    print(f"\nOK Conversion complete")
    print(f"  Total samples: {len(combined_df)}")
    print(f"  Columns: {len(combined_df.columns)}")

    return combined_df


def extract_affinity_data(df):
    """
    Extract and process affinity data from AbBiBench

    Args:
        df: DataFrame from AbBiBench

    Returns:
        Processed DataFrame with affinity values
    """
    print("\n" + "="*80)
    print("EXTRACTING AFFINITY DATA")
    print("="*80)

    print("\nAvailable columns:")
    print(df.columns.tolist())

    # Common affinity column names in AbBiBench
    possible_affinity_cols = [
        'binding_affinity',
        'affinity',
        'kd',
        'Kd',
        'KD',
        'delta_g',
        'ddg',
        'binding_score'
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
                          'affinity' in col.lower() or
                          'kd' in col.lower() or
                          'bind' in col.lower()]
        if affinity_related:
            print(f"Found related columns: {affinity_related}")
            affinity_col = affinity_related[0]
            print(f"Using: {affinity_col}")
        else:
            print("ERROR Could not find affinity data")
            return df

    # Check affinity units
    unit_cols = [col for col in df.columns if 'unit' in col.lower()]
    if unit_cols:
        print(f"Unit column found: {unit_cols[0]}")
        print(f"Units: {df[unit_cols[0]].unique()}")

    # Statistics
    print(f"\nAffinity statistics:")
    print(f"  Non-null values: {df[affinity_col].notna().sum()}")
    print(f"  Null values: {df[affinity_col].isna().sum()}")

    if df[affinity_col].notna().any():
        valid_data = df[affinity_col].dropna()
        print(f"  Min: {valid_data.min()}")
        print(f"  Max: {valid_data.max()}")
        print(f"  Mean: {valid_data.mean():.4f}")
        print(f"  Median: {valid_data.median():.4f}")

    return df


def convert_to_pkd(df, affinity_col='binding_affinity', unit_col='affinity_unit'):
    """
    Convert affinity values to pKd

    Args:
        df: DataFrame
        affinity_col: Column containing affinity values
        unit_col: Column containing units (if exists)

    Returns:
        DataFrame with pKd column added
    """
    print("\n" + "="*80)
    print("CONVERTING TO pKd")
    print("="*80)

    df = df.copy()

    if affinity_col not in df.columns:
        print(f"ERROR Column '{affinity_col}' not found")
        return df

    # Check if unit column exists
    has_units = unit_col in df.columns

    if has_units:
        print(f"\nUnits found: {df[unit_col].unique()}")

        # Convert based on unit
        df['pKd'] = np.nan

        for unit in df[unit_col].unique():
            mask = df[unit_col] == unit

            if pd.isna(unit):
                continue

            unit_lower = str(unit).lower()

            # Convert to Molar then to pKd
            if 'nm' in unit_lower or 'nanomolar' in unit_lower:
                # nM to M: divide by 1e9
                kd_m = df.loc[mask, affinity_col] / 1e9
                df.loc[mask, 'pKd'] = -np.log10(kd_m)

            elif 'um' in unit_lower or 'μm' in unit_lower or 'micromolar' in unit_lower:
                # μM to M: divide by 1e6
                kd_m = df.loc[mask, affinity_col] / 1e6
                df.loc[mask, 'pKd'] = -np.log10(kd_m)

            elif 'pm' in unit_lower or 'picomolar' in unit_lower:
                # pM to M: divide by 1e12
                kd_m = df.loc[mask, affinity_col] / 1e12
                df.loc[mask, 'pKd'] = -np.log10(kd_m)

            elif 'm' == unit_lower or 'molar' in unit_lower:
                # Already in Molar
                df.loc[mask, 'pKd'] = -np.log10(df.loc[mask, affinity_col])

            else:
                print(f"WARNING Unknown unit: {unit}")

    else:
        # Assume values are in standard format (check range to guess)
        print("\nNo unit column found, attempting to infer...")

        # If values are very small (< 1e-6), likely in M
        # If values are moderate (1-1000), likely in nM
        # If values are large (> 1000), might be μM or other

        sample_values = df[affinity_col].dropna().head(100)
        median_val = sample_values.median()

        if median_val < 1e-6:
            print(f"Values appear to be in Molar (median: {median_val:.2e})")
            df['pKd'] = -np.log10(df[affinity_col])
        elif 1 <= median_val <= 1000:
            print(f"Values appear to be in nM (median: {median_val:.2f})")
            df['pKd'] = -np.log10(df[affinity_col] / 1e9)
        else:
            print(f"WARNING Cannot determine units (median: {median_val:.2e})")
            print("Assuming Molar...")
            df['pKd'] = -np.log10(df[affinity_col])

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
        print_dataset_statistics(valid_pkd.values, "AbBiBench", binner)

    return df


def save_dataset(df, output_path='external_data/abbibench_raw.csv'):
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
    print("ABBIBENCH DOWNLOAD SCRIPT")
    print("="*80)
    print("\nThis script will:")
    print("1. Download AbBiBench from Hugging Face")
    print("2. Extract affinity data")
    print("3. Convert to pKd format")
    print("4. Save to CSV")
    print("\n" + "="*80)

    # 1. Download
    dataset = download_abbibench()
    if dataset is None:
        return

    # 2. Explore
    explore_dataset_structure(dataset)

    # 3. Convert to DataFrame
    df = convert_to_dataframe(dataset)

    # 4. Extract affinity data
    df = extract_affinity_data(df)

    # 5. Convert to pKd
    df = convert_to_pkd(df)

    # 6. Save
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
    print("2. Run integration script to merge with existing dataset")
    print("3. Generate ESM2 embeddings for new sequences")

    print("\nFiles created:")
    print(f"  - {output_path}")
    print(f"  - external_data/abbibench_cache/ (cached download)")


if __name__ == "__main__":
    main()
