"""
Download and process PDBbind dataset
PDBbind is the gold standard for protein-ligand and protein-protein binding affinity data

PDBbind 2020 (free) contains:
- 19,443 biomolecular complexes total
- 4,463 protein-protein complexes (includes antibody-antigen)
- Kd, Ki, IC50 values
- Standardized format

PDBbind 2024 (free registration) contains:
- 33,653 biomolecular complexes
- 4,594 protein-protein complexes
- 43% increase over 2020
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_utils import AffinityBinner, print_dataset_statistics


def check_for_downloaded_files(search_dir='./external_data'):
    """
    Check if PDBbind files have been downloaded

    Args:
        search_dir: Directory to search for downloaded files

    Returns:
        List of found PDBbind files
    """
    print("\n" + "="*80)
    print("CHECKING FOR PDBBIND FILES")
    print("="*80)

    search_path = Path(search_dir)
    search_path.mkdir(parents=True, exist_ok=True)

    # Common PDBbind file patterns
    patterns = [
        '*PDBbind*',
        '*pdbbind*',
        'PP_INDEX*',  # Protein-protein index
        '*general_set*',
        '*refined_set*',
        '*core_set*',
        'INDEX*'
    ]

    found_files = []

    for pattern in patterns:
        files = list(search_path.rglob(pattern))
        found_files.extend(files)

    # Remove duplicates
    found_files = list(set(found_files))

    if found_files:
        print(f"\nOK Found {len(found_files)} PDBbind-related files:")
        for f in found_files:
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.1f} KB)")
    else:
        print("\nWARNING No PDBbind files found")

    return found_files


def show_download_instructions():
    """
    Show instructions for downloading PDBbind data
    """
    print("\n" + "="*80)
    print("PDBBIND DOWNLOAD INSTRUCTIONS")
    print("="*80)

    print("\n📥 PDBbind 2020 (Free, No Registration):")
    print("-" * 80)
    print("1. Visit: http://www.pdbbind.org.cn/download.php")
    print("2. Scroll to 'PDBbind v2020' section")
    print("3. Download the following files:")
    print("   - General set: PP_INDEX_general_set.2020")
    print("   - OR Refined set: PP_INDEX_refined_set.2020")
    print("   - (These contain protein-protein binding data)")
    print("4. Save to: external_data/")

    print("\n📥 PDBbind 2024 (Free Registration Required):")
    print("-" * 80)
    print("1. Visit: https://www.pdbbind-plus.org.cn/")
    print("2. Click 'Register' (free for academics)")
    print("3. Verify email and login")
    print("4. Download protein-protein index file")
    print("5. Save to: external_data/")

    print("\n💡 Recommended:")
    print("-" * 80)
    print("Download the 2020 version first (no registration)")
    print("Then optionally register for 2024 for more data")

    print("\n📁 After downloading:")
    print("-" * 80)
    print("Run this script again to process the downloaded files")


def parse_pdbbind_index(file_path):
    """
    Parse PDBbind index file

    PDBbind index format:
    # PDB code / resolution / release year / -logKd/Ki / Kd/Ki / reference / ligand name

    Args:
        file_path: Path to PDBbind index file

    Returns:
        DataFrame with parsed data
    """
    print("\n" + "="*80)
    print("PARSING PDBBIND INDEX FILE")
    print("="*80)

    print(f"\nReading: {file_path}")

    data = []
    header_found = False

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Skip comments and empty lines
            if line.startswith('#') or not line.strip():
                if 'PDB code' in line:
                    header_found = True
                    print(f"  Header line {line_num}: {line.strip()}")
                continue

            # Parse data line
            try:
                # Split by whitespace (typically multiple spaces)
                parts = line.strip().split()

                if len(parts) >= 5:
                    pdb_code = parts[0]
                    resolution = parts[1]
                    year = parts[2]
                    log_affinity = parts[3]
                    affinity = parts[4]

                    # Parse remaining parts (reference, ligand name)
                    remaining = ' '.join(parts[5:])

                    data.append({
                        'pdb_code': pdb_code,
                        'resolution': resolution,
                        'release_year': year,
                        'log_affinity': log_affinity,
                        'affinity_value': affinity,
                        'additional_info': remaining
                    })

            except Exception as e:
                print(f"  WARNING Error parsing line {line_num}: {e}")
                continue

    df = pd.DataFrame(data)

    print(f"\nOK Parsed {len(df)} entries")
    print(f"  Columns: {list(df.columns)}")

    return df


def parse_affinity_values(df):
    """
    Parse affinity values and units from PDBbind data

    Common formats:
    - Kd=10nM
    - Ki=1.5uM
    - IC50=100pM
    - Kd>10mM (upper bound)
    - Kd<1nM (lower bound)

    Args:
        df: DataFrame with affinity_value column

    Returns:
        DataFrame with parsed affinity and unit columns
    """
    print("\n" + "="*80)
    print("PARSING AFFINITY VALUES")
    print("="*80)

    df = df.copy()

    # Extract affinity type (Kd, Ki, IC50)
    df['affinity_type'] = df['affinity_value'].str.extract(r'(Kd|Ki|IC50|Ka)', expand=False)

    # Extract operator (=, <, >, ~)
    df['operator'] = df['affinity_value'].str.extract(r'[Kd|Ki|IC50]+([=<>~])', expand=False)

    # Extract numeric value
    df['affinity_numeric'] = df['affinity_value'].str.extract(r'([0-9.]+)', expand=False).astype(float)

    # Extract unit (nM, uM, pM, mM)
    df['affinity_unit'] = df['affinity_value'].str.extract(r'([pnum]M)', expand=False)

    # Show statistics
    print("\nAffinity types:")
    print(df['affinity_type'].value_counts())

    print("\nAffinity units:")
    print(df['affinity_unit'].value_counts())

    print("\nOperators:")
    print(df['operator'].value_counts())

    # Show value range
    valid_values = df['affinity_numeric'].dropna()
    if len(valid_values) > 0:
        print(f"\nNumeric values:")
        print(f"  Range: [{valid_values.min():.2e}, {valid_values.max():.2e}]")
        print(f"  Median: {valid_values.median():.2e}")

    return df


def convert_to_pkd(df):
    """
    Convert PDBbind affinity values to pKd

    Args:
        df: DataFrame with parsed affinity columns

    Returns:
        DataFrame with pKd column added
    """
    print("\n" + "="*80)
    print("CONVERTING TO pKd")
    print("="*80)

    df = df.copy()

    # Check if log_affinity column exists (this is often pKd or pKi)
    if 'log_affinity' in df.columns:
        try:
            df['log_affinity_numeric'] = pd.to_numeric(df['log_affinity'], errors='coerce')
            has_log = df['log_affinity_numeric'].notna().sum()
            if has_log > 0:
                print(f"\nOK Found log_affinity column with {has_log} values")
                print("  This appears to be pKd/pKi values")
        except:
            pass

    # Convert from affinity_numeric + affinity_unit
    if 'affinity_numeric' in df.columns and 'affinity_unit' in df.columns:
        df['pKd'] = np.nan

        for unit in ['pM', 'nM', 'uM', 'mM']:
            mask = df['affinity_unit'] == unit

            if mask.any():
                values = df.loc[mask, 'affinity_numeric']

                if unit == 'pM':
                    # pM to M: divide by 1e12
                    kd_m = values / 1e12
                elif unit == 'nM':
                    # nM to M: divide by 1e9
                    kd_m = values / 1e9
                elif unit == 'uM':
                    # μM to M: divide by 1e6
                    kd_m = values / 1e6
                elif unit == 'mM':
                    # mM to M: divide by 1e3
                    kd_m = values / 1e3
                else:
                    continue

                df.loc[mask, 'pKd'] = -np.log10(kd_m)

                print(f"\n  Converted {mask.sum()} {unit} values")

    # Use log_affinity if pKd not computed
    if 'pKd' not in df.columns or df['pKd'].isna().all():
        if 'log_affinity_numeric' in df.columns:
            print("\n  Using log_affinity column as pKd")
            df['pKd'] = df['log_affinity_numeric']

    # Remove invalid values
    if 'pKd' in df.columns:
        invalid_mask = np.isinf(df['pKd']) | np.isnan(df['pKd'])
        if invalid_mask.any():
            n_invalid = invalid_mask.sum()
            print(f"\n  WARNING Found {n_invalid} invalid pKd values")

        # Statistics
        valid_pkd = df['pKd'].dropna()
        if len(valid_pkd) > 0:
            print(f"\npKd conversion results:")
            print(f"  Valid: {len(valid_pkd)} values")
            print(f"  Range: [{valid_pkd.min():.2f}, {valid_pkd.max():.2f}]")
            print(f"  Mean: {valid_pkd.mean():.2f}")
            print(f"  Median: {valid_pkd.median():.2f}")

            # Show distribution
            binner = AffinityBinner()
            print_dataset_statistics(valid_pkd.values, "PDBbind", binner)

    return df


def filter_antibody_antigen(df):
    """
    Filter for antibody-antigen complexes

    Args:
        df: PDBbind DataFrame

    Returns:
        Filtered DataFrame
    """
    print("\n" + "="*80)
    print("FILTERING FOR ANTIBODY-ANTIGEN COMPLEXES")
    print("="*80)

    initial_count = len(df)

    # PDBbind protein-protein index typically includes all types
    # Need to identify antibody-antigen by:
    # 1. PDB code (some are known antibody structures)
    # 2. Additional info field
    # 3. Manual inspection

    print("\nWARNING Automatic antibody-antigen filtering not yet implemented")
    print("  PDBbind protein-protein index contains various complex types:")
    print("  - Antibody-antigen")
    print("  - Enzyme-inhibitor")
    print("  - Receptor-ligand")
    print("  - Other protein-protein interactions")

    print("\n💡 Recommendation:")
    print("  1. Save the full dataset")
    print("  2. Manually filter for antibody-antigen using PDB codes")
    print("  3. OR cross-reference with SAbDab/SAAINT databases")

    # Check additional_info for antibody keywords
    if 'additional_info' in df.columns:
        antibody_keywords = ['antibody', 'immunoglobulin', 'IgG', 'Fab', 'scFv']
        mask = df['additional_info'].str.contains(
            '|'.join(antibody_keywords), case=False, na=False
        )

        if mask.any():
            n_antibody = mask.sum()
            print(f"\n  Found {n_antibody} entries with antibody keywords")
            print("  This is a rough filter - manual curation recommended")

            # Optionally apply filter
            # df = df[mask].copy()

    print(f"\nTotal entries: {len(df)}")
    print("  (No automatic filtering applied)")

    return df


def save_dataset(df, output_path='external_data/pdbbind_raw.csv'):
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
    print("PDBBIND DOWNLOAD SCRIPT")
    print("="*80)
    print("\nThis script will:")
    print("1. Check for downloaded PDBbind files")
    print("2. Show download instructions if needed")
    print("3. Parse affinity data")
    print("4. Convert to pKd format")
    print("5. Save to CSV")
    print("\n" + "="*80)

    # 1. Check for existing files
    found_files = check_for_downloaded_files()

    if not found_files:
        # Show download instructions
        show_download_instructions()

        print("\n" + "="*80)
        print("PLEASE DOWNLOAD FILES FIRST")
        print("="*80)
        print("\nAfter downloading:")
        print("  python scripts/download_pdbbind.py")
        return

    # 2. Find index files
    index_files = [f for f in found_files if
                  'INDEX' in f.name.upper() or 'PP_INDEX' in f.name]

    if not index_files:
        print("\nWARNING No INDEX files found")
        print("Expected files like: PP_INDEX_general_set.2020")
        return

    print(f"\nOK Found {len(index_files)} index file(s)")

    # 3. Parse the first/largest index file
    index_file = max(index_files, key=lambda f: f.stat().st_size)
    print(f"\nUsing: {index_file.name}")

    df = parse_pdbbind_index(index_file)

    if df is None or len(df) == 0:
        print("\nERROR Failed to parse index file")
        return

    # 4. Parse affinity values
    df = parse_affinity_values(df)

    # 5. Convert to pKd
    df = convert_to_pkd(df)

    # 6. Filter for antibody-antigen (optional)
    df = filter_antibody_antigen(df)

    # 7. Save
    output_path = save_dataset(df)

    # Summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    print(f"\nOK Dataset saved to: {output_path}")
    print(f"OK Total samples: {len(df)}")

    if 'pKd' in df.columns:
        valid_pkd = df['pKd'].notna().sum()
        print(f"OK Samples with pKd: {valid_pkd}")

    print("\nNext steps:")
    print("1. Review the data: check the CSV file")
    print("2. Filter for antibody-antigen complexes")
    print("3. Cross-reference PDB codes with SAbDab/SAAINT")
    print("4. Run integration script to merge with existing dataset")

    print("\n💡 For antibody-antigen filtering:")
    print("  - Compare PDB codes with your existing dataset")
    print("  - Use SAbDab PDB code list")
    print("  - Manual inspection recommended")


if __name__ == "__main__":
    main()
