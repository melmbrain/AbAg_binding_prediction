#!/usr/bin/env python3
"""
Download and Process Ab-CoV Database

Ab-CoV: COVID-19 Antibody Database with Affinity Measurements
- 1,780 coronavirus antibodies
- 568 Kd measurements
- Expected: 100-200 very strong binders (pKd > 11)

Website: https://web.iitm.ac.in/ab-cov/home
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
import sys
from io import StringIO
import re

def convert_kd_to_pkd(kd_value, kd_unit):
    """
    Convert Kd value with unit to pKd

    Args:
        kd_value: float, the Kd value
        kd_unit: str, unit ('M', 'mM', 'uM', 'nM', 'pM', 'fM')

    Returns:
        pKd: float, -log10(Kd in M)
    """
    if pd.isna(kd_value) or pd.isna(kd_unit):
        return np.nan

    # Convert to Molar
    unit_multipliers = {
        'M': 1.0,
        'mM': 1e-3,
        'uM': 1e-6,
        'µM': 1e-6,
        'nM': 1e-9,
        'pM': 1e-12,
        'fM': 1e-15
    }

    kd_unit_clean = kd_unit.strip()
    if kd_unit_clean not in unit_multipliers:
        print(f"[WARNING] Unknown unit: {kd_unit_clean}, skipping")
        return np.nan

    kd_molar = float(kd_value) * unit_multipliers[kd_unit_clean]

    if kd_molar <= 0:
        return np.nan

    pkd = -np.log10(kd_molar)
    return pkd

def download_abcov():
    """Download Ab-CoV database"""
    print("="*80)
    print("DOWNLOADING Ab-CoV DATABASE")
    print("="*80)
    print("\nDatabase: COVID-19 Antibody Database")
    print("Source: IIT Madras")
    print("URL: https://web.iitm.ac.in/ab-cov/home")
    print("")

    # Try different potential download URLs
    urls_to_try = [
        "https://web.iitm.ac.in/ab-cov/download",
        "https://web.iitm.ac.in/ab-cov/static/downloads/ab-cov-data.csv",
        "https://web.iitm.ac.in/ab-cov/api/download",
    ]

    output_dir = Path("external_data/therapeutic")
    output_dir.mkdir(parents=True, exist_ok=True)

    for url in urls_to_try:
        print(f"\n[INFO] Trying URL: {url}")
        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                # Check if it's CSV data
                content = response.text
                if ',' in content[:1000]:  # Basic CSV check
                    output = output_dir / "abcov_raw.csv"
                    output.write_text(content)

                    size_mb = output.stat().st_size / (1024 * 1024)
                    print(f"[OK] Downloaded {size_mb:.2f} MB")
                    print(f"     Saved to: {output}")

                    # Try to parse
                    try:
                        df = pd.read_csv(output)
                        print(f"[OK] Loaded {len(df)} entries")
                        print(f"     Columns: {list(df.columns)}")
                        return df
                    except Exception as e:
                        print(f"[WARNING] Could not parse as CSV: {e}")
                        continue
            else:
                print(f"[INFO] Status code: {response.status_code}")

        except Exception as e:
            print(f"[INFO] Failed: {e}")
            continue

    # If automated download failed
    print("\n" + "="*80)
    print("AUTOMATED DOWNLOAD FAILED")
    print("="*80)
    print("\n[INFO] Ab-CoV may require manual download through web interface")
    print("\nMANUAL DOWNLOAD INSTRUCTIONS:")
    print("  1. Visit: https://web.iitm.ac.in/ab-cov/home")
    print("  2. Click 'Download' or 'Export' button")
    print("  3. Save as: external_data/therapeutic/abcov_raw.csv")
    print("  4. Run this script again to process the data")
    print("")

    # Check if manual file exists
    manual_file = output_dir / "abcov_raw.csv"
    if manual_file.exists():
        print(f"[OK] Found manually downloaded file: {manual_file}")
        try:
            df = pd.read_csv(manual_file)
            print(f"[OK] Loaded {len(df)} entries from manual download")
            return df
        except Exception as e:
            print(f"[ERROR] Could not parse manual file: {e}")

    return None

def process_abcov(df):
    """Process Ab-CoV data to extract high-affinity antibodies"""
    if df is None:
        return None

    print("\n" + "="*80)
    print("PROCESSING Ab-CoV DATA")
    print("="*80)

    print(f"\n[INFO] Raw data: {len(df)} entries")
    print(f"[INFO] Columns: {list(df.columns)}")

    # Look for Kd columns
    kd_columns = [col for col in df.columns if 'kd' in col.lower() or 'affinity' in col.lower()]
    print(f"[INFO] Potential affinity columns: {kd_columns}")

    # Common column names in antibody databases
    common_mappings = {
        'Kd': 'kd_value',
        'KD': 'kd_value',
        'Kd_value': 'kd_value',
        'Affinity': 'kd_value',
        'Unit': 'kd_unit',
        'Kd_unit': 'kd_unit',
        'heavy_chain': 'heavy_chain_seq',
        'light_chain': 'light_chain_seq',
        'VH': 'heavy_chain_seq',
        'VL': 'light_chain_seq',
        'Antibody': 'antibody_name',
        'Target': 'antigen_name',
        'Antigen': 'antigen_name',
    }

    # Try to standardize columns
    df_processed = df.copy()
    for old_col, new_col in common_mappings.items():
        if old_col in df_processed.columns:
            df_processed = df_processed.rename(columns={old_col: new_col})

    # Calculate pKd if we have Kd values
    if 'kd_value' in df_processed.columns:
        print("\n[INFO] Converting Kd to pKd...")

        if 'kd_unit' in df_processed.columns:
            df_processed['pKd'] = df_processed.apply(
                lambda row: convert_kd_to_pkd(row['kd_value'], row['kd_unit']),
                axis=1
            )
        else:
            # Assume nM if no unit specified (common in COVID antibody data)
            print("[WARNING] No unit column found, assuming nM")
            df_processed['pKd'] = df_processed['kd_value'].apply(
                lambda x: convert_kd_to_pkd(x, 'nM') if pd.notna(x) else np.nan
            )

        # Filter for entries with pKd
        df_with_pkd = df_processed[df_processed['pKd'].notna()].copy()
        print(f"[OK] {len(df_with_pkd)} entries with pKd values")

        # Analyze distribution
        if len(df_with_pkd) > 0:
            print(f"\npKd Statistics:")
            print(f"  Min:  {df_with_pkd['pKd'].min():.2f}")
            print(f"  Max:  {df_with_pkd['pKd'].max():.2f}")
            print(f"  Mean: {df_with_pkd['pKd'].mean():.2f}")
            print(f"  Median: {df_with_pkd['pKd'].median():.2f}")

            # Count very strong binders
            very_strong = df_with_pkd[df_with_pkd['pKd'] > 11]
            print(f"\n[OK] VERY STRONG BINDERS (pKd > 11): {len(very_strong)}")

            if len(very_strong) > 0:
                print(f"\nTop 10 Strongest Binders:")
                top_10 = very_strong.nlargest(10, 'pKd')
                for idx, row in top_10.iterrows():
                    kd_pm = 10**(-row['pKd']) * 1e12  # Convert to pM
                    antibody_name = row.get('antibody_name', 'Unknown')
                    print(f"  pKd = {row['pKd']:.2f} ({kd_pm:.1f} pM) - {antibody_name}")

            # Save processed data
            output = Path("external_data/therapeutic/abcov_processed.csv")
            df_with_pkd.to_csv(output, index=False)
            print(f"\n[OK] Saved processed data to: {output}")

            # Save very strong binders separately
            if len(very_strong) > 0:
                output_strong = Path("external_data/therapeutic/abcov_very_strong.csv")
                very_strong.to_csv(output_strong, index=False)
                print(f"[OK] Saved very strong binders to: {output_strong}")

            return df_with_pkd

    else:
        print("[WARNING] Could not find Kd columns in data")
        print("[INFO] Available columns:", list(df_processed.columns))

        # Save anyway for manual inspection
        output = Path("external_data/therapeutic/abcov_processed.csv")
        df_processed.to_csv(output, index=False)
        print(f"[INFO] Saved raw processed data to: {output}")
        return df_processed

    return None

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("Ab-CoV DATABASE DOWNLOADER AND PROCESSOR")
    print("="*80)
    print("\nTarget: COVID-19 neutralizing antibodies with high affinity")
    print("Expected: 100-200 very strong binders (pKd > 11)")
    print("")

    # Download
    df = download_abcov()

    # Process
    df_processed = process_abcov(df)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if df_processed is not None and len(df_processed) > 0:
        print(f"\n[OK] Successfully processed {len(df_processed)} Ab-CoV entries")

        if 'pKd' in df_processed.columns:
            very_strong = df_processed[df_processed['pKd'] > 11]
            print(f"[OK] Found {len(very_strong)} very strong binders (pKd > 11)")
            print("\n[INFO] Ready for integration with existing dataset")
        else:
            print("[WARNING] No pKd values calculated")
            print("[INFO] Manual inspection required")
    else:
        print("\n[ERROR] Failed to download or process Ab-CoV data")
        print("\n[INFO] Please try manual download:")
        print("       1. Visit https://web.iitm.ac.in/ab-cov/home")
        print("       2. Download data as CSV")
        print("       3. Save to external_data/therapeutic/abcov_raw.csv")
        print("       4. Run this script again")

    return df_processed

if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0 if result is not None else 1)
    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
