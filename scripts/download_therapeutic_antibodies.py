#!/usr/bin/env python3
"""
Download Therapeutic and Vaccine Antibody Data

Downloads high-affinity antibody data from:
1. Ab-CoV (COVID-19 antibodies with Kd values)
2. CoV-AbDab (Coronavirus antibody database)
3. Thera-SAbDab (Therapeutic antibodies)
4. SAbDab (Structural antibody database with affinity filter)

Focus: Very strong binders (pKd > 11, Kd < 100 pM)
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
import sys
import time
from io import StringIO

def download_file(url, output_path, description=""):
    """Download file from URL with progress"""
    print(f"\n[INFO] Downloading {description}...")
    print(f"       URL: {url}")
    print(f"       Output: {output_path}")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[OK] Downloaded {size_mb:.2f} MB")
        return True

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Download failed: {e}")
        return False

def download_covabdab():
    """Download CoV-AbDab database"""
    print("\n" + "="*80)
    print("DOWNLOADING CoV-AbDab (Coronavirus Antibody Database)")
    print("="*80)

    url = "http://opig.stats.ox.ac.uk/webapps/covabdab/static/downloads/CoV-AbDab_260125.csv"
    output = Path("external_data/therapeutic/covabdab_raw.csv")

    if download_file(url, output, "CoV-AbDab"):
        try:
            df = pd.read_csv(output)
            print(f"[OK] Loaded {len(df)} antibody entries")
            print(f"     Columns: {', '.join(df.columns[:10].tolist())}...")
            return df
        except Exception as e:
            print(f"[ERROR] Failed to parse CSV: {e}")
            return None
    return None

def download_therasabdab():
    """Download Thera-SAbDab database"""
    print("\n" + "="*80)
    print("DOWNLOADING Thera-SAbDab (Therapeutic Antibodies)")
    print("="*80)

    # Note: The actual download URL may require manual access
    # This is a placeholder - will try the documented URL
    url = "http://opig.stats.ox.ac.uk/webapps/newsabdab/therasabdab/download"
    output = Path("external_data/therapeutic/therasabdab_raw.csv")

    print("[INFO] Thera-SAbDab download may require manual access")
    print("       Attempting automated download...")

    if download_file(url, output, "Thera-SAbDab"):
        try:
            df = pd.read_csv(output)
            print(f"[OK] Loaded {len(df)} therapeutic antibodies")
            return df
        except Exception as e:
            print(f"[ERROR] Failed to parse CSV: {e}")
            print("[INFO] You may need to download manually from:")
            print("       http://opig.stats.ox.ac.uk/webapps/newsabdab/therasabdab/")
            return None
    return None

def download_sabdab_affinity():
    """Download SAbDab entries with affinity data"""
    print("\n" + "="*80)
    print("DOWNLOADING SAbDab (With Affinity Filter)")
    print("="*80)

    # SAbDab search/download endpoint
    # This may need to be done through their web interface
    print("[INFO] SAbDab requires web interface for filtered downloads")
    print("       Visit: http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab")
    print("       Filter: Affinity Data = Yes")
    print("       Then download as CSV")
    print("")
    print("[INFO] Attempting to download summary file...")

    # Try to get the summary file
    url = "http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all"
    output = Path("external_data/therapeutic/sabdab_summary.tsv")

    if download_file(url, output, "SAbDab Summary"):
        try:
            df = pd.read_csv(output, sep='\t')
            print(f"[OK] Loaded {len(df)} SAbDab entries")

            # Filter for entries with affinity
            if 'affinity' in df.columns:
                df_affinity = df[df['affinity'].notna()]
                print(f"[OK] Found {len(df_affinity)} entries with affinity data")

                output_filtered = Path("external_data/therapeutic/sabdab_affinity.tsv")
                df_affinity.to_csv(output_filtered, sep='\t', index=False)
                return df_affinity
            else:
                print("[WARNING] No 'affinity' column found in SAbDab summary")
                return df
        except Exception as e:
            print(f"[ERROR] Failed to parse TSV: {e}")
            return None
    return None

def process_covabdab(df):
    """Process CoV-AbDab data to extract affinity information"""
    print("\n[INFO] Processing CoV-AbDab data...")

    if df is None:
        return None

    # CoV-AbDab may have neutralization data but not direct Kd
    # We'll cross-reference with structures
    print(f"[INFO] CoV-AbDab contains {len(df)} antibody entries")
    print("[INFO] This database is primarily for sequences and structures")
    print("[INFO] Will be used for cross-referencing with other databases")

    # Save processed version
    output = Path("external_data/therapeutic/covabdab_processed.csv")
    df.to_csv(output, index=False)
    print(f"[OK] Saved to {output}")

    return df

def process_therasabdab(df):
    """Process Thera-SAbDab data"""
    print("\n[INFO] Processing Thera-SAbDab data...")

    if df is None:
        return None

    print(f"[INFO] Thera-SAbDab contains {len(df)} therapeutic antibodies")

    # Save processed version
    output = Path("external_data/therapeutic/therasabdab_processed.csv")
    df.to_csv(output, index=False)
    print(f"[OK] Saved to {output}")

    return df

def main():
    """Main download orchestration"""
    print("="*80)
    print("THERAPEUTIC ANTIBODY DATABASE DOWNLOADER")
    print("="*80)
    print("\nTarget: High-affinity therapeutic and vaccine antibodies")
    print("Goal: Boost very strong binders (pKd > 11, Kd < 100 pM)")
    print("")

    # Create output directory
    output_dir = Path("external_data/therapeutic")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Download databases
    print("\n" + "="*80)
    print("PHASE 1: DOWNLOADING DATABASES")
    print("="*80)

    # 1. CoV-AbDab (largest, easiest to download)
    df_covabdab = download_covabdab()
    if df_covabdab is not None:
        results['covabdab'] = process_covabdab(df_covabdab)

    time.sleep(2)  # Be polite to servers

    # 2. Thera-SAbDab
    df_thera = download_therasabdab()
    if df_thera is not None:
        results['therasabdab'] = process_therasabdab(df_thera)

    time.sleep(2)

    # 3. SAbDab with affinity filter
    df_sabdab = download_sabdab_affinity()
    if df_sabdab is not None:
        results['sabdab'] = df_sabdab

    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)

    for name, df in results.items():
        if df is not None:
            print(f"[OK] {name}: {len(df)} entries")
        else:
            print(f"[ERROR] {name}: Failed to download")

    print("\n[INFO] Next Steps:")
    print("       1. Check external_data/therapeutic/ directory")
    print("       2. For databases requiring manual download:")
    print("          - Visit the URLs mentioned above")
    print("          - Download CSV files manually")
    print("          - Place in external_data/therapeutic/")
    print("       3. Run integration script to merge with existing data")

    return results

if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n[INFO] Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
