#!/usr/bin/env python3
"""
Download and analyze new antibody-antigen binding affinity data sources.

Sources:
1. CoV-AbDab - Coronavirus Antibody Database (Oxford)
2. Ab-CoV - Coronavirus antibody affinity data (IIT Madras)
3. TDC AntibodyAff - Therapeutics Data Commons
"""
import os
import sys
from pathlib import Path
import urllib.request
import ssl

# Disable SSL verification for downloads (some servers have certificate issues)
ssl._create_default_https_context = ssl._create_unverified_context

DATA_DIR = Path("C:/Users/401-24/Desktop/Ab_Ag_dataset/data/new_sources")
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("DOWNLOADING NEW DATA SOURCES")
print("="*60)
print()

# =============================================================================
# 1. CoV-AbDab - Coronavirus Antibody Database
# =============================================================================
print("1. CoV-AbDab (Coronavirus Antibody Database)")
print("-"*50)

covabdab_url = "https://opig.stats.ox.ac.uk/webapps/covabdab/static/downloads/CoV-AbDab_080224.csv"
covabdab_file = DATA_DIR / "CoV-AbDab.csv"

if not covabdab_file.exists():
    print(f"   Downloading from: {covabdab_url}")
    try:
        urllib.request.urlretrieve(covabdab_url, covabdab_file)
        print(f"   Saved to: {covabdab_file}")
    except Exception as e:
        print(f"   ERROR: {e}")
        print("   Please download manually from: https://opig.stats.ox.ac.uk/webapps/covabdab/")
else:
    print(f"   Already exists: {covabdab_file}")

# =============================================================================
# 2. TDC AntibodyAff - Using PyTDC library
# =============================================================================
print()
print("2. TDC AntibodyAff (Therapeutics Data Commons)")
print("-"*50)

tdc_file = DATA_DIR / "tdc_antibody_sabdab.csv"

if not tdc_file.exists():
    try:
        from tdc.multi_pred import AntibodyAff
        print("   Loading TDC AntibodyAff dataset...")
        data = AntibodyAff(name='Protein_SAbDab')
        df = data.get_data()
        df.to_csv(tdc_file, index=False)
        print(f"   Saved {len(df)} samples to: {tdc_file}")
    except ImportError:
        print("   PyTDC not installed. Installing...")
        os.system("pip install PyTDC")
        try:
            from tdc.multi_pred import AntibodyAff
            data = AntibodyAff(name='Protein_SAbDab')
            df = data.get_data()
            df.to_csv(tdc_file, index=False)
            print(f"   Saved {len(df)} samples to: {tdc_file}")
        except Exception as e:
            print(f"   ERROR: {e}")
    except Exception as e:
        print(f"   ERROR: {e}")
else:
    print(f"   Already exists: {tdc_file}")

# =============================================================================
# 3. Zenodo Antibody Kd Dataset
# =============================================================================
print()
print("3. Zenodo Antibody Kd Dataset")
print("-"*50)

zenodo_url = "https://zenodo.org/records/13120765/files/antibody_affinity_protein_sabdab.csv"
zenodo_file = DATA_DIR / "zenodo_antibody_kd.csv"

if not zenodo_file.exists():
    print(f"   Downloading from: {zenodo_url}")
    try:
        urllib.request.urlretrieve(zenodo_url, zenodo_file)
        print(f"   Saved to: {zenodo_file}")
    except Exception as e:
        print(f"   ERROR: {e}")
        print("   Please download manually from: https://zenodo.org/records/13120765")
else:
    print(f"   Already exists: {zenodo_file}")

print()
print("="*60)
print("DOWNLOAD COMPLETE")
print("="*60)
print()
print("Next: Run analyze_new_data_sources.py to evaluate the data")
