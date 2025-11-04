#!/usr/bin/env python3
"""
Fetch antibody sequences from SAbDab for specific PDB entries
"""

import pandas as pd
import requests
import time
from pathlib import Path

def fetch_sequences_for_pdb(pdb_code, h_chain, l_chain):
    """Fetch antibody sequences from SAbDab API"""

    # SAbDab API endpoint for fetching sequences
    base_url = "http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/search/"

    try:
        # Query with PDB code
        params = {'pdb': pdb_code}
        response = requests.get(base_url, params=params, timeout=30)

        if response.status_code == 200:
            # Try to parse JSON response
            try:
                data = response.json()
                return data
            except:
                # Might be HTML, try different endpoint
                pass

        # Try direct structure download endpoint
        struct_url = f"http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb/{pdb_code}/"
        response2 = requests.get(struct_url, timeout=30)

        if response2.status_code == 200:
            return response2.text

    except Exception as e:
        print(f"[ERROR] Failed to fetch {pdb_code}: {e}")
        return None

    return None

def get_sequences_from_rcsb(pdb_code):
    """Fetch FASTA sequences from RCSB PDB"""

    url = f"https://www.rcsb.org/fasta/entry/{pdb_code}/display"

    try:
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            fasta_text = response.text

            # Parse FASTA
            sequences = {}
            current_chain = None
            current_seq = []

            for line in fasta_text.split('\n'):
                if line.startswith('>'):
                    # Save previous sequence
                    if current_chain:
                        sequences[current_chain] = ''.join(current_seq)

                    # Parse header: >4HBC_1|Chain A|...
                    parts = line.split('|')
                    if len(parts) >= 2:
                        chain_info = parts[1]
                        if 'Chain' in chain_info:
                            current_chain = chain_info.split()[-1]  # Get chain ID
                            current_seq = []
                else:
                    current_seq.append(line.strip())

            # Save last sequence
            if current_chain:
                sequences[current_chain] = ''.join(current_seq)

            return sequences

    except Exception as e:
        print(f"[ERROR] Failed to fetch FASTA for {pdb_code}: {e}")
        return None

    return None

def main():
    """Main execution"""
    print("="*80)
    print("FETCHING SEQUENCES FOR SABDAB VERY STRONG BINDERS")
    print("="*80)

    # Load very strong binders
    df = pd.read_csv('external_data/therapeutic/sabdab_very_strong.csv')
    print(f"\n[INFO] Processing {len(df)} very strong binders")

    # Track results
    sequences_found = []
    sequences_missing = []

    # Get unique PDB codes
    unique_pdbs = df[['pdb', 'Hchain', 'Lchain']].drop_duplicates()
    print(f"[INFO] Unique PDB entries: {len(unique_pdbs)}")

    for idx, row in unique_pdbs.iterrows():
        pdb = row['pdb']
        h_chain = row['Hchain']
        l_chain = row['Lchain']

        print(f"\n[INFO] Fetching {pdb} (H:{h_chain}, L:{l_chain})...")

        # Fetch sequences from RCSB
        sequences = get_sequences_from_rcsb(pdb)

        if sequences:
            # Extract heavy and light chain sequences
            h_seq = sequences.get(h_chain, '')
            l_seq = sequences.get(l_chain, '') if pd.notna(l_chain) else ''

            if h_seq:
                sequences_found.append({
                    'pdb_code': pdb,
                    'heavy_chain': h_chain,
                    'light_chain': l_chain if pd.notna(l_chain) else None,
                    'heavy_chain_seq': h_seq,
                    'light_chain_seq': l_seq
                })
                print(f"[OK] Found sequences (H:{len(h_seq)} aa, L:{len(l_seq)} aa)")
            else:
                sequences_missing.append(pdb)
                print(f"[WARNING] No sequence for heavy chain {h_chain}")
        else:
            sequences_missing.append(pdb)
            print(f"[ERROR] Failed to fetch sequences")

        # Be polite to servers
        time.sleep(1)

    # Create DataFrame with sequences
    if sequences_found:
        df_seq = pd.DataFrame(sequences_found)

        # Merge with affinity data
        df_with_seq = df.merge(
            df_seq,
            left_on=['pdb', 'Hchain', 'Lchain'],
            right_on=['pdb_code', 'heavy_chain', 'light_chain'],
            how='left'
        )

        # Add source
        df_with_seq['source'] = 'SAbDab'

        # Save
        output = Path('external_data/therapeutic/sabdab_very_strong_with_sequences.csv')
        df_with_seq.to_csv(output, index=False)

        print(f"\n[OK] Saved {len(df_with_seq)} entries with sequences to: {output}")
        print(f"[INFO] Sequences found for {len(sequences_found)} unique PDB entries")
        print(f"[INFO] Sequences missing for {len(sequences_missing)} unique PDB entries")

        # Count how many have both heavy and light sequences
        both_chains = (df_with_seq['heavy_chain_seq'].notna() &
                      df_with_seq['light_chain_seq'].notna()).sum()
        print(f"[INFO] Entries with both H+L chains: {both_chains}")

        return df_with_seq
    else:
        print("\n[ERROR] No sequences found")
        return None

if __name__ == "__main__":
    result = main()
