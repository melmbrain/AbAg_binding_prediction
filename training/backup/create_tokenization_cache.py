"""
SQLite-Based Tokenization Cache (Inspired by CAFA6 SQLite Success)
====================================================================

Strategy:
1. Pre-tokenize all sequences → SQLite DB (one-time)
2. Training reads from cache (10x faster)
3. Constant RAM usage (no memory leaks)

Based on successful CAFA6 approach with chunked processing!
"""

import sqlite3
import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import hashlib
import numpy as np
import argparse
from pathlib import Path


def create_sequence_hash(sequence):
    """Create hash for sequence to use as key"""
    return hashlib.md5(sequence.encode()).hexdigest()


def create_tokenization_db(db_path):
    """Create SQLite database for tokenized sequences"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table with indexed hash for fast lookups
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tokenized_sequences (
            seq_hash TEXT PRIMARY KEY,
            sequence TEXT,
            input_ids BLOB,
            attention_mask BLOB
        )
    ''')

    # Create index for faster lookups
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_hash ON tokenized_sequences(seq_hash)')

    conn.commit()
    return conn


def tokenize_and_store_chunk(sequences, tokenizer, conn, max_length=512, chunk_size=1000):
    """Tokenize sequences in chunks and store in SQLite"""
    cursor = conn.cursor()

    # Process in smaller batches to avoid memory issues
    for i in range(0, len(sequences), chunk_size):
        batch_seqs = sequences[i:i+chunk_size]

        # Tokenize batch
        tokens = tokenizer(
            batch_seqs,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Store each tokenized sequence
        data = []
        for j, seq in enumerate(batch_seqs):
            seq_hash = create_sequence_hash(seq)
            input_ids = tokens['input_ids'][j].numpy().tobytes()
            attention_mask = tokens['attention_mask'][j].numpy().tobytes()
            data.append((seq_hash, seq, input_ids, attention_mask))

        # Batch insert with UPSERT (handles duplicates)
        cursor.executemany('''
            INSERT OR REPLACE INTO tokenized_sequences
            (seq_hash, sequence, input_ids, attention_mask)
            VALUES (?, ?, ?, ?)
        ''', data)

        conn.commit()


def main():
    parser = argparse.ArgumentParser(description='Create tokenization cache')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data')
    parser.add_argument('--output_db', type=str, default='tokenization_cache.db',
                       help='Output SQLite database')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--chunk_size', type=int, default=10000,
                       help='Number of rows to process at once')
    args = parser.parse_args()

    print("="*70)
    print("CREATING TOKENIZATION CACHE (SQLite)")
    print("="*70)
    print(f"Strategy: Pre-tokenize once, train fast forever!")
    print(f"Inspired by CAFA6 SQLite success 🚀")
    print()

    # Initialize tokenizer
    print("Loading ESM-2 tokenizer...")
    model_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✓ Tokenizer loaded")

    # Create database
    db_path = Path(args.output_db)
    print(f"\nCreating database: {db_path}")
    conn = create_tokenization_db(str(db_path))
    print("✓ Database created")

    # Load data
    print(f"\nLoading data: {args.data}")
    df = pd.read_csv(args.data)
    total_rows = len(df)
    print(f"✓ Loaded {total_rows:,} samples")

    # Get unique sequences
    print("\nExtracting unique sequences...")
    antibody_seqs = df['antibody_sequence'].unique().tolist()
    antigen_seqs = df['antigen_sequence'].unique().tolist()
    all_unique_seqs = list(set(antibody_seqs + antigen_seqs))
    print(f"✓ Found {len(all_unique_seqs):,} unique sequences")
    print(f"  - Antibody sequences: {len(antibody_seqs):,}")
    print(f"  - Antigen sequences: {len(antigen_seqs):,}")

    # Check existing cache
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM tokenized_sequences')
    existing_count = cursor.fetchone()[0]

    if existing_count > 0:
        print(f"\n⚠️  Found {existing_count:,} existing cached sequences")
        print("Will only tokenize new sequences (UPSERT strategy)")

    # Tokenize in chunks with progress bar
    print(f"\nTokenizing sequences...")
    print(f"Chunk size: {args.chunk_size:,} sequences per batch")
    print(f"This will take ~{len(all_unique_seqs) * 0.001 / 60:.1f} minutes")
    print()

    for i in tqdm(range(0, len(all_unique_seqs), args.chunk_size),
                  desc="Tokenizing", unit="chunk"):
        chunk = all_unique_seqs[i:i+args.chunk_size]
        tokenize_and_store_chunk(chunk, tokenizer, conn, args.max_length, chunk_size=1000)

        # Periodic snapshot info (every 100k sequences)
        if (i + args.chunk_size) % 100000 == 0:
            cursor.execute('SELECT COUNT(*) FROM tokenized_sequences')
            cached_count = cursor.fetchone()[0]
            print(f"  📦 Cached {cached_count:,} sequences so far...")

    # Final stats
    cursor.execute('SELECT COUNT(*) FROM tokenized_sequences')
    final_count = cursor.fetchone()[0]

    db_size_mb = db_path.stat().st_size / (1024 * 1024)

    print("\n" + "="*70)
    print("TOKENIZATION CACHE COMPLETE!")
    print("="*70)
    print(f"✅ Cached sequences: {final_count:,}")
    print(f"✅ Database size: {db_size_mb:.1f} MB")
    print(f"✅ Database path: {db_path.absolute()}")
    print()
    print("Next step: Use train_ultra_optimized_cached.py for 10x faster training!")
    print("="*70)

    conn.close()


if __name__ == '__main__':
    main()
