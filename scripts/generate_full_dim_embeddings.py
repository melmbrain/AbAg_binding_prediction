#!/usr/bin/env python3
"""
Generate Full-Dimensional ESM-2 Embeddings for v3 Training

This script generates 1,280-dimensional ESM-2 embeddings from sequences.
It creates the files needed by prepare_full_dimensional_features.py:
  - new_embeddings.npy (1,280-dim embeddings)
  - new_embedding_indices.npy (index mapping)
  - merged_with_therapeutics.csv (base dataset)

Time estimate: 4-8 hours on CPU, 1-2 hours on GPU
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from tqdm import tqdm
import argparse
import time
import sys

def load_esm2_model(device='cuda'):
    """Load ESM-2 model and tokenizer"""
    print("\n" + "="*80)
    print("LOADING ESM-2 MODEL")
    print("="*80)

    model_name = "facebook/esm2_t33_650M_UR50D"
    print(f"Model: {model_name}")
    print(f"Device: {device}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model (this may take a few minutes)...")
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully")
    print(f"   Parameters: ~650M")
    print(f"   Output dimensions: 1,280")

    return model, tokenizer

def generate_embedding(sequence, model, tokenizer, device='cuda', max_length=1022):
    """Generate 1,280-dimensional embedding for a single sequence"""
    # Tokenize
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate embedding
    with torch.no_grad():
        outputs = model(**inputs)
        # Get last hidden state and mean pool
        embeddings = outputs.last_hidden_state
        # Mean pooling (excluding special tokens)
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask

    return embedding.cpu().numpy()[0]  # Shape: (1280,)

def concatenate_sequences(row):
    """Concatenate antibody heavy + light + antigen sequences"""
    sequences = []

    if 'antibody_heavy' in row and pd.notna(row['antibody_heavy']):
        sequences.append(row['antibody_heavy'])
    elif 'heavy_chain' in row and pd.notna(row['heavy_chain']):
        sequences.append(row['heavy_chain'])

    if 'antibody_light' in row and pd.notna(row['antibody_light']):
        sequences.append(row['antibody_light'])
    elif 'light_chain' in row and pd.notna(row['light_chain']):
        sequences.append(row['light_chain'])
    elif 'antibody_sequence' in row and pd.notna(row['antibody_sequence']):
        sequences.append(row['antibody_sequence'])

    if 'antigen' in row and pd.notna(row['antigen']):
        sequences.append(row['antigen'])
    elif 'antigen_sequence' in row and pd.notna(row['antigen_sequence']):
        sequences.append(row['antigen_sequence'])

    return "<sep>".join(sequences) if sequences else None

def main():
    parser = argparse.ArgumentParser(description='Generate full-dimensional ESM-2 embeddings')
    parser.add_argument('--input',
                       default=None,
                       help='Input CSV file with sequences')
    parser.add_argument('--output-dir',
                       default='external_data',
                       help='Output directory')
    parser.add_argument('--batch-size',
                       type=int,
                       default=8,
                       help='Batch size for embedding generation')
    parser.add_argument('--max-samples',
                       type=int,
                       default=None,
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--device',
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Auto-detect input file if not specified
    if args.input is None:
        # Try common locations
        possible_files = [
            '/mnt/c/Users/401-24/Desktop/drive-download-20251106T005713Z-1-001/AbAg_data/merged_with_all_features.csv',
            '/mnt/c/Users/401-24/Desktop/Ab_Ag_dataset/data/agab/agab_phase2_full.csv',
            'external_data/merged_with_therapeutics.csv'
        ]

        for file in possible_files:
            if Path(file).exists():
                args.input = file
                print(f"Auto-detected input file: {file}")
                break

        if args.input is None:
            print("ERROR: Could not find input file. Please specify with --input")
            sys.exit(1)

    print("\n" + "="*80)
    print("FULL-DIMENSIONAL EMBEDDING GENERATION")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}/")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/4] Loading dataset...")
    start_time = time.time()
    df = pd.read_csv(args.input, low_memory=False)
    print(f"✅ Loaded {len(df):,} rows")
    print(f"   Columns: {list(df.columns[:10])}...")
    print(f"   Time: {time.time() - start_time:.1f}s")

    # Limit samples if requested
    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"⚠️  Limited to {len(df):,} samples for testing")

    # Load model
    print("\n[2/4] Loading ESM-2 model...")
    start_time = time.time()
    model, tokenizer = load_esm2_model(args.device)
    print(f"   Time: {time.time() - start_time:.1f}s")

    # Generate embeddings
    print("\n[3/4] Generating embeddings...")
    print(f"   Expected time: ~{len(df) * 2 / 3600:.1f} hours on CPU")
    print(f"   Progress will be saved every 1000 samples\n")

    embeddings_list = []
    indices_list = []
    failed_indices = []

    start_time = time.time()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating embeddings"):
        try:
            # Concatenate sequences
            sequence = concatenate_sequences(row)

            if sequence is None or len(sequence) < 10:
                failed_indices.append(idx)
                continue

            # Generate embedding
            embedding = generate_embedding(sequence, model, tokenizer, args.device)

            # Verify dimensions
            if embedding.shape[0] != 1280:
                print(f"ERROR: Unexpected embedding shape: {embedding.shape}")
                failed_indices.append(idx)
                continue

            embeddings_list.append(embedding)
            indices_list.append(idx)

            # Save checkpoint every 1000 samples
            if len(embeddings_list) % 1000 == 0:
                checkpoint_file = f"{args.output_dir}/embeddings_checkpoint_{len(embeddings_list)}.npy"
                np.save(checkpoint_file, np.array(embeddings_list))
                print(f"\n✅ Checkpoint saved: {len(embeddings_list):,} embeddings")

        except Exception as e:
            print(f"\n⚠️  Failed at index {idx}: {e}")
            failed_indices.append(idx)
            continue

    elapsed_time = time.time() - start_time

    print(f"\n✅ Embedding generation complete!")
    print(f"   Successful: {len(embeddings_list):,}/{len(df):,}")
    print(f"   Failed: {len(failed_indices):,}")
    print(f"   Total time: {elapsed_time/3600:.2f} hours")

    # Convert to numpy arrays
    print("\n[4/4] Saving outputs...")

    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    indices_array = np.array(indices_list, dtype=np.int32)

    print(f"   Embeddings shape: {embeddings_array.shape}")
    print(f"   Indices shape: {indices_array.shape}")

    # Save embeddings
    embeddings_file = f"{args.output_dir}/new_embeddings.npy"
    np.save(embeddings_file, embeddings_array)
    size_mb = Path(embeddings_file).stat().st_size / (1024**2)
    print(f"✅ Saved: {embeddings_file} ({size_mb:.1f} MB)")

    # Save indices
    indices_file = f"{args.output_dir}/new_embedding_indices.npy"
    np.save(indices_file, indices_array)
    size_mb = Path(indices_file).stat().st_size / (1024**2)
    print(f"✅ Saved: {indices_file} ({size_mb:.1f} MB)")

    # Save base dataset (copy of input or filtered version)
    dataset_file = f"{args.output_dir}/merged_with_therapeutics.csv"
    df.to_csv(dataset_file, index=False)
    size_mb = Path(dataset_file).stat().st_size / (1024**2)
    print(f"✅ Saved: {dataset_file} ({size_mb:.1f} MB)")

    # Summary
    print("\n" + "="*80)
    print("EMBEDDING GENERATION COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in {args.output_dir}/:")
    print(f"  1. new_embeddings.npy - {len(embeddings_list):,} embeddings (1,280 dims each)")
    print(f"  2. new_embedding_indices.npy - Index mapping")
    print(f"  3. merged_with_therapeutics.csv - Base dataset")
    print(f"\nNext step:")
    print(f"  python3 scripts/prepare_full_dimensional_features.py")
    print(f"\nThis will create:")
    print(f"  external_data/merged_with_full_features.csv (~1GB)")
    print(f"\nThen upload to Google Drive and train on Colab!")
    print("="*80)

if __name__ == "__main__":
    main()
