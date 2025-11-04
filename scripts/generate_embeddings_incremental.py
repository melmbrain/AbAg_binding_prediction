#!/usr/bin/env python3
"""
Incremental ESM2 Embedding Generation

Generates embeddings in small batches to avoid GPU memory conflicts.
Can run in background with low priority while your main training is running.

Features:
- Batch processing (small batches to limit GPU memory)
- Auto-pause if GPU memory is high
- CPU fallback option
- Resume from checkpoint
- Progress saving
"""

import pandas as pd
import numpy as np
import torch
import time
import argparse
from pathlib import Path
import sys
import os

def check_gpu_memory():
    """Check current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3    # GB
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB

            usage_pct = (mem_allocated / mem_total) * 100

            print(f"[INFO] GPU {i}: {mem_allocated:.2f}GB / {mem_total:.2f}GB ({usage_pct:.1f}% used)")

            return usage_pct
    return 0

def wait_for_gpu_availability(threshold=80.0, check_interval=60):
    """Wait until GPU memory usage drops below threshold"""

    while True:
        usage = check_gpu_memory()

        if usage < threshold:
            print(f"[OK] GPU memory usage ({usage:.1f}%) below threshold ({threshold}%)")
            return True
        else:
            print(f"[WAIT] GPU memory usage ({usage:.1f}%) above threshold ({threshold}%)")
            print(f"       Waiting {check_interval}s before retry...")
            print(f"       (Your other training is using the GPU)")
            time.sleep(check_interval)

def generate_embeddings_batch(sequences, model, tokenizer, device, max_length=512):
    """Generate ESM2 embeddings for a batch of sequences"""

    try:
        # Tokenize
        inputs = tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling over sequence length
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.cpu().numpy()

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"[ERROR] GPU out of memory for batch size {len(sequences)}")
            torch.cuda.empty_cache()
            return None
        else:
            raise e

def load_esm2_model(device='cuda', use_cpu=False):
    """Load ESM2 model"""

    print("\n" + "="*80)
    print("LOADING ESM2 MODEL")
    print("="*80)

    if use_cpu:
        device = 'cpu'
        print("[INFO] Using CPU for embedding generation (slower but won't conflict with GPU)")

    try:
        from transformers import AutoTokenizer, AutoModel

        model_name = "facebook/esm2_t33_650M_UR50D"
        print(f"[INFO] Loading {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        model = model.to(device)
        model.eval()

        print(f"[OK] Model loaded on {device}")

        return model, tokenizer, device

    except Exception as e:
        print(f"[ERROR] Failed to load ESM2 model: {e}")
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description="Incremental ESM2 embedding generation")
    parser.add_argument("--data", type=str, default="external_data/merged_with_therapeutics.csv",
                       help="Input data file")
    parser.add_argument("--output", type=str, default="external_data/merged_with_all_features.csv",
                       help="Output file with embeddings")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for embedding generation (smaller = less GPU memory)")
    parser.add_argument("--gpu_threshold", type=float, default=80.0,
                       help="GPU memory threshold % - pause if above this")
    parser.add_argument("--check_interval", type=int, default=300,
                       help="Seconds to wait between GPU checks when paused")
    parser.add_argument("--use_cpu", action="store_true",
                       help="Use CPU instead of GPU (slower but no conflict)")
    parser.add_argument("--checkpoint_file", type=str, default="external_data/embedding_checkpoint.pkl",
                       help="Checkpoint file to resume from")
    parser.add_argument("--save_every", type=int, default=50,
                       help="Save checkpoint every N batches")
    parser.add_argument("--pca_model", type=str, default=None,
                       help="Path to existing PCA model for transformation")

    args = parser.parse_args()

    print("="*80)
    print("INCREMENTAL ESM2 EMBEDDING GENERATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data file: {args.data}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  GPU threshold: {args.gpu_threshold}%")
    print(f"  Use CPU: {args.use_cpu}")
    print(f"  Checkpoint: {args.checkpoint_file}")

    # Load data
    print(f"\n[INFO] Loading data...")
    df = pd.read_csv(args.data)
    print(f"[OK] Loaded {len(df)} samples")

    # Find samples without embeddings
    esm2_cols = [col for col in df.columns if col.startswith('esm2_pca_')]
    if esm2_cols:
        first_col = esm2_cols[0]
        needs_embedding = df[first_col].isna()
        df_need = df[needs_embedding].copy()

        print(f"\n[INFO] Samples needing embeddings: {len(df_need)}")
        print(f"[INFO] Samples with embeddings: {(~needs_embedding).sum()}")
    else:
        print("[ERROR] No ESM2 PCA columns found in data!")
        return

    if len(df_need) == 0:
        print("\n[OK] All samples already have embeddings!")
        return

    # Check for checkpoint
    checkpoint_path = Path(args.checkpoint_file)
    start_idx = 0
    embeddings_generated = []
    samples_processed = []

    if checkpoint_path.exists():
        print(f"\n[INFO] Loading checkpoint from {checkpoint_path}...")
        import pickle
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                start_idx = checkpoint.get('last_index', 0)
                embeddings_generated = checkpoint.get('embeddings', [])
                samples_processed = checkpoint.get('samples_processed', [])

            progress_pct = (start_idx / len(df_need)) * 100 if len(df_need) > 0 else 0
            print(f"[OK] Resuming from index {start_idx}")
            print(f"     Progress: {start_idx:,} / {len(df_need):,} ({progress_pct:.1f}%)")
            print(f"     Embeddings already generated: {len(embeddings_generated)}")
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint: {e}")
            print(f"[INFO] Starting from beginning")
            start_idx = 0
            embeddings_generated = []
            samples_processed = []

    # Load ESM2 model
    device = 'cpu' if args.use_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, device = load_esm2_model(device, args.use_cpu)

    if model is None:
        print("[ERROR] Failed to load model")
        return

    # Process in batches
    print("\n" + "="*80)
    print("GENERATING EMBEDDINGS")
    print("="*80)

    total_batches = (len(df_need) - start_idx + args.batch_size - 1) // args.batch_size

    for batch_idx in range(start_idx // args.batch_size, total_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(df_need))

        # Check GPU availability (skip if using CPU)
        if not args.use_cpu:
            wait_for_gpu_availability(args.gpu_threshold, args.check_interval)

        # Get batch
        batch_df = df_need.iloc[batch_start:batch_end]

        # Combine heavy and light chain sequences
        sequences = []
        for _, row in batch_df.iterrows():
            h_seq = str(row.get('heavy_chain_seq', ''))
            l_seq = str(row.get('light_chain_seq', ''))
            combined = h_seq + l_seq  # Simple concatenation
            sequences.append(combined)

        print(f"\n[INFO] Batch {batch_idx + 1}/{total_batches} ({batch_start}-{batch_end})")
        print(f"       Processing {len(sequences)} sequences...")

        # Generate embeddings
        batch_embeddings = generate_embeddings_batch(
            sequences, model, tokenizer, device
        )

        if batch_embeddings is None:
            print("[ERROR] Failed to generate embeddings for batch, skipping...")
            continue

        embeddings_generated.extend(batch_embeddings)
        samples_processed.extend(batch_df.index.tolist())

        print(f"[OK] Generated embeddings: shape {batch_embeddings.shape}")

        # Save checkpoint periodically
        if (batch_idx + 1) % args.save_every == 0:
            print(f"[INFO] Saving checkpoint...")
            import pickle
            import datetime

            checkpoint = {
                'last_index': batch_end,
                'embeddings': embeddings_generated,
                'samples_processed': samples_processed,
                'timestamp': datetime.datetime.now().isoformat(),
                'progress_pct': (batch_end / len(df_need)) * 100
            }

            try:
                # Save to temporary file first
                temp_checkpoint = str(checkpoint_path) + '.tmp'
                with open(temp_checkpoint, 'wb') as f:
                    pickle.dump(checkpoint, f)

                # Then rename (atomic operation)
                import shutil
                shutil.move(temp_checkpoint, checkpoint_path)

                print(f"[OK] Checkpoint saved: {batch_end:,}/{len(df_need):,} ({checkpoint['progress_pct']:.1f}%)")
            except Exception as e:
                print(f"[ERROR] Failed to save checkpoint: {e}")

        # Small delay to be nice to GPU
        if not args.use_cpu:
            time.sleep(2)

    print("\n" + "="*80)
    print("EMBEDDING GENERATION COMPLETE")
    print("="*80)
    print(f"\n[OK] Generated {len(embeddings_generated)} embeddings")

    # Save final checkpoint
    print(f"\n[INFO] Saving final checkpoint...")
    import pickle
    import datetime

    final_checkpoint = {
        'last_index': len(df_need),
        'embeddings': embeddings_generated,
        'samples_processed': samples_processed,
        'timestamp': datetime.datetime.now().isoformat(),
        'progress_pct': 100.0,
        'status': 'complete'
    }

    with open(checkpoint_path, 'wb') as f:
        pickle.dump(final_checkpoint, f)
    print(f"[OK] Final checkpoint saved")

    # Save embeddings
    embeddings_file = Path("external_data/new_embeddings.npy")
    np.save(embeddings_file, np.array(embeddings_generated))
    print(f"[OK] Saved raw embeddings to: {embeddings_file}")

    # Save sample indices
    indices_file = Path("external_data/new_embedding_indices.npy")
    np.save(indices_file, np.array(samples_processed))
    print(f"[OK] Saved sample indices to: {indices_file}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n[1] Apply PCA transformation to reduce to 150 components")
    print("[2] Merge embeddings with original dataframe")
    print("[3] Train model with full 390k dataset")
    print("\n[INFO] Run: python scripts/apply_pca_and_merge.py")

if __name__ == "__main__":
    main()
