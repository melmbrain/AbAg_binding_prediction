#!/usr/bin/env python3
"""
Check Embedding Generation Progress

Displays current progress of background embedding generation.
"""

import pickle
from pathlib import Path
import os
import sys
from datetime import datetime

def format_time_remaining(samples_done, samples_total, time_elapsed_sec):
    """Estimate time remaining"""
    if samples_done == 0:
        return "Unknown"

    rate = samples_done / time_elapsed_sec  # samples per second
    samples_remaining = samples_total - samples_done
    seconds_remaining = samples_remaining / rate

    # Convert to hours and minutes
    hours = int(seconds_remaining // 3600)
    minutes = int((seconds_remaining % 3600) // 60)

    if hours > 24:
        days = hours // 24
        hours = hours % 24
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"

def main():
    print("="*80)
    print("EMBEDDING GENERATION PROGRESS")
    print("="*80)

    checkpoint_file = Path("external_data/embedding_checkpoint.pkl")

    if not checkpoint_file.exists():
        print("\n[INFO] No checkpoint file found")
        print("[INFO] Embedding generation may not have started yet")
        print("\nTo start: scripts/start_embedding_generation.bat")
        return

    # Load checkpoint
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)

        last_index = checkpoint.get('last_index', 0)
        total_samples = 185771  # Total samples needing embeddings
        progress_pct = checkpoint.get('progress_pct', 0)
        timestamp = checkpoint.get('timestamp', 'Unknown')
        status = checkpoint.get('status', 'in_progress')

        print(f"\n{'Metric':<25} | {'Value'}")
        print("-"*80)
        print(f"{'Status':<25} | {status.upper()}")
        print(f"{'Samples processed':<25} | {last_index:,} / {total_samples:,}")
        print(f"{'Progress':<25} | {progress_pct:.2f}%")
        print(f"{'Samples remaining':<25} | {total_samples - last_index:,}")
        print(f"{'Last updated':<25} | {timestamp}")

        # Progress bar (ASCII for Korean Windows compatibility)
        bar_length = 50
        filled = int(bar_length * progress_pct / 100)
        bar = '#' * filled + '-' * (bar_length - filled)
        print(f"\n[{bar}] {progress_pct:.1f}%")

        if status == 'complete':
            print("\n" + "="*80)
            print("EMBEDDING GENERATION COMPLETE!")
            print("="*80)
            print("\nNext steps:")
            print("  1. Run: python scripts/apply_pca_and_merge.py")
            print("  2. Train with full dataset")
        else:
            print(f"\n[INFO] Embedding generation in progress...")

            # Check if log file exists
            log_file = Path("embedding_generation.log")
            if log_file.exists():
                print(f"[INFO] Log file: {log_file}")
                print(f"[INFO] Log size: {os.path.getsize(log_file) / 1024:.1f} KB")

                # Show last few lines
                print("\nLast 5 log lines:")
                print("-"*80)
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-5:]:
                        print(line.rstrip())

        # Calculate file sizes
        print("\n" + "="*80)
        print("FILES")
        print("="*80)

        files_to_check = [
            ("external_data/embedding_checkpoint.pkl", "Checkpoint"),
            ("external_data/new_embeddings.npy", "Raw embeddings"),
            ("external_data/new_embedding_indices.npy", "Sample indices"),
            ("embedding_generation.log", "Log file")
        ]

        print(f"\n{'File':<35} | {'Size':<15} | {'Status'}")
        print("-"*80)

        for file_path, description in files_to_check:
            path = Path(file_path)
            if path.exists():
                size_mb = os.path.getsize(path) / (1024 * 1024)
                if size_mb < 1:
                    size_str = f"{size_mb * 1024:.1f} KB"
                else:
                    size_str = f"{size_mb:.1f} MB"
                status_str = "EXISTS"
            else:
                size_str = "N/A"
                status_str = "Not yet"

            print(f"{description:<35} | {size_str:<15} | {status_str}")

    except Exception as e:
        print(f"\n[ERROR] Failed to read checkpoint: {e}")
        print("\n[INFO] The checkpoint file may be corrupted or still being written")
        print("[INFO] Wait a moment and try again")

if __name__ == "__main__":
    main()
