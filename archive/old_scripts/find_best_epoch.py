"""
Find Best Epoch from Training Results
Analyzes checkpoints and training metrics to identify optimal stopping point
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

def analyze_from_csv(csv_path):
    """Find best epoch from training metrics CSV"""

    df = pd.read_csv(csv_path)

    print("\n" + "="*70)
    print("BEST EPOCH ANALYSIS FROM TRAINING METRICS")
    print("="*70)

    # Filter to validation epochs only
    val_df = df[df['val_spearman'].notna()].copy()

    if len(val_df) == 0:
        print("ERROR: No validation metrics found in CSV!")
        return None

    # Find best epoch by Spearman
    best_idx = val_df['val_spearman'].idxmax()
    best_epoch = int(val_df.loc[best_idx, 'epoch'])
    best_spearman = val_df.loc[best_idx, 'val_spearman']
    best_recall = val_df.loc[best_idx, 'val_recall_pkd9']
    best_train_loss = val_df.loc[best_idx, 'train_loss']

    print(f"\n🏆 BEST EPOCH: {best_epoch}")
    print(f"   Validation Spearman: {best_spearman:.4f}")
    print(f"   Recall@pKd≥9: {best_recall:.2f}%")
    print(f"   Training Loss: {best_train_loss:.4f}")

    # Compare with final epoch
    final_idx = val_df.index[-1]
    final_epoch = int(val_df.loc[final_idx, 'epoch'])
    final_spearman = val_df.loc[final_idx, 'val_spearman']
    final_recall = val_df.loc[final_idx, 'val_recall_pkd9']

    print(f"\n📊 FINAL EPOCH: {final_epoch}")
    print(f"   Validation Spearman: {final_spearman:.4f}")
    print(f"   Recall@pKd≥9: {final_recall:.2f}%")

    if best_epoch != final_epoch:
        performance_drop = (best_spearman - final_spearman) / best_spearman * 100
        wasted_epochs = final_epoch - best_epoch

        print(f"\n⚠️  OVERFITTING DETECTED")
        print(f"   Performance dropped by: {performance_drop:.1f}%")
        print(f"   Wasted epochs: {wasted_epochs}")
        print(f"   Should have stopped at epoch {best_epoch}")

        # Calculate time wasted (assuming 3 min/epoch from your results)
        time_per_epoch = 3  # minutes
        wasted_time = wasted_epochs * time_per_epoch
        print(f"   Time wasted: ~{wasted_time} minutes")
    else:
        print(f"\n✓ Training ended at optimal point!")

    # Analyze when to stop with early stopping
    print(f"\n" + "="*70)
    print("EARLY STOPPING SIMULATION")
    print("="*70)

    # Simulate different patience values
    for patience in [5, 10, 15]:
        stopped_epoch = simulate_early_stopping(val_df, patience)
        stopped_spearman = val_df[val_df['epoch'] == stopped_epoch]['val_spearman'].values[0]

        print(f"\nPatience={patience} epochs:")
        print(f"  Would stop at epoch: {stopped_epoch}")
        print(f"  Spearman at stop: {stopped_spearman:.4f}")
        print(f"  vs Best: {best_spearman:.4f} (diff: {(stopped_spearman-best_spearman):.4f})")

    print("="*70)

    return {
        'best_epoch': best_epoch,
        'best_spearman': best_spearman,
        'final_epoch': final_epoch,
        'final_spearman': final_spearman
    }


def simulate_early_stopping(val_df, patience, min_delta=0.0001):
    """Simulate early stopping with given patience"""
    best_score = -np.inf
    counter = 0
    best_epoch = 0

    for idx, row in val_df.iterrows():
        epoch = int(row['epoch'])
        score = row['val_spearman']

        if score > best_score + min_delta:
            best_score = score
            best_epoch = epoch
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            return epoch

    return int(val_df['epoch'].iloc[-1])


def analyze_from_checkpoints(checkpoint_dir):
    """Analyze available checkpoints"""

    checkpoint_dir = Path(checkpoint_dir)

    print("\n" + "="*70)
    print("CHECKPOINT ANALYSIS")
    print("="*70)

    # Find all checkpoints
    checkpoints = list(checkpoint_dir.glob("*.pth"))

    if len(checkpoints) == 0:
        print("No checkpoints found!")
        return

    print(f"\nFound {len(checkpoints)} checkpoint files:")

    checkpoint_info = []

    for ckpt_path in checkpoints:
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

            info = {
                'file': ckpt_path.name,
                'epoch': checkpoint.get('epoch', 'N/A'),
                'val_spearman': checkpoint.get('val_spearman', checkpoint.get('best_val_spearman', -1)),
                'val_recall': checkpoint.get('val_recall_pkd9', 'N/A'),
                'train_loss': checkpoint.get('train_loss', 'N/A')
            }

            checkpoint_info.append(info)

        except Exception as e:
            print(f"  ⚠ Error loading {ckpt_path.name}: {e}")

    # Sort by Spearman
    checkpoint_info = sorted(checkpoint_info,
                            key=lambda x: x['val_spearman'] if isinstance(x['val_spearman'], (int, float)) else -1,
                            reverse=True)

    # Print table
    print(f"\n{'File':<35} {'Epoch':<8} {'Spearman':<12} {'Recall':<12} {'Loss':<12}")
    print("-" * 80)

    for info in checkpoint_info:
        epoch_str = str(info['epoch']) if info['epoch'] != 'N/A' else 'N/A'
        spearman_str = f"{info['val_spearman']:.4f}" if isinstance(info['val_spearman'], (int, float)) and info['val_spearman'] > 0 else 'N/A'
        recall_str = f"{info['val_recall']:.2f}%" if isinstance(info['val_recall'], (int, float)) else 'N/A'
        loss_str = f"{info['train_loss']:.4f}" if isinstance(info['train_loss'], (int, float)) else 'N/A'

        print(f"{info['file']:<35} {epoch_str:<8} {spearman_str:<12} {recall_str:<12} {loss_str:<12}")

    # Identify best
    if checkpoint_info[0]['val_spearman'] > 0:
        print(f"\n🏆 Best checkpoint: {checkpoint_info[0]['file']}")
        print(f"   Epoch: {checkpoint_info[0]['epoch']}")
        print(f"   Spearman: {checkpoint_info[0]['val_spearman']:.4f}")

    print("="*70)


def plot_performance_curve(csv_path, output_path=None):
    """Create a focused plot showing best epoch vs final"""

    df = pd.read_csv(csv_path)
    val_df = df[df['val_spearman'].notna()].copy()

    if len(val_df) == 0:
        return

    # Find best epoch
    best_idx = val_df['val_spearman'].idxmax()
    best_epoch = val_df.loc[best_idx, 'epoch']
    best_spearman = val_df.loc[best_idx, 'val_spearman']

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Spearman over epochs
    ax.plot(val_df['epoch'], val_df['val_spearman'], 'b-o', linewidth=2, markersize=6, label='Validation Spearman')

    # Mark best epoch
    ax.scatter([best_epoch], [best_spearman], color='green', s=300, marker='*',
               zorder=5, label=f'Best: Epoch {int(best_epoch)}')

    # Mark final epoch
    final_epoch = val_df['epoch'].iloc[-1]
    final_spearman = val_df['val_spearman'].iloc[-1]
    ax.scatter([final_epoch], [final_spearman], color='red', s=200, marker='X',
               zorder=5, label=f'Final: Epoch {int(final_epoch)}')

    # Add annotations
    ax.annotate(f'Best\n{best_spearman:.4f}',
                xy=(best_epoch, best_spearman),
                xytext=(10, 20), textcoords='offset points',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.8),
                arrowprops=dict(arrowstyle='->', lw=2))

    if best_epoch != final_epoch:
        ax.annotate(f'Final\n{final_spearman:.4f}',
                    xy=(final_epoch, final_spearman),
                    xytext=(10, -30), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightcoral', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', lw=2))

        # Shade overfitting region
        ax.axvspan(best_epoch, final_epoch, alpha=0.2, color='red', label='Overfitting Region')

    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Spearman Correlation', fontsize=13, fontweight='bold')
    ax.set_title('Training Progress: Optimal vs Final Epoch', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    ax.set_ylim([0, 1.0])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot to: {output_path}")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find optimal epoch from training results')
    parser.add_argument('--csv', type=str, default=None,
                       help='Path to training_metrics.csv')
    parser.add_argument('--checkpoint_dir', type=str, default='output',
                       help='Directory containing checkpoint files')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plot')

    args = parser.parse_args()

    # Try to find CSV if not provided
    if args.csv is None:
        csv_candidates = [
            Path(args.checkpoint_dir) / 'training_metrics.csv',
            Path('output/training_metrics.csv'),
            Path('outputs_max_speed/training_metrics.csv'),
        ]

        for candidate in csv_candidates:
            if candidate.exists():
                args.csv = str(candidate)
                print(f"Found CSV: {args.csv}")
                break

    # Analyze CSV if available
    results = None
    if args.csv and Path(args.csv).exists():
        results = analyze_from_csv(args.csv)

        if args.plot:
            output_path = Path(args.csv).parent / 'best_epoch_analysis.png'
            plot_performance_curve(args.csv, output_path)
    else:
        print("No training_metrics.csv found. Analyzing checkpoints only...")

    # Analyze checkpoints
    if Path(args.checkpoint_dir).exists():
        analyze_from_checkpoints(args.checkpoint_dir)
    else:
        print(f"\nCheckpoint directory not found: {args.checkpoint_dir}")

    # Recommendations
    if results:
        print("\n" + "="*70)
        print("RECOMMENDATIONS FOR NEXT TRAINING")
        print("="*70)

        best_epoch = results['best_epoch']
        final_epoch = results['final_epoch']

        if final_epoch > best_epoch + 10:
            print(f"\n✓ Enable early stopping with patience=10")
            print(f"  This would have saved {(final_epoch - best_epoch) * 3:.0f} minutes")

        print(f"\n✓ Run for ~{int(best_epoch * 1.2)} epochs")
        print(f"  (Best was at {best_epoch}, so +20% buffer)")

        print(f"\n✓ Validate every epoch (--validation_frequency 1)")
        print(f"  Better tracking of performance peaks")

        print("="*70)
