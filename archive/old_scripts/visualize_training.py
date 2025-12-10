"""
Visualize Training Progress
Reads training_metrics.csv and creates comprehensive training curves
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def plot_training_curves(csv_path, output_dir=None):
    """Create comprehensive training visualization plots"""

    # Read metrics
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        print("No data found in CSV file!")
        return

    print(f"Loaded {len(df)} epochs of training data")
    print(f"Columns: {list(df.columns)}")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress Overview', fontsize=16, fontweight='bold')

    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['train_loss'], 'b-o', linewidth=2, markersize=4, label='Train Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Over Time', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Add trend line
    if len(df) > 2:
        z = np.polyfit(df['epoch'], df['train_loss'], 2)
        p = np.poly1d(z)
        ax1.plot(df['epoch'], p(df['epoch']), "r--", alpha=0.5, label='Trend')

    # Plot 2: Validation Spearman Correlation
    ax2 = axes[0, 1]
    val_df = df[df['val_spearman'].notna()].copy()
    if len(val_df) > 0:
        ax2.plot(val_df['epoch'], val_df['val_spearman'], 'g-o',
                linewidth=2, markersize=6, label='Val Spearman')
        ax2.plot(val_df['epoch'], val_df['best_spearman'], 'r--',
                linewidth=2, alpha=0.7, label='Best Spearman')

        # Mark best epoch
        best_idx = val_df['val_spearman'].idxmax()
        best_epoch = val_df.loc[best_idx, 'epoch']
        best_spearman = val_df.loc[best_idx, 'val_spearman']
        ax2.scatter([best_epoch], [best_spearman], color='red', s=200,
                   marker='*', zorder=5, label=f'Best: {best_spearman:.4f}')
        ax2.annotate(f'Best\n{best_spearman:.4f}',
                    xy=(best_epoch, best_spearman),
                    xytext=(10, -10), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Spearman Correlation', fontsize=12)
    ax2.set_title('Validation Spearman Correlation', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim([-0.1, 1.0])

    # Plot 3: Recall@pKd≥9
    ax3 = axes[1, 0]
    if len(val_df) > 0 and 'val_recall_pkd9' in val_df.columns:
        ax3.plot(val_df['epoch'], val_df['val_recall_pkd9'], 'purple',
                marker='s', linewidth=2, markersize=6, label='Recall@pKd≥9')
        ax3.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Perfect (100%)')

    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Recall (%)', fontsize=12)
    ax3.set_title('High-Affinity Recall (pKd ≥ 9)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_ylim([0, 105])

    # Plot 4: Learning Rate
    ax4 = axes[1, 1]
    lr_df = df[df['learning_rate'].notna()].copy()
    if len(lr_df) > 0:
        ax4.plot(lr_df['epoch'], lr_df['learning_rate'], 'orange',
                linewidth=2, label='Learning Rate')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Learning Rate', fontsize=12)
        ax4.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)
        ax4.set_yscale('log')

    plt.tight_layout()

    # Save figure
    if output_dir:
        output_path = Path(output_dir) / 'training_curves.png'
    else:
        output_path = Path(csv_path).parent / 'training_curves.png'

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved training curves to: {output_path}")

    plt.show()

    # Print summary statistics
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Total Epochs: {len(df)}")
    print(f"Final Train Loss: {df['train_loss'].iloc[-1]:.4f}")

    if len(val_df) > 0:
        print(f"\nValidation Metrics:")
        print(f"  Best Spearman: {val_df['val_spearman'].max():.4f} (Epoch {val_df.loc[val_df['val_spearman'].idxmax(), 'epoch']:.0f})")
        print(f"  Final Spearman: {val_df['val_spearman'].iloc[-1]:.4f}")
        print(f"  Best Recall@pKd≥9: {val_df['val_recall_pkd9'].max():.2f}%")
        print(f"  Final Recall@pKd≥9: {val_df['val_recall_pkd9'].iloc[-1]:.2f}%")

    if len(lr_df) > 0:
        print(f"\nLearning Rate:")
        print(f"  Initial: {lr_df['learning_rate'].iloc[0]:.6f}")
        print(f"  Final: {lr_df['learning_rate'].iloc[-1]:.6f}")

    print("="*70)

    # Check for overfitting
    if len(val_df) > 5:
        # Get indices for first half and second half
        mid_idx = len(val_df) // 2
        first_half_spearman = val_df['val_spearman'].iloc[:mid_idx].mean()
        second_half_spearman = val_df['val_spearman'].iloc[mid_idx:].mean()

        print("\n" + "="*70)
        print("OVERFITTING CHECK")
        print("="*70)

        if second_half_spearman < first_half_spearman * 0.95:
            print("⚠️  Possible overfitting detected!")
            print(f"   First half avg Spearman: {first_half_spearman:.4f}")
            print(f"   Second half avg Spearman: {second_half_spearman:.4f}")
            print(f"   Performance drop: {(1 - second_half_spearman/first_half_spearman)*100:.1f}%")
            print("\n   Recommendations:")
            print("   - Consider using early stopping")
            print("   - Try reducing learning rate")
            print("   - Increase regularization (dropout, weight decay)")
        else:
            print("✓ No significant overfitting detected")
            print(f"  First half avg Spearman: {first_half_spearman:.4f}")
            print(f"  Second half avg Spearman: {second_half_spearman:.4f}")

        print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize training progress')
    parser.add_argument('--csv', type=str, default='outputs_max_speed/training_metrics.csv',
                       help='Path to training_metrics.csv file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: same as CSV)')

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        print("\nLooking for CSV files in current directory...")
        found = list(Path('.').rglob('training_metrics.csv'))
        if found:
            print(f"Found {len(found)} training_metrics.csv files:")
            for f in found:
                print(f"  - {f}")
            print(f"\nUsing: {found[0]}")
            csv_path = found[0]
        else:
            print("No training_metrics.csv files found!")
            exit(1)

    plot_training_curves(csv_path, args.output_dir)
