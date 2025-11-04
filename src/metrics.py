"""
Evaluation metrics for binding affinity prediction
Tracks performance by affinity range to monitor extreme value prediction
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


class AffinityMetrics:
    """
    Comprehensive metrics for affinity prediction
    Tracks overall and per-bin performance
    """
    def __init__(self, bins: Optional[List[float]] = None,
                 bin_labels: Optional[List[str]] = None):
        """
        Args:
            bins: Bin edges for pKd values
            bin_labels: Labels for each bin
        """
        if bins is None:
            self.bins = [0, 5, 7, 9, 11, 16]
        else:
            self.bins = bins

        if bin_labels is None:
            self.bin_labels = ['very_weak', 'weak', 'moderate', 'strong', 'very_strong']
        else:
            self.bin_labels = bin_labels

    def compute_overall_metrics(self, y_true: np.ndarray,
                                y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute overall performance metrics

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Regression metrics
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)

        # Correlation metrics
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        spearman_r, spearman_p = spearmanr(y_true, y_pred)

        metrics['pearson_r'] = pearson_r
        metrics['pearson_p'] = pearson_p
        metrics['spearman_r'] = spearman_r
        metrics['spearman_p'] = spearman_p

        # Additional metrics
        errors = y_pred - y_true
        metrics['mean_error'] = np.mean(errors)
        metrics['std_error'] = np.std(errors)
        metrics['median_absolute_error'] = np.median(np.abs(errors))

        # Percentage within thresholds
        metrics['within_0.5'] = np.mean(np.abs(errors) < 0.5) * 100
        metrics['within_1.0'] = np.mean(np.abs(errors) < 1.0) * 100
        metrics['within_2.0'] = np.mean(np.abs(errors) < 2.0) * 100

        return metrics

    def compute_per_bin_metrics(self, y_true: np.ndarray,
                                y_pred: np.ndarray) -> pd.DataFrame:
        """
        Compute metrics for each affinity bin

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            DataFrame with per-bin metrics
        """
        bin_indices = np.digitize(y_true, self.bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(self.bin_labels) - 1)

        results = []

        for i, label in enumerate(self.bin_labels):
            mask = bin_indices == i
            n_samples = mask.sum()

            if n_samples == 0:
                # No samples in this bin
                results.append({
                    'bin': label,
                    'range': f'{self.bins[i]:.1f}-{self.bins[i+1]:.1f}',
                    'n_samples': 0,
                    'rmse': np.nan,
                    'mae': np.nan,
                    'r2': np.nan,
                    'pearson_r': np.nan,
                    'mean_error': np.nan,
                    'within_1.0': np.nan
                })
                continue

            y_true_bin = y_true[mask]
            y_pred_bin = y_pred[mask]

            # Compute metrics for this bin
            rmse = np.sqrt(mean_squared_error(y_true_bin, y_pred_bin))
            mae = mean_absolute_error(y_true_bin, y_pred_bin)

            # R2 and Pearson might fail if no variance
            try:
                r2 = r2_score(y_true_bin, y_pred_bin)
            except:
                r2 = np.nan

            try:
                pearson_r, _ = pearsonr(y_true_bin, y_pred_bin)
            except:
                pearson_r = np.nan

            errors = y_pred_bin - y_true_bin
            mean_error = np.mean(errors)
            within_1 = np.mean(np.abs(errors) < 1.0) * 100

            results.append({
                'bin': label,
                'range': f'{self.bins[i]:.1f}-{self.bins[i+1]:.1f}',
                'n_samples': n_samples,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'pearson_r': pearson_r,
                'mean_error': mean_error,
                'within_1.0': within_1
            })

        return pd.DataFrame(results)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                verbose: bool = True) -> Dict:
        """
        Complete evaluation with overall and per-bin metrics

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            verbose: Whether to print results

        Returns:
            Dictionary containing all metrics
        """
        overall = self.compute_overall_metrics(y_true, y_pred)
        per_bin = self.compute_per_bin_metrics(y_true, y_pred)

        if verbose:
            self.print_results(overall, per_bin)

        return {
            'overall': overall,
            'per_bin': per_bin
        }

    @staticmethod
    def print_results(overall: Dict, per_bin: pd.DataFrame):
        """
        Print formatted evaluation results
        """
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)

        print("\nOverall Performance:")
        print("-" * 80)
        print(f"  RMSE:              {overall['rmse']:.4f}")
        print(f"  MAE:               {overall['mae']:.4f}")
        print(f"  R²:                {overall['r2']:.4f}")
        print(f"  Pearson r:         {overall['pearson_r']:.4f} (p={overall['pearson_p']:.2e})")
        print(f"  Spearman r:        {overall['spearman_r']:.4f}")
        print(f"  Mean error:        {overall['mean_error']:.4f} ± {overall['std_error']:.4f}")
        print(f"  Median abs error:  {overall['median_absolute_error']:.4f}")
        print(f"\nAccuracy within threshold:")
        print(f"  ±0.5 pKd units:    {overall['within_0.5']:.1f}%")
        print(f"  ±1.0 pKd units:    {overall['within_1.0']:.1f}%")
        print(f"  ±2.0 pKd units:    {overall['within_2.0']:.1f}%")

        print("\n" + "-" * 80)
        print("Performance by Affinity Range:")
        print("-" * 80)
        print(per_bin.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        print("="*80 + "\n")

    def plot_results(self, y_true: np.ndarray, y_pred: np.ndarray,
                    save_path: Optional[str] = None):
        """
        Create visualization plots for evaluation

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Scatter plot with identity line
        ax = axes[0, 0]
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        ax.set_xlabel('True pKd', fontsize=12)
        ax.set_ylabel('Predicted pKd', fontsize=12)
        ax.set_title('Predicted vs True Affinity', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add R² and RMSE to plot
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 2. Residual plot
        ax = axes[0, 1]
        residuals = y_pred - y_true
        ax.scatter(y_true, residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.axhline(y=1, color='orange', linestyle=':', lw=1, label='±1 pKd')
        ax.axhline(y=-1, color='orange', linestyle=':', lw=1)
        ax.set_xlabel('True pKd', fontsize=12)
        ax.set_ylabel('Residual (Predicted - True)', fontsize=12)
        ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Error distribution by bin
        ax = axes[1, 0]
        bin_indices = np.digitize(y_true, self.bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(self.bin_labels) - 1)

        errors_by_bin = [residuals[bin_indices == i] for i in range(len(self.bin_labels))]
        bp = ax.boxplot(errors_by_bin, labels=self.bin_labels, patch_artist=True)

        # Color boxes
        colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99', '#ff99ff']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Affinity Range', fontsize=12)
        ax.set_ylabel('Prediction Error (pKd units)', fontsize=12)
        ax.set_title('Error Distribution by Affinity Range', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 4. Per-bin metrics comparison
        ax = axes[1, 1]
        per_bin_metrics = self.compute_per_bin_metrics(y_true, y_pred)

        x = np.arange(len(self.bin_labels))
        width = 0.35

        # Plot RMSE and MAE
        ax2 = ax.twinx()
        bars1 = ax.bar(x - width/2, per_bin_metrics['rmse'], width,
                       label='RMSE', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, per_bin_metrics['mae'], width,
                       label='MAE', color='lightcoral', alpha=0.8)

        # Plot sample counts on secondary axis
        line = ax2.plot(x, per_bin_metrics['n_samples'], 'go-', linewidth=2,
                       markersize=8, label='N samples')

        ax.set_xlabel('Affinity Range', fontsize=12)
        ax.set_ylabel('Error (pKd units)', fontsize=12)
        ax2.set_ylabel('Number of samples', fontsize=12)
        ax.set_title('Metrics by Affinity Range', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.bin_labels, rotation=45, ha='right')

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        return fig


class MetricsTracker:
    """
    Track metrics over training epochs
    """
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'val_mae': [],
            'val_r2': [],
            'val_pearson': []
        }

    def update(self, metrics: Dict, prefix: str = 'val'):
        """
        Update history with new metrics

        Args:
            metrics: Dictionary of metrics
            prefix: Prefix for metric names ('train' or 'val')
        """
        for key, value in metrics.items():
            history_key = f'{prefix}_{key}'
            if history_key not in self.history:
                self.history[history_key] = []
            self.history[history_key].append(value)

    def plot_history(self, save_path: Optional[str] = None):
        """
        Plot training history

        Args:
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot loss
        ax = axes[0, 0]
        if 'train_loss' in self.history and len(self.history['train_loss']) > 0:
            ax.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in self.history and len(self.history['val_loss']) > 0:
            ax.plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot RMSE
        ax = axes[0, 1]
        if 'val_rmse' in self.history and len(self.history['val_rmse']) > 0:
            ax.plot(self.history['val_rmse'], label='Val RMSE', linewidth=2, color='coral')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('Validation RMSE', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot R²
        ax = axes[1, 0]
        if 'val_r2' in self.history and len(self.history['val_r2']) > 0:
            ax.plot(self.history['val_r2'], label='Val R²', linewidth=2, color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('R²')
        ax.set_title('Validation R²', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot Pearson correlation
        ax = axes[1, 1]
        if 'val_pearson' in self.history and len(self.history['val_pearson']) > 0:
            ax.plot(self.history['val_pearson'], label='Val Pearson r',
                   linewidth=2, color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Pearson r')
        ax.set_title('Validation Pearson Correlation', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")

        return fig

    def get_best_epoch(self, metric: str = 'val_rmse',
                      mode: str = 'min') -> Tuple[int, float]:
        """
        Get the epoch with best performance

        Args:
            metric: Metric to use for selection
            mode: 'min' or 'max'

        Returns:
            (best_epoch, best_value)
        """
        if metric not in self.history or len(self.history[metric]) == 0:
            return 0, 0.0

        values = self.history[metric]
        if mode == 'min':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

        return best_idx, values[best_idx]


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics...")

    # Create synthetic test data
    np.random.seed(42)
    n_samples = 1000

    # True values with realistic distribution
    y_true = np.concatenate([
        np.random.uniform(0, 5, 20),
        np.random.uniform(5, 7, 320),
        np.random.uniform(7, 9, 350),
        np.random.uniform(9, 11, 290),
        np.random.uniform(11, 15, 20)
    ])

    # Predictions with some error
    noise = np.random.normal(0, 0.5, n_samples)
    y_pred = y_true + noise

    # Add systematic error for extreme values (model underperforms on extremes)
    extreme_mask = (y_true < 5) | (y_true > 11)
    y_pred[extreme_mask] += np.random.normal(0, 1.5, extreme_mask.sum())

    # Evaluate
    metrics = AffinityMetrics()
    results = metrics.evaluate(y_true, y_pred, verbose=True)

    # Create plots
    print("\nGenerating plots...")
    fig = metrics.plot_results(y_true, y_pred,
                              save_path='test_evaluation_plots.png')

    # Test metrics tracker
    print("\nTesting metrics tracker...")
    tracker = MetricsTracker()

    for epoch in range(20):
        # Simulate improving metrics
        tracker.update({'loss': 1.0 - epoch * 0.04}, prefix='train')
        tracker.update({'loss': 1.2 - epoch * 0.05,
                       'rmse': 1.0 - epoch * 0.03,
                       'r2': epoch * 0.04,
                       'pearson': 0.5 + epoch * 0.02}, prefix='val')

    tracker.plot_history(save_path='test_training_history.png')

    best_epoch, best_rmse = tracker.get_best_epoch('val_rmse', mode='min')
    print(f"\nBest epoch: {best_epoch}, Best RMSE: {best_rmse:.4f}")

    print("\n✓ All metrics tests passed!")
