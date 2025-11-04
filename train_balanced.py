"""
Training script with class imbalance handling for affinity prediction

This script implements:
- Stratified sampling
- Class weights
- Focal loss
- Per-bin evaluation
- All recommended strategies from references
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import custom modules
from src.data_utils import (
    AffinityBinner, AffinityDataset, StratifiedBatchSampler,
    WeightedRandomSamplerByBin, calculate_class_weights,
    create_stratified_split, print_dataset_statistics
)
from src.losses import (
    FocalMSELoss, WeightedMSELoss, RangeFocusedLoss, get_loss_function
)
from src.metrics import AffinityMetrics, MetricsTracker


class SimpleAffinityModel(nn.Module):
    """
    Simple MLP model for affinity prediction
    Replace this with your actual model architecture
    """
    def __init__(self, input_dim: int, hidden_dims: list = [512, 256, 128],
                 dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class BalancedAffinityTrainer:
    """
    Trainer with class imbalance handling
    """
    def __init__(self, model, config):
        """
        Args:
            model: PyTorch model
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)

        # Initialize components
        self.binner = AffinityBinner()
        self.metrics = AffinityMetrics()
        self.tracker = MetricsTracker()

        # Loss function
        self.loss_fn = self._create_loss_function()

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def _create_loss_function(self):
        """Create loss function based on config"""
        loss_type = self.config.get('loss_type', 'weighted_mse')

        if loss_type == 'focal_mse':
            return FocalMSELoss(
                gamma=self.config.get('focal_gamma', 2.0)
            )
        elif loss_type == 'weighted_mse':
            return WeightedMSELoss()
        elif loss_type == 'range_focused':
            return RangeFocusedLoss(
                range_weights=self.config.get('range_weights', [10.0, 1.0, 1.0, 1.0, 10.0])
            )
        else:
            return nn.MSELoss()

    def _create_optimizer(self):
        """Create optimizer"""
        optimizer_type = self.config.get('optimizer', 'adam')
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-5)

        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9,
                           weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'plateau')

        if scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.get('epochs', 100)
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        else:
            return None

    def train_epoch(self, train_loader, sample_weights=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            features = batch['features'].to(self.device)
            targets = batch['target'].to(self.device)
            bin_indices = batch['bin']

            # Forward pass
            predictions = self.model(features)

            # Get sample weights for this batch if using weighted loss
            if sample_weights is not None and 'weighted' in self.config.get('loss_type', ''):
                batch_weights = sample_weights[bin_indices].to(self.device)
                loss = self.loss_fn(predictions, targets, batch_weights)
            else:
                loss = self.loss_fn(predictions, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.get('clip_grad', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.config['clip_grad'])

            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            n_batches += 1

            pbar.set_postfix({'loss': f'{total_loss/n_batches:.4f}'})

        return total_loss / n_batches

    @torch.no_grad()
    def evaluate(self, val_loader):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        n_batches = 0

        all_predictions = []
        all_targets = []

        for batch in val_loader:
            features = batch['features'].to(self.device)
            targets = batch['target'].to(self.device)

            # Forward pass
            predictions = self.model(features)

            # Calculate loss
            loss = self.loss_fn(predictions, targets)

            total_loss += loss.item()
            n_batches += 1

            # Collect predictions
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        # Concatenate all predictions
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)

        # Compute metrics
        results = self.metrics.evaluate(all_targets, all_predictions, verbose=False)

        return total_loss / n_batches, results

    def train(self, train_loader, val_loader, sample_weights=None):
        """
        Complete training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            sample_weights: Per-sample weights for training (optional)
        """
        epochs = self.config.get('epochs', 100)
        save_dir = Path(self.config.get('save_dir', 'checkpoints'))
        save_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Loss function: {self.config.get('loss_type', 'mse')}")
        print(f"Optimizer: {self.config.get('optimizer', 'adam')}")
        print(f"Learning rate: {self.config.get('learning_rate', 0.001)}")
        print(f"Batch size: {self.config.get('batch_size', 32)}")
        print(f"Sampling strategy: {self.config.get('sampling_strategy', 'standard')}")
        print("="*80 + "\n")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 80)

            # Train
            train_loss = self.train_epoch(train_loader, sample_weights)

            # Validate
            val_loss, val_results = self.evaluate(val_loader)

            # Update tracker
            self.tracker.update({'loss': train_loss}, prefix='train')
            self.tracker.update({
                'loss': val_loss,
                'rmse': val_results['overall']['rmse'],
                'mae': val_results['overall']['mae'],
                'r2': val_results['overall']['r2'],
                'pearson': val_results['overall']['pearson_r']
            }, prefix='val')

            # Print results
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")
            print(f"Val RMSE:   {val_results['overall']['rmse']:.4f}")
            print(f"Val MAE:    {val_results['overall']['mae']:.4f}")
            print(f"Val R²:     {val_results['overall']['r2']:.4f}")
            print(f"Val Pearson: {val_results['overall']['pearson_r']:.4f}")

            # Print per-bin performance
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print("\nPer-bin performance:")
                print(val_results['per_bin'][['bin', 'n_samples', 'rmse', 'mae']].to_string(index=False))

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()

                # Save checkpoint
                checkpoint_path = save_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, checkpoint_path)
                print(f"\n✓ New best model saved (val_loss: {val_loss:.4f})")

        # Training complete
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        # Load best model
        self.model.load_state_dict(self.best_model_state)

        # Final evaluation
        print("\nFinal evaluation on validation set:")
        _, final_results = self.evaluate(val_loader)
        self.metrics.print_results(final_results['overall'], final_results['per_bin'])

        # Save plots
        plots_dir = save_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)

        # Training history
        self.tracker.plot_history(save_path=plots_dir / 'training_history.png')

        # Evaluation plots
        all_predictions = []
        all_targets = []
        for batch in val_loader:
            features = batch['features'].to(self.device)
            targets = batch['target']
            predictions = self.model(features)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.numpy())

        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        self.metrics.plot_results(all_targets, all_predictions,
                                 save_path=plots_dir / 'final_evaluation.png')

        print(f"\nPlots saved to: {plots_dir}")

        return self.tracker, final_results


def load_data(data_path: str, config: dict):
    """
    Load and prepare data

    Args:
        data_path: Path to CSV file
        config: Configuration dictionary

    Returns:
        train_loader, val_loader, sample_weights
    """
    print("\n" + "="*80)
    print("LOADING AND PREPARING DATA")
    print("="*80)

    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")

    # Extract features and targets
    # Assuming features are ESM2 PCA columns
    feature_cols = [col for col in df.columns if col.startswith('esm2_pca_')]
    if not feature_cols:
        raise ValueError("No feature columns found (expected 'esm2_pca_*' columns)")

    features = df[feature_cols].values
    targets = df['pKd'].values

    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")

    # Create binner and print statistics
    binner = AffinityBinner()
    print_dataset_statistics(targets, "Full Dataset", binner)

    # Stratified split
    print("\nCreating stratified train/validation split...")
    X_train, X_val, y_train, y_val = create_stratified_split(
        features, targets,
        train_size=config.get('train_size', 0.8),
        random_state=config.get('random_seed', 42),
        binner=binner
    )

    print_dataset_statistics(y_train, "Training Set", binner)
    print_dataset_statistics(y_val, "Validation Set", binner)

    # Create datasets
    train_dataset = AffinityDataset(X_train, y_train, binner)
    val_dataset = AffinityDataset(X_val, y_val, binner)

    # Calculate sample weights
    sample_weights = None
    if 'weighted' in config.get('loss_type', '') or config.get('use_sample_weights', False):
        print("\nCalculating sample weights...")
        sample_weights = calculate_class_weights(
            y_train, binner,
            method=config.get('weight_method', 'inverse_frequency')
        )
        print(f"  Weights range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")

    # Create data loaders
    batch_size = config.get('batch_size', 32)
    sampling_strategy = config.get('sampling_strategy', 'stratified')

    print(f"\nCreating data loaders (strategy: {sampling_strategy})...")

    if sampling_strategy == 'stratified':
        # Stratified batch sampling
        train_sampler = StratifiedBatchSampler(
            train_dataset.get_bin_indices_for_samples(),
            batch_size=batch_size,
            shuffle=True
        )
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)

    elif sampling_strategy == 'weighted':
        # Weighted random sampling
        train_sampler = WeightedRandomSamplerByBin(
            train_dataset.get_bin_indices_for_samples(),
            num_samples=len(train_dataset)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    else:
        # Standard sampling
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    return train_loader, val_loader, sample_weights, X_train.shape[1]


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train affinity prediction model with class imbalance handling')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--loss', type=str, default='weighted_mse',
                       choices=['mse', 'weighted_mse', 'focal_mse', 'range_focused'],
                       help='Loss function')
    parser.add_argument('--sampling', type=str, default='stratified',
                       choices=['standard', 'stratified', 'weighted'],
                       help='Sampling strategy')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto, cuda, cpu)')

    args = parser.parse_args()

    # Load or create config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}

    # Update config with command line args
    config.update({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'loss_type': args.loss,
        'sampling_strategy': args.sampling,
        'save_dir': args.save_dir
    })

    # Device
    if args.device == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config['device'] = args.device

    # Load data
    train_loader, val_loader, sample_weights, input_dim = load_data(args.data, config)

    # Create model
    print("\nCreating model...")
    model = SimpleAffinityModel(
        input_dim=input_dim,
        hidden_dims=config.get('hidden_dims', [512, 256, 128]),
        dropout=config.get('dropout', 0.3)
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = BalancedAffinityTrainer(model, config)

    # Train
    tracker, results = trainer.train(train_loader, val_loader, sample_weights)

    # Save config
    config_path = Path(config['save_dir']) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfiguration saved to: {config_path}")

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
