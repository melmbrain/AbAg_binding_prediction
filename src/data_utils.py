"""
Data utilities for handling class imbalance in affinity prediction
Includes stratified sampling, class weights, and data loaders
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class AffinityBinner:
    """
    Bins affinity values (pKd) into categories for stratification
    """
    def __init__(self, bins: Optional[List[float]] = None):
        """
        Args:
            bins: Bin edges for pKd values. Default: [0, 5, 7, 9, 11, 16]
                  Creates: very_weak (<5), weak (5-7), moderate (7-9),
                          strong (9-11), very_strong (>11)
        """
        if bins is None:
            self.bins = [0, 5, 7, 9, 11, 16]
        else:
            self.bins = bins

        self.bin_labels = [
            'very_weak',
            'weak',
            'moderate',
            'strong',
            'very_strong'
        ]

    def get_bin_index(self, pkd_value: float) -> int:
        """Get bin index for a pKd value"""
        return np.digitize(pkd_value, self.bins) - 1

    def get_bin_label(self, pkd_value: float) -> str:
        """Get bin label for a pKd value"""
        idx = self.get_bin_index(pkd_value)
        return self.bin_labels[min(idx, len(self.bin_labels) - 1)]

    def bin_array(self, pkd_values: np.ndarray) -> np.ndarray:
        """Bin an array of pKd values"""
        return np.digitize(pkd_values, self.bins) - 1

    def get_bin_counts(self, pkd_values: np.ndarray) -> Dict[str, int]:
        """Get count of samples in each bin"""
        bins = self.bin_array(pkd_values)
        counts = {}
        for i, label in enumerate(self.bin_labels):
            counts[label] = np.sum(bins == i)
        return counts

    def get_bin_statistics(self, pkd_values: np.ndarray) -> pd.DataFrame:
        """Get detailed statistics for each bin"""
        bins = self.bin_array(pkd_values)
        stats = []

        for i, label in enumerate(self.bin_labels):
            mask = bins == i
            bin_values = pkd_values[mask]

            if len(bin_values) > 0:
                stats.append({
                    'bin': label,
                    'range': f'{self.bins[i]:.1f}-{self.bins[i+1]:.1f}',
                    'count': len(bin_values),
                    'percentage': 100 * len(bin_values) / len(pkd_values),
                    'mean': np.mean(bin_values),
                    'std': np.std(bin_values),
                    'min': np.min(bin_values),
                    'max': np.max(bin_values)
                })
            else:
                stats.append({
                    'bin': label,
                    'range': f'{self.bins[i]:.1f}-{self.bins[i+1]:.1f}',
                    'count': 0,
                    'percentage': 0,
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan
                })

        return pd.DataFrame(stats)


class AffinityDataset(Dataset):
    """
    PyTorch Dataset for antibody-antigen binding affinity prediction
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 binner: Optional[AffinityBinner] = None):
        """
        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target values (pKd)
            binner: AffinityBinner instance for stratification
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.binner = binner if binner is not None else AffinityBinner()

        # Precompute bin indices for each sample
        self.bin_indices = self.binner.bin_array(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'target': self.targets[idx],
            'bin': self.bin_indices[idx]
        }

    def get_bin_indices_for_samples(self) -> np.ndarray:
        """Return bin indices for all samples"""
        return self.bin_indices


class StratifiedBatchSampler(Sampler):
    """
    Stratified batch sampler that ensures each batch contains samples from all bins
    """
    def __init__(self, bin_indices: np.ndarray, batch_size: int,
                 drop_last: bool = False, shuffle: bool = True):
        """
        Args:
            bin_indices: Array of bin indices for each sample
            batch_size: Total batch size
            drop_last: Whether to drop the last incomplete batch
            shuffle: Whether to shuffle samples within bins
        """
        self.bin_indices = bin_indices
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Group sample indices by bin
        self.bins = defaultdict(list)
        for idx, bin_idx in enumerate(bin_indices):
            self.bins[bin_idx].append(idx)

        self.n_bins = len(self.bins)
        self.samples_per_bin = max(1, batch_size // self.n_bins)

        # Calculate number of batches
        min_bin_size = min(len(indices) for indices in self.bins.values())
        self.n_batches = min_bin_size // self.samples_per_bin

        if not self.drop_last:
            # Add one more batch if there are remaining samples
            max_bin_size = max(len(indices) for indices in self.bins.values())
            if max_bin_size > self.n_batches * self.samples_per_bin:
                self.n_batches += 1

    def __iter__(self):
        # Shuffle indices within each bin if requested
        bin_iterators = {}
        for bin_idx, indices in self.bins.items():
            if self.shuffle:
                indices = np.random.permutation(indices).tolist()
            bin_iterators[bin_idx] = iter(indices)

        # Generate batches
        for _ in range(self.n_batches):
            batch = []
            for bin_idx in sorted(self.bins.keys()):
                # Take samples_per_bin samples from this bin
                bin_samples = []
                iterator = bin_iterators[bin_idx]

                for _ in range(self.samples_per_bin):
                    try:
                        bin_samples.append(next(iterator))
                    except StopIteration:
                        # Restart iterator for this bin (with replacement)
                        if self.shuffle:
                            indices = np.random.permutation(self.bins[bin_idx]).tolist()
                        else:
                            indices = self.bins[bin_idx]
                        bin_iterators[bin_idx] = iter(indices)
                        try:
                            bin_samples.append(next(bin_iterators[bin_idx]))
                        except StopIteration:
                            break

                batch.extend(bin_samples)

            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self):
        return self.n_batches


class WeightedRandomSamplerByBin(Sampler):
    """
    Weighted random sampler that oversamples rare bins
    """
    def __init__(self, bin_indices: np.ndarray, num_samples: Optional[int] = None,
                 replacement: bool = True):
        """
        Args:
            bin_indices: Array of bin indices for each sample
            num_samples: Number of samples to draw. If None, uses len(bin_indices)
            replacement: Whether to sample with replacement
        """
        self.bin_indices = bin_indices
        self.num_samples = num_samples if num_samples is not None else len(bin_indices)
        self.replacement = replacement

        # Calculate weights inversely proportional to bin frequency
        bin_counts = np.bincount(bin_indices)
        total_samples = len(bin_indices)

        # Weight = 1 / (bin_frequency)
        bin_weights = np.zeros(len(bin_counts))
        for i in range(len(bin_counts)):
            if bin_counts[i] > 0:
                bin_weights[i] = total_samples / (len(bin_counts) * bin_counts[i])

        # Assign weight to each sample
        self.weights = torch.DoubleTensor([bin_weights[bin_idx] for bin_idx in bin_indices])

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples,
                                      self.replacement).tolist())

    def __len__(self):
        return self.num_samples


def calculate_class_weights(targets: np.ndarray, binner: Optional[AffinityBinner] = None,
                            method: str = 'inverse_frequency') -> torch.Tensor:
    """
    Calculate class weights for imbalanced affinity data

    Args:
        targets: Array of target values (pKd)
        binner: AffinityBinner instance
        method: 'inverse_frequency' or 'balanced' or 'effective_samples'

    Returns:
        Tensor of weights for each sample
    """
    if binner is None:
        binner = AffinityBinner()

    bin_indices = binner.bin_array(targets)
    bin_counts = np.bincount(bin_indices, minlength=len(binner.bin_labels))
    total_samples = len(targets)
    n_bins = len(binner.bin_labels)

    if method == 'inverse_frequency':
        # Weight = total_samples / (n_bins * bin_count)
        bin_weights = np.zeros(n_bins)
        for i in range(n_bins):
            if bin_counts[i] > 0:
                bin_weights[i] = total_samples / (n_bins * bin_counts[i])
            else:
                bin_weights[i] = 0.0

    elif method == 'balanced':
        # Weight = 1 / bin_count (normalized)
        bin_weights = np.zeros(n_bins)
        for i in range(n_bins):
            if bin_counts[i] > 0:
                bin_weights[i] = 1.0 / bin_counts[i]
            else:
                bin_weights[i] = 0.0
        # Normalize
        bin_weights = bin_weights / bin_weights.sum() * n_bins

    elif method == 'effective_samples':
        # Based on "Class-Balanced Loss Based on Effective Number of Samples"
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, bin_counts)
        bin_weights = (1.0 - beta) / (effective_num + 1e-8)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Assign weight to each sample based on its bin
    sample_weights = torch.FloatTensor([bin_weights[bin_idx] for bin_idx in bin_indices])

    return sample_weights


def create_stratified_split(features: np.ndarray, targets: np.ndarray,
                           train_size: float = 0.8, random_state: int = 42,
                           binner: Optional[AffinityBinner] = None) -> Tuple:
    """
    Create stratified train/validation split

    Args:
        features: Feature matrix
        targets: Target values (pKd)
        train_size: Proportion of data for training
        random_state: Random seed
        binner: AffinityBinner instance

    Returns:
        X_train, X_val, y_train, y_val
    """
    from sklearn.model_selection import train_test_split

    if binner is None:
        binner = AffinityBinner()

    # Bin the targets for stratification
    bin_indices = binner.bin_array(targets)

    # Stratified split
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        train_size=train_size,
        stratify=bin_indices,
        random_state=random_state
    )

    X_train = features[train_idx]
    X_val = features[val_idx]
    y_train = targets[train_idx]
    y_val = targets[val_idx]

    return X_train, X_val, y_train, y_val


def print_dataset_statistics(targets: np.ndarray, name: str = "Dataset",
                             binner: Optional[AffinityBinner] = None):
    """
    Print detailed statistics about affinity distribution
    """
    if binner is None:
        binner = AffinityBinner()

    stats = binner.get_bin_statistics(targets)

    print(f"\n{'='*80}")
    print(f"{name} Statistics")
    print(f"{'='*80}")
    print(f"Total samples: {len(targets)}")
    print(f"Mean pKd: {np.mean(targets):.2f}")
    print(f"Std pKd: {np.std(targets):.2f}")
    print(f"Range: [{np.min(targets):.2f}, {np.max(targets):.2f}]")
    print(f"\nDistribution by affinity range:")
    print(stats.to_string(index=False))
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Example usage
    print("Testing data utilities...")

    # Create synthetic data mimicking your distribution
    np.random.seed(42)
    n_samples = 1000

    # Simulate your distribution: mostly moderate, few extremes
    targets = np.concatenate([
        np.random.uniform(0, 5, 20),      # very weak (2%)
        np.random.uniform(5, 7, 320),     # weak (32%)
        np.random.uniform(7, 9, 350),     # moderate (35%)
        np.random.uniform(9, 11, 290),    # strong (29%)
        np.random.uniform(11, 15, 20)     # very strong (2%)
    ])
    np.random.shuffle(targets)

    features = np.random.randn(n_samples, 150)  # 150 features

    # Test AffinityBinner
    binner = AffinityBinner()
    print_dataset_statistics(targets, "Synthetic Dataset", binner)

    # Test stratified split
    X_train, X_val, y_train, y_val = create_stratified_split(
        features, targets, train_size=0.8, binner=binner
    )

    print_dataset_statistics(y_train, "Training Set", binner)
    print_dataset_statistics(y_val, "Validation Set", binner)

    # Test class weights
    weights = calculate_class_weights(y_train, binner, method='inverse_frequency')
    print(f"Sample weights - Min: {weights.min():.4f}, Max: {weights.max():.4f}, "
          f"Mean: {weights.mean():.4f}")

    # Test dataset and dataloaders
    train_dataset = AffinityDataset(X_train, y_train, binner)

    # Regular dataloader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Stratified batch sampler
    stratified_sampler = StratifiedBatchSampler(
        train_dataset.get_bin_indices_for_samples(),
        batch_size=32,
        shuffle=True
    )
    stratified_loader = DataLoader(train_dataset, batch_sampler=stratified_sampler)

    # Weighted random sampler
    weighted_sampler = WeightedRandomSamplerByBin(
        train_dataset.get_bin_indices_for_samples(),
        num_samples=len(train_dataset)
    )
    weighted_loader = DataLoader(train_dataset, batch_size=32, sampler=weighted_sampler)

    print("\nTesting dataloaders...")
    print(f"Regular loader: {len(train_loader)} batches")
    print(f"Stratified loader: {len(stratified_loader)} batches")
    print(f"Weighted loader: {len(weighted_loader)} batches")

    # Check batch composition
    batch = next(iter(stratified_loader))
    print(f"\nStratified batch composition:")
    batch_bins = batch['bin'].numpy()
    for i, label in enumerate(binner.bin_labels):
        count = np.sum(batch_bins == i)
        print(f"  {label}: {count} samples")

    print("\n✓ All tests passed!")
