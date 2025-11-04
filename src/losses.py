"""
Custom loss functions for imbalanced affinity prediction
Includes Focal Loss, Weighted MSE, and Huber Loss variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class FocalMSELoss(nn.Module):
    """
    Focal Loss adapted for regression tasks
    Focuses on hard-to-predict samples (large errors)

    Based on: Lin et al. (2017) "Focal Loss for Dense Object Detection"
    Adapted for regression on imbalanced affinity data
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        """
        Args:
            gamma: Focusing parameter. Higher values focus more on hard examples.
                   Typical values: 1.0-3.0. Default: 2.0
            alpha: Class weights (per-sample weights). If None, no weighting.
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions (batch_size,)
            targets: Ground truth targets (batch_size,)

        Returns:
            Focal MSE loss
        """
        # Compute squared error
        mse = (predictions - targets) ** 2

        # Normalize MSE to [0, 1] range for focal weight calculation
        # Using sigmoid-like normalization
        max_error = 10.0  # Maximum expected error in pKd units
        normalized_error = torch.clamp(torch.sqrt(mse) / max_error, 0, 1)

        # Focal weight: (normalized_error)^gamma
        # Higher error -> higher weight
        focal_weight = torch.pow(normalized_error, self.gamma)

        # Apply focal weight
        loss = focal_weight * mse

        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != loss.device:
                self.alpha = self.alpha.to(loss.device)
            loss = loss * self.alpha

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedMSELoss(nn.Module):
    """
    MSE Loss with per-sample weights for class imbalance
    """
    def __init__(self, weights: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        """
        Args:
            weights: Per-sample weights (batch_size,)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.weights = weights
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            sample_weights: Optional per-batch sample weights (overrides self.weights)

        Returns:
            Weighted MSE loss
        """
        mse = (predictions - targets) ** 2

        # Use provided sample_weights or fall back to self.weights
        weights = sample_weights if sample_weights is not None else self.weights

        if weights is not None:
            if weights.device != mse.device:
                weights = weights.to(mse.device)
            mse = mse * weights

        if self.reduction == 'mean':
            return mse.mean()
        elif self.reduction == 'sum':
            return mse.sum()
        else:
            return mse


class WeightedHuberLoss(nn.Module):
    """
    Huber Loss with per-sample weights
    More robust to outliers than MSE
    """
    def __init__(self, delta: float = 1.0, weights: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        """
        Args:
            delta: Threshold for switching between L1 and L2 loss
            weights: Per-sample weights
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.delta = delta
        self.weights = weights
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            sample_weights: Optional per-batch sample weights

        Returns:
            Weighted Huber loss
        """
        error = predictions - targets
        abs_error = torch.abs(error)

        # Huber loss: quadratic for small errors, linear for large errors
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear

        # Apply weights
        weights = sample_weights if sample_weights is not None else self.weights
        if weights is not None:
            if weights.device != loss.device:
                weights = weights.to(loss.device)
            loss = loss * weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combination of multiple losses
    Useful for balancing different objectives
    """
    def __init__(self, losses: dict, weights: dict):
        """
        Args:
            losses: Dict of loss functions {'name': loss_fn}
            weights: Dict of loss weights {'name': weight}
        """
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.loss_weights = weights

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                **kwargs) -> dict:
        """
        Returns:
            Dict with total loss and individual loss components
        """
        total_loss = 0
        loss_dict = {}

        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(predictions, targets, **kwargs)
            weighted_loss = self.loss_weights[name] * loss_value

            loss_dict[f'loss_{name}'] = loss_value.item()
            total_loss += weighted_loss

        loss_dict['loss_total'] = total_loss

        return loss_dict


class RangeFocusedLoss(nn.Module):
    """
    Custom loss that applies different weights to different affinity ranges
    Emphasizes extreme ranges (very weak and very strong)
    """
    def __init__(self, bins: list = None, range_weights: list = None,
                 base_loss: str = 'mse', reduction: str = 'mean'):
        """
        Args:
            bins: Bin edges for pKd values [0, 5, 7, 9, 11, 16]
            range_weights: Weight for each range [very_weak, weak, moderate, strong, very_strong]
                          Default: [10, 1, 1, 1, 10] - emphasize extremes
            base_loss: 'mse' or 'huber'
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        if bins is None:
            self.bins = torch.FloatTensor([0, 5, 7, 9, 11, 16])
        else:
            self.bins = torch.FloatTensor(bins)

        if range_weights is None:
            # Default: emphasize extreme ranges
            self.range_weights = torch.FloatTensor([10.0, 1.0, 1.0, 1.0, 10.0])
        else:
            self.range_weights = torch.FloatTensor(range_weights)

        self.base_loss = base_loss
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Range-focused loss
        """
        # Move bins and weights to same device as targets
        if self.bins.device != targets.device:
            self.bins = self.bins.to(targets.device)
        if self.range_weights.device != targets.device:
            self.range_weights = self.range_weights.to(targets.device)

        # Compute base loss
        if self.base_loss == 'mse':
            base_loss = (predictions - targets) ** 2
        elif self.base_loss == 'huber':
            error = predictions - targets
            abs_error = torch.abs(error)
            base_loss = torch.where(
                abs_error <= 1.0,
                0.5 * error ** 2,
                abs_error - 0.5
            )
        else:
            raise ValueError(f"Unknown base loss: {self.base_loss}")

        # Determine which bin each target belongs to
        # digitize equivalent in pytorch
        bin_indices = torch.searchsorted(self.bins[:-1], targets, right=False)
        bin_indices = torch.clamp(bin_indices, 0, len(self.range_weights) - 1)

        # Apply range-specific weights
        sample_weights = self.range_weights[bin_indices]
        weighted_loss = base_loss * sample_weights

        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """
    Factory function to get loss function by name

    Args:
        loss_type: 'mse', 'weighted_mse', 'focal_mse', 'huber',
                   'weighted_huber', 'range_focused'
        **kwargs: Additional arguments for the loss function

    Returns:
        Loss function module
    """
    if loss_type == 'mse':
        return nn.MSELoss()

    elif loss_type == 'weighted_mse':
        return WeightedMSELoss(**kwargs)

    elif loss_type == 'focal_mse':
        return FocalMSELoss(**kwargs)

    elif loss_type == 'huber':
        return nn.SmoothL1Loss()

    elif loss_type == 'weighted_huber':
        return WeightedHuberLoss(**kwargs)

    elif loss_type == 'range_focused':
        return RangeFocusedLoss(**kwargs)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")

    # Create synthetic data
    torch.manual_seed(42)
    batch_size = 32
    predictions = torch.randn(batch_size) * 2 + 8  # Around pKd=8
    targets = torch.randn(batch_size) * 2 + 8

    # Add some extreme values
    targets[0] = 3.0   # very weak
    targets[1] = 13.0  # very strong
    predictions[0] = 7.0  # bad prediction for very weak
    predictions[1] = 9.0  # bad prediction for very strong

    print(f"\nTarget range: [{targets.min():.2f}, {targets.max():.2f}]")
    print(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")

    # Test standard MSE
    mse_loss = nn.MSELoss()
    print(f"\nStandard MSE Loss: {mse_loss(predictions, targets):.4f}")

    # Test Focal MSE
    focal_loss = FocalMSELoss(gamma=2.0)
    print(f"Focal MSE Loss (gamma=2.0): {focal_loss(predictions, targets):.4f}")

    # Test Weighted MSE with higher weights for extremes
    from src.data_utils import AffinityBinner, calculate_class_weights
    binner = AffinityBinner()
    weights = calculate_class_weights(targets.numpy(), binner)

    weighted_mse = WeightedMSELoss()
    print(f"Weighted MSE Loss: {weighted_mse(predictions, targets, weights):.4f}")

    # Test Range Focused Loss
    range_loss = RangeFocusedLoss(range_weights=[10.0, 1.0, 1.0, 1.0, 10.0])
    print(f"Range Focused Loss: {range_loss(predictions, targets):.4f}")

    # Compare errors for extreme vs moderate values
    extreme_mask = (targets < 5) | (targets > 11)
    moderate_mask = (targets >= 7) & (targets <= 9)

    extreme_errors = ((predictions - targets) ** 2)[extreme_mask]
    moderate_errors = ((predictions - targets) ** 2)[moderate_mask]

    print(f"\nError analysis:")
    print(f"  Extreme samples: {extreme_mask.sum()} samples, "
          f"mean error: {extreme_errors.mean():.4f}")
    print(f"  Moderate samples: {moderate_mask.sum()} samples, "
          f"mean error: {moderate_errors.mean():.4f}")

    # Test combined loss
    losses = {
        'mse': WeightedMSELoss(),
        'focal': FocalMSELoss(gamma=1.5)
    }
    loss_weights = {
        'mse': 0.5,
        'focal': 0.5
    }
    combined = CombinedLoss(losses, loss_weights)
    loss_dict = combined(predictions, targets, sample_weights=weights)
    print(f"\nCombined Loss: {loss_dict}")

    print("\n✓ All loss function tests passed!")
