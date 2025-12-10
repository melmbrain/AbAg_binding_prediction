import torch
import os
from pathlib import Path

output_dir = Path("C:/Users/401-24/Desktop/AbAg_binding_prediction/output")

# Find all checkpoint files
checkpoint_files = sorted(output_dir.glob("*.pth"))

print("=" * 80)
print("TRAINING RESULTS ANALYSIS")
print("=" * 80)
print(f"\nFound {len(checkpoint_files)} checkpoint files\n")

# Analyze best model
best_model = output_dir / "best_model-008.pth"
if best_model.exists():
    print("\n" + "=" * 80)
    print("BEST MODEL (Epoch 8)")
    print("=" * 80)
    try:
        checkpoint = torch.load(best_model, map_location='cpu', weights_only=False)
        print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        if 'best_val_loss' in checkpoint:
            print(f"Best Validation Loss: {checkpoint['best_val_loss']:.4f}")
        if 'val_loss' in checkpoint:
            print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
        if 'train_loss' in checkpoint:
            print(f"Training Loss: {checkpoint['train_loss']:.4f}")
        if 'val_rmse' in checkpoint:
            print(f"Validation RMSE: {checkpoint['val_rmse']:.4f}")
        if 'val_mae' in checkpoint:
            print(f"Validation MAE: {checkpoint['val_mae']:.4f}")
        if 'val_r2' in checkpoint:
            print(f"Validation R²: {checkpoint['val_r2']:.4f}")
        if 'val_pearson' in checkpoint:
            print(f"Validation Pearson: {checkpoint['val_pearson']:.4f}")

    except Exception as e:
        print(f"Error loading best model: {e}")

# Analyze latest checkpoint
latest_checkpoint = output_dir / "checkpoint_latest-003.pth"
if latest_checkpoint.exists():
    print("\n" + "=" * 80)
    print("LATEST CHECKPOINT")
    print("=" * 80)
    try:
        checkpoint = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
        print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

        if 'epoch' in checkpoint:
            print(f"Current Epoch: {checkpoint['epoch']}")
        if 'best_val_loss' in checkpoint:
            print(f"Best Validation Loss So Far: {checkpoint['best_val_loss']:.4f}")
        if 'val_loss' in checkpoint:
            print(f"Current Validation Loss: {checkpoint['val_loss']:.4f}")
        if 'train_loss' in checkpoint:
            print(f"Current Training Loss: {checkpoint['train_loss']:.4f}")
        if 'val_rmse' in checkpoint:
            print(f"Current Validation RMSE: {checkpoint['val_rmse']:.4f}")
        if 'val_mae' in checkpoint:
            print(f"Current Validation MAE: {checkpoint['val_mae']:.4f}")
        if 'val_r2' in checkpoint:
            print(f"Current Validation R²: {checkpoint['val_r2']:.4f}")
        if 'val_pearson' in checkpoint:
            print(f"Current Validation Pearson: {checkpoint['val_pearson']:.4f}")
        if 'best_val_spearman' in checkpoint:
            print(f"Best Validation Spearman: {checkpoint['best_val_spearman']:.4f}")
        if 'val_spearman' in checkpoint:
            print(f"Current Validation Spearman: {checkpoint['val_spearman']:.4f}")

        # Print all available metrics
        print(f"\nAll available metrics:")
        for key, value in checkpoint.items():
            if key not in ['model_state_dict', 'optimizer_state_dict', 'model', 'optimizer']:
                if isinstance(value, (int, float, str)):
                    print(f"  {key}: {value}")
                elif isinstance(value, (list, tuple)) and len(value) < 50:
                    print(f"  {key}: {value}")

        # Check for training history
        if 'train_losses' in checkpoint:
            print(f"\nTraining Loss History: {checkpoint['train_losses']}")
        if 'val_losses' in checkpoint:
            print(f"Validation Loss History: {checkpoint['val_losses']}")

    except Exception as e:
        print(f"Error loading latest checkpoint: {e}")

# Get file sizes
print("\n" + "=" * 80)
print("CHECKPOINT FILE SIZES")
print("=" * 80)
for f in checkpoint_files:
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"{f.name}: {size_mb:.2f} MB")

print("\n" + "=" * 80)
