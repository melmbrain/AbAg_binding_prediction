"""
Training Script for IgT5 + ESM-2 Hybrid Model

State-of-the-art architecture (2024-2025):
- IgT5 for antibody embeddings (best binding affinity prediction)
- ESM-2 for antigen embeddings (best epitope prediction)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import argparse
from pathlib import Path
import time

from model_igt5_esm2 import IgT5ESM2Model, FocalMSELoss, count_parameters


class AbAgDataset(Dataset):
    """Simple dataset for antibody-antigen pairs"""

    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        print(f"Dataset created with {len(df):,} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            'antibody_sequence': self.df.iloc[idx]['antibody_sequence'],
            'antigen_sequence': self.df.iloc[idx]['antigen_sequence'],
            'pKd': torch.tensor(self.df.iloc[idx]['pKd'], dtype=torch.float32)
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    antibody_seqs = [item['antibody_sequence'] for item in batch]
    antigen_seqs = [item['antigen_sequence'] for item in batch]
    pKds = torch.stack([item['pKd'] for item in batch])

    return {
        'antibody_seqs': antibody_seqs,
        'antigen_seqs': antigen_seqs,
        'pKd': pKds
    }


def train_epoch(model, loader, optimizer, criterion, device, scaler):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        antibody_seqs = batch['antibody_seqs']
        antigen_seqs = batch['antigen_seqs']
        targets = batch['pKd'].to(device)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            predictions = model(antibody_seqs, antigen_seqs, device)
            loss = criterion(predictions, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.2e}'})

    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Evaluate model on validation/test set"""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            antibody_seqs = batch['antibody_seqs']
            antigen_seqs = batch['antigen_seqs']
            batch_targets = batch['pKd'].to(device)

            with torch.amp.autocast('cuda'):
                batch_predictions = model(antibody_seqs, antigen_seqs, device)

            predictions.extend(batch_predictions.cpu().numpy())
            targets.extend(batch_targets.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    spearman = stats.spearmanr(targets, predictions)[0]
    pearson = np.corrcoef(targets, predictions)[0, 1]

    # Recall for strong binders (pKd >= 9)
    strong_binders = targets >= 9.0
    predicted_strong = predictions >= 9.0
    recall_pkd9 = (strong_binders & predicted_strong).sum() / strong_binders.sum() if strong_binders.sum() > 0 else 0

    # Recall for weak binders (pKd < 7)
    weak_binders = targets < 7.0
    predicted_weak = predictions < 7.0
    recall_pkd7 = (weak_binders & predicted_weak).sum() / weak_binders.sum() if weak_binders.sum() > 0 else 0

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'spearman': spearman,
        'pearson': pearson,
        'recall_pkd9': recall_pkd9 * 100,
        'recall_pkd7': recall_pkd7 * 100,
        'predictions': predictions,
        'targets': targets
    }


def main(args):
    # Setup
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"IGT5 + ESM-2 HYBRID TRAINING (State-of-the-art 2024)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"{'='*70}\n")

    # Load data
    print(f"Loading data: {args.data}")
    df = pd.read_csv(args.data)
    print(f"✅ Loaded {len(df):,} samples\n")

    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Data split:")
    print(f"  Train: {len(train_df):,}")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}\n")

    # Create datasets and loaders
    train_dataset = AbAgDataset(train_df)
    val_dataset = AbAgDataset(val_df)
    test_dataset = AbAgDataset(test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Create model
    print("Creating model...")
    model = IgT5ESM2Model(dropout=args.dropout, freeze_encoders=True).to(device)

    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}\n")

    # Loss, optimizer, scheduler
    criterion = FocalMSELoss(gamma=args.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')

    # Resume from checkpoint if provided
    start_epoch = 0
    best_spearman = -1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.resume and Path(args.resume).exists():
        print(f"\n🔄 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_spearman = checkpoint.get('best_val_spearman', -1)
        print(f"✓ Resuming from epoch {start_epoch}")
        print(f"✓ Best Spearman: {best_spearman:.4f}\n")

    # Training loop
    print(f"{'='*70}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)

        # Validate
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val RMSE: {val_metrics['rmse']:.4f} | MAE: {val_metrics['mae']:.4f}")
        print(f"Val Spearman: {val_metrics['spearman']:.4f} | Pearson: {val_metrics['pearson']:.4f}")
        print(f"Val Recall@pKd≥9: {val_metrics['recall_pkd9']:.2f}% | Recall@pKd<7: {val_metrics['recall_pkd7']:.2f}%")

        # Save best model
        if val_metrics['spearman'] > best_spearman:
            best_spearman = val_metrics['spearman']
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            print("✓ Saved best model")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_spearman': best_spearman,
            'val_metrics': val_metrics
        }
        torch.save(checkpoint, output_dir / 'checkpoint_latest.pth')

    print(f"\n{'='*70}")
    print(f"Training complete! Best Spearman: {best_spearman:.4f}")
    print(f"{'='*70}")

    # Final test evaluation
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    print("\nFinal test evaluation:")
    model.load_state_dict(torch.load(output_dir / 'best_model.pth'))
    test_metrics = evaluate(model, test_loader, device)

    print(f"Test RMSE: {test_metrics['rmse']:.4f} | MAE: {test_metrics['mae']:.4f}")
    print(f"Test Spearman: {test_metrics['spearman']:.4f} | Pearson: {test_metrics['pearson']:.4f}")
    print(f"Test Recall@pKd≥9: {test_metrics['recall_pkd9']:.2f}% | Recall@pKd<7: {test_metrics['recall_pkd7']:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train IgT5 + ESM-2 Hybrid Model')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--output_dir', type=str, default='outputs_igt5_esm2', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')

    args = parser.parse_args()
    main(args)
