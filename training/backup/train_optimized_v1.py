"""
Optimized Training Script - Phase 1 (Quick Wins)
================================================

This script implements immediate speed and accuracy improvements:
- FlashAttention for 3-10x faster inference
- Mixed precision (bfloat16) for 1.5-2x speedup
- Focal MSE Loss for better extreme value prediction
- Optimized batch processing

Expected: 4-6 hours training (vs 15-20 hours)
Expected: 30-40% recall on strong binders (vs 17%)

Usage:
    python train_optimized_v1.py --data /path/to/data.csv --epochs 50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import json
import time
from pathlib import Path


# ============================================================================
# OPTIMIZED FOCAL MSE LOSS
# ============================================================================

class FocalMSELoss(nn.Module):
    """
    Focal MSE Loss - puts more weight on hard examples (extreme values)
    gamma: focusing parameter (higher = more focus on hard examples)
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        mse = (pred - target) ** 2
        # Weight hard examples more
        focal_weight = (1 + mse) ** self.gamma
        return (focal_weight * mse).mean()


# ============================================================================
# DATASET WITH STRATIFIED SAMPLING SUPPORT
# ============================================================================

class AbAgDataset(Dataset):
    """Dataset for antibody-antigen pairs with affinity labels"""

    def __init__(self, df, tokenizer, max_length=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Create affinity bins for stratified sampling
        self.df['affinity_bin'] = pd.cut(
            self.df['pKd'],
            bins=[-np.inf, 6, 7, 8, 9, np.inf],
            labels=['very_weak', 'weak', 'moderate', 'strong', 'very_strong']
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        return {
            'antibody_seq': row['antibody_sequence'],
            'antigen_seq': row['antigen_sequence'],
            'pKd': torch.tensor(row['pKd'], dtype=torch.float32),
            'affinity_bin': row['affinity_bin']
        }

    def get_sample_weights(self):
        """Calculate weights for stratified sampling (oversample rare strong binders)"""
        bin_counts = self.df['affinity_bin'].value_counts()
        # Inverse frequency weighting
        weights = self.df['affinity_bin'].map(lambda x: 1.0 / bin_counts[x])
        return weights.values


def collate_fn(batch, tokenizer, max_length=512):
    """
    Custom collate function with optimized batching
    Processes antibody and antigen sequences separately
    """
    antibody_seqs = [item['antibody_seq'] for item in batch]
    antigen_seqs = [item['antigen_seq'] for item in batch]
    pKds = torch.stack([item['pKd'] for item in batch])

    # Tokenize with padding to max length in batch (not global max)
    ab_tokens = tokenizer(
        antibody_seqs,
        return_tensors='pt',
        padding=True,  # Pad to longest in batch
        truncation=True,
        max_length=max_length
    )

    ag_tokens = tokenizer(
        antigen_seqs,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length
    )

    return {
        'ab_tokens': ab_tokens,
        'ag_tokens': ag_tokens,
        'pKd': pKds
    }


# ============================================================================
# OPTIMIZED MODEL ARCHITECTURE
# ============================================================================

class OptimizedAbAgModel(nn.Module):
    """
    Optimized model with:
    - FlashAttention support
    - Mixed precision (bfloat16)
    - Better architecture (LayerNorm, GELU)
    """

    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", dropout=0.2):
        super().__init__()

        # Load ESM-2 with FlashAttention
        print("Loading ESM-2 model with optimizations...")
        try:
            self.esm = AutoModel.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2"  # 3-10x faster!
            )
            print("✓ FlashAttention enabled")
        except Exception as e:
            print(f"⚠ FlashAttention not available: {e}")
            print("  Falling back to standard attention")
            self.esm = AutoModel.from_pretrained(model_name)

        # Freeze ESM-2 (we're using frozen embeddings for Phase 1)
        for param in self.esm.parameters():
            param.requires_grad = False

        # Improved regression head with LayerNorm and GELU
        hidden_size = self.esm.config.hidden_size  # 1280 for ESM2_650M
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 1)
        )

    def forward(self, ab_tokens, ag_tokens):
        """Forward pass with mixed precision"""

        # Use autocast for mixed precision (use float16 for compatibility)
        with torch.cuda.amp.autocast():
            # Process antibody
            ab_out = self.esm(**ab_tokens)
            ab_emb = ab_out.last_hidden_state.mean(dim=1)  # Mean pooling

            # Process antigen
            ag_out = self.esm(**ag_tokens)
            ag_emb = ag_out.last_hidden_state.mean(dim=1)

        # Concatenate embeddings
        combined = torch.cat([ab_emb, ag_emb], dim=1)

        # Regression head
        pred = self.regressor(combined)
        return pred.squeeze(-1)


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """Train for one epoch with mixed precision"""
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc='Training')

    for batch in pbar:
        # Move to device
        ab_tokens = {k: v.to(device) for k, v in batch['ab_tokens'].items()}
        ag_tokens = {k: v.to(device) for k, v in batch['ag_tokens'].items()}
        targets = batch['pKd'].to(device)

        optimizer.zero_grad()

        # Mixed precision training (use float16 for compatibility)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds = model(ab_tokens, ag_tokens)
                loss = criterion(preds, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(ab_tokens, ag_tokens)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_targets = []

    for batch in tqdm(loader, desc='Evaluating'):
        ab_tokens = {k: v.to(device) for k, v in batch['ab_tokens'].items()}
        ag_tokens = {k: v.to(device) for k, v in batch['ag_tokens'].items()}
        targets = batch['pKd'].to(device)

        with torch.cuda.amp.autocast():
            preds = model(ab_tokens, ag_tokens)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate metrics
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds - all_targets))
    r2 = 1 - (np.sum((all_targets - all_preds) ** 2) /
              np.sum((all_targets - all_targets.mean()) ** 2))
    spearman = stats.spearmanr(all_targets, all_preds).correlation
    pearson = stats.pearsonr(all_targets, all_preds)[0]

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'spearman': spearman,
        'pearson': pearson,
        'predictions': all_preds,
        'targets': all_targets
    }


def calculate_recall_by_range(targets, preds, threshold=9.0):
    """Calculate recall for strong binders (pKd >= threshold)"""
    strong_mask = targets >= threshold
    if strong_mask.sum() == 0:
        return 0.0

    # Predict as strong if pred >= threshold - 0.5 (to account for underprediction)
    pred_strong_mask = preds >= (threshold - 0.5)

    # True positives
    tp = (strong_mask & pred_strong_mask).sum()
    recall = tp / strong_mask.sum()

    return recall


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main(args):
    print("="*60)
    print("OPTIMIZED TRAINING - PHASE 1")
    print("="*60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load data
    print(f"\nLoading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"Total samples: {len(df):,}")

    # Check required columns
    required_cols = ['antibody_sequence', 'antigen_sequence', 'pKd']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        # Try alternative column names
        col_mapping = {
            'ab_sequence': 'antibody_sequence',
            'ag_sequence': 'antigen_sequence',
            'affinity': 'pKd',
            'pkd': 'pKd'
        }
        for old, new in col_mapping.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})

        # Check again
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    # Data distribution
    print(f"\npKd distribution:")
    print(df['pKd'].describe())

    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df):,}")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # Create datasets
    train_dataset = AbAgDataset(train_df, tokenizer, max_length=args.max_length)
    val_dataset = AbAgDataset(val_df, tokenizer, max_length=args.max_length)
    test_dataset = AbAgDataset(test_df, tokenizer, max_length=args.max_length)

    # Create stratified sampler for training
    if args.use_stratified_sampling:
        print("\nUsing stratified sampling (oversampling strong binders)...")
        weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        collate_fn=lambda x: collate_fn(x, tokenizer, args.max_length),
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer, args.max_length),
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer, args.max_length),
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create model
    print("\nCreating model...")
    model = OptimizedAbAgModel(dropout=args.dropout)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function
    criterion = FocalMSELoss(gamma=args.focal_gamma)
    print(f"\nUsing Focal MSE Loss (gamma={args.focal_gamma})")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr / 100
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Check for checkpoint to resume from
    start_epoch = 0
    best_val_spearman = -1

    if args.resume:
        checkpoint_path = Path(args.resume)
    else:
        # Auto-detect checkpoint_latest.pth in output directory
        checkpoint_path = args.output_dir / 'checkpoint_latest.pth'

    if checkpoint_path.exists():
        print(f"\n{'='*60}")
        print(f"Found checkpoint: {checkpoint_path}")
        print("Loading checkpoint to resume training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_spearman = checkpoint.get('best_val_spearman', -1)

        print(f"✓ Resumed from epoch {checkpoint['epoch'] + 1}")
        print(f"✓ Best validation Spearman so far: {best_val_spearman:.4f}")
        print("="*60)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    if start_epoch > 0:
        print(f"Resuming from epoch {start_epoch + 1}/{args.epochs}")
    print("="*60)

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        # Calculate recall on strong binders
        val_recall = calculate_recall_by_range(
            val_metrics['targets'],
            val_metrics['predictions'],
            threshold=9.0
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val RMSE: {val_metrics['rmse']:.4f} | MAE: {val_metrics['mae']:.4f}")
        print(f"Val Spearman: {val_metrics['spearman']:.4f} | Pearson: {val_metrics['pearson']:.4f}")
        print(f"Val R²: {val_metrics['r2']:.4f}")
        print(f"Val Recall@pKd≥9: {val_recall:.2%}")

        # Save best model
        if val_metrics['spearman'] > best_val_spearman:
            best_val_spearman = val_metrics['spearman']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'best_val_spearman': best_val_spearman,
                'args': vars(args)
            }, args.output_dir / 'best_model.pth')
            print("✓ Saved best model")

        # Save periodic checkpoint (every epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_metrics': val_metrics,
            'best_val_spearman': best_val_spearman,
            'args': vars(args)
        }, args.output_dir / 'checkpoint_latest.pth')
        print(f"✓ Saved checkpoint (epoch {epoch+1})")

        # Step scheduler
        scheduler.step()

    training_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed in {training_time/3600:.2f} hours")

    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(args.output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final test evaluation
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, device)
    test_recall = calculate_recall_by_range(
        test_metrics['targets'],
        test_metrics['predictions'],
        threshold=9.0
    )

    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"R²: {test_metrics['r2']:.4f}")
    print(f"Spearman ρ: {test_metrics['spearman']:.4f}")
    print(f"Pearson r: {test_metrics['pearson']:.4f}")
    print(f"Recall@pKd≥9: {test_recall:.2%}")

    # Save results
    results = {
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'test_r2': test_metrics['r2'],
        'test_spearman': test_metrics['spearman'],
        'test_pearson': test_metrics['pearson'],
        'test_recall_strong': float(test_recall),
        'training_time_hours': training_time / 3600,
        'total_samples': len(df),
        'test_samples': len(test_df)
    }

    with open(args.output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save predictions
    pred_df = pd.DataFrame({
        'true_pKd': test_metrics['targets'],
        'predicted_pKd': test_metrics['predictions'],
        'residual': test_metrics['targets'] - test_metrics['predictions']
    })
    pred_df.to_csv(args.output_dir / 'test_predictions.csv', index=False)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimized AbAg Training - Phase 1')

    # Data
    parser.add_argument('--data', type=str, required=True,
                        help='Path to CSV file with antibody_sequence, antigen_sequence, pKd')
    parser.add_argument('--output_dir', type=str, default='./outputs_optimized_v1',
                        help='Output directory')

    # Model
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (try 16, 32, 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')

    # Loss
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter')

    # Sampling
    parser.add_argument('--use_stratified_sampling', action='store_true',
                        help='Use stratified sampling to oversample strong binders')

    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume from (default: auto-detect checkpoint_latest.pth)')

    args = parser.parse_args()

    # Create output directory
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    main(args)
