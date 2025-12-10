"""
Ultra-Fast Training - LOW STORAGE VERSION
Optimized for Google Drive with <10 GB available
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
from pathlib import Path
import time
import os
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer

try:
    from faesm.esm import FAEsmForMaskedLM
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    from transformers import AutoModel
    FLASH_ATTN_AVAILABLE = False


class IgT5ESM2ModelFast(nn.Module):
    """Optimized model with FlashAttention and torch.compile support"""

    def __init__(self, dropout=0.3, freeze_encoders=True):
        super().__init__()

        print("Loading IgT5 for antibody...")
        self.igt5_tokenizer = T5Tokenizer.from_pretrained("Exscientia/IgT5", do_lower_case=False)
        self.igt5_model = T5EncoderModel.from_pretrained("Exscientia/IgT5")

        print("Loading ESM-2 for antigen...")
        self.esm2_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

        if FLASH_ATTN_AVAILABLE:
            self.esm2_model = FAEsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
        else:
            from transformers import AutoModel
            self.esm2_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

        if freeze_encoders:
            for param in self.igt5_model.parameters():
                param.requires_grad = False
            for param in self.esm2_model.parameters():
                param.requires_grad = False

        igt5_dim = self.igt5_model.config.d_model
        esm2_dim = self.esm2_model.config.hidden_size
        combined_dim = igt5_dim + esm2_dim

        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def get_antibody_embedding(self, antibody_seq, device):
        inputs = self.igt5_tokenizer(
            antibody_seq, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = self.igt5_model(**inputs)
            ab_emb = outputs.last_hidden_state.mean(dim=1)
        return ab_emb.squeeze(0)

    def get_antigen_embedding(self, antigen_seq, device):
        inputs = self.esm2_tokenizer(
            antigen_seq, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = self.esm2_model(**inputs)
            ag_emb = outputs.last_hidden_state[:, 0, :]
        return ag_emb.squeeze(0)

    def forward(self, antibody_seqs, antigen_seqs, device):
        ab_embeddings = []
        for ab_seq in antibody_seqs:
            ab_emb = self.get_antibody_embedding(ab_seq, device)
            ab_embeddings.append(ab_emb)
        ab_embeddings = torch.stack(ab_embeddings).to(device)

        ag_embeddings = []
        for ag_seq in antigen_seqs:
            ag_emb = self.get_antigen_embedding(ag_seq, device)
            ag_embeddings.append(ag_emb)
        ag_embeddings = torch.stack(ag_embeddings).to(device)

        combined = torch.cat([ab_embeddings, ag_embeddings], dim=1)
        predictions = self.regressor(combined).squeeze(-1)
        return predictions


class FocalMSELoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        mse = (pred - target) ** 2
        focal_weight = (1 + mse) ** self.gamma
        return (focal_weight * mse).mean()


class AbAgDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            'antibody_sequence': self.df.iloc[idx]['antibody_sequence'],
            'antigen_sequence': self.df.iloc[idx]['antigen_sequence'],
            'pKd': torch.tensor(self.df.iloc[idx]['pKd'], dtype=torch.float32)
        }


def collate_fn(batch):
    antibody_seqs = [item['antibody_sequence'] for item in batch]
    antigen_seqs = [item['antigen_sequence'] for item in batch]
    pKds = torch.stack([item['pKd'] for item in batch])
    return {'antibody_seqs': antibody_seqs, 'antigen_seqs': antigen_seqs, 'pKd': pKds}


def save_checkpoint_smart(model, optimizer, scheduler, epoch, batch_idx,
                          best_spearman, output_dir, save_type='latest'):
    """
    Smart checkpoint saving for limited storage

    save_type:
    - 'latest': Overwrite latest checkpoint (for frequent saves)
    - 'best': Save as best model (smaller, model weights only)
    - 'epoch': Save end-of-epoch (with scheduler)
    """
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    output_dir = Path(output_dir)

    if save_type == 'best':
        # Only save model weights (smallest, ~2.0 GB)
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'best_val_spearman': best_spearman,
            'epoch': epoch,
            'batch_idx': batch_idx
        }
        checkpoint_path = output_dir / 'best_model.pth'

    elif save_type == 'latest':
        # Rotating checkpoint: save as temp, then replace latest
        checkpoint = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_spearman': best_spearman,
            'timestamp': time.time()
        }

        # Save to temp file first
        temp_path = output_dir / 'checkpoint_temp.pth'
        latest_path = output_dir / 'checkpoint_latest.pth'
        backup_path = output_dir / 'checkpoint_backup.pth'

        # Rotate: latest → backup, temp → latest
        if latest_path.exists():
            if backup_path.exists():
                backup_path.unlink()  # Delete old backup
            latest_path.rename(backup_path)  # latest → backup

        torch.save(checkpoint, temp_path)
        temp_path.rename(latest_path)  # temp → latest
        checkpoint_path = latest_path

    elif save_type == 'epoch':
        # Full checkpoint with scheduler (end of epoch)
        checkpoint = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_spearman': best_spearman,
            'timestamp': time.time()
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Overwrite epoch checkpoint
        checkpoint_path = output_dir / 'checkpoint_epoch.pth'
        torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def cleanup_old_checkpoints(output_dir, keep_files=['best_model.pth', 'checkpoint_latest.pth',
                                                     'checkpoint_backup.pth', 'checkpoint_epoch.pth']):
    """Remove all checkpoint files except the ones we want to keep"""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return

    for file in output_dir.glob('*.pth'):
        if file.name not in keep_files:
            file.unlink()
            print(f"  Cleaned up: {file.name}")


def quick_eval(model, loader, device, max_batches=50, use_bfloat16=True):
    """Quick evaluation"""
    model.eval()
    predictions = []
    targets = []

    dtype = torch.bfloat16 if use_bfloat16 else torch.float16

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break

            antibody_seqs = batch['antibody_seqs']
            antigen_seqs = batch['antigen_seqs']
            batch_targets = batch['pKd'].to(device)

            with torch.amp.autocast('cuda', dtype=dtype):
                batch_predictions = model(antibody_seqs, antigen_seqs, device)

            predictions.extend(batch_predictions.cpu().numpy())
            targets.extend(batch_targets.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    spearman = stats.spearmanr(targets, predictions)[0]
    strong_binders = targets >= 9.0
    predicted_strong = predictions >= 9.0
    recall = (strong_binders & predicted_strong).sum() / strong_binders.sum() if strong_binders.sum() > 0 else 0

    return {'spearman': spearman, 'recall_pkd9': recall * 100}


def train_epoch(model, loader, optimizer, criterion, device,
               epoch, start_batch, output_dir, save_every_n_batches=500, use_bfloat16=True):
    """Training with infrequent checkpointing to save storage"""
    model.train()
    total_loss = 0
    best_spearman = -1

    dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}")

    for batch_idx, batch in pbar:
        if batch_idx < start_batch:
            continue

        antibody_seqs = batch['antibody_seqs']
        antigen_seqs = batch['antigen_seqs']
        targets = batch['pKd'].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=dtype):
            predictions = model(antibody_seqs, antigen_seqs, device)
            loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.2e}', 'batch': f'{batch_idx+1}/{len(loader)}'})

        # Save checkpoint less frequently (every 500 batches = ~20 min)
        if (batch_idx + 1) % save_every_n_batches == 0:
            checkpoint_path = save_checkpoint_smart(
                model, optimizer, None, epoch, batch_idx,
                best_spearman, output_dir, save_type='latest'
            )
            print(f"\n✓ Saved checkpoint: {checkpoint_path.name}")

            # Clean up any old files
            cleanup_old_checkpoints(output_dir)

    return total_loss / len(loader)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*70)
    print("ULTRA-FAST TRAINING - LOW STORAGE MODE")
    print("="*70)
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"\nStorage-Saving Features:")
    print(f"  ✓ Rotating checkpoints (max 3 files)")
    print(f"  ✓ Save every {args.save_every_n_batches} batches (~20 min)")
    print(f"  ✓ Auto-cleanup old files")
    print(f"  ✓ Total storage: ~7.5 GB")
    print("="*70 + "\n")

    # Load data
    print("Loading data...")
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df):,} samples\n")

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    val_df_quick = val_df.sample(frac=0.1, random_state=42)

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Val (quick): {len(val_df_quick):,}\n")

    train_dataset = AbAgDataset(train_df)
    val_dataset = AbAgDataset(val_df_quick)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=2, collate_fn=collate_fn, pin_memory=True,
                             persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=2, collate_fn=collate_fn, pin_memory=True,
                           persistent_workers=True)

    # Initialize model
    print("Initializing model...")
    model = IgT5ESM2ModelFast(dropout=args.dropout, freeze_encoders=True).to(device)

    if args.use_compile:
        print("\nCompiling model with torch.compile...")
        model = torch.compile(model)
        print("✓ Model compiled\n")

    criterion = FocalMSELoss(gamma=args.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    start_batch = 0
    best_spearman = -1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Clean up old checkpoints first
    cleanup_old_checkpoints(output_dir)

    # Auto-resume
    latest_checkpoint = output_dir / 'checkpoint_latest.pth'
    if latest_checkpoint.exists():
        print(f"Found checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)

        model_to_load = model._orig_mod if hasattr(model, '_orig_mod') else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch_idx'] + 1
        best_spearman = checkpoint.get('best_val_spearman', -1)
        print(f"Resuming from Epoch {start_epoch+1}, Batch {start_batch}, Spearman: {best_spearman:.4f}\n")

    print(f"Starting training for {args.epochs} epochs...")
    print(f"Checkpoints every {args.save_every_n_batches} batches\n")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, start_batch if epoch == start_epoch else 0,
            output_dir, args.save_every_n_batches, args.use_bfloat16
        )

        print("\nQuick validation...")
        val_metrics = quick_eval(model, val_loader, device, max_batches=50, use_bfloat16=args.use_bfloat16)
        scheduler.step()

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Spearman: {val_metrics['spearman']:.4f} | Recall@pKd≥9: {val_metrics['recall_pkd9']:.2f}%")

        if val_metrics['spearman'] > best_spearman:
            best_spearman = val_metrics['spearman']
            save_checkpoint_smart(
                model, optimizer, scheduler, epoch, len(train_loader)-1,
                best_spearman, output_dir, save_type='best'
            )
            print("✓ Saved best model")

        # Save end-of-epoch checkpoint
        save_checkpoint_smart(
            model, optimizer, scheduler, epoch, len(train_loader)-1,
            best_spearman, output_dir, save_type='epoch'
        )

        # Clean up
        cleanup_old_checkpoints(output_dir)

        start_batch = 0

    print(f"\n{'='*70}")
    print(f"Training complete! Best Spearman: {best_spearman:.4f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs_low_storage')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--save_every_n_batches', type=int, default=500,
                       help='Save every N batches (default: 500 = ~20 min)')
    parser.add_argument('--use_bfloat16', type=bool, default=True)
    parser.add_argument('--use_compile', type=bool, default=True)
    args = parser.parse_args()
    main(args)
