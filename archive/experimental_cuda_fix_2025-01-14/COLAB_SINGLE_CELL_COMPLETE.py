# ============================================================================
# ULTRA SPEED v2.6 - SINGLE CELL VERSION FOR COLAB
# ============================================================================
# Copy and paste this ENTIRE cell into Colab and run it!
# No file uploads needed - everything is self-contained
# ============================================================================

# STEP 1: Install dependencies
print("Installing dependencies...")
import subprocess
import sys

packages = ['transformers', 'pandas', 'scipy', 'scikit-learn', 'tqdm',
            'sentencepiece', 'faesm', 'bitsandbytes', 'accelerate']
for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])
print("✓ Dependencies installed\n")

# STEP 2: Import everything
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.checkpoint import checkpoint
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
import time
import shutil
import gc
import random
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer, BitsAndBytesConfig
import threading

try:
    from faesm.esm import FAEsmForMaskedLM
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    from transformers import AutoModel
    FLASH_ATTN_AVAILABLE = False

# STEP 3: Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

print("="*70)
print("ULTRA SPEED v2.6 - SINGLE CELL VERSION")
print("="*70)
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"FlashAttention: {FLASH_ATTN_AVAILABLE}")
print("="*70 + "\n")

# STEP 4: Configuration
DATA_PATH = 'agab_phase2_full.csv'
OUTPUT_DIR = 'outputs_max_speed'
EPOCHS = 50
BATCH_SIZE = 16
ACCUMULATION_STEPS = 3
LEARNING_RATE = 4e-3
WEIGHT_DECAY = 0.01
DROPOUT = 0.3
FOCAL_GAMMA = 2.0
SAVE_EVERY_N_BATCHES = 500
NUM_WORKERS = 4
PREFETCH_FACTOR = 4
VALIDATION_FREQUENCY = 2
USE_BFLOAT16 = True
USE_COMPILE = True
USE_FUSED_OPTIMIZER = True
USE_QUANTIZATION = True
USE_CHECKPOINTING = True
USE_BUCKETING = True

# STEP 5: Define all functions and classes
def ultra_aggressive_cleanup():
    """Ultra aggressive cleanup"""
    print("\n🚨 ULTRA AGGRESSIVE CLEANUP")
    print("="*60)
    freed_space = 0

    try:
        subprocess.run(['pip', 'cache', 'purge'], capture_output=True)
        print("  ✓ Pip cache cleared")
    except:
        pass

    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        print("  ✓ CUDA cache cleared")
    except:
        pass

    try:
        cache_dir = Path.home() / '.cache' / 'huggingface'
        if cache_dir.exists():
            keep_models = ['Exscientia--IgT5', 'facebook--esm2_t33_650M_UR50D']
            hub_dir = cache_dir / 'hub'
            if hub_dir.exists():
                for item in hub_dir.iterdir():
                    if item.is_dir():
                        should_keep = any(model in item.name for model in keep_models)
                        if not should_keep:
                            try:
                                size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                                shutil.rmtree(item)
                                freed_space += size
                                print(f"  ✓ Removed: {item.name[:35]}... ({size/1e9:.2f}GB)")
                            except:
                                pass

            downloads_dir = cache_dir / 'downloads'
            if downloads_dir.exists():
                try:
                    size = sum(f.stat().st_size for f in downloads_dir.rglob('*') if f.is_file())
                    shutil.rmtree(downloads_dir)
                    downloads_dir.mkdir()
                    freed_space += size
                    print(f"  ✓ Cleared downloads ({size/1e9:.2f}GB)")
                except:
                    pass
    except Exception as e:
        print(f"  ⚠ HuggingFace cleanup: {e}")

    for _ in range(3):
        gc.collect()

    print(f"📊 Freed: ~{freed_space/1e9:.2f}GB")
    print("="*60 + "\n")


def monitor_disk_usage(threshold_gb=150):
    """Monitor disk and trigger cleanup if needed"""
    try:
        result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if '/' in line and 'Filesystem' not in line:
                parts = line.split()
                if len(parts) >= 5:
                    used_str = parts[2]
                    total_str = parts[1]
                    percent_str = parts[4]

                    used_gb = float(used_str.replace('G', '').replace('M', '0.'))
                    total_gb = float(total_str.replace('G', '').replace('M', '0.'))
                    percent = int(percent_str.replace('%', ''))

                    print(f"  📊 Disk: {used_gb:.1f}GB/{total_gb:.1f}GB ({percent}%)")

                    if used_gb > threshold_gb:
                        print(f"\n⚠️  CRITICAL: Disk {used_gb:.1f}GB > {threshold_gb}GB!")
                        ultra_aggressive_cleanup()
                        return True

                    return False
    except:
        pass
    return False


def cleanup_disk_space():
    """Standard cleanup at start of each epoch"""
    print("\n🧹 Disk cleanup...")
    try:
        subprocess.run(['pip', 'cache', 'purge'], capture_output=True)
        torch.cuda.empty_cache()
        gc.collect()
        print("  ✓ Standard cleanup done")
    except:
        pass
    monitor_disk_usage(threshold_gb=180)


class BucketBatchSampler(Sampler):
    """Group sequences by similar lengths"""

    def __init__(self, dataset, batch_size, drop_last=True, buckets=[256, 384, 512]):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.buckets = sorted(buckets)
        self.bucket_indices = {b: [] for b in self.buckets}

        for idx in range(len(dataset)):
            item = dataset[idx]
            seq_len = len(item['antibody_sequence'])
            bucket = min([b for b in self.buckets if b >= seq_len], default=self.buckets[-1])
            self.bucket_indices[bucket].append(idx)

        print(f"\n📊 Bucket Distribution:")
        for bucket in self.buckets:
            count = len(self.bucket_indices[bucket])
            print(f"  ≤{bucket}: {count:,} samples ({count/len(dataset)*100:.1f}%)")

    def __iter__(self):
        bucket_order = list(self.buckets)
        random.shuffle(bucket_order)

        for bucket in bucket_order:
            indices = self.bucket_indices[bucket].copy()
            random.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self):
        count = 0
        for bucket in self.buckets:
            n = len(self.bucket_indices[bucket])
            count += n // self.batch_size
            if not self.drop_last and n % self.batch_size > 0:
                count += 1
        return count


class IgT5ESM2ModelUltraSpeed(nn.Module):
    """Ultra-optimized model with batch processing"""

    def __init__(self, dropout=0.3, freeze_encoders=True, use_quantization=True, use_checkpointing=True):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        print("Loading models with optimizations...")

        if use_quantization:
            print("  → Using INT8 quantization for encoders")
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False
                )
                use_quant = True
            except:
                print("  ⚠ Quantization not available, using BFloat16")
                quantization_config = None
                use_quant = False
        else:
            quantization_config = None
            use_quant = False

        print("  Loading IgT5 for antibody...")
        self.igt5_tokenizer = T5Tokenizer.from_pretrained("Exscientia/IgT5", do_lower_case=False, use_fast=True)

        if use_quant and quantization_config:
            self.igt5_model = T5EncoderModel.from_pretrained("Exscientia/IgT5", quantization_config=quantization_config, device_map="auto")
        else:
            self.igt5_model = T5EncoderModel.from_pretrained("Exscientia/IgT5")

        print("  Loading ESM-2 for antigen...")
        self.esm2_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", use_fast=True)

        if FLASH_ATTN_AVAILABLE:
            print("  → Using FAESM with FlashAttention")
            self.esm2_model = FAEsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
        else:
            print("  → Using standard ESM-2 with PyTorch SDPA")
            if use_quant and quantization_config:
                self.esm2_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D", quantization_config=quantization_config, device_map="auto")
            else:
                self.esm2_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

        if freeze_encoders:
            for param in self.igt5_model.parameters():
                param.requires_grad = False
            for param in self.esm2_model.parameters():
                param.requires_grad = False

        igt5_dim = self.igt5_model.config.d_model
        esm2_dim = self.esm2_model.config.hidden_size
        combined_dim = igt5_dim + esm2_dim

        self.regressor_block1 = nn.Sequential(nn.Linear(combined_dim, 1024), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(1024))
        self.regressor_block2 = nn.Sequential(nn.Linear(1024, 512), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(512))
        self.regressor_block3 = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(256))
        self.regressor_block4 = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout))
        self.regressor_final = nn.Linear(128, 1)

    def get_batch_embeddings(self, sequences, model, tokenizer, device, pooling='mean'):
        """Batch process all sequences at once"""
        inputs = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device, non_blocking=True)

        with torch.no_grad():
            outputs = model(**inputs)
            if pooling == 'mean':
                embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings

    def forward(self, antibody_seqs, antigen_seqs, device):
        """Batch-parallel embedding generation"""
        ab_embeddings = self.get_batch_embeddings(antibody_seqs, self.igt5_model, self.igt5_tokenizer, device, pooling='mean')
        ag_embeddings = self.get_batch_embeddings(antigen_seqs, self.esm2_model, self.esm2_tokenizer, device, pooling='cls')
        combined = torch.cat([ab_embeddings, ag_embeddings], dim=1)

        if self.use_checkpointing and self.training:
            x = checkpoint(self.regressor_block1, combined, use_reentrant=False)
            x = checkpoint(self.regressor_block2, x, use_reentrant=False)
            x = checkpoint(self.regressor_block3, x, use_reentrant=False)
            x = checkpoint(self.regressor_block4, x, use_reentrant=False)
        else:
            x = self.regressor_block1(combined)
            x = self.regressor_block2(x)
            x = self.regressor_block3(x)
            x = self.regressor_block4(x)

        predictions = self.regressor_final(x).squeeze(-1)
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
    """Collate function"""
    antibody_seqs = [item['antibody_sequence'] for item in batch]
    antigen_seqs = [item['antigen_sequence'] for item in batch]
    pKds = torch.stack([item['pKd'] for item in batch])
    return {'antibody_seqs': antibody_seqs, 'antigen_seqs': antigen_seqs, 'pKd': pKds}


def async_save_checkpoint(checkpoint, path):
    """Save checkpoint in background"""
    def _save():
        torch.save(checkpoint, path)
    thread = threading.Thread(target=_save, daemon=True)
    thread.start()
    return thread


def save_checkpoint_smart(model, optimizer, scheduler, epoch, batch_idx, best_spearman, output_dir, save_type='latest'):
    """Smart checkpoint saving"""
    if hasattr(model, '_orig_mod'):
        model_to_save = model._orig_mod
    elif hasattr(model, 'module'):
        model_to_save = model.module
    else:
        model_to_save = model

    output_dir = Path(output_dir)

    if save_type == 'best':
        checkpoint = {'model_state_dict': model_to_save.state_dict(), 'best_val_spearman': best_spearman, 'epoch': epoch, 'batch_idx': batch_idx}
        checkpoint_path = output_dir / 'best_model.pth'
        torch.save(checkpoint, checkpoint_path)

    elif save_type == 'latest':
        checkpoint = {'epoch': epoch, 'batch_idx': batch_idx, 'model_state_dict': model_to_save.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(), 'best_val_spearman': best_spearman, 'timestamp': time.time()}

        temp_path = output_dir / 'checkpoint_temp.pth'
        latest_path = output_dir / 'checkpoint_latest.pth'
        backup_path = output_dir / 'checkpoint_backup.pth'

        if latest_path.exists():
            if backup_path.exists():
                backup_path.unlink()
            latest_path.rename(backup_path)

        async_save_checkpoint(checkpoint, temp_path)
        time.sleep(0.1)
        if temp_path.exists():
            temp_path.rename(latest_path)
        checkpoint_path = latest_path

    elif save_type == 'epoch':
        checkpoint = {'epoch': epoch, 'batch_idx': batch_idx, 'model_state_dict': model_to_save.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(), 'best_val_spearman': best_spearman, 'timestamp': time.time()}
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        checkpoint_path = output_dir / 'checkpoint_epoch.pth'
        torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def cleanup_old_checkpoints(output_dir):
    """Remove old checkpoints"""
    keep_files = ['best_model.pth', 'checkpoint_latest.pth', 'checkpoint_backup.pth', 'checkpoint_epoch.pth']
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return

    for file in output_dir.glob('*.pth'):
        if file.name not in keep_files:
            file.unlink()


def quick_eval(model, loader, device, max_batches=50, use_bfloat16=True):
    """Quick validation"""
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
            batch_targets = batch['pKd'].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=dtype):
                batch_predictions = model(antibody_seqs, antigen_seqs, device)

            predictions.extend(batch_predictions.float().cpu().numpy())
            targets.extend(batch_targets.float().cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    spearman = stats.spearmanr(targets, predictions)[0]
    strong_binders = targets >= 9.0
    predicted_strong = predictions >= 9.0
    recall = (strong_binders & predicted_strong).sum() / strong_binders.sum() if strong_binders.sum() > 0 else 0

    return {'spearman': spearman, 'recall_pkd9': recall * 100}


def train_epoch(model, loader, optimizer, criterion, device, epoch, start_batch, output_dir, accumulation_steps=4, save_every_n_batches=500, use_bfloat16=True):
    """Training with all optimizations"""
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
        targets = batch['pKd'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=dtype):
            predictions = model(antibody_seqs, antigen_seqs, device)
            loss = criterion(predictions, targets)
            loss = loss / accumulation_steps

        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.2e}', 'batch': f'{batch_idx+1}/{len(loader)}'})

        if (batch_idx + 1) % save_every_n_batches == 0:
            checkpoint_path = save_checkpoint_smart(model, optimizer, None, epoch, batch_idx, best_spearman, output_dir, save_type='latest')
            print(f"\n✓ Saved checkpoint: {checkpoint_path.name}")
            cleanup_old_checkpoints(output_dir)

        if (batch_idx + 1) % 250 == 0:
            monitor_disk_usage(threshold_gb=150)

    return total_loss / len(loader)


# STEP 6: Main training function
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*70)
    print("ULTRA SPEED TRAINING v2.6 - ALL ADVANCED OPTIMIZATIONS")
    print("="*70)
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    cleanup_disk_space()

    print(f"\nOptimizations Active:")
    print(f"  12. ⭐ Batch embedding generation (2-3× faster)")
    print(f"  13. ⭐ Sequence bucketing (1.3-1.5× faster)")
    print(f"  14. ⭐ Activation checkpointing: {USE_CHECKPOINTING}")
    print(f"  15. ⭐ INT8 quantization: {USE_QUANTIZATION}")
    print("="*70 + "\n")

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} samples\n")

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    val_df_quick = val_df.sample(frac=0.05, random_state=42)

    print(f"Train: {len(train_df):,} | Val (quick): {len(val_df_quick):,}\n")

    train_dataset = AbAgDataset(train_df)
    val_dataset = AbAgDataset(val_df_quick)

    if USE_BUCKETING:
        print("Creating bucket batch sampler...")
        train_sampler = BucketBatchSampler(train_dataset, batch_size=BATCH_SIZE, drop_last=True, buckets=[256, 384, 512])
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=NUM_WORKERS,
                                 prefetch_factor=PREFETCH_FACTOR, pin_memory=True, persistent_workers=True, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                                 prefetch_factor=PREFETCH_FACTOR, pin_memory=True, persistent_workers=True, drop_last=True, collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS,
                           prefetch_factor=PREFETCH_FACTOR, pin_memory=True, persistent_workers=True, collate_fn=collate_fn)

    print("Initializing ultra-optimized model...")
    model = IgT5ESM2ModelUltraSpeed(dropout=DROPOUT, freeze_encoders=True, use_quantization=USE_QUANTIZATION, use_checkpointing=USE_CHECKPOINTING).to(device)

    if USE_COMPILE:
        print("\nCompiling regressor blocks...")
        model.regressor_block1 = torch.compile(model.regressor_block1, mode='max-autotune')
        model.regressor_block2 = torch.compile(model.regressor_block2, mode='max-autotune')
        model.regressor_block3 = torch.compile(model.regressor_block3, mode='max-autotune')
        model.regressor_block4 = torch.compile(model.regressor_block4, mode='max-autotune')
        model.regressor_final = torch.compile(model.regressor_final, mode='max-autotune')
        print("✓ Regressor compiled\n")

    criterion = FocalMSELoss(gamma=FOCAL_GAMMA)

    if USE_FUSED_OPTIMIZER:
        try:
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=True)
            print("✓ Using fused optimizer\n")
        except:
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            print("⚠ Fused optimizer not available\n")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    start_epoch = 0
    start_batch = 0
    best_spearman = -1
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    cleanup_old_checkpoints(output_dir)

    # Try to load checkpoint
    latest_checkpoint = output_dir / 'checkpoint_latest.pth'
    if latest_checkpoint.exists():
        print(f"Found checkpoint: {latest_checkpoint}")
        print("Attempting to load checkpoint...")

        try:
            checkpoint = torch.load(latest_checkpoint, map_location=device)

            try:
                state_dict = checkpoint['model_state_dict']
                regressor_state = {k: v for k, v in state_dict.items() if k.startswith('regressor') or k.startswith('proj_')}
                model.load_state_dict(regressor_state, strict=False)
                print(f"✓ Loaded regressor weights from checkpoint")
                print(f"  Loaded {len(regressor_state)} parameters")
            except Exception as e:
                print(f"⚠ Could not load model state: {e}")

            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✓ Loaded optimizer state")
            except Exception as e:
                print(f"⚠ Could not load optimizer state: {e}")

            start_epoch = checkpoint['epoch']
            start_batch = checkpoint.get('batch_idx', 0) + 1
            best_spearman = checkpoint.get('best_val_spearman', -1)

            print(f"Resuming from Epoch {start_epoch+1}, Batch {start_batch}, Spearman: {best_spearman:.4f}\n")
        except Exception as e:
            print(f"⚠ Could not load checkpoint: {e}")
            print("Starting fresh training\n")

    print(f"Starting training for {EPOCHS} epochs...\n")

    for epoch in range(start_epoch, EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*70}")

        if epoch > 0:
            cleanup_disk_space()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, start_batch if epoch == start_epoch else 0,
                                output_dir, ACCUMULATION_STEPS, SAVE_EVERY_N_BATCHES, USE_BFLOAT16)

        if (epoch + 1) % VALIDATION_FREQUENCY == 0:
            print("\nQuick validation...")
            val_metrics = quick_eval(model, val_loader, device, max_batches=50, use_bfloat16=USE_BFLOAT16)
            print(f"Val Spearman: {val_metrics['spearman']:.4f} | Recall@pKd≥9: {val_metrics['recall_pkd9']:.2f}%")

            if val_metrics['spearman'] > best_spearman:
                best_spearman = val_metrics['spearman']
                save_checkpoint_smart(model, optimizer, scheduler, epoch, len(train_loader)-1, best_spearman, output_dir, save_type='best')
                print("✓ Saved best model")

        scheduler.step()
        print(f"\nTrain Loss: {train_loss:.4f}")

        save_checkpoint_smart(model, optimizer, scheduler, epoch, len(train_loader)-1, best_spearman, output_dir, save_type='epoch')
        cleanup_old_checkpoints(output_dir)

        start_batch = 0

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE!")
    print(f"Best Spearman: {best_spearman:.4f}")
    print(f"{'='*70}")


# STEP 7: Run training!
if __name__ == '__main__':
    main()
