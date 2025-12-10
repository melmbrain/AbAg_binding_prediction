"""
Evaluate Pre-trained Model v2.6 (100 Epochs)

This script evaluates your previously trained model (before optimizations)
on the test set with comprehensive metrics.

Usage:
    python evaluate_v26_model.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'model_path': 'best_model.pth',
    'data_path': 'agab_phase2_full.csv',
    'output_dir': 'evaluation_v26_output',
    'batch_size': 16,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
}

print("=" * 70)
print("EVALUATING MODEL v2.6 (100 EPOCHS - PRE-OPTIMIZATION)")
print("=" * 70)
print(f"\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")
print()

# Create output directory
output_dir = Path(CONFIG['output_dir'])
output_dir.mkdir(exist_ok=True)

device = torch.device(CONFIG['device'])
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

# ============================================================================
# Metrics Function
# ============================================================================

def compute_comprehensive_metrics(targets, predictions):
    """Compute all 12 standard metrics"""
    # Regression metrics
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    # Correlation metrics
    spearman, spearman_p = stats.spearmanr(targets, predictions)
    pearson, pearson_p = stats.pearsonr(targets, predictions)

    # Classification metrics for strong binders (pKd >= 9)
    strong_binders = targets >= 9.0
    predicted_strong = predictions >= 9.0

    tp = np.sum(strong_binders & predicted_strong)
    fp = np.sum(~strong_binders & predicted_strong)
    tn = np.sum(~strong_binders & ~predicted_strong)
    fn = np.sum(strong_binders & ~predicted_strong)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
        'spearman': spearman, 'spearman_p': spearman_p,
        'pearson': pearson, 'pearson_p': pearson_p,
        'recall_pkd9': recall * 100, 'precision_pkd9': precision * 100,
        'f1_pkd9': f1 * 100, 'specificity_pkd9': specificity * 100,
        'n_samples': len(targets), 'n_strong_binders': int(strong_binders.sum())
    }

# ============================================================================
# Model Architecture (Original v2.6)
# ============================================================================

class IgT5ESM2Model(nn.Module):
    """
    Original v2.6 model architecture.

    Must match exactly the architecture used during training!
    """
    def __init__(self, dropout=0.3, freeze_encoders=True, use_checkpointing=True):
        super().__init__()

        print("Building model architecture (v2.6)...")

        # Load IgT5
        print("  Loading IgT5 (antibody encoder)...")
        self.igt5_tokenizer = T5Tokenizer.from_pretrained("Exscientia/IgT5")
        self.igt5_model = T5EncoderModel.from_pretrained("Exscientia/IgT5")

        # Load ESM-2
        print("  Loading ESM-2 (antigen encoder)...")
        self.esm2_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.esm2_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

        # Freeze encoders
        if freeze_encoders:
            for param in self.igt5_model.parameters():
                param.requires_grad = False
            for param in self.esm2_model.parameters():
                param.requires_grad = False

        # Gradient checkpointing
        if use_checkpointing:
            self.igt5_model.gradient_checkpointing_enable()
            self.esm2_model.gradient_checkpointing_enable()

        # Get dimensions
        self.igt5_dim = self.igt5_model.config.d_model  # 512
        self.esm2_dim = self.esm2_model.config.hidden_size  # 1280
        self.combined_dim = self.igt5_dim + self.esm2_dim  # 1792

        # Regression head (v2.6 architecture)
        self.regression_head = nn.Sequential(
            nn.Linear(self.combined_dim, 1024),
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
            nn.LayerNorm(128),

            nn.Linear(128, 1)
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n  Model Statistics:")
        print(f"    Total params: {total_params/1e6:.1f}M")
        print(f"    Trainable params: {trainable_params/1e6:.1f}M")
        print(f"    Frozen params: {(total_params-trainable_params)/1e6:.1f}M\n")

    def forward(self, antibody_seqs, antigen_seqs, device):
        # Tokenize
        antibody_tokens = self.igt5_tokenizer(
            antibody_seqs, return_tensors='pt', padding=True,
            truncation=True, max_length=512
        ).to(device)

        antigen_tokens = self.esm2_tokenizer(
            antigen_seqs, return_tensors='pt', padding=True,
            truncation=True, max_length=1024
        ).to(device)

        # Encode
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            antibody_outputs = self.igt5_model(**antibody_tokens)
            antibody_embedding = antibody_outputs.last_hidden_state.mean(dim=1)

            antigen_outputs = self.esm2_model(**antigen_tokens)
            antigen_embedding = antigen_outputs.last_hidden_state.mean(dim=1)

            combined = torch.cat([antibody_embedding, antigen_embedding], dim=1)
            pKd_pred = self.regression_head(combined).squeeze(-1)

        return pKd_pred

# ============================================================================
# Dataset & DataLoader
# ============================================================================

class AbAgDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'antibody_seqs': row['antibody_sequence'],
            'antigen_seqs': row['antigen_sequence'],
            'pKd': torch.tensor(row['pKd'], dtype=torch.float32)
        }

def collate_fn(batch):
    return {
        'antibody_seqs': [item['antibody_seqs'] for item in batch],
        'antigen_seqs': [item['antigen_seqs'] for item in batch],
        'pKd': torch.stack([item['pKd'] for item in batch])
    }

# ============================================================================
# Load Data
# ============================================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(CONFIG['data_path'])
print(f"\nDataset: {len(df):,} samples")
print(f"  pKd range: {df['pKd'].min():.2f} - {df['pKd'].max():.2f}")
print(f"  Strong binders (≥9): {(df['pKd']>=9).sum():,} ({100*(df['pKd']>=9).sum()/len(df):.1f}%)")

# Split data (same as training - must use same random_state!)
print("\nSplitting data (70% train / 15% val / 15% test)...")
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"  Train: {len(train_df):,}")
print(f"  Val:   {len(val_df):,}")
print(f"  Test:  {len(test_df):,}")

# Create datasets
val_dataset = AbAgDataset(val_df)
test_dataset = AbAgDataset(test_df)

# Create dataloaders
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                        num_workers=CONFIG['num_workers'], collate_fn=collate_fn, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                         num_workers=CONFIG['num_workers'], collate_fn=collate_fn, pin_memory=True)

print(f"\nDataLoaders created:")
print(f"  Val:  {len(val_loader):,} batches")
print(f"  Test: {len(test_loader):,} batches")

# ============================================================================
# Load Model
# ============================================================================

print("\n" + "=" * 70)
print("LOADING MODEL v2.6")
print("=" * 70)

# Build model
model = IgT5ESM2Model(dropout=0.3, freeze_encoders=True, use_checkpointing=True)

# Load checkpoint
print(f"\nLoading checkpoint from: {CONFIG['model_path']}")
checkpoint = torch.load(CONFIG['model_path'], map_location=device, weights_only=False)

# Check what's in the checkpoint
print(f"\nCheckpoint contents:")
for key in checkpoint.keys():
    print(f"  - {key}")

# Load model state
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"\n✅ Model loaded successfully!")

# Print checkpoint info
if 'epoch' in checkpoint:
    print(f"   Trained for: {checkpoint['epoch']+1} epochs")
if 'val_spearman' in checkpoint:
    print(f"   Validation Spearman: {checkpoint['val_spearman']:.4f}")
if 'best_val_spearman' in checkpoint:
    print(f"   Best Val Spearman: {checkpoint['best_val_spearman']:.4f}")

# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_model(model, loader, device, desc="Evaluating"):
    """Evaluate model on dataset"""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            antibody_seqs = batch['antibody_seqs']
            antigen_seqs = batch['antigen_seqs']
            batch_targets = batch['pKd'].to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                batch_predictions = model(antibody_seqs, antigen_seqs, device)

            predictions.extend(batch_predictions.float().cpu().numpy())
            targets.extend(batch_targets.float().cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    metrics = compute_comprehensive_metrics(targets, predictions)
    return metrics, predictions, targets

# ============================================================================
# Run Evaluation
# ============================================================================

print("\n" + "=" * 70)
print("EVALUATION ON VALIDATION SET")
print("=" * 70)

val_metrics, val_preds, val_targets = evaluate_model(model, val_loader, device, "Validation")

print(f"\n📊 VALIDATION METRICS (Model v2.6):")
print(f"  Samples: {val_metrics['n_samples']:,}")
print(f"  Strong Binders (pKd≥9): {val_metrics['n_strong_binders']}")
print(f"\n  Regression Metrics:")
print(f"    RMSE:        {val_metrics['rmse']:.4f}")
print(f"    MAE:         {val_metrics['mae']:.4f}")
print(f"    MSE:         {val_metrics['mse']:.4f}")
print(f"    R²:          {val_metrics['r2']:.4f}")
print(f"\n  Correlation Metrics:")
print(f"    Spearman ρ:  {val_metrics['spearman']:.4f} (p={val_metrics['spearman_p']:.2e})")
print(f"    Pearson r:   {val_metrics['pearson']:.4f} (p={val_metrics['pearson_p']:.2e})")
print(f"\n  Classification Metrics (pKd≥9):")
print(f"    Recall:      {val_metrics['recall_pkd9']:.2f}%")
print(f"    Precision:   {val_metrics['precision_pkd9']:.2f}%")
print(f"    F1-Score:    {val_metrics['f1_pkd9']:.2f}%")
print(f"    Specificity: {val_metrics['specificity_pkd9']:.2f}%")

# ============================================================================
# Evaluate on Test Set
# ============================================================================

print("\n" + "=" * 70)
print("EVALUATION ON TEST SET (UNSEEN DATA)")
print("=" * 70)

test_metrics, test_preds, test_targets = evaluate_model(model, test_loader, device, "Test Set")

print(f"\n📊 TEST SET METRICS (Model v2.6 - TRUE PERFORMANCE):")
print(f"  Samples: {test_metrics['n_samples']:,}")
print(f"  Strong Binders (pKd≥9): {test_metrics['n_strong_binders']}")
print(f"\n  Regression Metrics:")
print(f"    RMSE:        {test_metrics['rmse']:.4f}")
print(f"    MAE:         {test_metrics['mae']:.4f}")
print(f"    MSE:         {test_metrics['mse']:.4f}")
print(f"    R²:          {test_metrics['r2']:.4f}")
print(f"\n  Correlation Metrics:")
print(f"    Spearman ρ:  {test_metrics['spearman']:.4f} (p={test_metrics['spearman_p']:.2e})")
print(f"    Pearson r:   {test_metrics['pearson']:.4f} (p={test_metrics['pearson_p']:.2e})")
print(f"\n  Classification Metrics (pKd≥9):")
print(f"    Recall:      {test_metrics['recall_pkd9']:.2f}%")
print(f"    Precision:   {test_metrics['precision_pkd9']:.2f}%")
print(f"    F1-Score:    {test_metrics['f1_pkd9']:.2f}%")
print(f"    Specificity: {test_metrics['specificity_pkd9']:.2f}%")

# ============================================================================
# Save Results
# ============================================================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Save predictions
val_results = pd.DataFrame({
    'true_pKd': val_targets,
    'pred_pKd': val_preds,
    'error': val_preds - val_targets,
    'abs_error': np.abs(val_preds - val_targets)
})
val_pred_path = output_dir / 'val_predictions_v26.csv'
val_results.to_csv(val_pred_path, index=False)
print(f"\n✅ Saved: {val_pred_path}")

test_results = pd.DataFrame({
    'true_pKd': test_targets,
    'pred_pKd': test_preds,
    'error': test_preds - test_targets,
    'abs_error': np.abs(test_preds - test_targets)
})
test_pred_path = output_dir / 'test_predictions_v26.csv'
test_results.to_csv(test_pred_path, index=False)
print(f"✅ Saved: {test_pred_path}")

# Save metrics
all_metrics = {
    'model_version': 'v2.6 (100 epochs, pre-optimization)',
    'model_path': str(CONFIG['model_path']),
    'checkpoint_epoch': int(checkpoint.get('epoch', -1)) + 1 if 'epoch' in checkpoint else 'unknown',
    'validation': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                   for k, v in val_metrics.items()},
    'test': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in test_metrics.items()},
}

metrics_path = output_dir / 'evaluation_metrics_v26.json'
with open(metrics_path, 'w') as f:
    json.dump(all_metrics, f, indent=2)
print(f"✅ Saved: {metrics_path}")

# ============================================================================
# Visualizations
# ============================================================================

print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)

# 1. Prediction vs Actual scatter plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Validation
ax1 = axes[0]
ax1.scatter(val_targets, val_preds, alpha=0.3, s=10, color='blue')
ax1.plot([4, 14], [4, 14], 'r--', linewidth=2, label='Perfect prediction')
ax1.set_xlabel('True pKd', fontsize=12)
ax1.set_ylabel('Predicted pKd', fontsize=12)
ax1.set_title(f'Validation Set (v2.6)\nSpearman: {val_metrics["spearman"]:.4f}, RMSE: {val_metrics["rmse"]:.4f}',
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(4, 14)
ax1.set_ylim(4, 14)

# Test
ax2 = axes[1]
ax2.scatter(test_targets, test_preds, alpha=0.3, s=10, color='orange')
ax2.plot([4, 14], [4, 14], 'r--', linewidth=2, label='Perfect prediction')
ax2.set_xlabel('True pKd', fontsize=12)
ax2.set_ylabel('Predicted pKd', fontsize=12)
ax2.set_title(f'Test Set (v2.6 - UNSEEN)\nSpearman: {test_metrics["spearman"]:.4f}, RMSE: {test_metrics["rmse"]:.4f}',
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(4, 14)
ax2.set_ylim(4, 14)

plt.tight_layout()
scatter_path = output_dir / 'predictions_scatter_v26.png'
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: {scatter_path}")
plt.close()

# 2. Error distribution
fig, ax = plt.subplots(figsize=(10, 6))
test_errors = test_preds - test_targets
ax.hist(test_errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
ax.axvline(x=np.mean(test_errors), color='g', linestyle='--', linewidth=2,
           label=f'Mean error: {np.mean(test_errors):.4f}')
ax.set_xlabel('Prediction Error (pKd units)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Test Set: Error Distribution (v2.6)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
error_path = output_dir / 'error_distribution_v26.png'
plt.savefig(error_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {error_path}")
plt.close()

# 3. Performance by pKd range
fig, ax = plt.subplots(figsize=(10, 6))
pkd_ranges = [(4, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 14)]
range_labels = ['4-6', '6-7', '7-8', '8-9', '9-10', '10+']
range_maes = []

for low, high in pkd_ranges:
    mask = (test_targets >= low) & (test_targets < high)
    if mask.sum() > 0:
        range_mae = np.mean(np.abs(test_preds[mask] - test_targets[mask]))
        range_maes.append(range_mae)
    else:
        range_maes.append(0)

ax.bar(range_labels, range_maes, color='steelblue', edgecolor='black')
ax.set_xlabel('pKd Range', fontsize=12)
ax.set_ylabel('MAE (pKd units)', fontsize=12)
ax.set_title('Test Set: MAE by pKd Range (v2.6)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
range_path = output_dir / 'mae_by_pkd_range_v26.png'
plt.savefig(range_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {range_path}")
plt.close()

# ============================================================================
# Final Summary
# ============================================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE!")
print("=" * 70)

print(f"\n📌 KEY RESULTS (Model v2.6 - 100 Epochs):")
print(f"  Validation Spearman: {val_metrics['spearman']:.4f}")
print(f"  Test Spearman:       {test_metrics['spearman']:.4f} ← TRUE PERFORMANCE")
print(f"  Test RMSE:           {test_metrics['rmse']:.4f}")
print(f"  Test MAE:            {test_metrics['mae']:.4f}")
print(f"  Test R²:             {test_metrics['r2']:.4f}")
print(f"  Test Recall@pKd≥9:   {test_metrics['recall_pkd9']:.2f}%")

print(f"\n📁 All results saved to: {output_dir}/")
print(f"  - val_predictions_v26.csv")
print(f"  - test_predictions_v26.csv")
print(f"  - evaluation_metrics_v26.json")
print(f"  - predictions_scatter_v26.png")
print(f"  - error_distribution_v26.png")
print(f"  - mae_by_pkd_range_v26.png")

print("\n" + "=" * 70)

# Error analysis
print(f"\n📊 Error Analysis (Test Set):")
print(f"   Mean error:       {np.mean(test_errors):.4f} pKd")
print(f"   Std error:        {np.std(test_errors):.4f} pKd")
print(f"   Median |error|:   {np.median(np.abs(test_errors)):.4f} pKd")
print(f"   95th %ile |error|: {np.percentile(np.abs(test_errors), 95):.4f} pKd")
print(f"   Max |error|:      {np.max(np.abs(test_errors)):.4f} pKd")

print("\n✅ Evaluation script completed successfully!")
