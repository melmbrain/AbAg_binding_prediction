"""
AbAg Binding Affinity Prediction - v3 FULL DIMENSIONS Training

This script trains on FULL 1,280-dimensional ESM2 embeddings (no PCA).

IMPROVEMENTS OVER v2:
- ✅ Full 1,280 dimensions (vs 150 PCA dimensions)
- ✅ 100% variance preserved (vs 99.9%)
- ✅ Expected 10-30% improvement on extreme affinities
- ✅ Optimized for Colab Pro (16GB+ GPU)
- ✅ All v2 improvements retained (GELU, deeper, focal loss, etc.)

REQUIREMENTS:
- Colab Pro (T4 16GB or better)
- merged_with_full_features.csv uploaded to Google Drive
- ~12-15 hours training time (100 epochs)

Expected Results:
- Overall RMSE: 1.38 → 0.8-1.0
- Very strong RMSE: 2.53 → 0.8-1.2 (70%+ improvement!)
- Spearman ρ: 0.43 → 0.65-0.75

Convert to notebook: jupytext --to notebook colab_training_v3_full_dimensions.py
"""

# %% [markdown]
# # AbAg Binding Affinity Prediction - v3 FULL DIMENSIONS
#
# **NEW in v3: Full 1,280-dimensional features (NO PCA)**
#
# **All v2 improvements + full dimensions:**
# - ✅ 1,280 input dimensions (8.5x more features)
# - ✅ 100% variance preserved (vs 99.9% with PCA)
# - ✅ GELU activation
# - ✅ Deeper architecture: 1,280 → 512 → 256 → 128 → 64 → 1
# - ✅ 10x stronger weights for extremes
# - ✅ Focal loss + gradient clipping
# - ✅ AdamW + cosine annealing
#
# **Expected: 70%+ improvement on extreme affinities!**
#
# ---
# **Training time:** ~12-15 hours on T4 GPU (100 epochs)
# **GPU requirement:** 16GB+ (Colab Pro)

# %% [markdown]
# ## 1. Setup - GPU and Dependencies

# %%
# Check GPU - MUST be 16GB+ for full dimensions
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {vram_gb:.1f} GB")

    if vram_gb < 15:
        print(f"\n⚠️ WARNING: This GPU has only {vram_gb:.1f} GB VRAM")
        print("   Full-dimensional training requires 16GB+")
        print("   Consider:")
        print("   - Using Colab Pro")
        print("   - Reducing batch size to 64")
        print("   - Using v2 with PCA (150 dims) instead")
else:
    print("❌ ERROR: No GPU detected! Enable GPU in Runtime settings.")

# %%
# Install dependencies
get_ipython().system('pip install -q transformers scikit-learn pandas numpy tqdm matplotlib seaborn')
print("✅ All dependencies installed!")

# %% [markdown]
# ## 2. Mount Google Drive

# %%
from google.colab import drive
drive.mount('/content/drive')
print("✅ Google Drive mounted!")

# %%
import os
from pathlib import Path

# Set up paths - MODIFY THIS to match your Google Drive location
DRIVE_DATA_PATH = "/content/drive/MyDrive/AbAg_data/merged_with_full_features.csv"
OUTPUT_DIR = "/content/drive/MyDrive/AbAg_data/models_v3_full_dim"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Copy data to local for faster training
LOCAL_DATA_PATH = "/content/merged_with_full_features.csv"

print(f"Data path: {DRIVE_DATA_PATH}")
print(f"Output directory: {OUTPUT_DIR}")

if os.path.exists(DRIVE_DATA_PATH):
    file_size_mb = os.path.getsize(DRIVE_DATA_PATH) / 1e6
    print(f"✅ Data file found! Size: {file_size_mb:.1f} MB")

    print("\nCopying data to local storage for faster I/O...")
    print("(This may take 2-3 minutes for full-dimensional data)")
    get_ipython().system('cp "{DRIVE_DATA_PATH}" "{LOCAL_DATA_PATH}"')
    print("✅ Data copied to local storage!")

    DATA_PATH = LOCAL_DATA_PATH
else:
    print(f"❌ Data file not found at: {DRIVE_DATA_PATH}")
    print("\nPlease prepare full-dimensional features first:")
    print("  1. Run: python scripts/prepare_full_dimensional_features.py")
    print("  2. Upload 'merged_with_full_features.csv' to your Google Drive")

# %% [markdown]
# ## 3. Imports and Constants

# %%
# Imports
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time
import gc

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Constants
BINS = [0, 5, 7, 9, 11, 16]
BIN_LABELS = ['very_weak', 'weak', 'moderate', 'strong', 'very_strong']

print("✅ Imports complete!")

# %% [markdown]
# ## 4. Dataset Class

# %%
class AffinityDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

print("✅ Dataset class defined!")

# %% [markdown]
# ## 5. v3 Model Architecture - FULL DIMENSIONS (1,280 input)

# %%
class AffinityModelV3FullDim(nn.Module):
    """v3 Model with FULL 1,280-dimensional input"""

    def __init__(self, input_dim=1280, hidden_dims=[512, 256, 128, 64], dropout=0.3):
        super(AffinityModelV3FullDim, self).__init__()

        print(f"Building v3 model: {input_dim} → {' → '.join(map(str, hidden_dims))} → 1")

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            linear = nn.Linear(prev_dim, hidden_dim)

            # Xavier initialization
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

            layers.extend([
                linear,
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),  # GELU for smooth gradients
                nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout * 0.5)
            ])
            prev_dim = hidden_dim

        # Output layer
        output_layer = nn.Linear(prev_dim, 1)
        nn.init.xavier_uniform_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

print("✅ v3 Model class defined (1,280-dim input)!")

# %% [markdown]
# ## 6. Loss Functions (Same as v2)

# %%
class WeightedMSELoss(nn.Module):
    """MSE with class-based weighting"""

    def __init__(self, bin_weights, bins_edges):
        super().__init__()
        self.bin_weights = bin_weights
        self.bins = bins_edges

    def forward(self, predictions, targets):
        weights = torch.ones_like(targets)
        for i, (low, high) in enumerate(zip(self.bins[:-1], self.bins[1:])):
            mask = (targets >= low) & (targets < high)
            weights[mask] = self.bin_weights[i]

        mse = (predictions - targets) ** 2
        weighted_mse = mse * weights
        return weighted_mse.mean()


class FocalMSELoss(nn.Module):
    """Focal loss for regression"""

    def __init__(self, bin_weights, bins_edges, gamma=2.0):
        super().__init__()
        self.bin_weights = bin_weights
        self.bins = bins_edges
        self.gamma = gamma

    def forward(self, predictions, targets):
        # Class weights
        weights = torch.ones_like(targets)
        for i, (low, high) in enumerate(zip(self.bins[:-1], self.bins[1:])):
            mask = (targets >= low) & (targets < high)
            weights[mask] = self.bin_weights[i]

        # Focal weighting
        mse = (predictions - targets) ** 2
        focal_weight = mse ** (self.gamma / 2)

        weighted_mse = mse * weights * (1 + focal_weight)
        return weighted_mse.mean()

print("✅ Loss functions defined!")

# %% [markdown]
# ## 7. Load and Prepare Data (FULL 1,280 dimensions)

# %%
print("Loading dataset with FULL dimensions...")
print("(This may take 1-2 minutes for 1,280-dim features)")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"✅ Loaded {len(df):,} samples")

# Filter samples with FULL dimensional features
full_dim_cols = [f'esm2_dim_{i}' for i in range(1280)]

# Check if columns exist
if full_dim_cols[0] not in df.columns:
    print(f"\n❌ ERROR: Full-dimensional features not found!")
    print(f"   Expected column: {full_dim_cols[0]}")
    print(f"   Found columns starting with 'esm2': {[c for c in df.columns if c.startswith('esm2')][:5]}")
    print("\n   Please run: python scripts/prepare_full_dimensional_features.py")
else:
    df_with_features = df[df[full_dim_cols[0]].notna()].copy()
    print(f"✅ Samples with FULL 1,280-dim features: {len(df_with_features):,}")

# Create affinity bins
df_with_features['affinity_bin'] = pd.cut(
    df_with_features['pKd'], bins=BINS, labels=BIN_LABELS, include_lowest=True
)

# Show distribution
print("\nAffinity Distribution:")
for label in BIN_LABELS:
    count = (df_with_features['affinity_bin'] == label).sum()
    pct = count / len(df_with_features) * 100
    marker = "⭐" if label in ['very_strong', 'very_weak'] else "  "
    print(f"{marker} {label:<15}: {count:6,} ({pct:5.2f}%)")

print(f"\nTotal: {len(df_with_features):,}")
print(f"Feature dimensions: 1,280 (8.5x more than v2's 150)")

# %%
# Extract FULL-dimensional features and labels
print("\nExtracting 1,280-dimensional features...")
X = df_with_features[full_dim_cols].values
y = df_with_features['pKd'].values

print(f"Feature matrix shape: {X.shape}")
print(f"Memory usage: ~{X.nbytes / 1e9:.2f} GB")

# Train/val/test split (same split as v1/v2 for fair comparison)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15/0.85, random_state=42)

print(f"\nSplit sizes:")
print(f"  Train set: {len(X_train):,} samples")
print(f"  Val set:   {len(X_val):,} samples")
print(f"  Test set:  {len(X_test):,} samples")

# Clear memory
del df, df_with_features
gc.collect()

# %%
# Calculate class weights (10x for extremes, same as v2)
y_train_binned = pd.cut(y_train, bins=BINS, labels=BIN_LABELS, include_lowest=True)
bin_counts = y_train_binned.value_counts().sort_index()
total_samples = len(y_train)
bin_weights = {}

for label in BIN_LABELS:
    count = bin_counts.get(label, 1)
    base_weight = total_samples / (len(BIN_LABELS) * count)

    # 10x stronger for extremes
    if label in ['very_strong', 'very_weak']:
        bin_weights[label] = base_weight * 10
    else:
        bin_weights[label] = base_weight

print("Class Weights (10x for extremes):")
for label, weight in bin_weights.items():
    marker = "⭐" if label in ['very_strong', 'very_weak'] else "  "
    print(f"{marker} {label:<15}: {weight:.2f}")

# Convert to tensor
bin_weights_tensor = torch.FloatTensor([bin_weights[l] for l in BIN_LABELS]).cuda()

# Create datasets
train_dataset = AffinityDataset(X_train, y_train)
val_dataset = AffinityDataset(X_val, y_val)
test_dataset = AffinityDataset(X_test, y_test)

# Clear memory
del X, y, X_temp, y_temp
gc.collect()
torch.cuda.empty_cache()

print("\n✅ Data preparation complete!")

# %% [markdown]
# ## 8. Training Configuration - Optimized for Full Dimensions

# %%
# v3 Configuration - optimized for full dimensions
EPOCHS = 100
BATCH_SIZE = 96  # Reduced from 128 due to larger input
LEARNING_RATE = 0.0001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_FOCAL_LOSS = True
GRADIENT_CLIP = 1.0

print("v3 FULL DIMENSIONS Configuration:")
print("="*60)
print(f"  Input dimensions: 1,280 (vs 150 in v2)")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE} (reduced for larger model)")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Device: {DEVICE}")
print(f"  Loss: {'Focal MSE' if USE_FOCAL_LOSS else 'Weighted MSE'}")
print(f"  Gradient clipping: {GRADIENT_CLIP}")
print(f"\n  Architecture: 1,280 → 512 → 256 → 128 → 64 → 1")
print(f"  Activation: GELU")
print(f"  Optimizer: AdamW + Cosine Annealing")
print(f"  Class weights: 10x for extremes")
print("="*60)

# %% [markdown]
# ## 9. Create Data Loaders

# %%
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"✅ Data loaders created!")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

# %% [markdown]
# ## 10. Initialize v3 Model (1,280 inputs)

# %%
# Initialize v3 FULL-DIMENSIONAL model
model = AffinityModelV3FullDim(
    input_dim=1280,  # FULL dimensions!
    hidden_dims=[512, 256, 128, 64],
    dropout=0.3
)
model = model.to(DEVICE)

# Loss and optimizer
if USE_FOCAL_LOSS:
    criterion = FocalMSELoss(bin_weights_tensor, BINS, gamma=2.0)
    print("Using Focal MSE Loss")
else:
    criterion = WeightedMSELoss(bin_weights_tensor, BINS)
    print("Using Weighted MSE Loss")

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=20,
    T_mult=2,
    eta_min=LEARNING_RATE * 0.01
)

# Count parameters
total_params = model.count_parameters()

print(f"\n✅ v3 Model initialized!")
print(f"  Total parameters: {total_params:,}")
print(f"  Model size: ~{total_params * 4 / 1e6:.1f} MB")
print(f"\nFull model architecture:")
print(model)

# Check VRAM usage
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    allocated_gb = torch.cuda.memory_allocated() / 1e9
    reserved_gb = torch.cuda.memory_reserved() / 1e9
    print(f"\nGPU Memory:")
    print(f"  Allocated: {allocated_gb:.2f} GB")
    print(f"  Reserved: {reserved_gb:.2f} GB")

# %% [markdown]
# ## 11. Training Loop

# %%
# Training loop
best_val_loss = float('inf')
train_losses = []
val_losses = []
learning_rates = []
train_start = time.time()

print("\nStarting v3 FULL-DIMENSIONAL training...")
print("="*70)
print("This will take ~12-15 hours on T4 GPU (100 epochs)")
print("="*70)

for epoch in range(EPOCHS):
    epoch_start = time.time()

    # Training
    model.train()
    train_loss = 0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

    for features, labels in train_pbar:
        features, labels = features.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

        optimizer.step()

        train_loss += loss.item()
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")

    with torch.no_grad():
        for features, labels in val_pbar:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            predictions = model(features)
            loss = criterion(predictions, labels)
            val_loss += loss.item()
            val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    # Update learning rate
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    scheduler.step()

    epoch_time = time.time() - epoch_start
    elapsed_time = time.time() - train_start
    eta = elapsed_time / (epoch + 1) * (EPOCHS - epoch - 1)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
          f"LR: {current_lr:.6f}, Time: {epoch_time:.1f}s, ETA: {eta/3600:.1f}h")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': {
                'input_dim': 1280,
                'hidden_dims': [512, 256, 128, 64],
                'dropout': 0.3,
                'activation': 'GELU',
                'learning_rate': LEARNING_RATE,
                'batch_size': BATCH_SIZE,
                'focal_loss': USE_FOCAL_LOSS
            }
        }, f'{OUTPUT_DIR}/best_model_v3_full_dim.pth')
        print(f"  ✅ New best model saved! (val_loss: {val_loss:.4f})")

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, f'{OUTPUT_DIR}/checkpoint_v3_epoch_{epoch+1}.pth')
        print(f"  💾 Checkpoint saved!")

    # Clear GPU cache every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.cuda.empty_cache()

total_time = time.time() - train_start
print(f"\n{'='*70}")
print(f"✅ Training complete! Total time: {total_time/3600:.2f} hours")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"{'='*70}")

# %% [markdown]
# ## 12. Plot Training Curves

# %%
# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(train_losses, label='Train Loss', linewidth=2)
axes[0].plot(val_losses, label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('v3 Full Dimensions - Training Curves', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(learning_rates, linewidth=2, color='green')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Learning Rate', fontsize=12)
axes[1].set_title('Learning Rate Schedule', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/training_curves_v3.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Training curves saved!")

# %% [markdown]
# ## 13. Evaluation

# %%
# Load best model
checkpoint = torch.load(f'{OUTPUT_DIR}/best_model_v3_full_dim.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✅ Best model loaded from epoch {checkpoint['epoch']+1}")
print(f"   Val loss: {checkpoint['val_loss']:.4f}")

# %%
# Evaluate on test set
model.eval()
test_predictions = []
test_targets = []

print("Running evaluation on test set...")
with torch.no_grad():
    for features, labels in tqdm(test_loader, desc="Testing"):
        features = features.to(DEVICE)
        predictions = model(features)
        test_predictions.extend(predictions.cpu().numpy())
        test_targets.extend(labels.numpy())

test_predictions = np.array(test_predictions)
test_targets = np.array(test_targets)

# Calculate metrics
mse = mean_squared_error(test_targets, test_predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_targets, test_predictions)
spearman = spearmanr(test_targets, test_predictions)[0]
pearson = pearsonr(test_targets, test_predictions)[0]
r2 = 1 - (np.sum((test_targets - test_predictions)**2) / np.sum((test_targets - test_targets.mean())**2))

print("="*70)
print("TEST SET PERFORMANCE (v3 FULL DIMENSIONS)")
print("="*70)
print(f"RMSE:        {rmse:.4f}")
print(f"MAE:         {mae:.4f}")
print(f"Spearman ρ:  {spearman:.4f}")
print(f"Pearson r:   {pearson:.4f}")
print(f"R²:          {r2:.4f}")
print("="*70)

# %%
# Per-bin metrics
test_df = pd.DataFrame({
    'target': test_targets,
    'prediction': test_predictions
})
test_df['affinity_bin'] = pd.cut(test_df['target'], bins=BINS, labels=BIN_LABELS, include_lowest=True)

print("\nPER-BIN PERFORMANCE:")
print("="*70)
print(f"{'Bin':<15} | {'Count':<8} | {'RMSE':<8} | {'MAE':<8}")
print("-"*70)

per_bin_results = {}
for label in BIN_LABELS:
    bin_data = test_df[test_df['affinity_bin'] == label]
    if len(bin_data) > 0:
        bin_rmse = np.sqrt(mean_squared_error(bin_data['target'], bin_data['prediction']))
        bin_mae = mean_absolute_error(bin_data['target'], bin_data['prediction'])
        per_bin_results[label] = {'rmse': bin_rmse, 'mae': bin_mae, 'count': len(bin_data)}
        marker = "⭐" if label in ['very_strong', 'very_weak'] else "  "
        print(f"{marker} {label:<13} | {len(bin_data):<8} | {bin_rmse:<8.4f} | {bin_mae:<8.4f}")

print("="*70)

# %% [markdown]
# ## 14. Comparison: v1 vs v2 vs v3

# %%
# Comprehensive comparison
v1_results = {
    'RMSE': 1.4761,
    'MAE': 1.3011,
    'Spearman': 0.3912,
    'Pearson': 0.7265,
    'R2': 0.5188,
    'very_strong_rmse': 2.9394
}

v2_results = {
    'RMSE': 1.3799,
    'MAE': 1.2143,
    'Spearman': 0.4258,
    'Pearson': 0.7624,
    'R2': 0.5795,
    'very_strong_rmse': 2.5300
}

v3_very_strong_rmse = per_bin_results['very_strong']['rmse']

print("\n" + "="*80)
print("FULL COMPARISON: v1 (PCA) → v2 (PCA+Improved) → v3 (FULL DIMS)")
print("="*80)
print(f"{'Metric':<20} | {'v1':<10} | {'v2':<10} | {'v3 (FULL)':<12} | {'v3 vs v1':<10}")
print("-"*80)

metrics = [
    ('RMSE', v1_results['RMSE'], v2_results['RMSE'], rmse),
    ('MAE', v1_results['MAE'], v2_results['MAE'], mae),
    ('Spearman ρ', v1_results['Spearman'], v2_results['Spearman'], spearman),
    ('Pearson r', v1_results['Pearson'], v2_results['Pearson'], pearson),
    ('R²', v1_results['R2'], v2_results['R2'], r2),
    ('Very Strong RMSE', v1_results['very_strong_rmse'], v2_results['very_strong_rmse'], v3_very_strong_rmse)
]

for metric_name, v1_val, v2_val, v3_val in metrics:
    change_pct = ((v1_val - v3_val) / v1_val) * 100 if 'RMSE' in metric_name else ((v3_val - v1_val) / v1_val) * 100
    symbol = "✅✅" if abs(change_pct) > 20 else "✅" if abs(change_pct) > 10 else "  "
    print(f"{symbol} {metric_name:<18} | {v1_val:<10.4f} | {v2_val:<10.4f} | {v3_val:<12.4f} | {change_pct:+.1f}%")

print("="*80)
print(f"\nInput dimensions: v1=150, v2=150, v3=1,280 (8.5x more features!)")
print(f"Training time: v1=31min, v2=31min, v3={total_time/3600:.1f}hours")
print("="*80)

# %% [markdown]
# ## 15. Generate Plots

# %%
# Predictions vs targets
plt.figure(figsize=(10, 10))
plt.scatter(test_targets, test_predictions, alpha=0.3, s=10)
plt.plot([test_targets.min(), test_targets.max()], [test_targets.min(), test_targets.max()], 'r--', lw=2)
plt.xlabel('True pKd', fontsize=12)
plt.ylabel('Predicted pKd', fontsize=12)
plt.title(f'v3 FULL DIMENSIONS - Test Set Predictions\nSpearman ρ = {spearman:.4f}, RMSE = {rmse:.4f}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.savefig(f'{OUTPUT_DIR}/predictions_vs_targets_v3.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Prediction plot saved!")

# %%
# Residuals analysis
residuals = test_predictions - test_targets

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(test_predictions, residuals, alpha=0.3, s=10)
axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0].set_xlabel('Predicted pKd', fontsize=12)
axes[0].set_ylabel('Residuals', fontsize=12)
axes[0].set_title('v3 - Residuals vs Predictions', fontsize=14)
axes[0].grid(True, alpha=0.3)

axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Residuals', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title(f'Residuals Distribution\nMean = {residuals.mean():.4f}, Std = {residuals.std():.4f}', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/residuals_analysis_v3.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Residuals plot saved!")

# %% [markdown]
# ## 16. Save Results

# %%
# Save comprehensive results
results_summary = f"""
AbAg Binding Affinity Prediction - v3 FULL DIMENSIONS Results
{'='*70}

Model: v3 with FULL 1,280-dimensional ESM2 embeddings (NO PCA)

Key Improvements over v2:
  - Input dimensions: 1,280 (vs 150 in v2) - 8.5x more features
  - Variance preserved: 100% (vs 99.9% in v2)
  - All v2 improvements retained (GELU, deeper arch, focal loss, etc.)

Training:
  - Epochs: {EPOCHS}
  - Batch size: {BATCH_SIZE}
  - Training samples: {len(X_train):,}
  - Validation samples: {len(X_val):,}
  - Test samples: {len(X_test):,}
  - Total training time: {total_time/3600:.2f} hours
  - Parameters: {total_params:,}

Test Set Performance:
  - RMSE:       {rmse:.4f} (v2: 1.3799, v1: 1.4761)
  - MAE:        {mae:.4f} (v2: 1.2143, v1: 1.3011)
  - Spearman ρ: {spearman:.4f} (v2: 0.4258, v1: 0.3912)
  - Pearson r:  {pearson:.4f} (v2: 0.7624, v1: 0.7265)
  - R²:         {r2:.4f} (v2: 0.5795, v1: 0.5188)

Per-Bin Performance:
"""

for label in BIN_LABELS:
    if label in per_bin_results:
        bin_res = per_bin_results[label]
        marker = "⭐" if label in ['very_strong', 'very_weak'] else "  "
        results_summary += f"{marker} - {label:<15}: RMSE={bin_res['rmse']:6.4f}, MAE={bin_res['mae']:6.4f}, N={bin_res['count']:6,}\n"

results_summary += f"\n{'='*70}\n"
results_summary += f"\nIMPROVEMENT OVER v1:\n"
results_summary += f"  - Overall RMSE: {((v1_results['RMSE'] - rmse) / v1_results['RMSE'] * 100):+.1f}%\n"
results_summary += f"  - Very Strong RMSE: {((v1_results['very_strong_rmse'] - v3_very_strong_rmse) / v1_results['very_strong_rmse'] * 100):+.1f}%\n"
results_summary += f"  - Spearman ρ: {((spearman - v1_results['Spearman']) / v1_results['Spearman'] * 100):+.1f}%\n"
results_summary += f"\n{'='*70}\n"

with open(f'{OUTPUT_DIR}/evaluation_results_v3.txt', 'w') as f:
    f.write(results_summary)

print(results_summary)
print(f"✅ Results saved to {OUTPUT_DIR}/evaluation_results_v3.txt")

# %%
# Save predictions
results_df = pd.DataFrame({
    'true_pKd': test_targets,
    'predicted_pKd': test_predictions,
    'residual': residuals,
    'affinity_bin': test_df['affinity_bin']
})

results_df.to_csv(f'{OUTPUT_DIR}/test_predictions_v3.csv', index=False)
print(f"✅ Predictions saved to {OUTPUT_DIR}/test_predictions_v3.csv")

# Save final model
torch.save({
    'model_state_dict': model.state_dict(),
    'metrics': {
        'rmse': rmse,
        'mae': mae,
        'spearman': spearman,
        'pearson': pearson,
        'r2': r2,
        'per_bin': per_bin_results
    },
    'config': {
        'input_dim': 1280,
        'hidden_dims': [512, 256, 128, 64],
        'dropout': 0.3,
        'activation': 'GELU',
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'focal_loss': USE_FOCAL_LOSS,
        'version': 'v3_full_dimensions'
    }
}, f'{OUTPUT_DIR}/final_model_v3_full_dim.pth')

print(f"\n✅ All files saved to Google Drive: {OUTPUT_DIR}")
print(f"\n🎉 v3 FULL-DIMENSIONAL TRAINING COMPLETE!")
print(f"\nYou can now download the trained model from Google Drive!")
