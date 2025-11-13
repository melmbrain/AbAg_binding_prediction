"""
Complete Google Colab Training Pipeline - FROM SCRATCH
This script does EVERYTHING: embedding generation + training + evaluation

Upload this to Google Colab and run!
Time: ~15-20 hours for full pipeline (most time is embedding generation)

Steps:
1. Upload your CSV to Google Drive
2. Run this notebook
3. Download trained model

Author: Jaeseong Yoon
Date: 2025-11-06
"""

# %% [markdown]
# # AbAg Binding Affinity Prediction - COMPLETE PIPELINE
#
# **This notebook does everything from scratch:**
# - ✅ Load raw data (sequences + pKd)
# - ✅ Generate ESM-2 embeddings (1,280 dimensions)
# - ✅ Train model (v3 full dimensions)
# - ✅ Evaluate and save results
#
# **Requirements:**
# - Google Colab Pro (16GB GPU recommended)
# - Dataset CSV uploaded to Google Drive
# - ~15-20 hours total time

# %% [markdown]
# ## Part 1: Setup and Installation

# %%
# Check GPU
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {vram_gb:.1f} GB")

    if vram_gb < 15:
        print(f"\n⚠️ WARNING: Only {vram_gb:.1f} GB VRAM available")
        print("   Consider using Colab Pro for better GPU")
else:
    print("\n❌ ERROR: No GPU detected! This will be VERY slow.")
    print("   Enable GPU: Runtime → Change runtime type → GPU")

# %%
# Install dependencies
print("Installing dependencies...")
get_ipython().system('pip install -q transformers torch pandas numpy scikit-learn scipy tqdm matplotlib seaborn')
print("✅ All dependencies installed!")

# %% [markdown]
# ## Part 2: Mount Google Drive and Setup Paths

# %%
from google.colab import drive
drive.mount('/content/drive')
print("✅ Google Drive mounted!")

# %%
import os
from pathlib import Path

# ===== MODIFY THESE PATHS =====
# Where is your dataset CSV in Google Drive?
DRIVE_DATA_PATH = "/content/drive/MyDrive/AbAg_data/agab_phase2_full.csv"

# Where to save outputs?
OUTPUT_DIR = "/content/drive/MyDrive/AbAg_outputs"

# ===== END OF PATHS TO MODIFY =====

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check if data file exists
if os.path.exists(DRIVE_DATA_PATH):
    file_size_mb = os.path.getsize(DRIVE_DATA_PATH) / 1e6
    print(f"✅ Data file found! Size: {file_size_mb:.1f} MB")
else:
    print(f"❌ Data file not found at: {DRIVE_DATA_PATH}")
    print("\nPlease upload your CSV to Google Drive first!")
    print("Expected columns: antibody_sequence, antigen_sequence, pKd")

# %% [markdown]
# ## Part 3: Load and Inspect Data

# %%
import pandas as pd
import numpy as np

print("Loading dataset...")
df = pd.read_csv(DRIVE_DATA_PATH)
print(f"✅ Loaded {len(df):,} samples")

print("\nDataset columns:")
print(df.columns.tolist())

print("\nFirst few rows:")
print(df.head())

print("\nDataset info:")
print(f"  Rows: {len(df):,}")
print(f"  Columns: {len(df.columns)}")
print(f"  Missing values:")
for col in ['antibody_sequence', 'antigen_sequence', 'pKd']:
    if col in df.columns:
        missing = df[col].isna().sum()
        print(f"    {col}: {missing:,} ({missing/len(df)*100:.1f}%)")

# %% [markdown]
# ## Part 4: Generate ESM-2 Embeddings
#
# **⏰ This will take 10-15 hours for 160K samples**
#
# The script will:
# - Process sequences in batches
# - Save progress every 1000 samples
# - Resume from checkpoint if interrupted

# %%
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import gc

print("Loading ESM-2 model...")
print("(This will download ~2.5GB on first run)")

model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
esm_model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
esm_model = esm_model.to(device)
esm_model.eval()

print(f"✅ ESM-2 model loaded on {device}")

# %%
def generate_embedding(sequence, max_length=1024):
    """Generate ESM-2 embedding for a single sequence"""
    try:
        # Truncate if too long
        if len(sequence) > max_length:
            sequence = sequence[:max_length]

        # Tokenize
        inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate embedding
        with torch.no_grad():
            outputs = esm_model(**inputs)
            # Mean pool over sequence length
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

        return embedding.cpu().numpy()
    except Exception as e:
        print(f"Error processing sequence: {e}")
        return None

def generate_embeddings_batch(df, batch_size=8, save_every=1000):
    """Generate embeddings for all samples with progress saving"""

    checkpoint_path = f"{OUTPUT_DIR}/embedding_checkpoint.csv"

    # Check for existing checkpoint
    start_idx = 0
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint, loading...")
        df_checkpoint = pd.read_csv(checkpoint_path)
        start_idx = len(df_checkpoint)
        print(f"Resuming from sample {start_idx:,}")

    # Filter valid samples
    df_valid = df.dropna(subset=['antibody_sequence', 'antigen_sequence', 'pKd']).copy()
    df_valid = df_valid[start_idx:].reset_index(drop=True)

    print(f"\nGenerating embeddings for {len(df_valid):,} samples...")
    print(f"Batch size: {batch_size}")
    print(f"Estimated time: {len(df_valid) * 0.3 / 3600:.1f} hours")

    embeddings = []

    for idx in tqdm(range(0, len(df_valid), batch_size), desc="Processing batches"):
        batch = df_valid.iloc[idx:idx+batch_size]

        for _, row in batch.iterrows():
            # Concatenate antibody + antigen sequences
            combined_seq = row['antibody_sequence'] + row['antigen_sequence']

            # Generate embedding
            emb = generate_embedding(combined_seq)

            if emb is not None:
                embeddings.append(emb)
            else:
                # Use zeros for failed embeddings
                embeddings.append(np.zeros(1280))

        # Save checkpoint every N samples
        if (idx + batch_size) % save_every == 0:
            current_df = df_valid.iloc[:idx+batch_size].copy()
            current_embeddings = np.array(embeddings)

            # Add embeddings as columns
            for i in range(1280):
                current_df[f'esm2_dim_{i}'] = current_embeddings[:, i]

            current_df.to_csv(checkpoint_path, index=False)
            print(f"\n💾 Checkpoint saved: {idx+batch_size:,} samples processed")

        # Clear GPU cache periodically
        if idx % (save_every * 10) == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Convert to array
    embeddings = np.array(embeddings)
    print(f"\n✅ Generated {len(embeddings):,} embeddings of shape {embeddings.shape}")

    return df_valid, embeddings

# %%
# Generate embeddings
print("=" * 70)
print("STARTING EMBEDDING GENERATION")
print("=" * 70)
print("\n⚠️  This will take 10-15 hours for full dataset")
print("⚠️  The session will save checkpoints every 1000 samples")
print("⚠️  If interrupted, just re-run this cell to resume\n")

df_processed, embeddings = generate_embeddings_batch(df, batch_size=4, save_every=1000)

# %%
# Add embeddings to dataframe
print("Adding embeddings to dataframe...")
for i in range(1280):
    df_processed[f'esm2_dim_{i}'] = embeddings[:, i]

# Save complete dataset
output_path = f"{OUTPUT_DIR}/dataset_with_embeddings.csv"
df_processed.to_csv(output_path, index=False)
print(f"✅ Complete dataset saved: {output_path}")
print(f"   Size: {os.path.getsize(output_path) / 1e9:.2f} GB")

# Clear memory
del esm_model, tokenizer
torch.cuda.empty_cache()
gc.collect()

# %% [markdown]
# ## Part 5: Prepare Data for Training

# %%
from sklearn.model_selection import train_test_split

# Copy data to local for faster training
LOCAL_DATA = "/content/dataset_with_embeddings.csv"
print(f"Copying data to local storage for faster I/O...")
get_ipython().system('cp "{output_path}" "{LOCAL_DATA}"')
print(f"✅ Data copied to local: {LOCAL_DATA}")

# %%
print("Loading data for training...")
df_train = pd.read_csv(LOCAL_DATA)
print(f"Loaded {len(df_train):,} samples with embeddings")

# Extract features and targets
feature_cols = [f'esm2_dim_{i}' for i in range(1280)]
X = df_train[feature_cols].values
y = df_train['pKd'].values

print(f"\nFeature matrix: {X.shape}")
print(f"Target vector: {y.shape}")

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15/0.85, random_state=42)

print(f"\nData split:")
print(f"  Train: {len(X_train):,} samples")
print(f"  Val:   {len(X_val):,} samples")
print(f"  Test:  {len(X_test):,} samples")

# Show affinity distribution
print(f"\nAffinity distribution:")
print(f"  Min pKd: {y.min():.2f}")
print(f"  Max pKd: {y.max():.2f}")
print(f"  Mean pKd: {y.mean():.2f}")
print(f"  Std pKd: {y.std():.2f}")

# %% [markdown]
# ## Part 6: Define Model and Training Components

# %%
import torch.nn as nn
import torch.nn.init as init

class AffinityModelV3(nn.Module):
    """v3 Model with full 1,280-dimensional input"""

    def __init__(self, input_dim=1280, hidden_dims=[512, 256, 128, 64], dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout if i < len(hidden_dims)-1 else dropout*0.5)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

        # Xavier initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x).squeeze(-1)

print("✅ Model architecture defined")

# %%
# Dataset class
from torch.utils.data import Dataset, DataLoader

class AffinityDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create datasets
train_dataset = AffinityDataset(X_train, y_train)
val_dataset = AffinityDataset(X_val, y_val)
test_dataset = AffinityDataset(X_test, y_test)

# Create dataloaders
BATCH_SIZE = 96
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"✅ Data loaders created (batch size: {BATCH_SIZE})")

# %% [markdown]
# ## Part 7: Training Loop

# %%
import torch.optim as optim
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

# Initialize model
model = AffinityModelV3(input_dim=1280, hidden_dims=[512, 256, 128, 64], dropout=0.3)
model = model.to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

# Count parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params:,}")
print(f"Model size: ~{total_params * 4 / 1e6:.1f} MB")

# %%
# Training configuration
EPOCHS = 100
GRADIENT_CLIP = 1.0

train_losses = []
val_losses = []
best_val_loss = float('inf')

print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: 1e-4")
print(f"Estimated time: ~12-15 hours on T4 GPU")
print("="*70 + "\n")

train_start = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()

    # Training
    model.train()
    train_loss = 0
    for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            predictions = model(features)
            loss = criterion(predictions, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    # Update scheduler
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    # Calculate ETA
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - train_start
    eta = elapsed / (epoch + 1) * (EPOCHS - epoch - 1)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
          f"LR: {current_lr:.6f}, Time: {epoch_time:.1f}s, ETA: {eta/3600:.1f}h")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, f'{OUTPUT_DIR}/best_model.pth')
        print(f"  ✅ Best model saved (val_loss: {val_loss:.4f})")

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, f'{OUTPUT_DIR}/checkpoint_epoch_{epoch+1}.pth')
        print(f"  💾 Checkpoint saved")

total_time = time.time() - train_start
print(f"\n{'='*70}")
print(f"✅ Training complete! Total time: {total_time/3600:.2f} hours")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"{'='*70}")

# %% [markdown]
# ## Part 8: Evaluation

# %%
# Load best model
checkpoint = torch.load(f'{OUTPUT_DIR}/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✅ Best model loaded (epoch {checkpoint['epoch']+1})")

# %%
# Evaluate on test set
model.eval()
test_predictions = []
test_targets = []

with torch.no_grad():
    for features, labels in tqdm(test_loader, desc="Testing"):
        features = features.to(device)
        predictions = model(features)
        test_predictions.extend(predictions.cpu().numpy())
        test_targets.extend(labels.numpy())

test_predictions = np.array(test_predictions)
test_targets = np.array(test_targets)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
mae = mean_absolute_error(test_targets, test_predictions)
spearman = spearmanr(test_targets, test_predictions)[0]
pearson = pearsonr(test_targets, test_predictions)[0]
r2 = 1 - (np.sum((test_targets - test_predictions)**2) / np.sum((test_targets - test_targets.mean())**2))

print("\n" + "="*70)
print("TEST SET RESULTS")
print("="*70)
print(f"RMSE:        {rmse:.4f}")
print(f"MAE:         {mae:.4f}")
print(f"Spearman ρ:  {spearman:.4f}")
print(f"Pearson r:   {pearson:.4f}")
print(f"R²:          {r2:.4f}")
print("="*70)

# %%
# Save results
results_df = pd.DataFrame({
    'true_pKd': test_targets,
    'predicted_pKd': test_predictions,
    'residual': test_predictions - test_targets
})
results_df.to_csv(f'{OUTPUT_DIR}/test_predictions.csv', index=False)

# Save summary
summary = {
    'rmse': float(rmse),
    'mae': float(mae),
    'spearman': float(spearman),
    'pearson': float(pearson),
    'r2': float(r2),
    'training_time_hours': total_time / 3600,
    'total_samples': len(df_train),
    'test_samples': len(test_targets)
}

import json
with open(f'{OUTPUT_DIR}/results_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ Results saved to {OUTPUT_DIR}")
print(f"\nYou can now download:")
print(f"  - best_model.pth (trained model)")
print(f"  - test_predictions.csv (predictions)")
print(f"  - results_summary.json (metrics)")

print(f"\n🎉 COMPLETE! Training finished successfully!")
