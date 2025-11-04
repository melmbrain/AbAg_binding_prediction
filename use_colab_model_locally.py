#!/usr/bin/env python3
"""
Use Colab-Trained Model Locally

This script allows you to use a model trained on Google Colab
on your local machine for inference.

Usage:
    python use_colab_model_locally.py --model path/to/best_model.pth

Requirements:
    - Download the model file from Google Drive
    - Have the same dependencies installed locally
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr

# Model architecture (MUST match Colab training)
class AffinityPredictor(nn.Module):
    def __init__(self, input_dim=150, hidden_dims=[256, 128], dropout=0.3):
        super(AffinityPredictor, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


def load_model(model_path, device='cpu'):
    """Load trained model from checkpoint"""

    print(f"Loading model from: {model_path}")

    # Initialize model
    model = AffinityPredictor(input_dim=150, hidden_dims=[256, 128], dropout=0.3)
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Model loaded successfully!")

    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"   Trained for {checkpoint['epoch']+1} epochs")
    if 'val_loss' in checkpoint:
        print(f"   Validation loss: {checkpoint['val_loss']:.4f}")
    if 'metrics' in checkpoint:
        print(f"   Metrics: {checkpoint['metrics']}")

    return model, checkpoint


def predict_single(model, features, device='cpu'):
    """Make prediction for a single sample"""

    model.eval()
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
        prediction = model(features_tensor).item()

    return prediction


def predict_batch(model, features_array, device='cpu', batch_size=128):
    """Make predictions for multiple samples"""

    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(0, len(features_array), batch_size):
            batch = features_array[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)
            batch_preds = model(batch_tensor).cpu().numpy()
            predictions.extend(batch_preds)

    return np.array(predictions)


def evaluate_model(model, data_path, device='cpu'):
    """Evaluate model on test dataset"""

    from sklearn.model_selection import train_test_split

    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)

    # Filter samples with features
    pca_cols = [f'esm2_pca_{i}' for i in range(150)]
    df_with_features = df[df[pca_cols[0]].notna()].copy()
    print(f"✅ Loaded {len(df_with_features):,} samples with features")

    # Extract features
    X = df_with_features[pca_cols].values
    y = df_with_features['pKd'].values

    # Same split as Colab (MUST use same random_state!)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    print(f"\nRunning evaluation on {len(X_test):,} test samples...")

    # Make predictions
    predictions = predict_batch(model, X_test, device=device)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    spearman = spearmanr(y_test, predictions)[0]
    pearson = pearsonr(y_test, predictions)[0]
    r2 = 1 - (np.sum((y_test - predictions)**2) / np.sum((y_test - y_test.mean())**2))

    # Print results
    print("\n" + "="*60)
    print("TEST SET PERFORMANCE")
    print("="*60)
    print(f"RMSE:        {rmse:.4f}")
    print(f"MAE:         {mae:.4f}")
    print(f"Spearman ρ:  {spearman:.4f}")
    print(f"Pearson r:   {pearson:.4f}")
    print(f"R²:          {r2:.4f}")
    print("="*60)

    # Per-bin metrics
    BINS = [0, 5, 7, 9, 11, 16]
    BIN_LABELS = ['very_weak', 'weak', 'moderate', 'strong', 'very_strong']

    test_df = pd.DataFrame({
        'target': y_test,
        'prediction': predictions
    })
    test_df['affinity_bin'] = pd.cut(test_df['target'], bins=BINS, labels=BIN_LABELS, include_lowest=True)

    print("\nPER-BIN PERFORMANCE:")
    print("="*60)
    print(f"{'Bin':<15} | {'Count':<8} | {'RMSE':<8} | {'MAE':<8}")
    print("-"*60)

    for label in BIN_LABELS:
        bin_data = test_df[test_df['affinity_bin'] == label]
        if len(bin_data) > 0:
            bin_rmse = np.sqrt(mean_squared_error(bin_data['target'], bin_data['prediction']))
            bin_mae = mean_absolute_error(bin_data['target'], bin_data['prediction'])
            print(f"{label:<15} | {len(bin_data):<8} | {bin_rmse:<8.4f} | {bin_mae:<8.4f}")

    print("="*60)

    return {
        'rmse': rmse,
        'mae': mae,
        'spearman': spearman,
        'pearson': pearson,
        'r2': r2,
        'predictions': predictions,
        'targets': y_test
    }


def main():
    parser = argparse.ArgumentParser(description='Use Colab-trained model locally')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint (.pth file)')
    parser.add_argument('--data', type=str, help='Path to dataset for evaluation (optional)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device to use (auto, cuda, or cpu)')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Check model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model file not found: {model_path}")
        print(f"\nMake sure you've downloaded the model from Google Drive!")
        return

    # Load model
    model, checkpoint = load_model(model_path, device=device)

    # If data provided, run evaluation
    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"❌ Error: Data file not found: {data_path}")
            return

        results = evaluate_model(model, data_path, device=device)
    else:
        print("\n✅ Model loaded and ready for inference!")
        print("\nTo evaluate on test data, run:")
        print(f"  python {__file__} --model {args.model} --data path/to/data.csv")

        print("\n" + "="*60)
        print("EXAMPLE: Using the model for prediction")
        print("="*60)
        print("""
from use_colab_model_locally import load_model, predict_single
import numpy as np

# Load model
model, checkpoint = load_model('path/to/best_model.pth')

# Prepare features (150-dimensional PCA features)
features = np.random.randn(150)  # Replace with actual features

# Make prediction
predicted_pKd = predict_single(model, features)
print(f"Predicted pKd: {predicted_pKd:.2f}")

# Convert to Kd
Kd_M = 10 ** (-predicted_pKd)
Kd_nM = Kd_M * 1e9
print(f"Predicted Kd: {Kd_nM:.2f} nM")
        """)


if __name__ == "__main__":
    main()
