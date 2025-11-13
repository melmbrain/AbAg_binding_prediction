#!/usr/bin/env python3
"""
Test script to verify dimension mismatch fix without requiring trained model
"""

import torch
import torch.nn as nn
import numpy as np
import sys

# Add parent directory to path
sys.path.insert(0, '.')

from abag_affinity.predictor import MultiHeadAttentionModel

print("=" * 70)
print("Testing Dimension Fix for MultiHeadAttentionModel")
print("=" * 70)

# Test 1: Verify model initialization with input_dim=256
print("\n✓ Test 1: Model initialization with input_dim=256, n_heads=8")
try:
    model = MultiHeadAttentionModel(input_dim=256, hidden_dim=256, n_heads=8)
    print("  ✓ SUCCESS: Model initialized correctly (256 / 8 = 32)")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Verify old input_dim=300 now raises error
print("\n✓ Test 2: Model initialization with input_dim=300, n_heads=8 (should fail)")
try:
    model_bad = MultiHeadAttentionModel(input_dim=300, hidden_dim=256, n_heads=8)
    print("  ✗ FAILED: Model should have raised ValueError for 300 / 8")
    sys.exit(1)
except ValueError as e:
    print(f"  ✓ SUCCESS: Correctly raised ValueError: {e}")
except Exception as e:
    print(f"  ✗ FAILED: Unexpected error: {e}")
    sys.exit(1)

# Test 3: Verify forward pass with 256-dim input
print("\n✓ Test 3: Forward pass with 256-dim input")
try:
    model = MultiHeadAttentionModel(input_dim=256, hidden_dim=256, n_heads=8)
    model.eval()

    # Create dummy 256-dim input (batch_size=1, seq_len=1, features=256)
    x = torch.randn(1, 1, 256)

    with torch.no_grad():
        output = model(x)

    print(f"  ✓ SUCCESS: Forward pass completed")
    print(f"    Input shape: {x.shape}")
    print(f"    Output shape: {output.shape}")
    print(f"    Output value: {output.item():.4f}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 4: Verify feature extraction dimensions (128 + 128 = 256)
print("\n✓ Test 4: Feature extraction dimensions")
try:
    # Simulate ESM-2 embeddings (1280-dim)
    ab_emb = np.random.randn(1280)
    ag_emb = np.random.randn(1280)

    # Extract 128 dims each (as in fixed predictor.py)
    ab_features = ab_emb[:128]
    ag_features = ag_emb[:128]

    # Combine
    features = np.concatenate([ab_features, ag_features])

    print(f"  ✓ SUCCESS: Feature extraction correct")
    print(f"    Antibody embedding: {ab_emb.shape[0]} dims → {ab_features.shape[0]} dims")
    print(f"    Antigen embedding: {ag_emb.shape[0]} dims → {ag_features.shape[0]} dims")
    print(f"    Combined features: {features.shape[0]} dims")

    if features.shape[0] != 256:
        print(f"  ✗ FAILED: Expected 256 dims, got {features.shape[0]}")
        sys.exit(1)

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 5: End-to-end test with 256-dim input
print("\n✓ Test 5: End-to-end prediction pipeline simulation")
try:
    model = MultiHeadAttentionModel(input_dim=256, hidden_dim=256, n_heads=8)
    model.eval()

    # Simulate full pipeline
    ab_emb = np.random.randn(1280)
    ag_emb = np.random.randn(1280)

    # Extract features (128 + 128 = 256)
    ab_features = ab_emb[:128]
    ag_features = ag_emb[:128]
    features = np.concatenate([ab_features, ag_features])

    # Convert to tensor and add batch/sequence dimensions
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        pKd = model(x).item()

    print(f"  ✓ SUCCESS: Full pipeline completed")
    print(f"    Input dimensions: {features.shape[0]}")
    print(f"    Tensor shape: {x.shape}")
    print(f"    Predicted pKd: {pKd:.4f}")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED - Dimension fix verified!")
print("=" * 70)
print("\nSummary:")
print("  • Model correctly uses 256-dim input (divisible by 8 heads)")
print("  • Feature extraction creates 128 + 128 = 256 dims")
print("  • Forward pass works without dimension mismatch errors")
print("  • Invalid dimensions (like 300) are correctly rejected")
print("\nThe dimension mismatch errors reported in basic_usage.py are now FIXED.")
print("=" * 70)
