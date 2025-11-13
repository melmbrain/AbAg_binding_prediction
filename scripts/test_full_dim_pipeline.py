#!/usr/bin/env python3
"""
Test Full-Dimensional Training Pipeline

This script validates that all components are ready for v3 training:
1. Data files exist
2. Feature dimensions are correct
3. Model architecture works
4. Small training run succeeds

Run this before uploading to Colab to catch issues early!
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from model_v3_full_dim import AffinityModelV3FullDim, get_model_v3
    print("✅ Model imports successful")
except ImportError as e:
    print(f"❌ Failed to import model: {e}")
    sys.exit(1)


def test_data_files():
    """Test 1: Check if required data files exist"""
    print("\n" + "="*60)
    print("TEST 1: Data Files")
    print("="*60)

    required_files = [
        'external_data/new_embeddings.npy',
        'external_data/new_embedding_indices.npy',
        'external_data/merged_with_therapeutics.csv'
    ]

    all_exist = True
    for file in required_files:
        if Path(file).exists():
            size_mb = os.path.getsize(file) / (1024**2)
            print(f"✅ {file} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {file} (NOT FOUND)")
            all_exist = False

    # Check full-dimensional output
    output_file = 'external_data/merged_with_full_features.csv'
    if Path(output_file).exists():
        size_mb = os.path.getsize(output_file) / (1024**2)
        print(f"✅ {output_file} ({size_mb:.1f} MB)")
    else:
        print(f"⚠️  {output_file} (NOT FOUND - needs to be generated)")

    return all_exist


def test_feature_dimensions():
    """Test 2: Validate feature dimensions in data"""
    print("\n" + "="*60)
    print("TEST 2: Feature Dimensions")
    print("="*60)

    # Check embeddings file
    embeddings_file = Path('external_data/new_embeddings.npy')
    if not embeddings_file.exists():
        print("❌ Embeddings file not found")
        return False

    try:
        embeddings = np.load(embeddings_file)
        print(f"Embeddings shape: {embeddings.shape}")

        if embeddings.shape[1] == 1280:
            print(f"✅ Correct dimensions: {embeddings.shape[1]} (expected 1,280)")
        else:
            print(f"❌ Wrong dimensions: {embeddings.shape[1]} (expected 1,280)")
            return False

        # Check memory requirements
        memory_gb = embeddings.nbytes / (1024**3)
        print(f"Memory usage: {memory_gb:.2f} GB")

        if memory_gb > 6:
            print(f"⚠️  Large memory footprint - ensure sufficient RAM")

        return True

    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return False


def test_model_architecture():
    """Test 3: Test model architecture"""
    print("\n" + "="*60)
    print("TEST 3: Model Architecture")
    print("="*60)

    try:
        # Test standard model
        print("\n1. Standard Model (1,280 → 512 → 256 → 128 → 64 → 1)")
        model = get_model_v3('standard', input_dim=1280)
        print(f"   Parameters: {model.count_parameters():,}")

        # Test forward pass
        x = torch.randn(16, 1280)  # Batch of 16
        y = model(x)
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {y.shape}")

        if y.shape == (16,):
            print("   ✅ Forward pass successful")
        else:
            print(f"   ❌ Wrong output shape: {y.shape}")
            return False

        # Test deep model
        print("\n2. Deep Model (1,280 → 1024 → 512 → 256 → 128 → 64 → 1)")
        model_deep = get_model_v3('deep', input_dim=1280)
        print(f"   Parameters: {model_deep.count_parameters():,}")

        y_deep = model_deep(x)
        if y_deep.shape == (16,):
            print("   ✅ Forward pass successful")
        else:
            print(f"   ❌ Wrong output shape: {y_deep.shape}")
            return False

        # Test attention model
        print("\n3. Attention Model")
        model_attn = get_model_v3('attention', input_dim=1280)
        print(f"   Parameters: {model_attn.count_parameters():,}")

        y_attn = model_attn(x)
        if y_attn.shape == (16,):
            print("   ✅ Forward pass successful")
        else:
            print(f"   ❌ Wrong output shape: {y_attn.shape}")
            return False

        return True

    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_availability():
    """Test 4: Check GPU availability and memory"""
    print("\n" + "="*60)
    print("TEST 4: GPU Availability")
    print("="*60)

    if torch.cuda.is_available():
        print(f"✅ CUDA available")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   VRAM: {vram_gb:.1f} GB")

        if vram_gb >= 15:
            print(f"   ✅ Sufficient VRAM for full-dimensional training")
        else:
            print(f"   ⚠️  Low VRAM - may need to reduce batch size")
            print(f"      Recommended: 16GB+ (Colab Pro)")

        return True
    else:
        print("⚠️  No GPU detected")
        print("   Full-dimensional training requires GPU")
        print("   This is OK if running locally (Colab will have GPU)")
        return False


def test_small_training_run():
    """Test 5: Run small training test"""
    print("\n" + "="*60)
    print("TEST 5: Small Training Run")
    print("="*60)

    try:
        # Create small synthetic dataset
        print("Creating synthetic dataset...")
        n_samples = 1000
        X = np.random.randn(n_samples, 1280).astype(np.float32)
        y = np.random.uniform(5, 11, n_samples).astype(np.float32)

        # Create tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        # Create model
        model = get_model_v3('standard', input_dim=1280)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        # Setup training
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

        # Train for a few iterations
        print(f"Training on {device}...")
        model.train()
        losses = []

        for i in range(10):
            # Random batch
            idx = np.random.choice(n_samples, 32, replace=False)
            batch_X = X_tensor[idx].to(device)
            batch_y = y_tensor[idx].to(device)

            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

            if i % 3 == 0:
                print(f"   Iteration {i+1}/10: Loss = {loss.item():.4f}")

        # Check if loss decreased
        if losses[-1] < losses[0]:
            print(f"✅ Training successful (loss decreased: {losses[0]:.4f} → {losses[-1]:.4f})")
            return True
        else:
            print(f"⚠️  Loss did not decrease (may need more iterations)")
            return True  # Still pass, just a warning

    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_requirements():
    """Test 6: Estimate memory requirements"""
    print("\n" + "="*60)
    print("TEST 6: Memory Requirements")
    print("="*60)

    # Model memory
    model = get_model_v3('standard', input_dim=1280)
    model_params = model.count_parameters()
    model_size_mb = (model_params * 4) / (1024**2)  # 4 bytes per float32
    print(f"Model size: ~{model_size_mb:.1f} MB ({model_params:,} parameters)")

    # Training batch memory
    batch_size = 96
    input_dim = 1280
    batch_memory_mb = (batch_size * input_dim * 4) / (1024**2)
    print(f"Batch memory (size={batch_size}): ~{batch_memory_mb:.1f} MB")

    # Gradients + optimizer states (roughly 3x model size)
    training_memory_mb = model_size_mb * 3
    print(f"Training overhead: ~{training_memory_mb:.1f} MB")

    # Total estimate
    total_vram_mb = model_size_mb + batch_memory_mb + training_memory_mb
    total_vram_gb = total_vram_mb / 1024
    print(f"\nEstimated VRAM usage: ~{total_vram_gb:.1f} GB")

    # Data file size
    data_file = Path('external_data/merged_with_full_features.csv')
    if data_file.exists():
        data_size_mb = os.path.getsize(data_file) / (1024**2)
        print(f"Data file size: {data_size_mb:.1f} MB")
    else:
        print(f"Data file: Not yet generated (~800-1000 MB expected)")

    # Recommendations
    print(f"\nRecommendations:")
    if total_vram_gb <= 12:
        print("  ✅ Will fit on Colab Pro T4 (16GB)")
    elif total_vram_gb <= 20:
        print("  ⚠️  Tight fit on T4, recommend L4 (24GB)")
    else:
        print("  ❌ Requires A100 (40GB)")

    return True


def main():
    """Run all tests"""
    print("="*60)
    print("FULL-DIMENSIONAL TRAINING PIPELINE VALIDATION")
    print("="*60)
    print("\nThis will test all components needed for v3 training")
    print("Run this BEFORE uploading to Colab to catch issues early!\n")

    results = {}

    # Run tests
    results['data_files'] = test_data_files()
    results['feature_dims'] = test_feature_dimensions()
    results['model_arch'] = test_model_architecture()
    results['gpu'] = test_gpu_availability()
    results['training'] = test_small_training_run()
    results['memory'] = test_memory_requirements()

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name.replace('_', ' ').title()}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Ready for v3 training on Colab Pro!")
        print("\nNext steps:")
        print("  1. Run: python scripts/prepare_full_dimensional_features.py")
        print("  2. Upload merged_with_full_features.csv to Google Drive")
        print("  3. Upload colab_training_v3_full_dimensions.ipynb to Colab")
        print("  4. Train for 12-15 hours")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix issues before training.")
        print("\nCommon fixes:")
        print("  - Missing data files: Run embedding generation scripts")
        print("  - Wrong dimensions: Re-generate embeddings with correct model")
        print("  - Model errors: Check src/model_v3_full_dim.py")
        print("  - GPU issues: Will be resolved on Colab (ignore if local)")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
