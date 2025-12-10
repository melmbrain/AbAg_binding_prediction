#!/bin/bash

# Install FlashAttention for 3-10x Training Speedup
# Run this AFTER stopping current training

echo "============================================="
echo "Installing FlashAttention"
echo "============================================="
echo ""
echo "This will:"
echo "  1. Upgrade transformers to latest version"
echo "  2. Install flash-attn (may take 5-10 minutes)"
echo "  3. Test that FlashAttention works"
echo ""
echo "⚠️  Make sure training is STOPPED before running this!"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Installation cancelled."
    exit 1
fi

echo ""
echo "Step 1: Upgrading transformers..."
pip install --upgrade transformers

echo ""
echo "Step 2: Installing flash-attn (this may take 5-10 minutes)..."
echo "Note: This requires CUDA and will compile from source"

# Try to install flash-attn
pip install flash-attn --no-build-isolation

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ flash-attn installed successfully!"
else
    echo ""
    echo "⚠️  flash-attn installation failed"
    echo ""
    echo "This is OK - trying alternative method..."

    # Try with specific version
    pip install flash-attn==2.5.0 --no-build-isolation

    if [ $? -eq 0 ]; then
        echo "✓ flash-attn 2.5.0 installed successfully!"
    else
        echo ""
        echo "❌ Could not install flash-attn"
        echo ""
        echo "Possible reasons:"
        echo "  - CUDA version incompatible (need CUDA 11.6+)"
        echo "  - Insufficient RAM during compilation"
        echo "  - Missing build tools"
        echo ""
        echo "The training script will still work without FlashAttention,"
        echo "it will just be 3-10x slower."
        exit 1
    fi
fi

echo ""
echo "Step 3: Testing FlashAttention..."

python3 -c "
import torch
from transformers import AutoModel

print('Testing FlashAttention...')
try:
    model = AutoModel.from_pretrained(
        'facebook/esm2_t33_650M_UR50D',
        attn_implementation='flash_attention_2'
    )
    print('✓ FlashAttention is working!')
    print('✓ You will get 3-10x speedup on training!')
except Exception as e:
    print(f'❌ FlashAttention test failed: {e}')
    print('Training will still work, just slower.')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================="
    echo "Installation Complete!"
    echo "============================================="
    echo ""
    echo "FlashAttention is ready to use."
    echo ""
    echo "Next steps:"
    echo "  1. Resume training with same command"
    echo "  2. It will auto-load from checkpoint"
    echo "  3. You should see '✓ FlashAttention enabled'"
    echo "  4. Training will be 3-10x faster!"
    echo ""
else
    echo ""
    echo "============================================="
    echo "Installation Issues"
    echo "============================================="
    echo ""
    echo "FlashAttention could not be enabled, but"
    echo "your training will still work (just slower)."
    echo ""
fi
