#!/bin/bash

# Training Script for RTX 2060 (6GB VRAM)
# Optimized settings for your GPU

echo "============================================="
echo "Starting Training on RTX 2060"
echo "============================================="
echo ""

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Recommended settings for RTX 2060
BATCH_SIZE=8          # Balanced for 6GB VRAM
EPOCHS=50
NUM_WORKERS=2
MAX_LENGTH=512
FOCAL_GAMMA=2.0

# Data path
DATA_PATH="/mnt/c/Users/401-24/Desktop/Ab_Ag_dataset/data/agab/agab_phase2_full.csv"

# Output directory
OUTPUT_DIR="outputs_optimized_v1"

echo "Settings:"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Data: $DATA_PATH"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Expected:"
echo "  Training Time: ~4 hours"
echo "  Memory Usage: ~5.2GB / 6GB"
echo "  Recall@pKd≥9: 35-45%"
echo ""

read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Training cancelled."
    exit 1
fi

echo ""
echo "Starting training..."
echo "You can monitor GPU with: watch -n 1 nvidia-smi"
echo ""

# Run training
python train_optimized_v1.py \
  --data "$DATA_PATH" \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --use_stratified_sampling \
  --focal_gamma $FOCAL_GAMMA \
  --max_length $MAX_LENGTH \
  --num_workers $NUM_WORKERS \
  --output_dir "$OUTPUT_DIR"

# Check if training completed
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================="
    echo "Training Completed Successfully!"
    echo "============================================="
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Check results:"
    echo "  cat $OUTPUT_DIR/results.json"
    echo ""
    echo "View predictions:"
    echo "  head $OUTPUT_DIR/test_predictions.csv"
    echo ""
else
    echo ""
    echo "============================================="
    echo "Training Failed!"
    echo "============================================="
    echo ""
    echo "Common issues:"
    echo "  - Out of memory: Try --batch_size 4"
    echo "  - CUDA error: Update PyTorch"
    echo "  - Data not found: Check data path"
    echo ""
fi
