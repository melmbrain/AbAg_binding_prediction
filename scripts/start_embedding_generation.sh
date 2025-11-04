#!/bin/bash
# Start Background Embedding Generation with Checkpoint System
# This script starts ESM2 embedding generation in background

echo "================================================================================"
echo "           STARTING BACKGROUND EMBEDDING GENERATION"
echo "================================================================================"
echo ""
echo "This will generate ESM2 embeddings for:"
echo "   - 185,718 AbBiBench samples"
echo "   -      53 Therapeutic antibody samples"
echo "   ------------------------------------------------"
echo "   - 185,771 total samples"
echo ""
echo "Configuration:"
echo "   - Mode: CPU (zero GPU conflict)"
echo "   - Batch size: 16 sequences"
echo "   - Checkpoint: Every 50 batches"
echo "   - Auto-resume: YES"
echo ""

# Check if checkpoint already exists
if [ -f "external_data/embedding_checkpoint.pkl" ]; then
    echo "[INFO] Found existing checkpoint - will resume from previous progress"
    echo ""

    # Show progress
    python3 -c "import pickle; cp = pickle.load(open('external_data/embedding_checkpoint.pkl', 'rb')); print(f'Previous progress: {cp[\"last_index\"]:,} / 185,771 ({cp.get(\"progress_pct\", 0):.1f}%)')"
    echo ""

    read -p "Resume from checkpoint? (Y/N): " resume
    if [[ "$resume" =~ ^[Nn]$ ]]; then
        echo "[INFO] Removing old checkpoint..."
        rm external_data/embedding_checkpoint.pkl
        echo "[OK] Starting fresh"
    else
        echo "[OK] Resuming from checkpoint"
    fi
else
    echo "[INFO] No checkpoint found - starting fresh"
fi

echo ""
echo "================================================================================"
echo "Starting embedding generation in background..."
echo "================================================================================"
echo ""
echo "The process will run in background. You can:"
echo "   - Check progress: python3 scripts/check_embedding_progress.py"
echo "   - View log:       tail -f embedding_generation.log"
echo "   - Stop process:   pkill -f generate_embeddings_incremental"
echo ""

# Start in background and redirect output to log
nohup python3 scripts/generate_embeddings_incremental.py --use_cpu --batch_size 16 --save_every 50 > embedding_generation.log 2>&1 &

PID=$!

echo ""
echo "[OK] Embedding generation started in background (PID: $PID)"
echo "[INFO] Log file: embedding_generation.log"
echo "[INFO] Checkpoint file: external_data/embedding_checkpoint.pkl"
echo ""
echo "Estimated completion time: 1-2 days"
echo ""
echo "Check progress with: python3 scripts/check_embedding_progress.py"
echo ""
