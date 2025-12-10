#!/bin/bash
# Optimized Training Configuration
# Based on analysis of your 50-epoch training results
# Expected to prevent overfitting and achieve better final performance

echo "=============================================================================="
echo "OPTIMIZED TRAINING RUN - v2.6 with Early Stopping"
echo "=============================================================================="
echo ""
echo "Key improvements over previous run:"
echo "  ✓ Early stopping (patience=10) to prevent overfitting"
echo "  ✓ Validation every epoch for better tracking"
echo "  ✓ Slightly lower learning rate for stability"
echo "  ✓ Higher dropout (0.35) for regularization"
echo "  ✓ Gradient clipping (max_norm=1.0) for stability"
echo "  ✓ Label smoothing (0.05) for better generalization"
echo "  ✓ Increased weight decay (0.02) for L2 regularization"
echo ""
echo "Expected results:"
echo "  • Training will stop automatically when performance plateaus"
echo "  • Should achieve ~0.42-0.45 Spearman (vs 0.42 previously)"
echo "  • Will maintain 100% Recall@pKd≥9"
echo "  • Estimated time: 1.5-2.5 hours (vs 3 hours for 50 epochs)"
echo ""
echo "=============================================================================="
echo ""

python train_ultra_speed_v26.py \
  --data agab_phase2_full.csv \
  --output_dir output_optimized \
  --epochs 50 \
  --batch_size 16 \
  --accumulation_steps 3 \
  --lr 3e-3 \
  --weight_decay 0.02 \
  --dropout 0.35 \
  --focal_gamma 2.0 \
  --save_every_n_batches 500 \
  --num_workers 4 \
  --prefetch_factor 4 \
  --validation_frequency 1 \
  --use_bfloat16 True \
  --use_compile False \
  --use_fused_optimizer True \
  --use_quantization True \
  --use_checkpointing True \
  --use_bucketing True \
  --use_early_stopping True \
  --early_stopping_patience 10 \
  --early_stopping_min_delta 0.0001 \
  --max_grad_norm 1.0 \
  --label_smoothing 0.05 \
  --l1_lambda 0.0 \
  --warmup_epochs 5

echo ""
echo "=============================================================================="
echo "Training complete! Next steps:"
echo "  1. python visualize_training.py --csv output_optimized/training_metrics.csv"
echo "  2. python find_best_epoch.py --checkpoint_dir output_optimized --plot"
echo "=============================================================================="
