#!/usr/bin/env python3
"""
Enhance training output to show comprehensive validation metrics
"""
import json
import re

# Read notebook
with open('C:/Users/401-24/Desktop/AbAg_binding_prediction/notebooks/colab_training_v2.7.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and modify cell 22 (training loop)
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

    if 'v2.7 Training Loop with REAL-TIME Prediction Monitoring' in src:
        print(f"Found training loop at cell {i}")

        # Find the section to replace using regex pattern
        pattern = r'(    # Show sample validation predictions.*?)(    # Step scheduler with validation Spearman)'

        new_section = '''    # Show sample validation predictions
    print(f"\\n  Validation samples (first 10):")
    for i in range(min(10, len(val_preds))):
        print(f"    True: {val_targets[i]:.2f} → Pred: {val_preds[i]:.2f}")

    elapsed = time.time() - start_time
    current_lr = optimizer.param_groups[0]['lr']

    # v2.7 CHANGE 6: Overfitting monitoring
    avg_train_loss = total_loss / max(batches_processed, 1)
    val_loss = metrics['rmse']  # Use RMSE as proxy for val loss
    overfit_ratio = val_loss / avg_train_loss if avg_train_loss > 0 else 1.0

    # Prediction distribution
    pred_mean = np.mean(val_preds)
    pred_std = np.std(val_preds)
    pred_min = np.min(val_preds)
    pred_max = np.max(val_preds)

    # Log epoch metrics to TensorBoard
    writer.add_scalar('Train/EpochLoss', avg_train_loss, epoch)
    writer.add_scalar('Val/Spearman', metrics['spearman'], epoch)
    writer.add_scalar('Val/Recall', metrics['recall'], epoch)
    writer.add_scalar('Val/Precision', metrics['precision'], epoch)
    writer.add_scalar('Val/RMSE', metrics['rmse'], epoch)
    writer.add_scalar('Val/MAE', metrics['mae'], epoch)
    writer.add_scalar('Val/Pearson', metrics['pearson'], epoch)
    writer.add_scalar('Val/R2', metrics['r2'], epoch)
    writer.add_scalar('Val/OverfitRatio', overfit_ratio, epoch)
    writer.add_scalar('Val/PredMean', pred_mean, epoch)
    writer.add_scalar('Val/PredStd', pred_std, epoch)
    writer.add_histogram('Val/Predictions', val_preds, epoch)
    writer.add_histogram('Val/Targets', val_targets, epoch)

    # ENHANCED OUTPUT: Show comprehensive metrics
    print()
    print("="*80)
    print(f"EPOCH {epoch+1}/{EPOCHS} COMPLETE - Training Time: {elapsed:.1f}s")
    print("="*80)

    print("\\nTRAINING METRICS:")
    print(f"  Train Loss:    {avg_train_loss:.4f}")
    print(f"  Learning Rate: {current_lr:.2e}")

    print("\\nVALIDATION METRICS:")
    print(f"  Val Loss (RMSE): {metrics['rmse']:.4f}")
    print(f"  MAE:             {metrics['mae']:.4f}")
    print(f"  R2:              {metrics['r2']:.4f}")

    print("\\nCORRELATION METRICS:")
    print(f"  Spearman:  {metrics['spearman']:.4f}", end="")
    if metrics['spearman'] > best_spearman:
        print(" <- NEW BEST!")
    else:
        print(f" (best: {best_spearman:.4f})")
    print(f"  Pearson:   {metrics['pearson']:.4f}")

    print("\\nCLASSIFICATION @ pKd>=9 (HIGH AFFINITY):")
    print(f"  Recall:    {metrics['recall']:.1f}% (how many strong binders we catch)")
    print(f"  Precision: {metrics['precision']:.1f}% (how accurate our predictions are)")

    print("\\nPREDICTION DISTRIBUTION:")
    print(f"  Range: [{pred_min:.2f}, {pred_max:.2f}]")
    print(f"  Mean:  {pred_mean:.2f} +/- {pred_std:.2f}")

    print("\\nOVERFITTING CHECK:")
    print(f"  Val/Train Loss Ratio: {overfit_ratio:.2f}x", end="")
    if overfit_ratio > 3.0:
        print(" <- WARNING: Overfitting detected!")
    elif overfit_ratio > 2.0:
        print(" <- Possible overfitting")
    else:
        print(" <- Good")

    # v2.7 Note: Predictions should now be in valid range
    if pred_min < 4.0 or pred_max > 14.0:
        print(f"\\nWARNING: Predictions outside valid range [4.0, 14.0]!")

    print("="*80)

    '''

        # Replace using regex
        src_modified = re.sub(pattern, new_section + '\\2', src, flags=re.DOTALL)

        if src_modified != src:
            # Update cell source (convert back to list format)
            cell['source'] = src_modified.split('\n')
            # Add newline to each line except last
            cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]

            print("  Cell updated successfully with enhanced output!")
        else:
            print("  WARNING: Could not find pattern to replace")

# Save notebook
with open('C:/Users/401-24/Desktop/AbAg_binding_prediction/notebooks/colab_training_v2.7.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\nNotebook saved with enhanced training output!")
print("\nNew epoch output format:")
print("="*80)
print("EPOCH X/50 COMPLETE - Training Time: XXXs")
print("="*80)
print("\nTRAINING METRICS:")
print("  Train Loss:    X.XXXX")
print("  Learning Rate: X.XXe-XX")
print("\nVALIDATION METRICS:")
print("  Val Loss (RMSE): X.XXXX")
print("  MAE:             X.XXXX")
print("  R2:              X.XXXX")
print("\nCORRELATION METRICS:")
print("  Spearman:  X.XXXX <- NEW BEST!")
print("  Pearson:   X.XXXX")
print("\nCLASSIFICATION @ pKd>=9 (HIGH AFFINITY):")
print("  Recall:    XX.X% (how many strong binders we catch)")
print("  Precision: XX.X% (how accurate our predictions are)")
print("\nPREDICTION DISTRIBUTION:")
print("  Range: [X.XX, XX.XX]")
print("  Mean:  X.XX +/- X.XX")
print("\nOVERFITTING CHECK:")
print("  Val/Train Loss Ratio: X.XXx <- Good")
print("="*80)
