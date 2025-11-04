# Quick Start: Background Embedding Generation

**Status:** Ready to start
**GPU Conflict:** SOLVED (uses CPU)
**Auto-Resume:** YES (checkpoint system)

---

## Quick Start (30 seconds)

### Windows

```bash
# Double-click or run:
scripts\start_embedding_generation.bat
```

### Linux/WSL

```bash
# Make executable (first time only):
chmod +x scripts/start_embedding_generation.sh

# Run:
./scripts/start_embedding_generation.sh
```

That's it! The process runs in background for 1-2 days.

---

## What It Does

Generates ESM2 embeddings for **185,771 samples** that currently don't have features:
- 185,718 AbBiBench samples
- 53 Therapeutic antibodies

**Configuration:**
- Mode: CPU (zero GPU conflict)
- Batch size: 16 sequences
- Checkpoint: Every 50 batches
- Auto-resume: YES
- Timeline: 1-2 days

---

## Monitoring Progress

### Check Progress Anytime

```bash
python scripts/check_embedding_progress.py
```

**Output:**
```
================================================================================
EMBEDDING GENERATION PROGRESS
================================================================================

Metric                    | Value
--------------------------------------------------------------------------------
Status                    | IN_PROGRESS
Samples processed         | 5,000 / 185,771
Progress                  | 2.69%
Samples remaining         | 180,771
Last updated              | 2025-11-03T10:45:23

[███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 2.7%
```

### View Live Log (Linux/WSL)

```bash
tail -f embedding_generation.log
```

### View Log (Windows)

```bash
type embedding_generation.log
```

---

## Checkpoint System Features

### ✅ Automatic Checkpoint Saving

- Saves every 50 batches (~800 samples)
- Saves progress, embeddings, and timestamp
- Atomic write (no corruption if interrupted)

### ✅ Auto-Resume

If interrupted (computer restart, power loss, manual stop):

```bash
# Just run the start script again:
scripts\start_embedding_generation.bat

# It will automatically detect the checkpoint and resume!
```

### ✅ Progress Tracking

Checkpoint includes:
- Number of samples processed
- All embeddings generated so far
- Progress percentage
- Timestamp

### ✅ Safe to Stop Anytime

Stop the process safely:

**Windows:**
```bash
# Find and kill the process
tasklist | findstr python
taskkill /F /PID <PID>
```

**Linux/WSL:**
```bash
# Kill the background process
pkill -f generate_embeddings_incremental
```

Your progress is saved! Just restart with the same command.

---

## Timeline & Performance

### Expected Performance (CPU Mode)

- Batch size: 16 sequences
- Time per batch: ~10-15 seconds
- Total batches: ~11,611 batches
- Total time: 32-48 hours

**Breakdown:**
```
185,771 samples ÷ 16 samples/batch = 11,611 batches
11,611 batches × 12s/batch = 139,332 seconds
139,332 seconds ÷ 3600s/hour = 38.7 hours
```

### Checkpoint Frequency

- Saves every 50 batches
- 50 batches × 16 samples = 800 samples per checkpoint
- Total checkpoints: ~233

**If interrupted:**
- Maximum work lost: 50 batches (~10 minutes)

---

## Files Created During Process

| File | Size | Description |
|------|------|-------------|
| `embedding_generation.log` | Grows | Live log output |
| `external_data/embedding_checkpoint.pkl` | ~500 MB | Progress checkpoint |
| `external_data/new_embeddings.npy` | ~1.5 GB | Raw embeddings (final) |
| `external_data/new_embedding_indices.npy` | ~1 MB | Sample indices (final) |

---

## Checking Status

### Method 1: Progress Script

```bash
python scripts/check_embedding_progress.py
```

Shows:
- Samples processed
- Progress percentage
- Progress bar
- Last 5 log lines
- File sizes

### Method 2: Check Checkpoint File

```bash
python -c "
import pickle
with open('external_data/embedding_checkpoint.pkl', 'rb') as f:
    cp = pickle.load(f)
    print(f'Progress: {cp[\"last_index\"]:,} / 185,771')
    print(f'Percent: {cp.get(\"progress_pct\", 0):.1f}%')
    print(f'Updated: {cp.get(\"timestamp\")}')
"
```

### Method 3: Check Process Status

**Windows:**
```bash
tasklist | findstr python
```

**Linux/WSL:**
```bash
ps aux | grep generate_embeddings
```

---

## Troubleshooting

### Process Not Running

**Check if it started:**
```bash
# Windows
tasklist | findstr python

# Linux
ps aux | grep generate_embeddings
```

**If not running:**
- Check `embedding_generation.log` for errors
- Make sure `transformers` library is installed: `pip install transformers`
- Check disk space: Need ~2 GB free

### Checkpoint Corrupted

**Rare but possible if interrupted during checkpoint save:**

```bash
# Remove corrupted checkpoint
rm external_data/embedding_checkpoint.pkl
rm external_data/embedding_checkpoint.pkl.tmp

# Restart (will start from beginning)
scripts/start_embedding_generation.bat
```

### Too Slow

**CPU mode is expected to be slow (1-2 days)**

To speed up (if GPU available):
```bash
python scripts/generate_embeddings_incremental.py \
  --batch_size 32 \
  --gpu_threshold 80.0
```

This will use GPU when available but auto-pause when busy.

### Out of Memory (CPU)

Reduce batch size:
```bash
python scripts/generate_embeddings_incremental.py \
  --use_cpu \
  --batch_size 8
```

---

## After Completion

Once embedding generation completes (100%):

### 1. Check Completion

```bash
python scripts/check_embedding_progress.py
```

Should show: `Status: COMPLETE`

### 2. Apply PCA Transformation

```bash
python scripts/apply_pca_and_merge.py
```

This will:
- Load your existing PCA model (150 components)
- Transform new embeddings
- Merge with original dataset
- Create: `external_data/merged_with_all_features.csv`

### 3. Train with Full Dataset

```bash
python train_balanced.py \
  --data external_data/merged_with_all_features.csv \
  --loss weighted_mse \
  --sampling stratified \
  --epochs 100
```

---

## Advanced Usage

### Custom Configuration

```bash
python scripts/generate_embeddings_incremental.py \
  --use_cpu \
  --batch_size 16 \
  --save_every 50 \
  --checkpoint_file external_data/my_checkpoint.pkl
```

### GPU Mode with Auto-Pause

```bash
python scripts/generate_embeddings_incremental.py \
  --batch_size 8 \
  --gpu_threshold 75.0 \
  --check_interval 300
```

Auto-pauses when GPU usage > 75%, checks every 5 minutes.

### Resume from Specific Checkpoint

```bash
python scripts/generate_embeddings_incremental.py \
  --use_cpu \
  --checkpoint_file external_data/embedding_checkpoint.pkl
```

Automatically resumes from checkpoint if it exists.

---

## Summary

**To Start:**
```bash
scripts\start_embedding_generation.bat  # Windows
./scripts/start_embedding_generation.sh  # Linux
```

**To Check Progress:**
```bash
python scripts/check_embedding_progress.py
```

**To Stop (Safe):**
```bash
# Windows
taskkill /F /PID <PID>

# Linux
pkill -f generate_embeddings_incremental
```

**To Resume:**
```bash
# Just run the start script again - auto-resumes!
scripts\start_embedding_generation.bat
```

---

## What Makes This Different from Before

**Old approach (would fail with your GPU busy):**
- Generate all embeddings at once
- Requires GPU availability
- No checkpoints → lose all progress if interrupted
- Conflicts with your other training

**New approach (works perfectly):**
- ✅ Uses CPU (zero GPU conflict)
- ✅ Checkpoint every 50 batches (max 10 min work lost)
- ✅ Auto-resume from checkpoint
- ✅ Safe to stop/restart anytime
- ✅ Runs in background (1-2 days)
- ✅ Monitor progress anytime

---

## Ready to Start?

```bash
# Just run this:
scripts\start_embedding_generation.bat

# Check progress:
python scripts/check_embedding_progress.py
```

Your embedding generation will run in background for 1-2 days while you do other work!

---

**Created:** 2025-11-03
**Status:** Ready to run
**Estimated Time:** 1-2 days
**GPU Conflict:** None (uses CPU)
