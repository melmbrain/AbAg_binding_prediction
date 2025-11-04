# Dual Computation Strategy Guide

**Problem:** Your GPU is busy training another deep learning model, making ESM2 embedding generation difficult.

**Solution:** Multiple strategies to work around GPU resource constraints.

---

## Strategy 1: Train with Existing Features ONLY (Immediate)

**Best for:** Getting results NOW without waiting for embeddings

### What It Does

- Uses only the 205k samples that already have ESM2 embeddings
- Skips the 185k AbBiBench + 53 therapeutic samples (no features yet)
- Applies improved class balancing methods (stratified sampling, class weights)
- **Zero GPU conflict** - no embedding generation needed

### How to Use

```bash
# Step 1: Filter dataset to samples with features
python scripts/train_with_existing_features.py

# Step 2: Train immediately
python train_balanced.py \
  --data external_data/train_ready_with_features.csv \
  --loss weighted_mse \
  --sampling stratified \
  --epochs 100
```

### Pros & Cons

✅ **Pros:**
- Train immediately (no waiting)
- No GPU resource conflict
- Still get improvement from better balancing methods
- 205k samples is substantial

❌ **Cons:**
- Doesn't use 185k AbBiBench samples
- Doesn't use 53 therapeutic antibodies
- Limited to original Phase 6 data

### Expected Results

- Very strong RMSE: ~2.2 → ~1.5 (32% improvement)
- Weak RMSE: ~2.5 → ~1.2 (52% improvement)
- From stratified sampling + class weights alone

---

## Strategy 2: Incremental Background Embedding Generation

**Best for:** Generating embeddings slowly in background while your main training runs

### What It Does

- Generates embeddings in small batches
- Auto-pauses when GPU memory is high
- Resumes automatically when GPU is available
- Saves checkpoints (can resume if interrupted)

### How to Use

```bash
# Option A: Auto-pause mode (waits for GPU availability)
python scripts/generate_embeddings_incremental.py \
  --batch_size 4 \
  --gpu_threshold 80.0 \
  --check_interval 300

# Option B: CPU mode (slower but no GPU conflict)
python scripts/generate_embeddings_incremental.py \
  --use_cpu \
  --batch_size 16
```

### Configuration Options

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--batch_size` | Sequences per batch | 4-8 (GPU), 16-32 (CPU) |
| `--gpu_threshold` | Pause if GPU usage above % | 80.0 |
| `--check_interval` | Seconds between GPU checks | 300 (5 min) |
| `--use_cpu` | Use CPU instead of GPU | For zero conflict |
| `--checkpoint_file` | Save progress file | Auto-save |
| `--save_every` | Checkpoint frequency | 100 batches |

### Pros & Cons

✅ **Pros:**
- Runs in background
- Automatic GPU conflict avoidance
- Resumes from checkpoint
- Eventually gets all embeddings

❌ **Cons:**
- Takes time (hours to days)
- Requires monitoring
- CPU mode is very slow

### Timeline Estimates

**GPU mode (batch_size=4, with pauses):**
- 185k samples × 2s/batch ÷ 4 samples/batch = ~26 hours
- With pauses: 2-4 days (depends on your main training)

**CPU mode (batch_size=16):**
- 185k samples × 10s/batch ÷ 16 samples/batch = ~32 hours
- Continuous: 1.5-2 days

---

## Strategy 3: Off-Peak Embedding Generation

**Best for:** If you have scheduled GPU availability

### What It Does

- Run embedding generation during off-peak hours
- E.g., overnight when main training is paused
- Use larger batches for faster processing

### How to Use

**Schedule with cron (Linux) or Task Scheduler (Windows):**

```bash
# Run overnight (11 PM - 7 AM)
# Windows Task Scheduler:
# - Create task to run at 11 PM
# - Action: python scripts/generate_embeddings_incremental.py --batch_size 16
# - Stop task at 7 AM

# Linux cron:
# 0 23 * * * cd /path/to/project && python scripts/generate_embeddings_incremental.py --batch_size 16
```

### Pros & Cons

✅ **Pros:**
- Faster (larger batches when GPU free)
- No daytime conflicts
- More efficient GPU use

❌ **Cons:**
- Requires scheduling
- May not complete in one night
- Needs checkpoint management

---

## Strategy 4: Split GPU Memory

**Best for:** If your main training doesn't use 100% GPU

### What It Does

- Limit GPU memory for embedding generation
- Run both processes simultaneously
- PyTorch memory fraction control

### How to Use

**In your embedding script:**

```python
# Limit ESM2 to 30% of GPU memory
import torch

# Set before loading model
torch.cuda.set_per_process_memory_fraction(0.3, device=0)

# Then load ESM2 model
from transformers import AutoModel
model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
```

**In your main training script:**

```python
# Use remaining 70% for main training
torch.cuda.set_per_process_memory_fraction(0.7, device=0)
```

### Pros & Cons

✅ **Pros:**
- Run both simultaneously
- Faster than CPU
- No scheduling needed

❌ **Cons:**
- May slow both processes
- Risk of OOM errors
- Requires memory tuning

---

## Strategy 5: Use Multiple GPUs (If Available)

**Best for:** If you have >1 GPU

### What It Does

- Main training on GPU 0
- Embedding generation on GPU 1
- Complete isolation

### How to Use

```bash
# Main training on GPU 0
CUDA_VISIBLE_DEVICES=0 python train.py

# Embedding generation on GPU 1 (separate terminal)
CUDA_VISIBLE_DEVICES=1 python scripts/generate_embeddings_incremental.py
```

### Pros & Cons

✅ **Pros:**
- Perfect isolation
- Both run at full speed
- No conflicts

❌ **Cons:**
- Requires multiple GPUs
- Not applicable if single GPU

---

## Recommended Workflow

### Phase 1: Immediate Training (TODAY)

```bash
# Filter to existing features
python scripts/train_with_existing_features.py

# Train with improved methods
python train_balanced.py \
  --data external_data/train_ready_with_features.csv \
  --loss weighted_mse \
  --sampling stratified
```

**Result:** Improved model from better balancing alone (205k samples)

### Phase 2: Background Embedding Generation (THIS WEEK)

**Option A - If GPU sometimes available:**
```bash
# Run in background with auto-pause
python scripts/generate_embeddings_incremental.py \
  --batch_size 4 \
  --gpu_threshold 80.0 \
  --check_interval 300
```

**Option B - If GPU always busy:**
```bash
# Use CPU (slow but zero conflict)
nohup python scripts/generate_embeddings_incremental.py --use_cpu &
```

### Phase 3: Full Dataset Training (NEXT WEEK)

```bash
# Once embeddings generated
python train_balanced.py \
  --data external_data/merged_with_all_features.csv \
  --loss weighted_mse \
  --sampling stratified
```

**Result:** Best model with full 390k samples

---

## Resource Usage Comparison

| Strategy | GPU Usage | Time | Dataset Size | Conflict Risk |
|----------|-----------|------|--------------|---------------|
| **Existing features only** | 0% (training only) | 0 hours | 205k | None |
| **Incremental (GPU)** | 10-30% intermittent | 26-96 hours | 390k | Low |
| **Incremental (CPU)** | 0% | 32-48 hours | 390k | None |
| **Off-peak** | 80-100% nights | 2-3 nights | 390k | Low |
| **Split GPU** | 30-40% constant | 12-24 hours | 390k | Medium |
| **Multi-GPU** | 100% GPU1 | 8-12 hours | 390k | None |

---

## Quick Decision Tree

```
Do you need results TODAY?
├─ YES → Use Strategy 1 (existing features only)
└─ NO → Continue...

Do you have multiple GPUs?
├─ YES → Use Strategy 5 (multi-GPU)
└─ NO → Continue...

Is your GPU ever <80% used?
├─ YES → Use Strategy 2 (incremental with auto-pause)
└─ NO → Continue...

Can you run overnight?
├─ YES → Use Strategy 3 (off-peak)
└─ NO → Use Strategy 2 with CPU flag
```

---

## Commands Summary

### Immediate Training (No Embeddings Needed)
```bash
python scripts/train_with_existing_features.py
python train_balanced.py --data external_data/train_ready_with_features.csv
```

### Background Embedding (Auto-Pause)
```bash
python scripts/generate_embeddings_incremental.py --gpu_threshold 80.0
```

### Background Embedding (CPU Only)
```bash
python scripts/generate_embeddings_incremental.py --use_cpu
```

### Overnight Embedding
```bash
# Schedule to run 11 PM - 7 AM
python scripts/generate_embeddings_incremental.py --batch_size 16
```

---

## Monitoring Progress

### Check Embedding Progress
```bash
# View checkpoint
python -c "
import pickle
with open('external_data/embedding_checkpoint.pkl', 'rb') as f:
    cp = pickle.load(f)
    print(f'Processed: {cp[\"last_index\"]} / 185771')
    print(f'Progress: {cp[\"last_index\"]/185771*100:.1f}%')
"
```

### Check GPU Usage
```bash
# Linux/WSL
nvidia-smi

# Or use Python
python -c "import torch; print(f'GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB')"
```

---

## My Recommendation for Your Situation

**Given that your GPU is busy with other training:**

1. **TODAY**: Use Strategy 1 (existing features)
   - Get immediate results
   - Train on 205k samples with improved methods
   - Zero GPU conflict

2. **THIS WEEK**: Use Strategy 2 with CPU flag
   - Run in background: `nohup python scripts/generate_embeddings_incremental.py --use_cpu &`
   - Takes 1-2 days but zero GPU conflict
   - Check progress occasionally

3. **NEXT WEEK**: Retrain with full dataset
   - Once embeddings complete
   - Full 390k samples with all features
   - Best possible model

**This gives you:**
- ✅ Immediate improved model (today)
- ✅ Zero GPU conflicts (all week)
- ✅ Eventually get best model (next week)

---

**Files Created:**
- `scripts/train_with_existing_features.py` - Filter and train on existing features
- `scripts/generate_embeddings_incremental.py` - Background embedding generation
- `DUAL_COMPUTATION_GUIDE.md` - This guide

**Next Step:** Run `python scripts/train_with_existing_features.py` to start training immediately!
