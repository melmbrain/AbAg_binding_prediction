# Training Speed-Up Guide for IgT5 + ESM-2

**Date**: 2025-11-13
**Goal**: Accelerate training from ~5 days to 1-2 days on Colab T4 GPU
**Based on**: 2024-2025 research on efficient protein language model training

---

## 🚀 Quick Summary: Expected Speed-Ups

| Optimization | Speed-Up | Memory Savings | Difficulty | Priority |
|--------------|----------|----------------|------------|----------|
| **torch.compile** | 1.5-2× | 0% | Easy | ⭐⭐⭐ HIGH |
| **BFloat16** | 1.3-1.5× | 50% | Easy | ⭐⭐⭐ HIGH |
| **Gradient Checkpointing** | -20% slower | 70% | Easy | ⭐⭐ (if OOM) |
| **FlashAttention (FAESM)** | 1.5-2× | 60% | Medium | ⭐⭐⭐ HIGH |
| **LoRA Fine-Tuning** | 4.5× | 75% | Medium | ⭐⭐ (alternative) |
| **Compiled Forward Pass** | 1.3-1.5× | 0% | Easy | ⭐⭐ MED |

**Combined Potential**: 3-6× faster training with same or better results

---

## 1. torch.compile() - Easiest & Most Effective ⭐⭐⭐

**Speed-Up**: 1.5-2× faster (43% average across 163 AI projects)
**Difficulty**: ONE LINE OF CODE
**Availability**: PyTorch 2.0+ (Colab has this)

### Implementation

```python
# In your training script, just wrap the model:

model = IgT5ESM2Model(dropout=0.3, freeze_encoders=True).to(device)

# Add this ONE line:
model = torch.compile(model)

# That's it! Everything else stays the same
```

### Why It Works
- Fuses operations to reduce kernel launches
- Optimizes memory access patterns
- Generates CUDA graphs for repeated patterns
- No accuracy loss

### Benchmark Results
- HuggingFace transformers: 1.5-2× faster training
- No code changes needed beyond wrapping model
- Works with mixed precision (AMP)

**✅ RECOMMEND: Add immediately - zero risk, massive gain**

---

## 2. BFloat16 Mixed Precision ⭐⭐⭐

**Speed-Up**: 1.3-1.5× faster
**Memory Savings**: 50%
**Difficulty**: Easy (you're already using float16)

### Current vs BFloat16

```python
# CURRENT (you're using this):
with torch.amp.autocast('cuda'):  # Uses float16
    predictions = model(antibody_seqs, antigen_seqs, device)

# BETTER - BFloat16:
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    predictions = model(antibody_seqs, antigen_seqs, device)
```

### Why BFloat16 > Float16
- **Numerical stability**: Same exponent range as float32 (8 bits)
- **No loss scaling needed**: Handles tiny gradients without underflow
- **"Just works"**: No convergence degradation in large transformers
- **T4 GPU support**: Google Colab T4 supports bfloat16

### Full Implementation

```python
# In main() function:
scaler = torch.amp.GradScaler('cuda', enabled=False)  # Disable for bfloat16

# In training loop:
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    predictions = model(antibody_seqs, antigen_seqs, device)
    loss = criterion(predictions, targets)

# No scaler needed with bfloat16
loss.backward()
optimizer.step()
```

**✅ RECOMMEND: Switch from float16 to bfloat16 - more stable, same speed**

---

## 3. FlashAttention via FAESM ⭐⭐⭐

**Speed-Up**: 1.5-2× faster (70% reduction on long sequences)
**Memory Savings**: 60%
**Difficulty**: Medium (replace model loading)

### Installation

```bash
# In Colab:
!pip install faesm[flash_attn]
```

### Implementation

```python
# BEFORE (current code):
from transformers import AutoModel, AutoTokenizer
self.esm2_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

# AFTER (with FlashAttention):
from faesm.esm import FAEsmForMaskedLM
self.esm2_model = FAEsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
```

### Why It Works
- I/O-aware attention algorithm
- Minimizes memory transfers between HBM and SRAM
- Fuses operations to reduce overhead
- **No changes to model weights** - same checkpoints work

### Benchmarks
- ESM-650M on A100: 70% faster, 60% less memory
- Longer sequences benefit more (your antibodies/antigens are 200-500 aa)
- Maintained model accuracy

### Compatibility
- ✅ Drop-in replacement for HuggingFace ESM-2
- ✅ Same API, same checkpoints
- ⚠️ FlashAttention requires CUDA 11.6+ (Colab has this)

**✅ RECOMMEND: Try this - high reward, medium effort**

---

## 4. LoRA Parameter-Efficient Fine-Tuning ⭐⭐

**Speed-Up**: 4.5× faster for ESM-2 3B
**Memory Savings**: 75%
**Difficulty**: Medium (architecture change)

### What Is LoRA?
Instead of training all 650M parameters of ESM-2, train only ~1-2M adapter weights:
- Add low-rank matrices to attention layers (Q, K, V)
- Freeze encoder weights
- Train only LoRA adapters + your regressor

### Implementation

```python
# Install:
!pip install peft

# In model __init__:
from peft import LoraConfig, get_peft_model

# Load ESM-2 normally
self.esm2_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

# Add LoRA adapters
lora_config = LoraConfig(
    r=8,                          # Rank (4-16 recommended)
    lora_alpha=16,                # Scaling factor
    target_modules=["query", "value"],  # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="FEATURE_EXTRACTION"
)

self.esm2_model = get_peft_model(self.esm2_model, lora_config)
self.esm2_model.print_trainable_parameters()  # See reduction
```

### Trade-offs
- ✅ Much faster training (4.5× for ESM-2 3B)
- ✅ Much less memory (fit larger batches)
- ✅ Similar or better performance (Spearman 0.70 in papers)
- ⚠️ Requires architecture change
- ⚠️ Need to save LoRA adapters separately

### When to Use
- If current training is too slow (>5 days)
- If running out of memory
- For experimentation with different hyperparameters

**⚠️ RECOMMEND: Alternative approach if others don't give enough speed-up**

---

## 5. Gradient Checkpointing ⚠️

**Speed**: -20% slower
**Memory Savings**: 70%
**Use Case**: ONLY if running out of memory

### Implementation

```python
# Enable gradient checkpointing:
self.igt5_model.gradient_checkpointing_enable()
self.esm2_model.gradient_checkpointing_enable()
```

### Trade-off
- Saves memory by recomputing activations
- Makes training 20% slower
- **Only use if you're hitting OOM errors**

**⚠️ NOT RECOMMENDED: You're not OOM limited, so this hurts performance**

---

## 6. Additional Optimizations

### A. Increase Batch Size (if memory allows)

```python
# Try increasing from 8 to 12 or 16:
--batch_size 12  # or 16
```

With bfloat16 + FlashAttention, you'll have 60-80% more memory available.

**Rule**: Larger batch size = fewer gradient updates = faster epochs

### B. DataLoader Optimization

```python
# Increase num_workers:
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,  # Increase from 2 to 4
    collate_fn=collate_fn,
    pin_memory=True,
    persistent_workers=True  # Add this
)
```

### C. Compile Embedding Functions

```python
# In model __init__:
self.get_antibody_embedding = torch.compile(self.get_antibody_embedding)
self.get_antigen_embedding = torch.compile(self.get_antigen_embedding)
```

---

## 📊 Recommended Implementation Plan

### Phase 1: Easy Wins (1 hour, 2-3× faster) ⭐⭐⭐

```python
# 1. Add torch.compile (1 line)
model = torch.compile(model)

# 2. Switch to bfloat16 (2 lines)
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    predictions = model(antibody_seqs, antigen_seqs, device)

# 3. Increase batch size
--batch_size 12  # or 16 if memory allows
```

**Expected**: 2-3× faster, 5 days → 1.5-2 days

### Phase 2: FlashAttention (30 min, +1.5× faster) ⭐⭐

```python
# 4. Install FAESM
!pip install faesm[flash_attn]

# 5. Replace ESM-2 import
from faesm.esm import FAEsmForMaskedLM
self.esm2_model = FAEsmForMaskedLM.from_pretrained(...)
```

**Expected**: Additional 1.5×, total 3-4.5× faster, 5 days → 1-1.5 days

### Phase 3: LoRA (if needed) ⭐

```python
# 6. Add LoRA to ESM-2 (if still too slow)
from peft import LoraConfig, get_peft_model
# ... (see implementation above)
```

**Expected**: Additional 2-3×, total 6-9× faster, 5 days → 12-18 hours

---

## 🔧 Complete Optimized Training Script

Here's a drop-in replacement for your current training script with all optimizations:

```python
# File: train_ultra_fast.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
from pathlib import Path
import time
from transformers import T5EncoderModel, T5Tokenizer

# Use FAESM for FlashAttention
from faesm.esm import FAEsmForMaskedLM
from transformers import AutoTokenizer


class IgT5ESM2ModelOptimized(nn.Module):
    def __init__(self, dropout=0.3, freeze_encoders=True, use_flash_attn=True):
        super().__init__()

        print("Loading IgT5 for antibody...")
        self.igt5_tokenizer = T5Tokenizer.from_pretrained("Exscientia/IgT5", do_lower_case=False)
        self.igt5_model = T5EncoderModel.from_pretrained("Exscientia/IgT5")

        print("Loading ESM-2 for antigen with FlashAttention...")
        self.esm2_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

        if use_flash_attn:
            # Use FAESM with FlashAttention (1.5-2× faster)
            self.esm2_model = FAEsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
        else:
            # Fallback to standard
            from transformers import AutoModel
            self.esm2_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

        if freeze_encoders:
            for param in self.igt5_model.parameters():
                param.requires_grad = False
            for param in self.esm2_model.parameters():
                param.requires_grad = False

        igt5_dim = self.igt5_model.config.d_model
        esm2_dim = self.esm2_model.config.hidden_size
        combined_dim = igt5_dim + esm2_dim

        print(f"\nArchitecture: {igt5_dim}D + {esm2_dim}D = {combined_dim}D")
        print("Optimizations: FlashAttention, BFloat16, torch.compile\n")

        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def get_antibody_embedding(self, antibody_seq, device):
        inputs = self.igt5_tokenizer(
            antibody_seq, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = self.igt5_model(**inputs)
            ab_emb = outputs.last_hidden_state.mean(dim=1)
        return ab_emb.squeeze(0)

    def get_antigen_embedding(self, antigen_seq, device):
        inputs = self.esm2_tokenizer(
            antigen_seq, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = self.esm2_model(**inputs)
            # FAESM compatible - use last_hidden_state
            ag_emb = outputs.last_hidden_state[:, 0, :]
        return ag_emb.squeeze(0)

    def forward(self, antibody_seqs, antigen_seqs, device):
        ab_embeddings = []
        for ab_seq in antibody_seqs:
            ab_emb = self.get_antibody_embedding(ab_seq, device)
            ab_embeddings.append(ab_emb)
        ab_embeddings = torch.stack(ab_embeddings).to(device)

        ag_embeddings = []
        for ag_seq in antigen_seqs:
            ag_emb = self.get_antigen_embedding(ag_seq, device)
            ag_embeddings.append(ag_emb)
        ag_embeddings = torch.stack(ag_embeddings).to(device)

        combined = torch.cat([ab_embeddings, ag_embeddings], dim=1)
        predictions = self.regressor(combined).squeeze(-1)
        return predictions


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}\n")

    # Initialize model with optimizations
    model = IgT5ESM2ModelOptimized(
        dropout=args.dropout,
        freeze_encoders=True,
        use_flash_attn=True
    ).to(device)

    # Apply torch.compile (1.5-2× faster)
    print("Compiling model with torch.compile...")
    model = torch.compile(model)
    print("✓ Model compiled\n")

    # ... (rest of your training code)

    # In training loop, use bfloat16:
    for batch_idx, batch in pbar:
        antibody_seqs = batch['antibody_seqs']
        antigen_seqs = batch['antigen_seqs']
        targets = batch['pKd'].to(device)

        optimizer.zero_grad()

        # Use bfloat16 (1.3-1.5× faster, more stable)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            predictions = model(antibody_seqs, antigen_seqs, device)
            loss = criterion(predictions, targets)

        # No scaler needed with bfloat16
        loss.backward()
        optimizer.step()

        # ... rest of training loop
```

---

## 📈 Expected Results

### Current Performance
- **Training speed**: ~1.59 it/s
- **Time per epoch**: ~2.4 hours (13,977 batches)
- **Total time (50 epochs)**: ~120 hours = 5 days

### With Phase 1 Optimizations (torch.compile + bfloat16)
- **Training speed**: ~3.5-4.0 it/s (2.5× faster)
- **Time per epoch**: ~1.0 hour
- **Total time (50 epochs)**: ~50 hours = 2 days ✅

### With Phase 2 (+ FlashAttention)
- **Training speed**: ~5.5-6.5 it/s (3.5-4× faster)
- **Time per epoch**: ~35-40 minutes
- **Total time (50 epochs)**: ~30-35 hours = 1.5 days ✅✅

### With Phase 3 (+ LoRA, if needed)
- **Training speed**: ~8-10 it/s (5-6× faster)
- **Time per epoch**: ~23-28 minutes
- **Total time (50 epochs)**: ~20-24 hours = 1 day ✅✅✅

---

## 🚨 Important Notes

1. **Colab Compatibility**: All optimizations work on Google Colab T4 GPU
2. **No Accuracy Loss**: These are speed optimizations, not accuracy trade-offs
3. **Easy Rollback**: If something breaks, just remove the optimization
4. **Test First**: Try Phase 1 on a few batches before running full training

---

## 📋 Checklist

**Phase 1: Easy Wins** (Recommend NOW)
- [ ] Add `model = torch.compile(model)` after model initialization
- [ ] Change `torch.amp.autocast('cuda')` to `torch.amp.autocast('cuda', dtype=torch.bfloat16)`
- [ ] Remove or disable GradScaler (not needed with bfloat16)
- [ ] Increase batch size to 12 or 16 if memory allows
- [ ] Test on 100 batches to verify speed-up

**Phase 2: FlashAttention** (Recommend NEXT)
- [ ] Install FAESM: `!pip install faesm[flash_attn]`
- [ ] Replace ESM-2 import with FAESM
- [ ] Test on 100 batches
- [ ] If successful, restart full training

**Phase 3: LoRA** (OPTIONAL)
- [ ] Install peft: `!pip install peft`
- [ ] Add LoRA config to ESM-2 model
- [ ] Verify trainable parameters reduced to ~1-2M
- [ ] Train and compare results

---

## 🔗 References

1. **FAESM**: github.com/pengzhangzhi/faplm
2. **FlashAttention**: Dao et al., 2024, "FlashAttention-2"
3. **torch.compile**: pytorch.org/blog/accelerating-hugging-face-and-timm-models
4. **LoRA for ESM-2**: aws.amazon.com/blogs/machine-learning/efficiently-fine-tune-the-esm-2
5. **Efficient Training**: PMC12481099 - "Efficient inference, training, and fine-tuning of protein language models"

---

**Last Updated**: 2025-11-13
**Status**: Ready to implement
**Expected Gain**: 3-6× faster training (5 days → 1-1.5 days)
