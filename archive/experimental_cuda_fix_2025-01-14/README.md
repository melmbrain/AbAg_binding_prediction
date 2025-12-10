# Experimental CUDA Graphs Fix - Archived 2025-01-14

This folder contains experimental files created during debugging of CUDA graphs errors.

## The Problem

Training was crashing with:
```
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been
overwritten by a subsequent run
```

**Root cause**: `torch.compile` + `activation checkpointing` conflict

## The Solution

**Nuclear fix** applied to `train_ultra_speed_v26.py` (lines 28-43):
- Global `torch.compiler.disable()` at import time
- Environment variables to disable CUDA graphs
- `use_compile=False` in config
- Kept activation checkpointing enabled

## Files Archived Here

### Experimental Python Scripts
- `COLAB_TRAINING_FIXED.py` - Early fix attempt
- `COLAB_SINGLE_CELL_COMPLETE.py` - Single-cell version
- `EMERGENCY_FIX.py` - Auto-patch script
- `train_ultra_speed_v26_COLAB_FIXED.py` - Another fix attempt

### Documentation Files
- `COLAB_FIXES_APPLIED.md` - Fix documentation
- `COLAB_NOTEBOOK_QUICK_GUIDE.md` - Quick guide
- `CUDA_GRAPHS_FIX_COMPLETE.md` - Complete fix docs
- `FINAL_FIX_APPLIED.md` - Final fix documentation
- `TROUBLESHOOTING_CUDA_GRAPHS.md` - Troubleshooting guide
- `V26_IMPLEMENTATION_COMPLETE.md` - Implementation docs
- `SIMPLE_COLAB_INSTRUCTIONS.md` - Simple instructions
- `SIMPLE_SETUP.md` - Simple setup guide
- `ADVANCED_OPTIMIZATIONS_V26.md` - Optimization details
- `TRAINING_SPEEDUP_STRATEGY.md` - Speedup strategy

### Configuration Files
- `SAFE_CONFIG.txt` - Safe fallback config
- `SWITCH_TO_V26_NOW.txt` - Migration guide
- `COLAB_TRAINING_FIX.txt` - Training fix notes
- `COLAB_SINGLE_CELL_V26.txt` - Single cell notes

## What Worked

The final working solution in `train_ultra_speed_v26.py`:

1. **Nuclear fix at import** (lines 28-43)
2. **use_compile=False** (line 938)
3. **use_checkpointing=True** (line 941)
4. **Batch size 16** (line 927)

Result: **4.45 it/s, no errors, ~21-22 hours for 50 epochs**

## Key Lessons

1. `type=bool` in argparse doesn't work - use `type=lambda x: x.lower() == 'true'`
2. Global `torch.compiler.disable()` is most reliable
3. Don't overcomplicate - simple fixes work best
4. Document the working config and don't change it

## Current Status

All files here are **archived** and **not used**.

The working version is:
- `/train_ultra_speed_v26.py` (parent directory)
- `/WORKING_CONFIG.md` (parent directory)

**Do not use files from this archive - they are experiments only!**

---

Archived: 2025-01-14
Reason: Experimentation complete, working solution found
