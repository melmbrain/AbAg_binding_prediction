# Troubleshooting: CUDA Graphs Error Persists

## If the error STILL happens after the fix...

### Step 1: Verify the script was actually updated in Colab

Run this in Colab to check:

```python
# Check if the file in Drive has the fix
!grep -n "default=False.*DISABLED" train_ultra_speed_v26.py

# Should show:
# 889:    parser.add_argument('--use_compile', type=lambda x: x.lower() == 'true', default=False)  # DISABLED for Colab
```

If you DON'T see this line, the file wasn't updated. Re-upload it!

---

### Step 2: Force disable torch.compile completely

Add this at the very top of the script (after imports):

```python
# Add to train_ultra_speed_v26.py after line 30 (after imports)

# NUCLEAR OPTION: Completely disable torch.compile globally
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch.compiler.disable()

print("⚠️ torch.compile FORCEFULLY DISABLED")
```

This will prevent ANY compilation from happening.

---

### Step 3: Disable activation checkpointing instead

If disabling compile doesn't work, the alternative is to disable checkpointing:

Change line 927 in the Colab args from:
```python
'--use_checkpointing', 'True',
```

To:
```python
'--use_checkpointing', 'False',  # Disable checkpointing to avoid CUDA graphs conflict
```

**Trade-off**:
- ✅ No more CUDA graphs error
- ❌ Can't use batch size 16 (will need to reduce to 12)
- ❌ Slightly slower training

---

### Step 4: Nuclear option - Disable BOTH

If nothing else works, disable both:

```python
'--use_compile', 'False',
'--use_checkpointing', 'False',
'--batch_size', '12',  # Reduce from 16
'--accumulation_steps', '4',  # Increase from 3 to keep effective batch at 48
```

This will definitely work, but you'll lose some speed optimizations.

---

### Step 5: Check PyTorch version compatibility

The CUDA graphs + checkpointing conflict might be PyTorch version specific.

Check your version:
```python
import torch
print(torch.__version__)
```

If you're on PyTorch 2.0-2.1, try upgrading:
```python
!pip install --upgrade torch torchvision torchaudio
```

---

### Step 6: Use environment variable to disable CUDA graphs

Add this at the start of your training cell:

```python
import os
os.environ['TORCH_CUDAGRAPH_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

!python train_ultra_speed_v26.py
```

---

### Step 7: Manual verification that compile is OFF

Add debug prints to verify compile is actually disabled.

Edit line 733 in `train_ultra_speed_v26.py`:

```python
# NEW: Compile only the regressor blocks (not the frozen encoders)
print(f"\n🔍 DEBUG: args.use_compile = {args.use_compile} (type: {type(args.use_compile)})")

if args.use_compile:
    print("\n⚠️⚠️⚠️ WARNING: COMPILE IS ENABLED - THIS WILL CRASH!")
    print("Compiling regressor blocks...")
    # ... rest of compile code
else:
    print("\n✅ Compilation DISABLED - skipping torch.compile")
    print("   This is correct for avoiding CUDA graphs errors\n")
```

Run training and check the output. You should see:
```
🔍 DEBUG: args.use_compile = False (type: <class 'bool'>)
✅ Compilation DISABLED - skipping torch.compile
```

If you see `args.use_compile = True`, then the fix didn't work!

---

### Step 8: Create a minimal test script

Test if the issue is specific to your model or a general PyTorch issue:

```python
# test_cuda_graphs.py
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Simple model with checkpointing
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(100, 50)
        self.layer2 = nn.Linear(50, 10)

    def forward(self, x):
        # Use checkpointing
        x = checkpoint(self.layer1, x, use_reentrant=False)
        x = checkpoint(self.layer2, x, use_reentrant=False)
        return x

model = TestModel().cuda()

# Try WITHOUT compile (should work)
print("Test 1: Without torch.compile")
x = torch.randn(16, 100).cuda()
y = model(x)
loss = y.sum()
loss.backward()
print("✅ Works without compile!")

# Try WITH compile (might crash)
print("\nTest 2: With torch.compile")
model_compiled = torch.compile(model, mode='reduce-overhead')
x = torch.randn(16, 100).cuda()
y = model_compiled(x)
loss = y.sum()
try:
    loss.backward()
    print("✅ Works with compile too!")
except RuntimeError as e:
    print(f"❌ CUDA graphs error: {e}")
    print("This confirms the conflict exists in your PyTorch version")
```

Run this to confirm the issue is real.

---

### Step 9: Alternative - Use gradient checkpointing without torch.utils.checkpoint

Replace `torch.utils.checkpoint.checkpoint()` with manual recomputation:

```python
# In the model's forward method, instead of:
if self.use_checkpointing and self.training:
    x = checkpoint(self.regressor_block1, combined, use_reentrant=False)
    # ...

# Use this:
if self.use_checkpointing and self.training:
    # Manual checkpointing - save memory without torch.utils.checkpoint
    with torch.no_grad():
        x = self.regressor_block1(combined)
    x.requires_grad = True
else:
    x = self.regressor_block1(combined)
```

But this is more complex and might not save as much memory.

---

### Step 10: Last resort - Train without optimizations

If ALL else fails, use the safe baseline config:

```python
args = parser.parse_args([
    '--data', 'agab_phase2_full.csv',
    '--output_dir', 'outputs_max_speed',
    '--epochs', '50',
    '--batch_size', '8',  # Reduce batch size
    '--accumulation_steps', '6',  # Increase accumulation
    '--lr', '4e-3',
    '--weight_decay', '0.01',
    '--dropout', '0.3',
    '--focal_gamma', '2.0',
    '--save_every_n_batches', '500',
    '--num_workers', '4',
    '--prefetch_factor', '4',
    '--validation_frequency', '2',
    '--use_bfloat16', 'True',  # Keep this
    '--use_compile', 'False',  # DISABLED
    '--use_fused_optimizer', 'True',  # Keep this
    '--use_quantization', 'True',  # Keep this
    '--use_checkpointing', 'False',  # DISABLED
    '--use_bucketing', 'True'  # Keep this
])
```

This will still be 8-12× faster than baseline, even without compile and checkpointing.

---

## Summary of Options (in order of preference)

1. ✅ **Verify file was updated** - Check the fix is actually in the file
2. ✅ **Add torch.compiler.disable()** - Force disable compilation globally
3. ✅ **Add environment variables** - Set TORCH_COMPILE_DISABLE=1
4. ✅ **Disable checkpointing** - Trade memory for stability
5. ✅ **Reduce batch size** - Use smaller batches (8 instead of 16)
6. ✅ **Upgrade PyTorch** - Update to latest version
7. ✅ **Test with minimal script** - Confirm it's a real PyTorch bug
8. ❌ **Manual checkpointing** - Complex, not recommended
9. ❌ **Disable all optimizations** - Last resort only

## Most Likely Solution

The issue is probably that **the file in Google Drive wasn't actually updated**.

Do this:
1. Delete `train_ultra_speed_v26.py` from Google Drive
2. Re-upload the fixed version from your local machine
3. Restart Colab runtime (Runtime → Restart runtime)
4. Run training again

This forces a clean state and ensures you're using the fixed file.
