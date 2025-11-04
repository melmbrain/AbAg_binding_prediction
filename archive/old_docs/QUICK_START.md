# Quick Start Guide - AbAg Binding Affinity Prediction

Get started with antibody-antigen binding affinity prediction in 5 minutes!

## Installation

```bash
# Navigate to package directory
cd /mnt/c/Users/401-24/Desktop/AbAg_binding_prediction

# Install dependencies
pip install -r requirements.txt

# Install package (development mode)
pip install -e .
```

## First Prediction

### Step 1: Import the Predictor

```python
from abag_affinity import AffinityPredictor

# Initialize (automatically loads model)
predictor = AffinityPredictor()
```

**First run note**: Downloads ESM-2 model (~140 MB), subsequent runs are fast!

### Step 2: Make a Prediction

```python
# Your antibody and antigen sequences
antibody_heavy = "EVQLQQSGPGLVKPSQTLSLTCAISG..."
antigen = "KVFGRCELAAAMKRHGLDNYRGYSLGN..."

# Predict binding affinity
result = predictor.predict(
    antibody_heavy=antibody_heavy,
    antigen=antigen
)

# View results
print(f"pKd: {result['pKd']:.2f}")
print(f"Kd: {result['Kd_nM']:.1f} nM")
print(f"Category: {result['category']}")
```

### Step 3: Interpret Results

| pKd | Kd | Interpretation |
|-----|-----|----------------|
| > 10 | < 1 nM | Exceptional binder (therapeutic quality) |
| 9-10 | 1-10 nM | Very strong binder |
| 7.5-9 | 10-100 nM | Strong binder (research grade) |
| 6-7.5 | 0.1-10 μM | Moderate binder |
| < 6 | > 10 μM | Weak or non-binder |

## Run Tests

Verify installation:

```bash
python tests/test_installation.py
```

Expected output:
```
🎉 ALL TESTS PASSED!
Your installation is working correctly!
```

## Run Examples

See full examples:

```bash
python examples/basic_usage.py
```

This will demonstrate:
1. Single prediction
2. Heavy-only antibodies
3. Batch processing
4. Variant comparison

## Common Use Cases

### Screen Antibody Library

```python
# Load your antibody library
library = [
    {'id': 'Ab001', 'heavy': 'EVQ...', 'antigen': target},
    {'id': 'Ab002', 'heavy': 'QVQ...', 'antigen': target},
    # ... more candidates
]

# Batch predict
results = predictor.predict_batch(library)

# Sort by affinity
best = sorted(results, key=lambda x: x['pKd'], reverse=True)

# Top 10 candidates
for r in best[:10]:
    print(f"{r['id']}: pKd={r['pKd']:.2f}, Kd={r['Kd_nM']:.1f} nM")
```

### Compare Mutants

```python
# Original antibody
original = predictor.predict(heavy=wild_type, antigen=target)

# Test mutation
mutant = predictor.predict(heavy=mutated, antigen=target)

# Compare
improvement = mutant['pKd'] - original['pKd']
print(f"ΔpKd: {improvement:+.2f}")
```

## Troubleshooting

**"Model not found"**
- Check that `models/agab_phase2_model.pth` exists
- Run from package root directory

**"Out of memory"**
- Use CPU: `predictor = AffinityPredictor(device='cpu')`
- Reduce batch size

**"Invalid amino acid"**
- Check sequences contain only: ACDEFGHIKLMNPQRSTVWY
- Remove gaps and special characters

## Next Steps

1. **Read full documentation**: [README.md](README.md)
2. **Try examples**: `python examples/basic_usage.py`
3. **Process your data**: Adapt batch processing example
4. **Explore API**: Check docstrings with `help(AffinityPredictor)`

## Performance Notes

- **Accuracy**: Spearman ρ = 0.85, Pearson r = 0.95
- **Speed**: ~1-2 seconds per prediction (GPU), ~5-10 seconds (CPU)
- **Batch processing**: Much faster than individual predictions
- **First run**: Slower due to ESM-2 download, subsequent runs are cached

## Questions?

- Check full README: [README.md](README.md)
- Run tests: `python tests/test_installation.py`
- Try examples: `python examples/basic_usage.py`

---

**You're ready to predict binding affinities! 🚀**
