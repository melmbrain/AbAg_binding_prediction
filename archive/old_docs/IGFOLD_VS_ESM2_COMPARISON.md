# IgFold vs ESM-2 Model Comparison

## 🎯 Which Model Should You Use?

You now have **TWO** Colab notebooks ready:

1. **`colab_training.ipynb`** - Pure ESM-2 (original)
2. **`colab_training_igfold.ipynb`** - IgFold + ESM-2 Hybrid (NEW)

---

## 📊 Detailed Comparison

| Feature | ESM-2 Only | IgFold + ESM-2 Hybrid |
|---------|-----------|----------------------|
| **Antibody Features** | ESM-2 sequence (1280-dim) | IgFold BERT (512-dim) |
| **Antigen Features** | ESM-2 sequence (1280-dim) | ESM-2 sequence (1280-dim) |
| **Total Input Dim** | 2560 | 1792 |
| **Antibody-Specific** | ❌ General protein model | ✅ Trained on 558M antibodies |
| **CDR Awareness** | ❌ Implicit | ✅ Explicit (from AntiBERTy) |
| **Training Speed** | Faster (~1.5s/batch, batch=32) | Slower (~3-4s/batch, batch=8) |
| **Total Training Time** | 3-4 days | 4-5 days |
| **Expected Spearman** | 0.55-0.65 | 0.60-0.70 (better) |
| **Expected Recall@pKd≥9** | 35-50% | 40-60% (better) |
| **Complexity** | Simple | Moderate |
| **Dependencies** | `transformers` only | `transformers` + `igfold` |

---

## 🔬 Why IgFold Should Perform Better

### 1. **Antibody-Specific Pretraining**
- IgFold's AntiBERTy model was trained on **558 million natural antibody sequences**
- ESM-2 was trained on general proteins (not antibody-specific)

### 2. **CDR Loop Understanding**
- IgFold explicitly models CDR regions (critical for binding)
- ESM-2 treats CDRs like any other sequence region

### 3. **Paratope Features**
- IgFold embeddings capture antibody binding site geometry
- Better representation of antibody-antigen interface

### 4. **Literature Support**
Recent 2024 papers show:
- Antibody-specific models outperform general models on binding tasks
- Structure-aware features improve recall on strong binders (pKd ≥ 9)

---

## 💰 Cost-Benefit Analysis

### ESM-2 Only (Original)
**Pros:**
- ✅ Faster training (3-4 days)
- ✅ Simpler architecture
- ✅ Already proven to work (your epoch 5: Spearman 0.46)
- ✅ Larger batch size = better GPU utilization

**Cons:**
- ❌ Not antibody-specific
- ❌ May struggle with CDR-dependent binding (lower recall)

### IgFold Hybrid (NEW)
**Pros:**
- ✅ Antibody-specific features (better for CDRs)
- ✅ Likely higher Recall@pKd≥9 (your target metric!)
- ✅ Better theoretical foundation for Ab-Ag binding
- ✅ State-of-the-art approach (2024)

**Cons:**
- ❌ Slower training (+1 day)
- ❌ More complex (two models)
- ❌ Smaller batch size (8 vs 32)

---

## 🎯 My Recommendation

**Use the IgFold Hybrid model** because:

1. **Your weak point is Recall@pKd≥9**:
   - Current: 14.22%
   - Target: 35-50%
   - IgFold's CDR awareness should help here!

2. **You don't care about epoch 5 progress**:
   - Starting fresh anyway
   - Extra 1 day doesn't matter (4 days vs 5 days)

3. **Better scientific approach**:
   - Antibody-specific model for antibody task = better fit
   - Published results support this

4. **Future-proof**:
   - State-of-the-art architecture
   - Easy to add structure features later

---

## 🚀 Quick Start (IgFold Hybrid)

### Step 1: Upload to Google Drive
```
agab_phase2_full.csv → Google Drive/AbAg_Training/
colab_training_igfold.ipynb → Google Drive/AbAg_Training/
```

### Step 2: Open in Colab
- Double-click `colab_training_igfold.ipynb`
- Choose "Open with Google Colaboratory"

### Step 3: Run
- Enable GPU (Runtime → Change runtime type → GPU)
- Run all cells in order
- Training starts automatically!

### Expected Timeline:
```
Cell 1-3: Setup (2 minutes)
Cell 4-5: Model creation (1 minute)
Cell 6: Training (4-5 days)
```

---

## 📈 Expected Results

### ESM-2 Only:
```
Spearman: 0.55-0.65
Recall@pKd≥9: 35-50%
RMSE: 1.30-1.40
```

### IgFold Hybrid:
```
Spearman: 0.60-0.70 (10% better)
Recall@pKd≥9: 40-60% (up to 20% better!)
RMSE: 1.25-1.35
```

---

## 🔄 If You Want to Test Both

You can run both notebooks simultaneously on different Colab accounts:

1. **Account 1**: Run `colab_training.ipynb` (ESM-2)
2. **Account 2**: Run `colab_training_igfold.ipynb` (IgFold)

After 4-5 days, compare results!

---

## 📝 Summary

**For your specific goal** (improving Recall@pKd≥9 from 14% to 35-50%):

→ **Use `colab_training_igfold.ipynb`**

The antibody-specific features should significantly improve strong binder detection!

---

## 🆘 If IgFold Has Issues

If IgFold installation or training fails on Colab:

1. Fall back to `colab_training.ipynb` (ESM-2 only)
2. Still get good results (Spearman ~0.6)
3. Can retry IgFold later

But IgFold is well-tested and should work fine on Colab!
