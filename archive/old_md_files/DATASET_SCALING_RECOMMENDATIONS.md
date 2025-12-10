# Dataset Scaling & Balance Recommendations

**Date**: 2025-11-27
**Current Dataset**: `agab_phase2_full.csv` (159,735 samples)
**After Filtering**: 152,827 valid samples (pKd ∈ [4.0, 14.0])
**Current Training**: v2.7 with stable MSE+BCE loss ✅

---

## 📊 Current Situation Analysis

### Your Current Dataset (AgAb Phase2 Full)
- **Total samples**: 159,735
- **Valid samples** (pKd 4-14): 152,827 (95.7%)
- **pKd range**: -2.96 to 12.43 (filtered to 4.0-12.43)
- **pKd mean**: 7.45 ± 2.11
- **Distribution**:
  - [4-6]: 106 samples (0.1%) ⚠️ SEVERELY UNDERREPRESENTED
  - [6-8]: 71,315 samples (46.6%)
  - [8-10]: 80,351 samples (52.6%) ✅ MAJORITY
  - [10-12]: 1,052 samples (0.7%)
  - [12-14]: 3 samples (0.002%)

### 🔴 Critical Balance Problem

Your dataset is **heavily imbalanced**:
- **Weak binders (pKd 4-6)**: Only 106 samples (0.1%)
- **Mid-range (pKd 6-8)**: 71,315 samples (46.6%)
- **Strong binders (pKd 8-10)**: 80,351 samples (52.6%)
- **Very strong (pKd 10+)**: Only 1,055 samples (0.7%)

**Impact**: Model will struggle to predict weak binders and very strong binders accurately.

---

## 🎯 Available Datasets for Scaling

### Option 1: AgAb Full (1.2M samples) ⭐ BEST FOR SCALING
**Location**: `data/agab/agab_full_dataset.csv`

**Pros**:
- **Massive size**: 1,227,083 total samples
- **Valid samples** (pKd 4-14): **187,894 samples** (23% more than current!)
- **Better distribution**:
  - [4-6]: 34,782 samples (18.5%) ✅ **330x more weak binders!**
  - [6-8]: 71,532 samples (38.1%)
  - [8-10]: 80,454 samples (42.8%)
  - [10-12]: 1,083 samples (0.6%)
  - [12-14]: 43 samples (0.02%)
- **Same source**: Same data format as current dataset

**Cons**:
- 43.4% missing/invalid affinity values (need filtering)
- Extreme outliers (pKd from -17 to 1461) need removal

**Recommendation**: ✅ **USE THIS** - Filter to [4.0, 14.0] range

---

### Option 2: AbbiBench (186k samples) ⭐ HIGH QUALITY
**Location**: `data/abbibench/abbibench_with_antigen.csv`

**Pros**:
- **Large size**: 185,732 samples
- **Has binding_score**: Values ~8-9 (pKd-like)
- **High quality**: From benchmarking dataset
- **Complete sequences**: Heavy + light chain + antigen

**Cons**:
- Need to verify binding_score = pKd
- Unknown distribution (needs analysis)

**Recommendation**: ✅ **ANALYZE FURTHER** - Could complement AgAb

---

### Option 3: SAbDab (954 samples) ⭐ GOLD STANDARD
**Location**: `data/sabdab/sabdab_sequences_with_affinity.csv`

**Pros**:
- **100% valid**: All samples in [4.0, 14.0] range
- **Structural quality**: From crystal structures
- **Good distribution**:
  - [4-6]: 41 samples (4.3%)
  - [6-8]: 302 samples (31.7%)
  - [8-10]: 470 samples (49.3%)
  - [10-12]: 139 samples (14.6%) ✅ **Better high-affinity coverage**
  - [12-14]: 2 samples (0.2%)
- **pKd mean**: 8.45 ± 1.40 (high quality)

**Cons**:
- Small size (only 954 samples)

**Recommendation**: ✅ **MERGE** - Excellent for improving high-affinity predictions

---

### Option 4: PPB Affinity & PDBbind ❌ NOT READY
**Issues**:
- No clear pKd column found
- Would require manual processing
- Lower priority

---

## 🚀 RECOMMENDED SCALING STRATEGY

### Phase 1: Scale to 188k with AgAb Full ⭐ IMMEDIATE
**Goal**: Fix weak binder imbalance, increase dataset size

**Steps**:
1. Load `agab_full_dataset.csv` (1.2M samples)
2. Filter pKd to [4.0, 14.0] range → 187,894 valid samples
3. Check for duplicates with current dataset
4. Merge deduplicated samples

**Expected Result**:
- **Total samples**: ~340,000 (if 50% overlap) or ~187,894 (if full replacement)
- **Weak binders [4-6]**: **34,782 samples** (vs current 106) ✅
- **Better balance**: 18.5% weak, 38.1% mid, 42.8% strong

**Training Impact**:
- Model will learn weak binder patterns
- Better generalization across full pKd range
- Expected Spearman: **0.50-0.60** (vs current target 0.45-0.55)

**Script Needed**:
```python
# scripts/data_collection/merge_agab_full_filtered.py
import pandas as pd

# Load full dataset
df_full = pd.read_csv('data/agab/agab_full_dataset.csv')

# Filter to valid pKd range
df_filtered = df_full[(df_full['affinity'] >= 4.0) & (df_full['affinity'] <= 14.0)]

# Rename 'affinity' to 'pKd' for consistency
df_filtered = df_filtered.rename(columns={'affinity': 'pKd'})

# Deduplicate (if needed)
df_filtered = df_filtered.drop_duplicates(subset=['antibody_sequence', 'antigen_sequence'])

# Save
df_filtered.to_csv('data/agab/agab_full_filtered_4to14.csv', index=False)

print(f"Filtered dataset: {len(df_filtered):,} samples")
```

---

### Phase 2: Add SAbDab for High-Affinity Balance ⭐ HIGH VALUE
**Goal**: Improve predictions for pKd > 10

**Steps**:
1. Load `sabdab_sequences_with_affinity.csv` (954 samples)
2. Check overlap with AgAb Full filtered
3. Merge non-duplicate samples

**Expected Result**:
- **Additional samples**: ~500-800 (after deduplication)
- **High-affinity boost**: +139 samples in [10-12] range
- **Total dataset**: ~188,500-340,800 samples

**Training Impact**:
- Better recall for very strong binders (pKd > 10)
- Structural quality from crystal structures
- Expected Spearman: **0.55-0.65**

---

### Phase 3: Evaluate AbbiBench Integration (OPTIONAL)
**Goal**: Further scaling if performance plateaus

**Requirements**:
1. Verify `binding_score` is equivalent to pKd
2. Check distribution
3. Test on small merge first

**Expected Result**:
- Potential +100-150k samples (if binding_score is valid pKd)
- Risk: Different scoring system might hurt homogeneity

**Decision Point**: Only proceed if Phase 1+2 doesn't achieve target performance

---

## 📈 Expected Performance Improvements

### Current (152k, imbalanced):
- Spearman: 0.45-0.55 (target)
- Weak binders: ❌ Poor (only 106 training samples)
- Strong binders: ✅ Good (80,351 samples)
- Very strong: ❌ Poor (only 1,055 samples)

### After Phase 1 (188k, balanced):
- Spearman: **0.50-0.60** ✅
- Weak binders: ✅ **Good** (34,782 samples)
- Strong binders: ✅ Good (80,454 samples)
- Very strong: ⚠️ Fair (1,083 samples)

### After Phase 1+2 (189k, optimal):
- Spearman: **0.55-0.65** ✅✅
- Weak binders: ✅ **Good** (34,823 samples)
- Strong binders: ✅ Good (80,772 samples)
- Very strong: ✅ **Much better** (1,222 samples)

---

## ⚡ IMMEDIATE ACTION PLAN

### Step 1: Create Filtered AgAb Full Dataset (TODAY)
```bash
cd C:\Users\401-24\Desktop\Ab_Ag_dataset
python scripts/data_collection/merge_agab_full_filtered.py
```

### Step 2: Analyze Overlap
```python
import pandas as pd

df_current = pd.read_csv('data/agab/agab_phase2_full.csv')
df_full_filtered = pd.read_csv('data/agab/agab_full_filtered_4to14.csv')

# Check overlap
current_pairs = set(zip(df_current['antibody_sequence'], df_current['antigen_sequence']))
full_pairs = set(zip(df_full_filtered['antibody_sequence'], df_full_filtered['antigen_sequence']))

overlap = len(current_pairs & full_pairs)
new_samples = len(full_pairs - current_pairs)

print(f"Overlap: {overlap:,} samples")
print(f"New samples: {new_samples:,}")
print(f"Total after merge: {len(full_pairs):,}")
```

### Step 3: Merge SAbDab
```python
df_sabdab = pd.read_csv('data/sabdab/sabdab_sequences_with_affinity.csv')
# Merge with AgAb Full filtered, deduplicate
```

### Step 4: Update Training Pipeline
- Upload new merged dataset to Google Drive
- Update `colab_training_v2.7.ipynb` Cell 11 to use new file
- Restart training

---

## 🎯 Final Recommendation

### **Immediate Actions** (Do This Now):

1. ✅ **Filter AgAb Full** to [4.0, 14.0] → 187,894 samples
   - Fixes weak binder imbalance (106 → 34,782 samples!)
   - 23% more total data
   - Same data source (no compatibility issues)

2. ✅ **Merge SAbDab** → +~800 samples
   - Boosts high-affinity predictions
   - Gold standard structural quality
   - Proven to work well (Spearman 0.552 standalone)

3. ✅ **Skip AbbiBench for now**
   - Needs validation of binding_score
   - Risk of heterogeneity
   - Can add later if needed

### **Expected Outcome**:
- **Total dataset**: ~188,700 samples (23% larger)
- **Balanced distribution**: 18.5% weak, 38.1% mid, 42.8% strong, 0.6% very strong
- **Expected performance**: Spearman **0.55-0.65** (vs current target 0.45-0.55)
- **Training time**: ~10-15% longer per epoch (still manageable on A100)

---

## 📋 Implementation Checklist

- [ ] Create `merge_agab_full_filtered.py` script
- [ ] Filter AgAb Full to pKd [4.0, 14.0]
- [ ] Analyze overlap with current dataset
- [ ] Deduplicate and merge
- [ ] Add SAbDab samples
- [ ] Create final merged dataset: `agab_phase2_full_v2_balanced.csv`
- [ ] Upload to Google Drive
- [ ] Update Colab notebook to use new dataset
- [ ] Restart v2.7 training with balanced data
- [ ] Monitor: weak binder predictions should improve significantly!

---

**Status**: Ready to implement
**Priority**: HIGH - Will significantly improve model performance
**Effort**: Low (1-2 hours)
**Impact**: High (23% more data, 330x more weak binders!)

**Let's scale up! 🚀**
