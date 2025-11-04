# Extreme Affinity Data Analysis Report

## Executive Summary

Your model is trained primarily on **moderate affinity data** (67% with pKd 5-9), with very few extreme values:
- **Very strong binders** (pKd > 11): Only 0.1% (240 out of 205k)
- **Very weak binders** (pKd < 5): Only 1.8% (3,778 out of 205k)

## Current Dataset Distribution (205k samples)

| Affinity Range | pKd Range | Kd Range | Count | Percentage |
|----------------|-----------|----------|-------|------------|
| Very weak | < 5 | > 10 μM | 3,778 | 1.8% |
| Weak | 5-7 | 100 nM - 10 μM | 66,114 | 32.2% |
| **Moderate** | **7-9** | **1-100 nM** | **71,789** | **35.0%** ← **Largest** |
| Strong | 9-11 | 0.01-1 nM | 58,567 | 28.6% |
| Very strong | > 11 | < 10 pM | 240 | **0.1%** ← **Critical gap** |

**Mean pKd: 7.76** (moderate affinity)

## Available Antibody-Antigen Extreme Data

### 1. Very Strong Binders (pKd > 11)

#### From SAbDab (5 complexes):
| PDB Code | Kd (nM) | pKd | Status |
|----------|---------|-----|--------|
| 2nyy | 0.00248 | 11.61 | **Already in dataset** |
| 3h42 | 0.00400 | 11.40 | **Already in dataset** |
| 3lhp | 0.00750 | 11.12 | **Already in dataset** |
| 2nz9 | 0.00680 | 11.17 | **Already in dataset** |
| 2bdn | 0.00460 | 11.34 | **Already in dataset** |

#### From SKEMPI2:
- 3 antibody-related complexes (but data quality issues in parsing)

**Total available: 8 complexes (but 5 already in your dataset!)**

### 2. Weak/Very Weak Binders (pKd < 7)

#### From SAbDab (4 complexes):
| PDB Code | Kd (nM) | pKd | Status |
|----------|---------|-----|--------|
| 2oqj | 200,000 | 3.70 | **Already in dataset** |
| **1aj7** | **135,000** | **3.87** | **NEW - Can add** |
| 1nby | 90,900 | 4.04 | **Already in dataset** |
| **1fl6** | **25,000** | **4.60** | **NEW - Can add** |

#### From SKEMPI2:
- 56 antibody-related weak binders (pKd 5-7)
- 13 antibody-related very weak binders (pKd < 5)

**Total available: 73 complexes**

## Key Finding: Data Already Exists!

**Important Discovery:** Most extreme affinity antibody-antigen complexes from SAbDab are **already in your dataset**. This suggests the problem is not lack of data, but rather:

1. **Class imbalance** in the existing dataset
2. Possible **filtering or preprocessing** that removes or downweights extreme values
3. Need for **stratified sampling** during training to ensure extreme cases are learned

## Recommendations

### Option 1: Rebalance Existing Dataset (RECOMMENDED)
Instead of adding new data, **rebalance your existing 205k dataset** to give more weight to extreme values:

```python
# Suggested rebalancing strategy
1. Identify all samples with pKd > 11 (240 samples)
2. Identify all samples with pKd < 5 (3,778 samples)
3. During training:
   - Oversample extreme cases (use higher sampling weight)
   - Or use stratified batching to ensure each batch has extreme samples
   - Or use focal loss to focus on hard-to-predict extreme cases
```

**Benefits:**
- No need to fetch/process new data
- Utilizes existing high-quality data more effectively
- Addresses the root cause (class imbalance)

### Option 2: Add New Data from External Sources

**High priority additions:**
1. **SKEMPI2 antibody weak binders**: 56 complexes with pKd 5-7
   - File: `extreme_affinity_data/skempi2_antibody_weak.csv`
   - These are mutants with reduced affinity

2. **SAbDab new weak binders**: 2 complexes (1aj7, 1fl6)
   - PDB: 1aj7 (Kd = 135 μM, pKd = 3.87)
   - PDB: 1fl6 (Kd = 25 μM, pKd = 4.60)

3. **SKEMPI2 very weak binders**: 13 antibody-antigen complexes
   - File: `extreme_affinity_data/skempi2_antibody_very_weak.csv`

**Total new data: ~71 antibody-antigen complexes**

### Option 3: Combine Both Approaches (BEST)

1. **Rebalance existing data** to ensure extreme cases are learned
2. **Add SKEMPI2 antibody weak binders** to increase diversity in weak binding region
3. **Validate on held-out extreme cases** to measure improvement

## Database Comparison

| Database | Total Entries | Very Strong (pKd>11) | Weak (pKd<7) | Ab-Ag Specific |
|----------|--------------|---------------------|--------------|----------------|
| **Your Phase6** | 204,986 | 240 (0.1%) | 69,892 (34%) | Yes |
| **SKEMPI2** | 7,085 | 741 (10%) | 1,695 (23%) | Partial (~10%) |
| **SAbDab** | 210 | 5 (2%) | 4 (2%) | Yes (100%) |

## Implementation Steps

### For Rebalancing (Recommended First Step):

1. **Analyze extreme cases in your dataset:**
   ```bash
   # Extract very strong binders (pKd > 11) from your dataset
   awk -F',' '$2 > 11' final_205k_dataset.csv > extreme_strong_existing.csv

   # Extract very weak binders (pKd < 5) from your dataset
   awk -F',' '$2 < 5 && $2 > 0' final_205k_dataset.csv > extreme_weak_existing.csv
   ```

2. **Implement stratified sampling in training:**
   - Create affinity bins: [0-5, 5-7, 7-9, 9-11, >11]
   - Ensure each training batch contains samples from all bins
   - Use class weights inversely proportional to frequency

3. **Add augmentation for extreme cases:**
   - Apply data augmentation specifically to rare extreme cases
   - Consider synthetic minority oversampling (SMOTE) for numerical features

### For Adding New Data:

The extracted data files are available in:
```
/mnt/c/Users/401-24/Desktop/AbAg_binding_prediction/extreme_affinity_data/
├── sabdab_very_strong.csv       (5 complexes, all already in dataset)
├── sabdab_very_weak.csv         (4 complexes, 2 new: 1aj7, 1fl6)
├── skempi2_antibody_very_strong.csv  (3 complexes)
├── skempi2_antibody_weak.csv         (56 complexes - NEW)
└── skempi2_antibody_very_weak.csv    (13 complexes - NEW)
```

## Next Steps

1. **Immediate action**: Implement stratified sampling/rebalancing in your training code
2. **Short term**: Add SKEMPI2 antibody weak binders (71 new complexes)
3. **Validation**: Test model performance on extreme affinity held-out set
4. **Monitoring**: Track metrics separately for different affinity ranges

## Files Generated

All analysis results saved to:
- `extreme_affinity_data/` - Extracted extreme affinity datasets
- `antibody_antigen_summary.txt` - Detailed summary
- `analyze_affinity_distribution.py` - Analysis script
- This report: `EXTREME_AFFINITY_ANALYSIS_REPORT.md`

---
*Report generated on 2025-11-03*
*Analysis of Docking prediction dataset at: C:\\Users\\401-24\\Desktop\\Docking prediction*
