# Reference Files Guide

## Overview

This directory contains comprehensive references for the antibody-antigen binding prediction project, with specific focus on extreme affinity values (very weak and very strong binders) and class imbalance handling.

## Reference Files

### 1. references_master.md ⭐ START HERE
**Master reference document** containing:
- Quick citation guide for all major papers
- Summary of key findings from each reference
- BibTeX entries for easy citation
- Project-specific context and recommendations

### 2. references_skempi2.md
**SKEMPI 2.0 Database** references including:
- Primary citation (Jankauskaitė et al., 2019)
- Database statistics (7,085 mutations)
- Antibody-antigen specific entries identified
- Related computational tools
- Complete citation formats

### 3. references_sabdab.md
**Structural Antibody Database (SAbDab)** references including:
- Primary citation (Dunbar et al., 2014)
- 2022 update (Schneider et al., 2022)
- Thera-SAbDab therapeutic antibody database
- Database access and API information
- Affinity data details (210 entries with binding data)

### 4. references_extreme_affinity.md
**Extreme binding affinity** research including:
- Femtomolar affinity achievement (Boder et al., 2000 - Kd = 48 fM)
- Picomolar affinity methods (ribosome display)
- Molecular basis of ultra-high affinity
- Weak affinity interactions (micromolar range)
- Measurement techniques for extreme affinities
- Natural affinity ceiling in B cell response (~100 pM)

### 5. references_class_imbalance.md
**Class imbalance handling** methods including:
- Stratified sampling for deep learning (Kim et al., 2023)
- SMOTE for minority oversampling (Chawla et al., 2002)
- Focal loss for hard examples (Lin et al., 2017)
- Application to binding affinity prediction (Jiang et al., 2022)
- Complete code examples in Python/PyTorch
- Evaluation metrics for imbalanced regression

## How to Use These References

### For Writing Papers

1. **Start with** `references_master.md` for quick citation templates
2. **Refer to** individual files for detailed abstracts and context
3. **Use BibTeX entries** provided in each file for bibliography management

### For Implementation

1. **Class imbalance methods:** See `references_class_imbalance.md` for code examples
2. **Understanding affinity ranges:** See `references_extreme_affinity.md` for biological context
3. **Database citations:** See `references_skempi2.md` and `references_sabdab.md`

### Citation Examples

**When using SKEMPI2 data:**
> "Mutation data were obtained from SKEMPI 2.0 (Jankauskaitė et al., 2019), a curated database of 7,085 protein-protein interaction mutations with binding affinity changes."

**When using SAbDab data:**
> "Antibody structures and binding affinity values were retrieved from SAbDab (Dunbar et al., 2014), a comprehensive database of all publicly available antibody structures."

**When implementing stratified sampling:**
> "To address class imbalance in affinity distribution, we employed stratified mini-batch sampling (Kim et al., 2023), ensuring balanced representation across affinity ranges during training."

**When discussing extreme affinities:**
> "While natural antibodies achieve affinities up to ~100 pM through somatic hypermutation, engineered antibodies can reach femtomolar affinities (Kd ~ 48 fM) through directed evolution (Boder et al., 2000)."

## Quick Reference: Key Papers

### Must-Cite Papers for This Project

1. **SKEMPI 2.0 Database**
   - Jankauskaitė et al. (2019) - *Bioinformatics*
   - DOI: 10.1093/bioinformatics/bty635

2. **SAbDab Database**
   - Dunbar et al. (2014) - *Nucleic Acids Research*
   - DOI: 10.1093/nar/gkt1043

3. **Class Imbalance Handling**
   - Kim et al. (2023) - *Electronics*
   - DOI: 10.3390/electronics12214423
   - Chawla et al. (2002) - SMOTE - *JAIR*
   - DOI: 10.1613/jair.953

4. **Extreme Affinity Context**
   - Boder et al. (2000) - Femtomolar affinity - *PNAS*
   - DOI: 10.1073/pnas.170297297
   - Kastritis & Bonvin (2013) - Weak interactions - *J. R. Soc. Interface*
   - DOI: 10.1098/rsif.2012.0835

## Project Context

### Problem Statement
Your model is trained primarily on moderate affinity data (pKd 7-9), with severe underrepresentation of extreme values:
- Very strong (pKd > 11): Only 0.1% of dataset (240/205k samples)
- Very weak (pKd < 5): Only 1.8% of dataset (3,778/205k samples)

### Solution Strategies (Supported by References)

**1. Rebalancing (Primary Recommendation)**
- Use stratified sampling (Kim et al., 2023)
- Apply class weights (see code in `references_class_imbalance.md`)
- Implement focal loss (Lin et al., 2017)

**2. Data Augmentation (Secondary)**
- Add SKEMPI2 antibody-antigen extremes (69 new complexes)
- Apply SMOTE-like interpolation (Chawla et al., 2002)

**3. Evaluation**
- Report per-affinity-range metrics
- Use AUPR for imbalanced evaluation (Jiang et al., 2022)

## Statistical Summary from References

### Affinity Distribution Comparison

| Database | Very Strong (>11) | Strong (9-11) | Moderate (7-9) | Weak (5-7) | Very Weak (<5) |
|----------|-------------------|---------------|----------------|------------|----------------|
| **Your data** | 0.1% | 28.6% | **35.0%** | 32.2% | 1.8% |
| **SKEMPI2** | 10.5% | 31.4% | 34.2% | 18.7% | 5.2% |
| **SAbDab** | 2.4% | 18.6% | 54.3% | 22.9% | 1.9% |

**Key insight:** SKEMPI2 has 100× more very strong binders (as % of dataset) compared to your current data.

### Natural vs. Engineered Affinity Limits

| Source | Typical Affinity Range | Best Achieved |
|--------|------------------------|---------------|
| **Natural immune response** | 1-100 nM (pKd 7-9) | ~100 pM (pKd 10) |
| **Therapeutic antibodies** | 1-100 nM (pKd 7-9) | ~1 nM (pKd 9) |
| **Engineered (in vitro)** | Variable | 48 fM (pKd 13.3) |

*Source: Boder et al. (2000), various therapeutic antibody reviews*

## File Relationships

```
references_master.md  ← Start here (overview + quick citations)
    ├── references_skempi2.md  ← Database details
    ├── references_sabdab.md   ← Database details
    ├── references_extreme_affinity.md  ← Scientific context
    └── references_class_imbalance.md   ← Implementation methods
```

## Export Options

### For LaTeX/BibTeX
All files include properly formatted BibTeX entries. Copy from individual files or use the consolidated entries in `references_master.md`.

### For Mendeley/Zotero
DOIs are provided for all papers. Use DOI lookup in your reference manager to import full citations.

### For Word/EndNote
PubMed IDs (PMIDs) are provided where available for direct import into EndNote.

## Updates and Maintenance

**Current version:** 2025-11-03

**To update:**
1. Check for new versions of SKEMPI and SAbDab databases
2. Search for recent papers on class imbalance in regression
3. Look for new extreme affinity achievements

**Suggested search terms:**
- "SKEMPI" + year
- "antibody affinity prediction" + year
- "class imbalance regression" + year
- "femtomolar antibody" + year

## Contact and Contributions

These reference files are part of the antibody-antigen binding prediction project. For questions or suggestions about additional references to include, please consult the project maintainer.

---

## Quick Start Checklist

- [ ] Read `references_master.md` for overview
- [ ] Review project-specific statistics in this file
- [ ] Check `references_class_imbalance.md` for implementation code
- [ ] Cite SKEMPI2 (Jankauskaitė et al., 2019) if using that data
- [ ] Cite SAbDab (Dunbar et al., 2014) if using that data
- [ ] Cite appropriate class imbalance methods used in your implementation

---

*Generated: 2025-11-03*
*Project: Antibody-Antigen Binding Prediction with Focus on Extreme Affinities*
*Total references compiled: 25+ papers*
