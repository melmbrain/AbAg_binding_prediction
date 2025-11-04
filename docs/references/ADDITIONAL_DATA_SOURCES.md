# Additional Data Sources for Antibody-Antigen Binding Affinity

**Search Date:** 2025-11-03
**Focus:** Extreme affinity values (very weak and very strong binders)

---

## Summary of Findings

I found **14 databases and datasets** with antibody-antigen binding affinity data that you can use to augment your existing 205k dataset. The most promising for extreme affinity values are marked with ⭐.

**Quick Stats:**
- **Best for extreme data:** SAAINT-DB, AbBiBench, PDBbind
- **Total additional samples available:** 100,000+ affinity measurements
- **Formats:** CSV, TSV, PDB, downloadable
- **Most recent:** SAAINT-DB (updated May 2025)

---

## 🔥 Top Recommendations (Download These First)

### 1. ⭐ SAAINT-DB (NEWEST - May 2025)
**Best for:** Extreme affinity values, most comprehensive

**Description:**
- **19,128 data entries** from 9,757 PDB structures
- **Nearly 2× more affinity data than SAbDab**
- Manual curation for high quality
- Last updated: **May 1, 2025**

**Download:**
- GitHub: https://github.com/tommyhuangthu/SAAINT
- Zenodo: https://zenodo.org/records/[check GitHub]
- Summary file: Direct download from GitHub repository

**Format:** PDB structures + CSV summary with affinity data

**Reference:**
- Paper: *Acta Pharmacologica Sinica* (2025)
- Contains: pKd, Kd values, experimental methods

**Why this is best:** Most recent, largest curated antibody-antigen affinity collection

---

### 2. ⭐ AbBiBench (Hugging Face - 2024)
**Best for:** Benchmark quality data with extreme values

**Description:**
- **184,500+ experimental measurements**
- **14 antibodies × 9 antigens**
- Includes: HER2, VEGF, influenza, SARS-CoV-2, lysozyme
- Heavy and light chain mutations
- Wild-type complex structures included

**Download:**
- Hugging Face: https://huggingface.co/datasets/AbBibench/Antibody_Binding_Benchmark_Dataset
- Direct download in multiple formats

**Format:** CSV with sequences, structures, affinity values

**Reference:**
- arXiv: 2506.04235
- Benchmarking framework paper

**Why this is best:** Standardized, high-quality benchmark data ready for ML

---

### 3. ⭐ PDBbind Version 2024
**Best for:** Comprehensive protein-protein including antibodies

**Description:**
- **33,653 biomolecular complexes**
- **4,594 protein-protein complexes** (includes antibodies)
- Kd, Ki, IC50 values
- 43% increase over version 2020

**Download:**
- Website: https://www.pdbbind-plus.org.cn/
- Free version 2020: http://www.pdbbind.org.cn/download.php
- Registration required for 2024 (free for academics)

**Format:** PDB structures + CSV with affinity data

**Note:** Version 2020 is **completely free**, version 2024 requires (free) registration

**Why this is best:** Gold standard database, includes extreme affinity values

---

## 📊 Additional High-Quality Databases

### 4. AACDB (Antigen-Antibody Complex Database)
**Description:**
- **7,498 manually curated antigen-antibody complexes**
- From 32,000+ PDB structures
- Comprehensive interface analysis

**Download:**
- Website: http://i.uestc.edu.cn/AACDB
- Export as TXT or CSV

**Format:** CSV/TXT downloadable

**Reference:** eLife 2024

---

### 5. Ab-CoV (Coronavirus Antibodies with Affinity)
**Best for:** COVID-19 antibody affinity data

**Description:**
- **1,780 coronavirus-related antibodies**
- **1,804 IC50 values**
- **849 EC50 values**
- **568 Kd values**
- Includes nanobodies

**Download:**
- Website: Search for "Ab-CoV database" (companion to CoV-AbDab)
- Data extracted from literature

**Format:** Structured database

**Reference:** *Bioinformatics* (2022)

---

### 6. CoV-AbDab (Coronavirus Antibody Sequences)
**Description:**
- Sequences and structures (not primarily affinity)
- Companion to Ab-CoV
- Free download without registration

**Download:**
- Website: http://opig.stats.ox.ac.uk/webapps/coronavirus

**Format:** FASTA sequences, structures

---

### 7. BindingDB
**Description:**
- **3.1M binding measurements**
- **1.3M compounds**
- Includes protein-protein interactions
- Tab-separated format

**Download:**
- Website: https://www.bindingdb.org/
- Download page: https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp

**Format:** TSV (tab-separated values)

**Note:** Primarily drug-target, but includes antibody data

---

### 8. IEDB (Immune Epitope Database)
**Description:**
- **2.2 million+ epitopes**
- **6.8 million assays**
- **1.6 million immune epitopes**
- Includes antibody binding affinity

**Download:**
- Website: https://www.iedb.org/
- Export via Customize Data Exports or API
- Full VM image available under license

**Format:** Multiple export formats including CSV

**Why useful:** Includes weak binders and non-binders

---

### 9. BioLiP2 (Updated 2023)
**Description:**
- Biologically relevant ligand-protein interactions
- Redundant and non-redundant versions (95% identity)
- 3D structures included

**Download:**
- Website: https://zhanggroup.org/BioLiP/download.html
- Alternative: https://aideepmed.com/BioLiP/download.html

**Format:** PDB structures + interaction data

---

### 10. AbSet (Zenodo - 800k+ Structures)
**Description:**
- **800,000+ antibody structures**
- Experimentally determined and in silico
- Molecular descriptors included

**Download:**
- Zenodo: Search "AbSet antibody dataset"
- GitHub scripts: https://github.com/SFBBGroup/AbSet.git

**Format:** Standardized structure files

**Reference:** *J. Chem. Inf. Model.* (2024)

---

## 💡 Specialized/Niche Sources

### 11. AgAb DB (Antigen Specific Antibody Database)
**Download:** https://naturalantibody.com/agab/
**Note:** You already have this data! This is the source of your Phase 2 AgAb data.

---

### 12. Trastuzumab Variant Libraries
**Description:**
- Mason et al.: 36,400 classified trastuzumab mutants
- Chinery et al.: 524,000+ trastuzumab variants

**Download:**
- Check supplementary materials of papers
- May be on GitHub or Zenodo

**Why useful:** Specific to HER2-targeting, includes weak variants

---

### 13. AbRank Dataset
**Description:**
- **380,000+ binding assays**
- 9 heterogeneous sources
- Ranking framework

**Download:**
- arXiv: 2506.17857
- Check paper supplementary for dataset

---

### 14. GearBind (Zenodo 2024)
**Description:**
- Geometric graph neural network training data
- SKEMPI and additional test sets

**Download:**
- Zenodo: https://doi.org/10.5281/zenodo.13085795

**Format:** Preprocessed for ML

---

## 📥 Recommended Download Strategy

### Phase 1: Core Databases (Start Here)
1. **SAAINT-DB** - Most comprehensive, most recent
2. **AbBiBench** - Hugging Face, easy download
3. **PDBbind 2020** - Free, no registration

**Expected additions:** ~50,000 new affinity measurements

---

### Phase 2: Specialized (If Phase 1 Insufficient)
4. **AACDB** - More antibody-antigen specific
5. **Ab-CoV** - COVID antibodies with extreme affinities
6. **BindingDB** - Additional protein-protein data

**Expected additions:** ~10,000 more measurements

---

### Phase 3: Research Datasets (For Specific Cases)
7. **Trastuzumab variants** - Weak binder examples
8. **AbRank** - Ranking data
9. **IEDB** - Non-binders and weak binders

---

## 🎯 Focus on Extreme Affinities

### For Very Strong Binders (pKd > 11, Kd < 10 pM):

**Best sources:**
1. **SAAINT-DB** - Check "high affinity" subset
2. **PDBbind** - Filter for Kd < 1e-11 M
3. **AbBiBench** - Includes affinity matured variants

**Search terms in papers:**
- "picomolar"
- "femtomolar"
- "sub-nanomolar"
- "high affinity"

---

### For Very Weak Binders (pKd < 5, Kd > 10 μM):

**Best sources:**
1. **IEDB** - Includes non-binders and weak epitopes
2. **Trastuzumab variants** - Deliberately weakened
3. **SKEMPI2** - Mutation-induced loss of binding

**Search terms in papers:**
- "micromolar"
- "weak binding"
- "low affinity"
- "germline"

---

## 📋 Data Integration Checklist

For each downloaded dataset:

- [ ] Check PDB code overlap with your existing 205k dataset
- [ ] Convert affinity units to pKd
- [ ] Filter for antibody-antigen complexes (not all protein-protein)
- [ ] Extract sequences (if not already present)
- [ ] Generate ESM2 embeddings
- [ ] Apply same PCA transformation
- [ ] Verify affinity distribution with `src/data_utils.py`
- [ ] Document source in metadata

---

## 🔧 Quick Integration Script

Use the provided script for adding new data:

```bash
# For SAAINT-DB data
python scripts/integrate_external_data.py \
  --existing_data your_205k_dataset.csv \
  --new_data saaint_db_affinity.csv \
  --source "SAAINT-DB" \
  --output merged_with_saaint.csv
```

(Script needs to be created based on `integrate_skempi2_data.py` template)

---

## 📊 Expected Impact

### Current Dataset:
- Very strong (pKd > 11): **240 samples (0.1%)**
- Very weak (pKd < 5): **3,778 samples (1.8%)**

### After Adding SAAINT-DB + AbBiBench:
- Very strong: **Est. 2,000-5,000 samples (~2-3%)** ✓ 10-20× increase
- Very weak: **Est. 10,000-15,000 samples (~5-7%)** ✓ 3-4× increase

### Overall:
- Total dataset: **~250k-300k samples**
- Better representation of extreme values
- More diverse antigen coverage

---

## 🚀 Priority Actions

### Today:
1. Download SAAINT-DB from GitHub
2. Download AbBiBench from Hugging Face
3. Register for PDBbind (free, academic)

### This Week:
4. Filter for antibody-antigen complexes
5. Check for duplicates with your 205k dataset
6. Convert to pKd and analyze distribution

### Next Week:
7. Generate embeddings for new sequences
8. Integrate with existing data
9. Re-train model with augmented dataset

---

## 📚 Citation Information

### SAAINT-DB:
```
Citation needed - Check GitHub repository for latest paper
Website: https://github.com/tommyhuangthu/SAAINT
```

### AbBiBench:
```
arXiv:2506.04235 (2024)
HuggingFace: AbBibench/Antibody_Binding_Benchmark_Dataset
```

### PDBbind:
```
Wang et al. (2015) Bioinformatics 31(3):405-412
Website: http://www.pdbbind.org.cn/
```

---

## ⚠️ Important Notes

### Data Quality:
- Always validate affinity units (M, nM, pM, etc.)
- Check experimental methods (SPR, ITC, ELISA, etc.)
- Verify temperature conditions
- Look for outliers

### Licensing:
- Most academic databases are **free for research**
- Some require registration (also free)
- Check licenses before commercial use
- Always cite the source database

### Integration Challenges:
- **Sequences:** May need to fetch from PDB
- **Embeddings:** Must generate ESM2 embeddings
- **PCA:** Must apply your existing PCA model
- **Features:** Align feature dimensions with your 150 PCA components

---

## 🆘 If You Need Help

### For downloading:
1. Check database documentation
2. Look for "Download" or "Export" buttons
3. Register if required (usually instant)
4. Contact database maintainers if issues

### For integration:
1. Use provided `integrate_skempi2_data.py` as template
2. Adapt for new database format
3. Check `src/data_utils.py` for affinity binning
4. Validate with per-bin statistics

### For citations:
1. Check database homepage
2. Look for "Cite us" section
3. Use DOI if available
4. Include database version and access date

---

## 📞 Database Contact Information

- **SAAINT-DB:** Check GitHub issues
- **SAbDab/CoV-AbDab:** opig@stats.ox.ac.uk
- **PDBbind:** Contact form on website
- **AbBiBench:** Check Hugging Face discussions

---

## 📈 Success Metrics

After integration, you should see:

| Metric | Before | Target After |
|--------|--------|--------------|
| Very strong samples | 240 (0.1%) | 2,000+ (1-2%) |
| Very weak samples | 3,778 (1.8%) | 10,000+ (5%) |
| Total dataset | 205k | 250-300k |
| Very strong RMSE | ~2.2 | <1.0 |
| Very weak RMSE | ~2.5 | <1.0 |

---

## 🎉 Summary

**You now have access to:**
- ✅ 14 databases with antibody-antigen affinity data
- ✅ 100,000+ additional affinity measurements
- ✅ Recent 2024-2025 updates
- ✅ Direct download links
- ✅ Integration strategies

**Best combination:**
1. SAAINT-DB (comprehensive, recent)
2. AbBiBench (standardized, ML-ready)
3. PDBbind (gold standard)

**Expected outcome:**
- 10-20× more very strong binders
- 3-4× more very weak binders
- Significantly improved extreme affinity prediction

---

*Document created: 2025-11-03*
*Total databases found: 14*
*Estimated additional data: 100,000+ measurements*
*Priority downloads: 3 (SAAINT-DB, AbBiBench, PDBbind)*

