# References and Data Sources

**Project:** Antibody-Antigen Binding Affinity Prediction with Extreme Affinity Enhancement
**Date:** 2025-11-03
**Purpose:** Complete documentation of all data sources, methods, and references used

---

## Primary Data Sources

### 1. AbBiBench - Antibody Binding Benchmark Dataset

**Citation:**
```
Ecker, N., Hie, B., & Regev, A. (2024).
AbBiBench: A Large-Scale Antibody Binding Benchmark Dataset.
Scientific Data (in review).
Available: https://huggingface.co/datasets/AbBibench/Antibody_Binding_Benchmark_Dataset
```

**Details:**
- **Source:** Hugging Face Datasets
- **URL:** https://huggingface.co/datasets/AbBibench/Antibody_Binding_Benchmark_Dataset
- **Download Date:** 2025-11-03
- **Total Samples:** 185,718 antibody-antigen binding measurements
- **Affinity Range:** pKd 0.0001 - 13.22
- **License:** Open access
- **Data Type:** Synthetic antibody sequences with binding scores

**Used For:**
- Increasing dataset size by 90.6%
- Adding 3,452 very weak binders (pKd < 5)
- Adding 101 very strong binders (pKd > 11)

**Processing:**
```python
# Downloaded using Hugging Face datasets library
from datasets import load_dataset
dataset = load_dataset("AbBibench/Antibody_Binding_Benchmark_Dataset")
```

---

### 2. SAAINT-DB - Structural Antibody and AAI Database

**Citation:**
```
Huang, X., Zhou, J., Chen, S., Xia, X., Chen, Y.E., & Xu, J. (2025).
SAAINT-DB: A comprehensive structural antibody database for antibody
modeling and design.
Acta Pharmacologica Sinica.
DOI: 10.1038/s41401-025-01608-5
```

**Details:**
- **Source:** GitHub Repository
- **URL:** https://github.com/tommyhuangthu/SAAINT
- **Clone Date:** 2025-11-03
- **Version:** 2025-10-24 (latest release)
- **Total Entries:** 20,385 antibody structures
- **Affinity Data:** 6,158 measurements
- **Valid Kd Values:** 2,695 entries
- **Very Strong Binders:** 173 entries (pKd > 11)
- **License:** Academic use

**Used For:**
- High-affinity therapeutic antibodies
- Femtomolar-range binders (0.03 pM)
- 100% sequence coverage for very strong binders
- Added 53 unique very strong binders (after deduplication)

**Data Files Used:**
```
saaintdb/saaintdb_affinity_all.tsv
saaintdb/saaintdb_20251024_all.tsv
```

**Processing:**
```python
# Affinity data
df_affinity = pd.read_csv('external_data/SAAINT/saaintdb/saaintdb_affinity_all.tsv', sep='\t')

# Sequence data
df_main = pd.read_csv('external_data/SAAINT/saaintdb/saaintdb_20251024_all.tsv', sep='\t')

# Merged on PDB_ID, H_chain_ID, L_chain_ID
```

**Top Antibody Added:**
- PDB: 7rew
- pKd: 13.47 (Kd = 0.03 pM = 30 femtomolar)
- Method: Kinetic Exclusion Assay

---

### 3. SAbDab - Structural Antibody Database

**Citation:**
```
Dunbar, J., Krawczyk, K., Leem, J., Baker, T., Fuchs, A., Georges, G.,
Shi, J., & Deane, C.M. (2014).
SAbDab: the structural antibody database.
Nucleic Acids Research, 42(D1), D1140-D1146.
DOI: 10.1093/nar/gkt1043
```

**Details:**
- **Source:** Oxford Protein Informatics Group (OPIG)
- **URL:** http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab
- **Download Date:** 2025-11-03
- **Total Structures:** 19,852 entries
- **With Affinity Data:** 1,307 entries
- **Very Strong Binders:** 31 entries (pKd > 11)
- **Update Frequency:** Weekly
- **License:** Academic use

**Used For:**
- Validated structural data with affinity measurements
- Cross-validation with SAAINT-DB
- Additional very strong binders

**API/Download:**
```
http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all
```

**Top Antibody:**
- PDB: 5c7x
- pKd: 12.40 (Kd = 0.4 pM)
- Target: Granulocyte-macrophage colony-stimulating factor

---

### 4. Original Phase 6 Dataset

**Details:**
- **Source:** Internal dataset (previous work)
- **File:** `data/processed/phase6/final_205k_dataset.csv`
- **Total Samples:** 204,986 antibody-antigen pairs
- **Affinity Range:** pKd 0.0 - 15.70
- **Very Strong Binders:** 230 (pKd > 11)
- **Very Weak Binders:** 3,794 (pKd < 5)
- **Features:** 150 ESM2 PCA components + metadata

**Used For:**
- Baseline training dataset
- Immediate training with existing features

---

## Methods and Models

### 5. ESM2 - Protein Language Model

**Citation:**
```
Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., ... & Rives, A. (2023).
Evolutionary-scale prediction of atomic-level protein structure with a
language model.
Science, 379(6637), 1123-1130.
DOI: 10.1126/science.ade2574
```

**Details:**
- **Model:** facebook/esm2_t33_650M_UR50D
- **Parameters:** 650 million
- **Source:** Meta AI (Facebook Research)
- **Library:** Hugging Face Transformers 4.57.1
- **URL:** https://huggingface.co/facebook/esm2_t33_650M_UR50D

**Used For:**
- Generating protein sequence embeddings
- Antibody heavy and light chain encoding
- 1280-dimensional embeddings → 150 PCA components

**Implementation:**
```python
from transformers import AutoTokenizer, AutoModel

model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
```

---

### 6. Class Imbalance Handling Methods

#### Stratified Sampling

**Citation:**
```
Kohavi, R. (1995).
A study of cross-validation and bootstrap for accuracy estimation and
model selection.
In Proceedings of the 14th International Joint Conference on Artificial
Intelligence (IJCAI), Vol. 2, pp. 1137-1143.
```

**Implementation:**
- Ensures each batch contains samples from all affinity bins
- Prevents model bias toward majority class
- Custom PyTorch `StratifiedBatchSampler`

#### Focal Loss

**Citation:**
```
Lin, T.Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
Focal loss for dense object detection.
In Proceedings of the IEEE International Conference on Computer Vision
(ICCV), pp. 2980-2988.
DOI: 10.1109/ICCV.2017.324
```

**Adapted For Regression:**
```python
class FocalMSELoss(nn.Module):
    def __init__(self, gamma=2.0):
        # Down-weights easy examples
        # Focuses on hard-to-predict samples
```

#### Class Weighting

**Citation:**
```
He, H., & Garcia, E.A. (2009).
Learning from imbalanced data.
IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263-1284.
DOI: 10.1109/TKDE.2008.239
```

**Method:**
- Inverse frequency weighting
- Higher loss weight for rare classes
- `weight = total_samples / (n_classes * class_samples)`

---

## Software and Libraries

### 7. PyTorch

**Citation:**
```
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ...
& Chintala, S. (2019).
PyTorch: An imperative style, high-performance deep learning library.
Advances in Neural Information Processing Systems, 32.
```

**Version:** 2.7.1+cu118
**Used For:** Deep learning model training, GPU/CPU computation

---

### 8. Pandas

**Citation:**
```
McKinney, W. (2010).
Data structures for statistical computing in Python.
In Proceedings of the 9th Python in Science Conference, Vol. 445,
pp. 51-56.
```

**Version:** Latest
**Used For:** Data manipulation, CSV processing, integration

---

### 9. Scikit-learn

**Citation:**
```
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B.,
Grisel, O., ... & Duchesnay, É. (2011).
Scikit-learn: Machine learning in Python.
Journal of Machine Learning Research, 12, 2825-2830.
```

**Used For:** PCA transformation, stratified splitting, metrics

---

### 10. Hugging Face Transformers

**Citation:**
```
Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ...
& Rush, A.M. (2020).
Transformers: State-of-the-art natural language processing.
In Proceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing: System Demonstrations, pp. 38-45.
```

**Version:** 4.57.1
**Used For:** ESM2 model loading and inference

---

## Additional Data Sources Documented (Not Downloaded Yet)

### 11. Ab-CoV - COVID-19 Antibody Database

**Citation:**
```
Deshpande, A., Pawar, S., Paschapur, A., Tiwari, A., & Chandra, N. (2021).
Ab-CoV: A database of antibodies to SARS-CoV-2 and related coronaviruses.
Database, 2021, baab054.
DOI: 10.1093/database/baab054
```

**Details:**
- **URL:** https://web.iitm.ac.in/ab-cov/home
- **Total Antibodies:** 1,780
- **Kd Measurements:** 568
- **Expected Very Strong:** 100-200
- **Status:** Documented for future integration

---

### 12. CoV-AbDab - Coronavirus Antibody Database

**Citation:**
```
Raybould, M.I.J., Kovaltsuk, A., Marks, C., & Deane, C.M. (2021).
CoV-AbDab: the coronavirus antibody database.
Bioinformatics, 37(5), 734-735.
DOI: 10.1093/bioinformatics/btaa739
```

**Details:**
- **URL:** http://opig.stats.ox.ac.uk/webapps/coronavirus
- **Total Entries:** 12,916
- **Update:** February 8, 2024
- **Status:** Documented for cross-referencing

---

### 13. Thera-SAbDab - Therapeutic Antibody Database

**Citation:**
```
Raybould, M.I.J., Marks, C., Krawczyk, K., Taddese, B., Nowak, J.,
Lewis, A.P., ... & Deane, C.M. (2020).
Thera-SAbDab: the Therapeutic Structural Antibody Database.
Nucleic Acids Research, 48(D1), D383-D388.
DOI: 10.1093/nar/gkz827
```

**Details:**
- **URL:** http://opig.stats.ox.ac.uk/webapps/newsabdab/therasabdab/
- **Therapeutic Antibodies:** 461
- **FDA-Approved Examples:** Many
- **Status:** Documented for future integration

---

## Data Processing and Integration

### Integration Workflow

**Date:** 2025-11-03

**Step 1: AbBiBench Download**
```bash
python scripts/download_abbibench.py
# Output: external_data/abbibench_raw.csv (44.80 MB, 185,718 samples)
```

**Step 2: SAAINT-DB Clone**
```bash
git clone https://github.com/tommyhuangthu/SAAINT.git external_data/SAAINT
# Processed: 173 very strong binders with 100% sequence coverage
```

**Step 3: SAbDab Download**
```bash
python scripts/download_therapeutic_antibodies.py
# Output: 1,307 antibodies with affinity, 31 very strong
```

**Step 4: Integration**
```bash
python scripts/integrate_all_databases.py
# Output: external_data/merged_with_abbibench.csv (390,704 samples)

python scripts/integrate_therapeutic_antibodies.py
# Output: external_data/merged_with_therapeutics.csv (390,757 samples)
```

**Deduplication:**
- Method: PDB code comparison
- AbBiBench: Assigned synthetic codes (abb0000, abb0001, ...)
- SAAINT-DB: 120 duplicates removed, 53 unique added
- SAbDab: Included in SAAINT duplicates

---

## Final Dataset Statistics

### Merged Dataset (merged_with_therapeutics.csv)

**Total Samples:** 390,757
**File Size:** 499.20 MB
**Columns:** 158 (150 ESM2 PCA + metadata)

**Affinity Distribution:**

| Category | Count | Percentage |
|----------|-------|------------|
| Very Weak (<5 pKd) | 7,246 | 1.85% |
| Weak (5-7 pKd) | 133,314 | 34.12% |
| Moderate (7-9 pKd) | 124,594 | 31.89% |
| Strong (9-11 pKd) | 116,223 | 29.74% |
| Very Strong (>11 pKd) | 384 | 0.10% |

**Data Sources Breakdown:**

| Source | Samples | Very Strong | Features |
|--------|---------|-------------|----------|
| Original Phase 6 | 204,986 | 230 | Complete |
| AbBiBench | 185,718 | 101 | Pending |
| SAAINT-DB | 53 | 53 | Pending |
| **Total** | **390,757** | **384** | Mixed |

---

## Computational Methods

### Embedding Generation

**Method:** ESM2 protein language model
**Configuration:**
- Model: facebook/esm2_t33_650M_UR50D
- Device: CPU (to avoid GPU conflict)
- Batch size: 16 sequences
- Input: Concatenated heavy + light chain sequences
- Output: 1280-dimensional embeddings
- Reduction: PCA to 150 components

**Checkpoint System:**
- Frequency: Every 50 batches
- File: embedding_checkpoint.pkl
- Auto-resume: Yes
- Maximum work lost: ~10 minutes

### Training Configuration

**Model Architecture:**
- Input: 150 ESM2 PCA features
- Hidden layers: [512, 256, 128]
- Output: 1 (pKd prediction)
- Activation: ReLU
- Dropout: 0.3

**Loss Functions:**
- Weighted MSE (inverse frequency)
- Focal MSE (gamma=2.0)

**Sampling:**
- Stratified batch sampling
- Ensures representation from all affinity bins

**Optimization:**
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32
- Epochs: 100

---

## Quality Control

### Data Validation

**Duplicate Removal:**
```python
# PDB code comparison
existing_pdbs = set(df_existing['pdb_code'].str.lower())
duplicates = df_new['pdb_code'].str.lower().isin(existing_pdbs)

# Sequence fingerprint comparison (backup)
df['seq_fp'] = df['heavy_chain_seq'] + '||' + df['light_chain_seq']
```

**Affinity Conversion:**
```python
# Standardized to pKd
pKd = -log10(Kd_M)

# Validation: 0 <= pKd <= 16
assert (df['pKd'] >= 0).all() and (df['pKd'] <= 16).all()
```

**Sequence Validation:**
```python
# Standard amino acids only
valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
assert all(set(seq).issubset(valid_aa) for seq in sequences)
```

---

## Reproducibility

### Random Seeds
```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

### Environment
```
Python: 3.10+
PyTorch: 2.7.1+cu118
CUDA: 11.8
Transformers: 4.57.1
Pandas: latest
NumPy: latest
Scikit-learn: latest
```

### Hardware
```
GPU: NVIDIA GeForce RTX 2060 (6GB)
CPU: Available for embedding generation
RAM: Sufficient for 650M parameter model
Disk: 568 GB available
```

---

## File Locations

### Data Files
```
external_data/merged_with_therapeutics.csv          # Final integrated dataset
external_data/train_ready_with_features.csv         # Filtered for immediate training
external_data/abbibench_raw.csv                     # AbBiBench download
external_data/SAAINT/                               # SAAINT-DB clone
external_data/therapeutic/                          # Processed therapeutic data
```

### Scripts
```
scripts/download_abbibench.py                       # AbBiBench downloader
scripts/download_therapeutic_antibodies.py          # Multi-database downloader
scripts/integrate_all_databases.py                  # Universal integrator
scripts/generate_embeddings_incremental.py          # ESM2 embedding generator
scripts/train_with_existing_features.py             # Feature filter
scripts/check_embedding_progress.py                 # Progress monitor
```

### Documentation
```
VACCINE_ANTIBODY_SOURCES.md                        # Therapeutic antibody guide
THERAPEUTIC_ANTIBODY_INTEGRATION_REPORT.md         # Integration analysis
EXTREME_AFFINITY_REPORT.md                         # Extreme affinity focus
DUAL_COMPUTATION_GUIDE.md                          # GPU conflict solutions
REFERENCES_AND_DATA_SOURCES.md                     # This file
```

---

## Acknowledgments

### Data Providers
- Meta AI (Facebook Research) - ESM2 model
- Hugging Face - AbBiBench dataset hosting
- University of Michigan - SAAINT-DB
- Oxford Protein Informatics Group - SAbDab, CoV-AbDab, Thera-SAbDab
- IIT Madras - Ab-CoV database

### Software
- PyTorch team
- Hugging Face Transformers team
- Pandas development team
- Scikit-learn developers
- NumPy community

---

## License and Usage

### Data Licenses
- **AbBiBench:** Open access
- **SAAINT-DB:** Academic use
- **SAbDab:** Academic use, cite original paper
- **ESM2:** MIT License (Meta AI)

### Proper Citation Required For:
1. All databases used (see citations above)
2. ESM2 model (Lin et al., 2023)
3. Methods (Focal Loss, Stratified Sampling, etc.)
4. Software libraries (PyTorch, Transformers, etc.)

---

## Document Information

**Created:** 2025-11-03
**Author:** Antibody-Antigen Binding Prediction Project
**Purpose:** Complete research documentation and reproducibility
**Status:** Active - Updated as new sources are added

**Last Updated:** 2025-11-03
**Version:** 1.0

---

## Future Data Sources (Planned)

The following sources are documented but not yet integrated:

1. **Ab-CoV** - 568 Kd measurements
2. **CoV-AbDab** - 12,916 sequences for cross-referencing
3. **Thera-SAbDab** - 461 therapeutic antibodies
4. **SKEMPI2** - Mutation effects on binding
5. **PDBbind** - Additional protein-protein complexes

When integrated, this document will be updated with download dates, processing methods, and citations.

---

**For questions or updates to this documentation:**
Contact: [Project Lead/Researcher]
Repository: [If applicable]
Date: 2025-11-03

---
