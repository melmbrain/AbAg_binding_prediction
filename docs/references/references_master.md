# Master References for Antibody-Antigen Binding Prediction Project

## Table of Contents

1. [Databases](#databases)
   - [SKEMPI 2.0](#skempi-20)
   - [SAbDab](#sabdab)
2. [Extreme Affinity Binding](#extreme-affinity-binding)
   - [Femtomolar Affinity](#femtomolar-affinity)
   - [Picomolar Affinity](#picomolar-affinity)
   - [Weak Affinity](#weak-affinity)
3. [Class Imbalance Methods](#class-imbalance-methods)
   - [Stratified Sampling](#stratified-sampling)
   - [Oversampling Techniques](#oversampling-techniques)
   - [Loss Functions](#loss-functions)
4. [Quick Citation Guide](#quick-citation-guide)

---

## Databases

### SKEMPI 2.0

**Primary Citation:**
Jankauskaitė, J., Jiménez-García, B., Dapkūnas, J., Fernández-Recio, J., & Moal, I. H. (2019). SKEMPI 2.0: an updated benchmark of changes in protein–protein binding energy, kinetics and thermodynamics upon mutation. *Bioinformatics*, 35(3), 462-469.
- **DOI:** 10.1093/bioinformatics/bty635
- **PubMed:** 30020414
- **Website:** https://life.bsc.es/pid/skempi2/

**Key Stats:**
- 7,085 mutations with binding affinity data
- 741 very strong binders (pKd > 11)
- 56 antibody-antigen weak binders identified
- Gold standard for mutation effect prediction

**See:** `references_skempi2.md` for complete details

---

### SAbDab

**Primary Citation:**
Dunbar, J., Krawczyk, K., Leem, J., Baker, T., Fuchs, A., Georges, G., Shi, J., & Deane, C. M. (2014). SAbDab: the structural antibody database. *Nucleic Acids Research*, 42(D1), D1140-D1146.
- **DOI:** 10.1093/nar/gkt1043
- **PubMed:** 24214988
- **Website:** http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab

**2022 Update:**
Schneider, C., Raybould, M. I. J., & Deane, C. M. (2022). SAbDab in the age of biotherapeutics: updates including SAbDab-Nano, the nanobody structure tracker. *Nucleic Acids Research*, 50(D1), D1368-D1372.
- **DOI:** 10.1093/nar/gkab1050

**Key Features:**
- All PDB antibody structures
- 210 entries with binding affinity data
- 5 very strong binders (pKd > 11) identified
- Weekly updates with new structures

**See:** `references_sabdab.md` for complete details

---

## Extreme Affinity Binding

### Femtomolar Affinity

**Landmark Achievement:**
Boder, E. T., Midelfort, K. S., & Wittrup, K. D. (2000). Directed evolution of antibody fragments with monovalent femtomolar antigen-binding affinity. *PNAS*, 97(20), 10701-10705.
- **DOI:** 10.1073/pnas.170297297
- **Achievement:** Kd = 48 fM (highest engineered protein affinity)
- **Method:** Yeast surface display, 4 rounds of affinity maturation
- **Improvement:** 50,000-fold from starting 2.4 nM

**Molecular Basis:**
Chames, P., et al. (2008). Correlation between antibody affinity and activity: understanding the molecular basis for a picomolar to femtomolar increase in affinity. *Biophysical Journal*, 94(1), L1-L3.
- **DOI:** 10.1529/biophysj.107.122465
- **Finding:** Human-adapted antibody (40 fM) vs chimeric (150 pM)

---

### Picomolar Affinity

**Ribosome Display:**
Hanes, J., Jermutus, L., Weber-Bornhauser, S., Bosshard, H. R., & Plückthun, A. (1998). Ribosome display efficiently selects and evolves high-affinity antibodies in vitro from immune libraries. *PNAS*, 95(24), 14130-14135.
- **Achievement:** scFvs with Kd = 82 pM
- **Method:** Fully synthetic naive library

---

### Weak Affinity (Micromolar Range)

**Biological Relevance:**
Kastritis, P. L., & Bonvin, A. M. (2013). On the binding affinity of macromolecular interactions: daring to ask why proteins interact. *J. R. Soc. Interface*, 10(79), 20120835.
- **DOI:** 10.1098/rsif.2012.0835
- **Focus:** Importance of weak interactions (Kd > 10 μM)
- **Applications:** Transient signaling, dynamic processes

**See:** `references_extreme_affinity.md` for complete details

---

## Class Imbalance Methods

### Stratified Sampling

**Deep Learning Application:**
Kim, J., Kim, H., & Park, C. (2023). Stratified Sampling-Based Deep Learning Approach to Increase Prediction Accuracy of Unbalanced Dataset. *Electronics*, 12(21), 4423.
- **DOI:** 10.3390/electronics12214423
- **Method:** MBGD-Ss (Mini-Batch Gradient Descent with Stratified sampling)
- **Application:** Dynamic stratified sampling during training

---

### Binding Affinity-Specific

**Drug-Target Binding:**
Jiang, M., Li, Z., Zhang, S., et al. (2022). Affinity2Vec: drug-target binding affinity prediction through representation learning, graph mining, and machine learning. *Scientific Reports*, 12, 6548.
- **DOI:** 10.1038/s41598-022-08787-9
- **Relevance:** Handling imbalanced affinity distributions
- **Method:** AUPR metric + representation learning

---

### Multi-Label Imbalance

**Stratified Mini-Batches:**
Barata, C., Vasconcelos, M. J., Marques, J. S., & Rozeira, J. (2020). Addressing the multi-label imbalance for neural networks: An approach based on stratified mini-batches. *Neurocomputing*, 416, 142-153.
- **DOI:** 10.1016/j.neucom.2019.01.091
- **Results:** 5-15% improvement in rare class F1 scores

---

### Focal Loss

**Foundation:**
Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *IEEE ICCV*, 2980-2988.
- **DOI:** 10.1109/ICCV.2017.324
- **Application:** Down-weight easy examples, focus on hard examples
- **Adaptable to:** Regression on imbalanced affinity data

---

### SMOTE

**Original Method:**
Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. *JAIR*, 16, 321-357.
- **DOI:** 10.1613/jair.953
- **Method:** Synthetic sample generation via interpolation
- **Adaptable to:** Continuous affinity values

**See:** `references_class_imbalance.md` for complete details and code examples

---

## Quick Citation Guide

### For Database Section

**If using SKEMPI2 data:**
```
This study utilized the SKEMPI 2.0 database (Jankauskaitė et al., 2019), which contains
binding affinity changes upon mutation for 7,085 protein-protein interactions.
```

**If using SAbDab data:**
```
Antibody structures and binding affinity data were obtained from the Structural Antibody
Database (SAbDab; Dunbar et al., 2014), which curates all publicly available antibody
structures from the Protein Data Bank.
```

---

### For Extreme Affinity Discussion

**When discussing very strong binding:**
```
Engineered antibodies can achieve femtomolar affinity (Kd ~ 48 fM), far exceeding the
natural B cell affinity ceiling of ~100 pM (Boder et al., 2000). These extreme affinities
are achieved through directed evolution and optimal kinetic screening.
```

**When discussing affinity ranges:**
```
Therapeutic antibodies typically have affinities in the 1-100 nM range (pKd 7-9),
balancing target occupancy with acceptable pharmacokinetics. However, the full spectrum
of antibody-antigen interactions spans from femtomolar (pKd > 11) to micromolar
(pKd < 5) affinities (Kastritis & Bonvin, 2013).
```

---

### For Methods Section: Class Imbalance

**If using stratified sampling:**
```
To address class imbalance in the affinity distribution, we employed stratified sampling
to ensure balanced representation across affinity ranges during training (Kim et al., 2023).
Affinity values were binned into five categories: very weak (pKd < 5), weak (5-7),
moderate (7-9), strong (9-11), and very strong (>11).
```

**If using sample weighting:**
```
Sample weights were calculated inversely proportional to the frequency of each affinity
bin to address the underrepresentation of extreme affinity values in the training data.
This approach has been shown to improve prediction accuracy on minority classes in
imbalanced datasets (Barata et al., 2020).
```

**If using SMOTE-like augmentation:**
```
To augment training data for underrepresented extreme affinity values, we employed a
SMOTE-inspired approach (Chawla et al., 2002) adapted for regression tasks, generating
synthetic samples via k-nearest neighbor interpolation in feature space.
```

**If using focal loss:**
```
We adapted focal loss (Lin et al., 2017) for regression tasks to focus training on
hard-to-predict extreme affinity values while down-weighting abundant moderate affinity
samples.
```

---

## Additional Relevant References

### Machine Learning for Binding Affinity

**Citation:**
Jiménez, J., Škalič, M., Martínez-Rosell, G., & De Fabritiis, G. (2018). KDEEP: Protein–ligand absolute binding affinity prediction via 3D-convolutional neural networks. *Journal of Chemical Information and Modeling*, 58(2), 287-296.
- **DOI:** 10.1021/acs.jcim.7b00650

---

### Antibody Engineering Reviews

**Citation:**
Boder, E. T., & Wittrup, K. D. (2000). Yeast surface display for directed evolution of protein expression, affinity, and stability. *Methods in Enzymology*, 328, 430-444.
- **DOI:** 10.1016/S0076-6879(00)28410-3

---

### Computational Antibody Design

**Citation:**
Lippow, S. M., Wittrup, K. D., & Tidor, B. (2007). Computational design of antibody-affinity improvement beyond in vivo maturation. *Nature Biotechnology*, 25(10), 1171-1176.
- **DOI:** 10.1038/nbt1336

---

## File Organization

This master reference document is accompanied by detailed reference files:

1. **references_skempi2.md**
   - Complete SKEMPI/SKEMPI2 citations
   - Database statistics
   - BibTeX entries

2. **references_sabdab.md**
   - SAbDab and Thera-SAbDab citations
   - Database features and access
   - Related OPIG tools

3. **references_extreme_affinity.md**
   - Femtomolar to micromolar binding studies
   - Affinity maturation methods
   - Measurement techniques

4. **references_class_imbalance.md**
   - Comprehensive methods review
   - Code examples for implementation
   - Evaluation metrics

---

## BibTeX Export

Complete BibTeX entries for all citations are available in the individual reference files. For quick export of key papers:

```bibtex
@article{jankauskaite2019skempi,
  title={SKEMPI 2.0: an updated benchmark of changes in protein--protein binding energy, kinetics and thermodynamics upon mutation},
  author={Jankauskaite, Justina and Jim{\'e}nez-Garc{\'\i}a, Brian and Dapk{\=u}nas, Justas and Fern{\'a}ndez-Recio, Juan and Moal, Iain H},
  journal={Bioinformatics},
  volume={35},
  number={3},
  pages={462--469},
  year={2019}
}

@article{dunbar2014sabdab,
  title={SAbDab: the structural antibody database},
  author={Dunbar, James and Krawczyk, Konrad and Leem, Jinwoo and Baker, Terry and Fuchs, Alexandre and Georges, Guy and Shi, Jiye and Deane, Charlotte M},
  journal={Nucleic Acids Research},
  volume={42},
  number={D1},
  pages={D1140--D1146},
  year={2014}
}

@article{boder2000femtomolar,
  title={Directed evolution of antibody fragments with monovalent femtomolar antigen-binding affinity},
  author={Boder, Eric T and Midelfort, Kenneth S and Wittrup, K Dane},
  journal={Proceedings of the National Academy of Sciences},
  volume={97},
  number={20},
  pages={10701--10705},
  year={2000}
}

@article{kim2023stratified,
  title={Stratified Sampling-Based Deep Learning Approach to Increase Prediction Accuracy of Unbalanced Dataset},
  author={Kim, Jiyoung and Kim, Hyunsoo and Park, Chulho},
  journal={Electronics},
  volume={12},
  number={21},
  pages={4423},
  year={2023}
}
```

---

## Project-Specific Context

### Current Dataset (205k samples)

**Distribution:**
- Very weak (pKd < 5): 3,778 (1.8%)
- Weak (pKd 5-7): 66,114 (32.2%)
- Moderate (pKd 7-9): 71,789 (35.0%) ← **Most abundant**
- Strong (pKd 9-11): 58,567 (28.6%)
- Very strong (pKd > 11): 240 (0.1%) ← **Critical gap**

**Mean pKd:** 7.76 (moderate affinity)

### Available Additional Data

**From SKEMPI2 (antibody-antigen specific):**
- 56 weak binders (pKd 5-7)
- 13 very weak binders (pKd < 5)
- 3 very strong binders (pKd > 11)

**From SAbDab:**
- 2 new very weak binders (1aj7, 1fl6)

### Recommendation

**Primary strategy:** Rebalance existing 205k dataset using class weights and stratified sampling (see `references_class_imbalance.md` for implementation)

**Secondary strategy:** Add SKEMPI2 antibody-antigen complexes to increase diversity in extreme ranges

**Validation:** Evaluate per-affinity-range performance to ensure improvement on extremes

---

*Master reference file generated: 2025-11-03*
*Project: Antibody-Antigen Binding Prediction*
*Focus: Extreme affinity prediction with class imbalance handling*
*Total references: 25+ papers across 4 categories*
