# SKEMPI Database References

## Primary Reference: SKEMPI 2.0

**Citation:**
Jankauskaitė, J., Jiménez-García, B., Dapkūnas, J., Fernández-Recio, J., & Moal, I. H. (2019). SKEMPI 2.0: an updated benchmark of changes in protein–protein binding energy, kinetics and thermodynamics upon mutation. *Bioinformatics*, 35(3), 462-469.

**DOI:** 10.1093/bioinformatics/bty635

**PubMed ID:** 30020414

**PMC ID:** PMC6361233

**Journal:** Bioinformatics, Oxford Academic Press

**Publication Date:** February 1, 2019

### Abstract Summary

SKEMPI 2.0 is a major update to a database of binding free energy changes upon mutation for structurally resolved protein-protein interactions. The database contains manually curated binding data for 7,085 mutations, representing a 133% increase from the previous version. It includes:

- **Binding affinity changes** for all 7,085 mutations
- **Kinetics data** for 1,844 mutations
- **Enthalpy and entropy changes** for 443 mutations
- **440 mutations** which abolish detectable binding

### Key Features

1. **Size:** 7,085 mutations (133% increase from SKEMPI 1.0)
2. **Data types:** ΔΔG, kinetics (kon, koff), thermodynamics (ΔH, ΔS)
3. **Coverage:** Includes antibody-antigen, enzyme-inhibitor, and other protein-protein interactions
4. **Accessibility:** Online at https://life.bsc.es/pid/skempi2/

### Applications

- Benchmark for computational methods predicting mutation effects
- Training data for machine learning models
- Analysis of structure-function relationships
- Study of protein binding thermodynamics and kinetics

### Database Statistics (Relevant to This Project)

From analysis of SKEMPI2 for antibody-antigen complexes:
- **Very strong binders** (pKd > 11): 741 entries (10.5% of database)
- **Strong binders** (pKd 9-11): 2,225 entries (31.4%)
- **Moderate binders** (pKd 7-9): 2,422 entries (34.2%)
- **Weak binders** (pKd 5-7): 1,327 entries (18.7%)
- **Very weak binders** (pKd < 5): 368 entries (5.2%)

**Antibody-antigen specific entries:**
- Very strong: 3 complexes
- Weak: 56 complexes
- Very weak: 13 complexes

### Data Format

CSV format with fields including:
- PDB code
- Mutation(s)
- Affinity wild-type (M)
- Affinity mutant (M)
- Protein 1 and Protein 2 names
- Temperature
- Kinetic parameters (kon, koff)
- Thermodynamic parameters (ΔH, ΔS)
- Experimental method

---

## Original SKEMPI Database

**Citation:**
Moal, I. H., & Fernández-Recio, J. (2012). SKEMPI: a Structural Kinetic and Energetic database of Mutant Protein Interactions and its use in empirical models. *Bioinformatics*, 28(20), 2600-2607.

**DOI:** 10.1093/bioinformatics/bts489

**Journal:** Bioinformatics, Oxford Academic Press

**Publication Date:** October 15, 2012

### Summary

Original version containing 3,047 mutations from 85 protein-protein complexes. Laid the foundation for systematic study of mutation effects on protein binding.

---

## Related Tools and Methods

### mCSM-PPI2

**Citation:**
Rodrigues, C. H., Pires, D. E., & Ascher, D. B. (2019). mCSM-PPI2: predicting the effects of mutations on protein–protein interactions. *Nucleic Acids Research*, 47(W1), W338-W344.

**DOI:** 10.1093/nar/gkz383

**Summary:**
Computational tool trained on SKEMPI database to predict mutation effects on protein-protein binding affinity.

---

## How to Cite SKEMPI2 in Your Work

```
Jankauskaitė, J., Jiménez-García, B., Dapkūnas, J., Fernández-Recio, J., & Moal, I. H. (2019).
SKEMPI 2.0: an updated benchmark of changes in protein–protein binding energy, kinetics and
thermodynamics upon mutation. Bioinformatics, 35(3), 462-469.
```

**BibTeX:**
```bibtex
@article{jankauskaite2019skempi,
  title={SKEMPI 2.0: an updated benchmark of changes in protein--protein binding energy, kinetics and thermodynamics upon mutation},
  author={Jankauskaite, Justina and Jim{\'e}nez-Garc{\'\i}a, Brian and Dapk{\=u}nas, Justas and Fern{\'a}ndez-Recio, Juan and Moal, Iain H},
  journal={Bioinformatics},
  volume={35},
  number={3},
  pages={462--469},
  year={2019},
  publisher={Oxford University Press}
}
```

---

*Reference file generated: 2025-11-03*
*For use in: Antibody-Antigen Binding Prediction Project*
