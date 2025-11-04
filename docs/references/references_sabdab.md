# SAbDab (Structural Antibody Database) References

## Primary Reference: SAbDab

**Citation:**
Dunbar, J., Krawczyk, K., Leem, J., Baker, T., Fuchs, A., Georges, G., Shi, J., & Deane, C. M. (2014). SAbDab: the structural antibody database. *Nucleic Acids Research*, 42(D1), D1140-D1146.

**DOI:** 10.1093/nar/gkt1043

**PubMed ID:** 24214988

**PMC ID:** PMC3965125

**Journal:** Nucleic Acids Research, Oxford Academic Press

**Publication Date:** January 2014

**Website:** http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab

### Abstract Summary

SAbDab is an online resource containing all publicly available antibody structures, annotated and presented in a consistent fashion. The database provides:

- **Structural data** for all antibody structures in the Protein Data Bank (PDB)
- **Annotations** including experimental information, gene details, and chain pairings
- **Antigen information** including binding affinity data where available
- **Weekly updates** to include new structures

### Key Features

1. **Comprehensive coverage:** All antibody structures from PDB
2. **Consistent annotation:** Standardized heavy/light chain identification
3. **Gene assignment:** IMGT germline gene identification
4. **Binding affinity data:** Where experimentally determined
5. **Antigen classification:** Categorization of bound antigens
6. **Search functionality:** Multiple search options by PDB code, sequence, species, etc.

### Database Statistics (Relevant to This Project)

From our analysis of SAbDab affinity data:
- **Total entries with affinity:** 210 antibody-antigen complexes
- **Very strong binders** (pKd > 11): 5 complexes
  - 2nyy (pKd = 11.61, Kd = 2.48 pM)
  - 3h42 (pKd = 11.40, Kd = 4.0 pM)
  - 3lhp (pKd = 11.12, Kd = 7.5 pM)
  - 2nz9 (pKd = 11.17, Kd = 6.8 pM)
  - 2bdn (pKd = 11.34, Kd = 4.6 pM)
- **Very weak binders** (pKd < 5): 4 complexes
  - 2oqj (pKd = 3.70, Kd = 200 μM)
  - 1aj7 (pKd = 3.87, Kd = 135 μM)
  - 1nby (pKd = 4.04, Kd = 90.9 μM)
  - 1fl6 (pKd = 4.60, Kd = 25 μM)

### Applications

- Training machine learning models for antibody engineering
- Studying antibody-antigen recognition patterns
- Benchmark for computational antibody design tools
- Analysis of CDR conformations and paratope-epitope interfaces

---

## Major Update: SAbDab 2022

**Citation:**
Schneider, C., Raybould, M. I. J., & Deane, C. M. (2022). SAbDab in the age of biotherapeutics: updates including SAbDab-Nano, the nanobody structure tracker. *Nucleic Acids Research*, 50(D1), D1368-D1372.

**DOI:** 10.1093/nar/gkab1050

**PubMed ID:** 34791371

**PMC ID:** PMC8728194

**Publication Date:** January 2022

### Key Updates

1. **SAbDab-Nano:** Specialized tracker for nanobody (VHH) structures
2. **Expanded coverage:** Includes single-domain antibodies
3. **Enhanced annotations:** Improved therapeutic antibody identification
4. **Updated interface:** Better search and visualization tools

---

## Related Database: Thera-SAbDab

**Citation:**
Raybould, M. I. J., Marks, C., Krawczyk, K., Taddese, B., Nowak, J., Lewis, A. P., Bujotzek, A., Shi, J., & Deane, C. M. (2020). Thera-SAbDab: the Therapeutic Structural Antibody Database. *Nucleic Acids Research*, 48(D1), D383-D388.

**DOI:** 10.1093/nar/gkz827

**PubMed ID:** 31555805

**PMC ID:** PMC6943036

**Website:** http://opig.stats.ox.ac.uk/webapps/newsabdab/therasabdab/

### Summary

Curated subset of SAbDab containing structures of therapeutic antibodies and their targets, with additional annotations:
- Clinical development stage
- Therapeutic indication
- Target antigen details
- Binding site analysis

---

## Data Access and Formats

### Direct Downloads
- **CSV/TSV files:** Available for bulk download
- **PDB files:** Individual structure files
- **FASTA sequences:** Antibody sequences
- **Annotated summaries:** Including affinity data

### API Access
SAbDab provides programmatic access for automated data retrieval and integration into computational pipelines.

---

## Affinity Data in SAbDab

### Measurement Types
- **Kd** (Equilibrium dissociation constant)
- **Ka** (Association constant)
- **IC50** (Half-maximal inhibitory concentration)
- **EC50** (Half-maximal effective concentration)

### Data Sources
Affinity values are extracted from:
1. Original PDB deposition
2. Literature citations
3. Binding databases (BindingDB, PDBbind)

### Quality Notes
- Not all structures have affinity data
- Measurement conditions may vary (temperature, pH, buffer)
- Some values are estimated or measured for related constructs

---

## How to Cite SAbDab in Your Work

```
Dunbar, J., Krawczyk, K., Leem, J., Baker, T., Fuchs, A., Georges, G., Shi, J., & Deane, C. M. (2014).
SAbDab: the structural antibody database. Nucleic Acids Research, 42(D1), D1140-D1146.
```

**BibTeX:**
```bibtex
@article{dunbar2014sabdab,
  title={SAbDab: the structural antibody database},
  author={Dunbar, James and Krawczyk, Konrad and Leem, Jinwoo and Baker, Terry and Fuchs, Alexandre and Georges, Guy and Shi, Jiye and Deane, Charlotte M},
  journal={Nucleic Acids Research},
  volume={42},
  number={D1},
  pages={D1140--D1146},
  year={2014},
  publisher={Oxford University Press}
}
```

**For 2022 Update:**
```bibtex
@article{schneider2022sabdab,
  title={SAbDab in the age of biotherapeutics: updates including SAbDab-Nano, the nanobody structure tracker},
  author={Schneider, Constantin and Raybould, Matthew IJ and Deane, Charlotte M},
  journal={Nucleic Acids Research},
  volume={50},
  number={D1},
  pages={D1368--D1372},
  year={2022},
  publisher={Oxford University Press}
}
```

---

## Related Oxford Protein Informatics Group (OPIG) Tools

### SAbPred
Antibody structure prediction tool integrated with SAbDab

### ABangle
Tool for analyzing antibody-antigen binding angles

### Parapred
Paratope (antibody binding site) prediction

**Website:** http://opig.stats.ox.ac.uk/

---

*Reference file generated: 2025-11-03*
*For use in: Antibody-Antigen Binding Prediction Project*
