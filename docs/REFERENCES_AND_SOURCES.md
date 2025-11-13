# References & Sources

**Project**: Antibody-Antigen Binding Affinity Prediction
**Compiled**: November 2025

---

## 📚 Primary Literature (Key Papers)

### IgT5 - State-of-the-Art Antibody Model

**Paper**: "Large scale paired antibody language models"
- **Authors**: Kenlay, H., Dreyer, F.A., Kovaltsuk, A., Miketa, D., Pires, D., Deane, C.M.
- **Journal**: PLOS Computational Biology
- **Date**: December 2024
- **DOI**: 10.1371/journal.pcbi.1012646
- **URL**: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012646
- **Model**: https://huggingface.co/Exscientia/IgT5

**Key Findings**:
- Trained on 2 billion unpaired + 2 million paired antibody sequences
- Best binding affinity prediction: R² 0.297-0.306
- Outperforms AntiBERTy, AbLang, ProtBert
- Paired training captures heavy+light chain interactions
- First T5 encoder-decoder antibody language model

**Relevance**: State-of-the-art model for antibody embeddings

---

### ESM-2 - Evolutionary Scale Modeling

**Paper**: "Evolutionary-scale prediction of atomic-level protein structure with a language model"
- **Authors**: Lin, Z., Akin, H., Rao, R., et al.
- **Journal**: Science
- **Date**: 2023
- **DOI**: 10.1126/science.ade2574
- **URL**: https://www.science.org/doi/10.1126/science.ade2574
- **Model**: https://huggingface.co/facebook/esm2_t33_650M_UR50D

**Key Findings**:
- 650M parameter protein language model
- Trained on 250 million protein sequences
- Achieves state-of-the-art on structure prediction
- Rich 1280-dim embeddings capture evolutionary information

**Relevance**: Used for antigen embeddings in our model

---

### EpiGraph - ESM-2 for Epitope Prediction

**Paper**: "B cell epitope prediction by capturing spatial clustering property of the epitopes using graph attention network"
- **Authors**: Wang, Y., et al.
- **Journal**: Scientific Reports
- **Date**: 2024
- **DOI**: 10.1038/s41598-024-78506-z
- **URL**: https://www.nature.com/articles/s41598-024-78506-z

**Key Findings**:
- Uses ESM-2 + ESM-IF1 embeddings
- Graph attention network for spatial features
- AUC 0.23-0.24 (best on benchmark)
- Outperforms BepiPred-3.0, DiscoTope-3.0

**Relevance**: Validates ESM-2 for antigen epitope prediction

---

### CALIBER - ESM-2 + Bi-LSTM

**Paper**: "Enhanced prediction of antigen and antibody binding interface using ESM-2 and Bi-LSTM"
- **Authors**: Various
- **Journal**: Molecular Immunology
- **Date**: 2025
- **DOI**: 10.1016/j.molimm.2025.000758
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S0198885925000758

**Key Findings**:
- ESM-2 encoder + Bi-LSTM network
- AUC 0.789 for conformational B cell epitopes
- AUC 0.776 for linear epitopes
- Superior performance on epitope-paratope prediction

**Relevance**: Confirms ESM-2 as best for antigen binding sites

---

### SEMA 2.0 - ESM-2 3B Benchmark

**Paper**: "SEMA 2.0: web-platform for B-cell conformational epitopes prediction using artificial intelligence"
- **Authors**: Various
- **Journal**: Nucleic Acids Research
- **Date**: 2024
- **DOI**: 10.1093/nar/gkae372
- **URL**: https://academic.oup.com/nar/article/52/W1/W533/7671315

**Key Findings**:
- ESM-2 with 3 billion parameters achieves best results
- ROC AUC 0.76 on benchmark datasets
- Larger ESM-2 models improve performance
- Web platform available for predictions

**Relevance**: Shows ESM-2 3B > ESM-2 650M for epitopes (but slower)

---

### IgFold - Antibody Structure Prediction

**Paper**: "Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies"
- **Authors**: Ruffolo, J.A., Sulam, J., Gray, J.J.
- **Journal**: Nature Communications
- **Date**: 2023
- **DOI**: 10.1038/s41467-023-38063-x
- **URL**: https://www.nature.com/articles/s41467-023-38063-x
- **Code**: https://github.com/Graylab/IgFold

**Key Findings**:
- Uses AntiBERTy (588M sequences) + graph networks
- Predicts antibody structure in <25 seconds
- Better or equal to AlphaFold for antibodies
- Embeddings useful for downstream tasks

**Relevance**: Compared to IgT5 - IgT5 is more recent and better for binding

---

### IgBERT - Paired Antibody Model

**Paper**: "Large scale paired antibody language models"
- **Same as IgT5 paper** (includes both IgT5 and IgBERT)
- **Model**: https://huggingface.co/Exscientia/IgBert

**Key Findings**:
- BERT-based paired antibody model
- R² 0.306 on binding energy (N=422 dataset)
- Trained on same 2B unpaired + 2M paired data as IgT5
- Second-best after IgT5

**Relevance**: Alternative to IgT5 if T5 architecture has issues

---

## 📊 Benchmark & Comparison Papers

### Antibody Language Models Comparison

**Paper**: "Do domain-specific protein language models outperform general models on immunology-related tasks?"
- **Journal**: ImmunoInformatics
- **Date**: 2024
- **DOI**: 10.1016/j.immuno.2024.00006
- **URL**: https://www.immunoinformaticsjournal.com/article/S2667-1190(24)00006-5/fulltext

**Key Findings**:
- ESM2 outperforms AbLang on antigen-binding tasks
- Domain-specific models (antibody) beat general (protein) for antibody tasks
- Comprehensive benchmark across multiple immunology tasks

**Relevance**: Validates use of antibody-specific models

---

### Protein Language Models Review

**Paper**: "Fine-tuning protein language models boosts predictions across diverse tasks"
- **Journal**: Nature Communications
- **Date**: 2024
- **DOI**: 10.1038/s41467-024-51844-2
- **URL**: https://www.nature.com/articles/s41467-024-51844-2

**Key Findings**:
- Compared ESM2, ProtT5, Ankh across protein tasks
- Fine-tuning improves all models
- Task-specific optimization crucial
- ESM2 generally performs best

**Relevance**: Benchmarks different protein language models

---

### Antibody Design Review

**Paper**: "Antibody design using deep learning: from sequence and structure design to affinity maturation"
- **Journal**: Briefings in Bioinformatics
- **Date**: 2024
- **DOI**: 10.1093/bib/bbae245
- **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11221890/

**Key Findings**:
- Comprehensive review of antibody deep learning methods
- IgFold, AntiBERTy, AbLang comparison
- Discussion of binding affinity prediction methods
- Future directions in antibody design

**Relevance**: Context for our work in broader field

---

## 🔧 Methods & Tools Papers

### Focal Loss

**Paper**: "Focal Loss for Dense Object Detection"
- **Authors**: Lin, T.Y., Goyal, P., Girshick, R., He, K., Dollár, P.
- **Conference**: ICCV 2017
- **URL**: https://arxiv.org/abs/1708.02002

**Key Concept**:
```python
FL(p_t) = -(1 - p_t)^γ * log(p_t)
```
- Emphasizes hard examples
- Reduces weight of easy examples
- γ controls focusing strength

**Relevance**: Our Focal MSE Loss adapted from this

---

### AdamW Optimizer

**Paper**: "Decoupled Weight Decay Regularization"
- **Authors**: Loshchilov, I., Hutter, F.
- **Conference**: ICLR 2019
- **URL**: https://arxiv.org/abs/1711.05101

**Key Concept**:
- Decouples weight decay from gradient update
- Better generalization than Adam
- Standard for transformer training

**Relevance**: Our optimizer choice

---

### Cosine Annealing

**Paper**: "SGDR: Stochastic Gradient Descent with Warm Restarts"
- **Authors**: Loshchilov, I., Hutter, F.
- **Conference**: ICLR 2017
- **URL**: https://arxiv.org/abs/1608.03983

**Key Concept**:
- Learning rate follows cosine schedule
- Smooth decay without jumps
- Helps escape local minima

**Relevance**: Our learning rate scheduler

---

## 📖 Background & Theory

### Antibody Structure

**Resource**: "Structural Basis of Antibody Function"
- **Source**: ImmunoInformatics textbook
- **Key Concepts**:
  - CDR regions (Complementarity Determining Regions)
  - Heavy and light chains
  - Paratope (antibody binding site)
  - Epitope (antigen binding site)

**Relevance**: Understanding what model needs to learn

---

### Binding Affinity

**Paper**: "The thermodynamics of protein-protein recognition"
- **Journal**: Reviews
- **Key Concepts**:
  - pKd = -log10(Kd)
  - Higher pKd = stronger binding
  - pKd ≥ 9 (Kd ≤ 1 nM) = strong binders
  - Temperature and pH effects

**Relevance**: Target variable definition

---

## 💻 Technical Resources

### Transformers Library

**Resource**: HuggingFace Transformers Documentation
- **URL**: https://huggingface.co/docs/transformers
- **Version Used**: 4.57.1
- **Models Used**:
  - `Exscientia/IgT5`
  - `facebook/esm2_t33_650M_UR50D`

**Relevance**: Implementation framework

---

### PyTorch

**Resource**: PyTorch Documentation
- **URL**: https://pytorch.org/docs/
- **Version Used**: 2.x (Colab default)
- **Features Used**:
  - Automatic Mixed Precision (AMP)
  - DataLoader with custom collate
  - Model checkpointing

**Relevance**: Deep learning framework

---

### Google Colab

**Resource**: Google Colaboratory
- **URL**: https://colab.research.google.com/
- **GPU Used**: Tesla T4 (16GB VRAM)
- **Features**:
  - Free GPU access
  - 12-hour session limit
  - Google Drive integration

**Relevance**: Training platform

---

## 📂 Datasets

### OAS (Observed Antibody Space)

**Resource**: "Observed Antibody Space: A Resource for Data Mining Next-Generation Sequencing of Antibody Repertoires"
- **URL**: http://opig.stats.ox.ac.uk/webapps/oas/
- **Size**: >2 billion antibody sequences
- **Usage**: IgT5 and IgBERT training data

**Relevance**: Source of antibody sequences for pretraining

---

### SAbDab (Structural Antibody Database)

**Resource**: "SAbDab: the structural antibody database"
- **URL**: http://opig.stats.ox.ac.uk/webapps/sabdab/
- **Size**: >5,000 antibody-antigen structures
- **Usage**: Benchmark for structure prediction

**Relevance**: Potential validation dataset

---

### Our Dataset

**File**: `agab_phase2_full.csv`
- **Size**: 159,735 antibody-antigen pairs
- **Features**: Antibody sequence, antigen sequence, pKd
- **Source**: [Internal dataset, specify origin if known]
- **Split**: 70% train, 15% val, 15% test

**Relevance**: Our training data

---

## 🔗 Related Work

### TCR-ESM

**Paper**: "TCR-ESM: Employing protein language embeddings to predict TCR-peptide-MHC binding"
- **Journal**: PMC
- **Date**: 2024
- **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC10749252/

**Key Concept**: Uses ESM-2 for T-cell receptor binding prediction

**Relevance**: Similar task (immune protein binding) using ESM-2

---

### DeepProSite

**Paper**: "DeepProSite: structure-aware protein binding site prediction using ESMFold and pretrained language model"
- **Journal**: Bioinformatics
- **Date**: 2023
- **URL**: https://academic.oup.com/bioinformatics/article/39/12/btad718/7453375

**Key Concept**: ESMFold + ProtT5 for binding site prediction

**Relevance**: Combines structure + sequence like we might in future

---

### ParaAntiProt

**Paper**: "ParaAntiProt provides paratope prediction using antibody and protein language models"
- **Journal**: Scientific Reports
- **Date**: 2024 (November)
- **URL**: https://www.nature.com/articles/s41598-024-80940-y

**Key Concept**: Uses ESM-2, AntiBERTy, AbLang, IgBERT for paratope prediction

**Relevance**: Comparison of antibody language models

---

## 📝 Code & Implementation References

### IgT5 HuggingFace

**URL**: https://huggingface.co/Exscientia/IgT5
**Code Example**:
```python
from transformers import T5EncoderModel, T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("Exscientia/IgT5", do_lower_case=False)
model = T5EncoderModel.from_pretrained("Exscientia/IgT5")
```

---

### ESM-2 HuggingFace

**URL**: https://huggingface.co/facebook/esm2_t33_650M_UR50D
**Code Example**:
```python
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
```

---

### IgFold GitHub

**URL**: https://github.com/Graylab/IgFold
**Installation**: `pip install igfold`
**Usage**: Structure prediction and embedding extraction

---

## 🎓 Tutorials & Guides

### Protein Language Models Tutorial

**Resource**: "A Primer on Protein Language Models"
- **URL**: Various blog posts and tutorials
- **Topics**:
  - What are protein language models?
  - How do they work?
  - Applications in biology

---

### Antibody Structure Basics

**Resource**: IMGT (International ImMunoGeneTics information system)
- **URL**: http://www.imgt.org/
- **Topics**:
  - Antibody numbering schemes
  - CDR definitions
  - Heavy and light chain pairing

---

## 📊 Benchmarks & Leaderboards

### CASP (Critical Assessment of Structure Prediction)

**URL**: https://predictioncenter.org/
**Relevance**: Protein structure prediction benchmark (ESM-2 performs well)

---

### CAFA (Critical Assessment of Functional Annotation)

**URL**: https://biofunctionprediction.org/cafa/
**Relevance**: Protein function prediction benchmark

---

## 🌐 Online Resources

### Papers with Code

**URL**: https://paperswithcode.com/task/antibody-design
**Usage**: Find latest papers and benchmarks on antibody design

---

### arXiv - Quantitative Biology

**URL**: https://arxiv.org/list/q-bio/recent
**Usage**: Preprints of latest computational biology research

---

### bioRxiv - Bioinformatics

**URL**: https://www.biorxiv.org/
**Usage**: Preprints of latest bioinformatics research

---

## 📚 Citation Format (For Future Paper)

### BibTeX Entries

```bibtex
@article{kenlay2024igt5,
  title={Large scale paired antibody language models},
  author={Kenlay, Henry and Dreyer, Fergus A and Kovaltsuk, Aleksandr and Miketa, Dominik and Pires, Douglas and Deane, Charlotte M},
  journal={PLOS Computational Biology},
  year={2024},
  month={December},
  doi={10.1371/journal.pcbi.1012646}
}

@article{lin2023esm2,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and others},
  journal={Science},
  year={2023},
  doi={10.1126/science.ade2574}
}

@article{wang2024epigraph,
  title={B cell epitope prediction by capturing spatial clustering property of the epitopes using graph attention network},
  author={Wang, Yiqi and others},
  journal={Scientific Reports},
  year={2024},
  doi={10.1038/s41598-024-78506-z}
}
```

---

## 🔍 Keywords for Literature Search

**Primary Keywords**:
- Antibody-antigen binding prediction
- Protein language models
- Antibody embeddings
- Binding affinity prediction
- IgT5, IgBERT, IgFold, AntiBERTy
- ESM-2, ProtT5

**Secondary Keywords**:
- CDR prediction
- Epitope-paratope interaction
- Therapeutic antibody design
- Deep learning for immunology
- Protein-protein interaction

**Search Queries**:
```
"antibody binding affinity" AND "deep learning"
"IgT5" OR "IgBERT" OR "IgFold"
"ESM-2" AND "epitope prediction"
"protein language model" AND "antibody"
```

---

## 📅 Paper Reading Schedule (Completed)

### Week 1 (Nov 10-16, 2025)
- [✅] IgT5 paper (Kenlay et al. 2024)
- [✅] ESM-2 paper (Lin et al. 2023)
- [✅] IgFold paper (Ruffolo et al. 2023)
- [✅] EpiGraph paper (Wang et al. 2024)
- [✅] CALIBER paper (2025)
- [✅] Domain-specific models comparison (2024)

### Week 2 (Future)
- [ ] Antibody design review papers
- [ ] Related work on TCR binding
- [ ] Structure-aware methods
- [ ] Multi-modal learning approaches

---

## 🎯 Most Influential Papers for This Project

1. **Kenlay et al. 2024 (IgT5)** - Our antibody model choice
2. **Lin et al. 2023 (ESM-2)** - Our antigen model choice
3. **Wang et al. 2024 (EpiGraph)** - Validates ESM-2 for epitopes
4. **CALIBER 2025** - Confirms ESM-2 + Bi-LSTM for binding
5. **Lin et al. 2017 (Focal Loss)** - Our loss function inspiration

---

## 📧 Contact Information for Model Authors

**IgT5/IgBERT**:
- Organization: Exscientia + Oxford Protein Informatics Group
- GitHub: https://github.com/Exscientia
- Issues: https://huggingface.co/Exscientia/IgT5/discussions

**ESM-2**:
- Organization: Meta AI (Facebook Research)
- GitHub: https://github.com/facebookresearch/esm
- Issues: https://github.com/facebookresearch/esm/issues

---

**Document Status**: Comprehensive reference list
**Last Updated**: 2025-11-13
**Maintained By**: Project team
**Purpose**: Citation and background for future publication
