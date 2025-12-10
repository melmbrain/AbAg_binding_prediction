# Antibody-Antigen Binding Affinity Prediction

> State-of-the-art antibody-antigen binding prediction using IgT5 + ESM-2 hybrid architecture

**Status**: 🔄 Training in Progress
**Model**: IgT5 (antibody) + ESM-2 (antigen)
**Platform**: Google Colab T4 GPU
**Expected Completion**: November 17-18, 2025

---

## 🚀 Quick Start

👉 **New here?** Read [START_HERE_FINAL.md](START_HERE_FINAL.md)

👉 **Ready to train?** Use `colab_training_SOTA.ipynb` on Google Colab

👉 **Want details?** See [PROJECT_LOG.md](PROJECT_LOG.md)

---

## 📊 Current Results

| Metric | Baseline (E5) | Target | Expected (E50) |
|--------|---------------|--------|----------------|
| **Spearman** | 0.46 | 0.60-0.70 | 0.65 |
| **Recall@pKd≥9** | 14.22% | 40-60% | 52% |
| **RMSE** | 1.45 | 1.25-1.35 | 1.30 |

*Baseline from incomplete ESM-2 training (epoch 5/50). Expected values based on IgT5 + ESM-2 literature.*

---

## 📁 Documentation

| Document | Purpose |
|----------|---------|
| [START_HERE_FINAL.md](START_HERE_FINAL.md) | Quick start guide |
| [PROJECT_LOG.md](PROJECT_LOG.md) | Complete work history & decisions |
| [OUTCOMES_AND_FUTURE_PLAN.md](OUTCOMES_AND_FUTURE_PLAN.md) | Results & future research |
| [MODEL_COMPARISON_FINAL.md](MODEL_COMPARISON_FINAL.md) | Why IgT5 + ESM-2? |
| [REFERENCES_AND_SOURCES.md](REFERENCES_AND_SOURCES.md) | All citations & papers |
| [COLAB_SETUP_GUIDE.md](COLAB_SETUP_GUIDE.md) | Google Colab instructions |
| [FILE_ORGANIZATION.md](FILE_ORGANIZATION.md) | Project structure |

---

## 🧬 Model Architecture

```
Antibody Seq → IgT5 (1024-dim) ─┐
                                 ├─→ Regressor → pKd
Antigen Seq  → ESM-2 (1280-dim) ─┘
```

**Why this works:**
- **IgT5**: State-of-the-art antibody model (Dec 2024, R² 0.297-0.306)
- **ESM-2**: Best for antigen epitopes (AUC 0.76-0.789 in 2024-2025 papers)
- **Hybrid**: Combines antibody-specific + proven antigen features

---

## 📚 Key References

1. **IgT5** (Dec 2024): Kenlay et al., PLOS Computational Biology
2. **ESM-2** (2023): Lin et al., Science
3. **EpiGraph** (2024): ESM-2 for epitope prediction, AUC 0.23
4. **CALIBER** (2025): ESM-2 + Bi-LSTM, AUC 0.789

[Full references →](REFERENCES_AND_SOURCES.md)

---

## 🎯 Project Goal

**Predict strong binders (pKd ≥ 9) for drug discovery**

Current: 14% recall → Target: 40-60% recall

---

**Last Updated**: 2025-11-13
**Project**: Antibody binding prediction for therapeutic development
