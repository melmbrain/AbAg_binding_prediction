# Antibody-Antigen Binding Affinity Prediction

**Deep learning model for predicting antibody-antigen binding affinity (pKd) using dual protein language models.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**Current Version**: v2.8.0 (2025-12-10)
**Status**: Production Ready
**Architecture**: IgT5 (antibody) + ProtT5 (antigen) with Cross-Attention Fusion

---

## Performance

| Model | Val R² | Val MAE | Architecture |
|-------|--------|---------|--------------|
| **v2.8 (Stage 2)** | **0.7865** | **0.4254** | Cross-attention + ResidualMLP |
| v2.8 (Stage 1) | 0.7220 | 0.5500 | IgT5 + ProtT5 with LoRA |
| v2.6-beta | 0.390 (ρ) | - | IgT5 + ESM-2 3B |
| v2.5 | 0.42 (ρ) | - | ESM-2 650M |

---

## Quick Start

### Option 1: Python Inference

```python
from inference import BindingAffinityPredictor

# Load model
predictor = BindingAffinityPredictor('models/v2.8_stage2/stage2_experiment_20251210_033341.pth')

# Predict binding affinity
pKd = predictor.predict_from_sequences(
    antibody_seq="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS...",
    antigen_seq="MKTIIALSYIFCLVFADYKDDDDK..."
)

print(f"Predicted pKd: {pKd:.2f}")
print(f"Predicted Kd: {10**(9-pKd):.2f} nM")
```

### Option 2: Command Line

```bash
# Single prediction
python inference.py --antibody "EVQLVESGGGL..." --antigen "MKTIIALSYIF..."

# Batch prediction from CSV
python inference.py --csv input.csv --output predictions.csv
```

### Option 3: Web API

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
python api.py

# Open http://localhost:8000 in browser
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/melmbrain/AbAg_binding_prediction.git
cd AbAg_binding_prediction

# Install dependencies
pip install -r requirements.txt

# Download model (or use your own trained model)
# Place model file in models/v2.8_stage2/
```

---

## Model Architecture

```
Input: Antibody Sequence + Antigen Sequence
           ↓                    ↓
    ┌─────────────┐      ┌─────────────┐
    │    IgT5     │      │   ProtT5    │
    │  (frozen)   │      │  (frozen)   │
    │  + LoRA     │      │  + LoRA     │
    └─────────────┘      └─────────────┘
           ↓                    ↓
      [1024-dim]           [1024-dim]
           ↓                    ↓
    ┌─────────────────────────────────┐
    │    Cross-Attention Fusion       │
    │  - Ab→Ag + Ag→Ab attention      │
    │  - 8 heads, 512-dim             │
    └─────────────────────────────────┘
                    ↓
    ┌─────────────────────────────────┐
    │      Residual MLP Head          │
    │  512 → 256 → 128 → 1            │
    │  BatchNorm + GELU + Dropout     │
    └─────────────────────────────────┘
                    ↓
            Output: pKd
```

---

## Training (Google Colab)

### Two-Stage Training Pipeline

**Stage 1: Fine-tune Encoders & Generate Embeddings**
```
notebooks/Stage1_Generate_Embeddings.ipynb
- Fine-tune IgT5 + ProtT5 with LoRA
- Generate and cache embeddings for all samples
- ~20 hours on A100 40GB
```

**Stage 2: Train Fusion Model**
```
notebooks/Stage2_Fast_Training.ipynb
- Load cached embeddings (instant)
- Train fusion + head model
- ~1 hour on A100 40GB
```

### Setup for Colab

1. Create folder structure in Google Drive:
```
MyDrive/AbAg_Project/
├── data/
│   └── ab_ag_affinity_complete.csv
├── embeddings/
├── checkpoints/
└── models/
```

2. Upload dataset to `data/`

3. Run Stage 1 notebook, then Stage 2

---

## pKd Interpretation

| pKd Range | Binding Strength | Kd (approx) |
|-----------|------------------|-------------|
| ≥ 9 | Very Strong | < 1 nM |
| 7-9 | Strong | 1-100 nM |
| 5-7 | Moderate | 0.1-10 µM |
| < 5 | Weak | > 10 µM |

---

## Project Structure

```
AbAg_binding_prediction/
├── inference.py              # Inference script
├── api.py                    # FastAPI web server
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── models/
│   ├── v2.8_stage2/          # Latest model
│   │   ├── *.pth             # Model weights
│   │   └── README.md         # Model card
│   └── backup/
├── notebooks/
│   ├── Stage1_Generate_Embeddings.ipynb
│   ├── Stage2_Fast_Training.ipynb
│   └── ...
├── data/
│   └── ab_ag_affinity_complete.csv
├── src/
│   └── ...
└── archive/
    └── ...
```

---

## API Documentation

Once the API is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| GET | `/health` | Health check |
| GET | `/model-info` | Model information |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch prediction (max 100) |

---

## Pre-trained Models

Download from [Hugging Face](https://huggingface.co/Kroea/AbAg-binding-prediction):

```python
from huggingface_hub import hf_hub_download

# v2.8 Stage 2 model (recommended)
model_path = hf_hub_download(
    repo_id="Kroea/AbAg-binding-prediction",
    filename="stage2_final.pth"
)
```

---

## Citation

If you use this code, please cite:
- **IgT5**: [Exscientia/IgT5](https://huggingface.co/Exscientia/IgT5)
- **ProtT5**: [Rostlab/prot_t5_xl_half_uniref50-enc](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc)

---

## Version History

| Version | Date | R² | Notes |
|---------|------|-----|-------|
| **v2.8** | 2025-12-10 | **0.7865** | Two-stage pipeline, cross-attention fusion |
| v2.6-beta | 2025-11-25 | 0.39 (ρ) | ESM-2 3B experimental |
| v2.5 | 2025-11-13 | 0.42 (ρ) | ESM-2 650M stable |

---

## License

MIT License

---

**Last Updated**: 2025-12-10
**Status**: Production Ready
