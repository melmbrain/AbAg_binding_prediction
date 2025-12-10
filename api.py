"""
Antibody-Antigen Binding Affinity Prediction API
=================================================

A FastAPI-based REST API for predicting binding affinity.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000

Or run directly:
    python api.py
"""

import os
import sys
from typing import List, Optional
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# ============================================================================
# API Models
# ============================================================================

class PredictionRequest(BaseModel):
    """Single prediction request."""
    antibody_sequence: str = Field(..., min_length=10, description="Antibody amino acid sequence")
    antigen_sequence: str = Field(..., min_length=10, description="Antigen amino acid sequence")

    class Config:
        json_schema_extra = {
            "example": {
                "antibody_sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKG",
                "antigen_sequence": "MKTIIALSYIFCLVFADYKDDDDKMRVLRGNAVVLQVVSNGTFQVYEQLLKAGEYVFNTSLVLRG"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    pairs: List[PredictionRequest] = Field(..., max_length=100, description="List of Ab-Ag pairs (max 100)")


class PredictionResponse(BaseModel):
    """Prediction response."""
    pKd: float = Field(..., description="Predicted pKd value")
    Kd_nM: float = Field(..., description="Predicted Kd in nanomolar")
    Kd_uM: float = Field(..., description="Predicted Kd in micromolar")
    binding_strength: str = Field(..., description="Qualitative binding strength")
    antibody_length: int = Field(..., description="Antibody sequence length")
    antigen_length: int = Field(..., description="Antigen sequence length")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    count: int


class ModelInfo(BaseModel):
    """Model information."""
    version: str
    architecture: str
    validation_r2: float
    validation_mae: float
    encoders: dict


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="AbAg Binding Affinity Predictor",
    description="""
    Predict antibody-antigen binding affinity (pKd) using deep learning.

    ## Features
    - Single or batch predictions
    - Supports raw amino acid sequences
    - Returns pKd, Kd (nM/µM), and binding strength interpretation

    ## Model
    - Architecture: IgT5 + ProtT5 embeddings → Cross-attention fusion → Residual MLP
    - Validation R²: 0.7865
    - Validation MAE: 0.4254
    """,
    version="2.8.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor (lazy loaded)
predictor = None


def get_predictor():
    """Lazy load the predictor."""
    global predictor
    if predictor is None:
        from inference import BindingAffinityPredictor

        # Find model file
        model_paths = [
            "models/v2.8_stage2/stage2_experiment_20251210_033341.pth",
            "models/stage2_final.pth",
            "stage2_experiment_20251210_033341.pth",
        ]

        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            raise FileNotFoundError("Model file not found. Please check model paths.")

        print(f"Loading model from: {model_path}")
        predictor = BindingAffinityPredictor(model_path)

    return predictor


def interpret_binding(pKd: float) -> str:
    """Interpret binding strength from pKd."""
    if pKd >= 9:
        return "Very Strong"
    elif pKd >= 7:
        return "Strong"
    elif pKd >= 5:
        return "Moderate"
    else:
        return "Weak"


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Home page with simple UI."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AbAg Binding Affinity Predictor</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #2c3e50; }
            .form-group { margin: 20px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            textarea { width: 100%; height: 100px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
            button { background: #3498db; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
            button:hover { background: #2980b9; }
            #result { margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 4px; display: none; }
            .metric { margin: 10px 0; }
            .metric-label { font-weight: bold; color: #666; }
            .metric-value { font-size: 24px; color: #2c3e50; }
            .binding-strong { color: #27ae60; }
            .binding-moderate { color: #f39c12; }
            .binding-weak { color: #e74c3c; }
        </style>
    </head>
    <body>
        <h1>AbAg Binding Affinity Predictor</h1>
        <p>Predict antibody-antigen binding affinity using deep learning.</p>

        <div class="form-group">
            <label>Antibody Sequence:</label>
            <textarea id="antibody" placeholder="Enter antibody amino acid sequence...">EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKG</textarea>
        </div>

        <div class="form-group">
            <label>Antigen Sequence:</label>
            <textarea id="antigen" placeholder="Enter antigen amino acid sequence...">MKTIIALSYIFCLVFADYKDDDDKMRVLRGNAVVLQVVSNGTFQVYEQLLKAGEYVFNTSLVLRG</textarea>
        </div>

        <button onclick="predict()">Predict Binding Affinity</button>

        <div id="result">
            <div class="metric">
                <div class="metric-label">Predicted pKd</div>
                <div class="metric-value" id="pKd">-</div>
            </div>
            <div class="metric">
                <div class="metric-label">Predicted Kd</div>
                <div class="metric-value" id="Kd">-</div>
            </div>
            <div class="metric">
                <div class="metric-label">Binding Strength</div>
                <div class="metric-value" id="strength">-</div>
            </div>
        </div>

        <script>
            async function predict() {
                const antibody = document.getElementById('antibody').value;
                const antigen = document.getElementById('antigen').value;

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ antibody_sequence: antibody, antigen_sequence: antigen })
                    });

                    const data = await response.json();

                    document.getElementById('pKd').textContent = data.pKd.toFixed(2);
                    document.getElementById('Kd').textContent = data.Kd_nM.toFixed(2) + ' nM';

                    const strengthEl = document.getElementById('strength');
                    strengthEl.textContent = data.binding_strength;
                    strengthEl.className = 'metric-value';
                    if (data.binding_strength.includes('Strong')) {
                        strengthEl.classList.add('binding-strong');
                    } else if (data.binding_strength === 'Moderate') {
                        strengthEl.classList.add('binding-moderate');
                    } else {
                        strengthEl.classList.add('binding-weak');
                    }

                    document.getElementById('result').style.display = 'block';
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
        </script>

        <hr style="margin-top: 40px;">
        <p>API Documentation: <a href="/docs">/docs</a> | <a href="/redoc">/redoc</a></p>
    </body>
    </html>
    """


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/model-info", response_model=ModelInfo)
async def model_info():
    """Get model information."""
    return {
        "version": "2.8.0",
        "architecture": "Cross-attention fusion + Residual MLP",
        "validation_r2": 0.7865,
        "validation_mae": 0.4254,
        "encoders": {
            "antibody": "IgT5 (Exscientia/IgT5)",
            "antigen": "ProtT5 (Rostlab/prot_t5_xl_half_uniref50-enc)"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict binding affinity for a single antibody-antigen pair.

    Returns pKd (predicted binding affinity) and Kd in nM and µM.
    """
    try:
        pred = get_predictor()
        pKd = pred.predict_from_sequences(
            request.antibody_sequence,
            request.antigen_sequence
        )

        Kd_nM = 10 ** (9 - pKd)

        return PredictionResponse(
            pKd=round(pKd, 4),
            Kd_nM=round(Kd_nM, 4),
            Kd_uM=round(Kd_nM / 1000, 6),
            binding_strength=interpret_binding(pKd),
            antibody_length=len(request.antibody_sequence),
            antigen_length=len(request.antigen_sequence)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict binding affinity for multiple antibody-antigen pairs.

    Maximum 100 pairs per request.
    """
    try:
        pred = get_predictor()

        antibody_seqs = [p.antibody_sequence for p in request.pairs]
        antigen_seqs = [p.antigen_sequence for p in request.pairs]

        pKds = pred.predict_batch(antibody_seqs, antigen_seqs)

        predictions = []
        for i, pKd in enumerate(pKds):
            Kd_nM = 10 ** (9 - pKd)
            predictions.append(PredictionResponse(
                pKd=round(pKd, 4),
                Kd_nM=round(Kd_nM, 4),
                Kd_uM=round(Kd_nM / 1000, 6),
                binding_strength=interpret_binding(pKd),
                antibody_length=len(request.pairs[i].antibody_sequence),
                antigen_length=len(request.pairs[i].antigen_sequence)
            ))

        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Starting AbAg Binding Affinity Predictor API...")
    print("Open http://localhost:8000 in your browser")
    print("API docs at http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)
