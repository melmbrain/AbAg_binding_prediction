"""
Antibody-Antigen Binding Affinity Predictor

Production-ready API for predicting antibody-antigen binding affinity.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional
import logging
import warnings

logger = logging.getLogger(__name__)


class MultiHeadAttentionModel(nn.Module):
    """Multi-head attention model for binding affinity prediction"""

    def __init__(self, input_dim=300, hidden_dim=256, n_heads=6, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_out)
        x = x.squeeze(1)
        return self.ff(x).squeeze(-1)


class AffinityPredictor:
    """
    Antibody-Antigen Binding Affinity Predictor

    Predicts binding affinity (pKd) from amino acid sequences using
    ESM-2 protein language model embeddings and multi-head attention.

    Performance:
        Spearman ρ = 0.8501
        Pearson r = 0.9461
        R² = 0.8779
        Trained on 7,015 Ab-Ag pairs

    Example:
        >>> predictor = AffinityPredictor()
        >>> result = predictor.predict(
        ...     antibody_heavy="EVQLQQSG...",
        ...     antibody_light="DIQMTQSP...",
        ...     antigen="KVFGRCELA..."
        ... )
        >>> print(f"pKd: {result['pKd']:.2f}, Kd: {result['Kd_nM']:.1f} nM")

    Args:
        model_path: Path to trained model file (default: auto-detect)
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
        verbose: Print initialization messages
    """

    VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        self.verbose = verbose

        # Auto-detect model path
        if model_path is None:
            pkg_dir = Path(__file__).parent.parent
            model_path = pkg_dir / 'models' / 'agab_phase2_model.pth'

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Please ensure the model file is in the correct location."
            )

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Load ESM-2
        self.esm_model, self.esm_tokenizer = self._load_esm2()

        if self.verbose:
            logger.info(f"AffinityPredictor initialized")
            logger.info(f"  Device: {self.device}")
            logger.info(f"  Model: {self.model_path.name}")
            logger.info(f"  Performance: Spearman ρ = 0.85")

    def _load_model(self) -> nn.Module:
        """Load the trained model"""
        model = MultiHeadAttentionModel(input_dim=300, hidden_dim=256, n_heads=6)
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        return model.to(self.device)

    def _load_esm2(self):
        """Load ESM-2 protein language model"""
        try:
            from transformers import EsmModel, EsmTokenizer
            warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

            model = EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
            tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

            return model.to(self.device).eval(), tokenizer

        except ImportError:
            raise ImportError(
                "transformers library required. Install with:\n"
                "pip install transformers"
            )

    def _validate_sequence(self, seq: str, name: str = "Sequence") -> str:
        """Validate amino acid sequence"""
        if not seq:
            raise ValueError(f"{name} cannot be empty")

        clean = seq.replace('XXX', '').replace(' ', '').replace('\n', '').upper()
        invalid = set(clean) - self.VALID_AA

        if invalid:
            raise ValueError(
                f"Invalid amino acids in {name}: {invalid}\n"
                f"Valid: {sorted(self.VALID_AA)}"
            )

        if len(clean) < 10:
            raise ValueError(f"{name} too short ({len(clean)} aa, minimum: 10)")

        return seq

    def _get_embedding(self, sequence: str) -> np.ndarray:
        """Generate ESM-2 embedding for sequence"""
        with torch.no_grad():
            inputs = self.esm_tokenizer(
                sequence,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.device)

            outputs = self.esm_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        return embedding.flatten()

    def predict(
        self,
        antibody_heavy: str,
        antigen: str,
        antibody_light: str = ""
    ) -> Dict:
        """
        Predict binding affinity for antibody-antigen pair

        Args:
            antibody_heavy: Heavy chain sequence (required)
            antigen: Antigen sequence (required)
            antibody_light: Light chain sequence (optional)

        Returns:
            Dictionary containing:
                - pKd: Predicted pKd value
                - Kd_nM: Kd in nanomolar
                - Kd_uM: Kd in micromolar
                - Kd_M: Kd in molar
                - category: Binding strength category
                - interpretation: Human-readable description

        Example:
            >>> result = predictor.predict(
            ...     antibody_heavy="EVQLQQS...",
            ...     antibody_light="DIQMTQS...",
            ...     antigen="KVFGRCE..."
            ... )
            >>> print(f"pKd: {result['pKd']:.2f}")
        """
        # Validate inputs
        self._validate_sequence(antibody_heavy, "Antibody heavy chain")
        self._validate_sequence(antigen, "Antigen")
        if antibody_light:
            self._validate_sequence(antibody_light, "Antibody light chain")

        # Combine antibody chains
        if antibody_light:
            antibody_seq = antibody_heavy + 'XXX' + antibody_light
        else:
            antibody_seq = antibody_heavy

        # Get embeddings
        ab_emb = self._get_embedding(antibody_seq)
        ag_emb = self._get_embedding(antigen)

        # Reduce to 150 dims each (model input)
        ab_features = ab_emb[:150]
        ag_features = ag_emb[:150]

        # Combine and predict
        features = np.concatenate([ab_features, ag_features])
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pKd = self.model(x).item()

        # Convert to various units
        Kd_M = 10 ** (-pKd)
        Kd_nM = Kd_M * 1e9
        Kd_uM = Kd_M * 1e6

        return {
            'pKd': round(pKd, 2),
            'Kd_M': Kd_M,
            'Kd_nM': round(Kd_nM, 2),
            'Kd_uM': round(Kd_uM, 4),
            'category': self._categorize(pKd),
            'interpretation': self._interpret(pKd)
        }

    def predict_batch(
        self,
        pairs: List[Dict[str, str]],
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Predict affinity for multiple antibody-antigen pairs

        Args:
            pairs: List of dicts with keys 'antibody_heavy', 'antigen',
                   and optionally 'antibody_light' and 'id'
            show_progress: Show progress bar (requires tqdm)

        Returns:
            List of prediction dictionaries

        Example:
            >>> pairs = [
            ...     {'antibody_heavy': 'EVQ...', 'antigen': 'KVF...'},
            ...     {'antibody_heavy': 'QVQ...', 'antigen': 'KVF...'}
            ... ]
            >>> results = predictor.predict_batch(pairs)
        """
        iterator = pairs
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(pairs, desc="Predicting")
            except ImportError:
                pass

        results = []
        for i, pair in enumerate(iterator):
            try:
                result = self.predict(
                    antibody_heavy=pair['antibody_heavy'],
                    antigen=pair['antigen'],
                    antibody_light=pair.get('antibody_light', '')
                )
                result['id'] = pair.get('id', i)
                result['status'] = 'success'
            except Exception as e:
                result = {
                    'id': pair.get('id', i),
                    'status': 'error',
                    'error': str(e)
                }
            results.append(result)

        return results

    def _interpret(self, pKd: float) -> str:
        """Human-readable interpretation"""
        if pKd > 10:
            return "Exceptional binder (picomolar, Kd < 1 nM)"
        elif pKd > 9:
            return "Very strong binder (sub-nanomolar, Kd ~ 1-10 nM)"
        elif pKd > 7.5:
            return "Strong binder (nanomolar, Kd ~ 10-100 nM)"
        elif pKd > 6:
            return "Moderate binder (micromolar, Kd ~ 0.1-10 μM)"
        elif pKd > 4:
            return "Weak binder (Kd ~ 10-100 μM)"
        else:
            return "Very weak or non-binder (Kd > 100 μM)"

    def _categorize(self, pKd: float) -> str:
        """Simple category"""
        if pKd > 9:
            return "excellent"
        elif pKd > 7.5:
            return "good"
        elif pKd > 6:
            return "moderate"
        else:
            return "poor"
