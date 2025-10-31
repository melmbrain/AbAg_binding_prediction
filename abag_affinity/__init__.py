"""
AbAg Binding Affinity Prediction

A production-ready package for predicting antibody-antigen binding affinity
using deep learning and protein language models.

Performance: Spearman ρ = 0.85 (7,015 training pairs)

Example:
    >>> from abag_affinity import AffinityPredictor
    >>> predictor = AffinityPredictor()
    >>> result = predictor.predict(antibody_seq, antigen_seq)
    >>> print(f"Predicted pKd: {result['pKd']:.2f}")
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .predictor import AffinityPredictor

__all__ = ['AffinityPredictor', '__version__']
