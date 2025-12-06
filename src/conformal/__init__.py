"""
Conformal Prediction Module

Provides LAC and APS scoring methods for uncertainty quantification
in language model predictions.
"""

from src.conformal.conformal_base import (
    BaseConformalPredictor,
    ConformalConfig,
    ConformalPredictionResult,
    ConformalPredictionValidator,
    PredictionSet,
)
from src.conformal.scorers import APSScorer, ConformalAnalyzer, LACScorer

__all__ = [
    # Base classes and data structures
    "BaseConformalPredictor",
    "ConformalConfig",
    "ConformalPredictionResult",
    "ConformalPredictionValidator",
    "PredictionSet",
    # Scorers
    "LACScorer",
    "APSScorer",
    "ConformalAnalyzer",
]
