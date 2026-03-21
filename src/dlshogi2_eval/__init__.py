"""Eval-only reference extraction of python-dlshogi2."""

from .features import FEATURES_NUM, MOVE_LABELS_NUM, MOVE_PLANES_NUM
from .infer import DensePV, LegalPV, PolicyValueEvaluator
from .loader import load_model_from_checkpoint
from .model import PolicyValueNetwork

__all__ = [
    "FEATURES_NUM",
    "MOVE_LABELS_NUM",
    "MOVE_PLANES_NUM",
    "DensePV",
    "LegalPV",
    "PolicyValueEvaluator",
    "PolicyValueNetwork",
    "load_model_from_checkpoint",
]

__version__ = "0.1.0"
