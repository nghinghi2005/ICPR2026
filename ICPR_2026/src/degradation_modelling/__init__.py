"""
Domain-specific Degradation Modelling (Stage 0).

Goal:
  For each track and each paired (HR, LR) frame, find degradation parameters
  that minimize the difference between:
    LR_real  vs  Downsample_bicubic( Degrade(HR, params) )

Constraints:
  - Downsampling must be bicubic.
  - Downsampled image size must match LR_real size exactly.
  - Search is greedy over a fixed degradation operator order.
"""

from .config import GreedySearchConfig
from .runner import run_degradation_modelling

__all__ = [
    "GreedySearchConfig",
    "run_degradation_modelling",
]


