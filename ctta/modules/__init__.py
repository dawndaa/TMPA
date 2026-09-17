"""DAF-derived stabilization modules for TMPA CTTA.

Implemented:
- source prediction consistency: DAF-style symmetric KL against a frozen source model.
- prompt feature consistency: TMPA-specific counterpart to DAF visual feature
  consistency, applied in the prompt/text space that TMPA actually adapts.
- CMAC source anchoring: DAF-style directional source-anchor constraint mapped
  to TMPA's pixel-wise class probabilities.
- temporal SAFS: streaming update gate adapted from DAF's batch-level feature
  shift filtering for strict batch-size-one CTTA.
"""

from .cmac import cmac_source_anchor_loss
from .consistency import prompt_feature_consistency, source_prediction_consistency
from .safs import TemporalSAFSGate, prediction_shift_score

__all__ = [
    'source_prediction_consistency',
    'prompt_feature_consistency',
    'cmac_source_anchor_loss',
    'TemporalSAFSGate',
    'prediction_shift_score',
]
