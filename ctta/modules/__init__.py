"""DAF-derived stabilization modules for TMPA CTTA.

Implemented:
- source prediction consistency: DAF-style symmetric KL against a frozen source model.
- prompt feature consistency: TMPA-specific counterpart to DAF visual feature
  consistency, applied in the prompt/text space that TMPA actually adapts.

Planned next: CMAC and SAFS, after the two consistency modules are evaluated
independently against the fixed TMPA-Continual baseline.
"""

from .consistency import prompt_feature_consistency, source_prediction_consistency

__all__ = ['source_prediction_consistency', 'prompt_feature_consistency']
