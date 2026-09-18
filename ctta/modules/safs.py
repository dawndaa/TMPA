"""Temporal SAFS gating for strict single-stream TMPA CTTA.

DAF SAFS computes a per-sample feature-shift score inside a batch and keeps
samples whose shift is above mean - alpha * std. Strict CTTA here uses
batch_size=1, so the original batch statistic degenerates. This module preserves
the same selection principle but estimates the shift distribution over a recent
temporal window.

Because TMPA freezes its visual encoder, the sample-specific drift signal is the
cosine distance between adapted and frozen-source pixel-class probability
vectors. The gate decides whether the current sample is allowed to update the
adaptive prompt state. It is not an additional loss.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Union

import torch
import torch.nn.functional as F

from .consistency import _class_dim, _normalize_probabilities


def prediction_shift_score(
    adapted_prob_maps: torch.Tensor,
    source_prob_maps: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean pixel-wise cosine distance between adapted and source predictions."""
    if adapted_prob_maps.shape != source_prob_maps.shape:
        raise ValueError(
            'SAFS requires adapted/source probability maps with identical shapes, '
            f'got {tuple(adapted_prob_maps.shape)} and {tuple(source_prob_maps.shape)}.'
        )

    class_dim = _class_dim(adapted_prob_maps)
    p_adapt = _normalize_probabilities(adapted_prob_maps.detach(), eps=eps)
    p_source = _normalize_probabilities(source_prob_maps.detach(), eps=eps)

    p_adapt = F.normalize(p_adapt, p=2, dim=class_dim, eps=eps)
    p_source = F.normalize(p_source, p=2, dim=class_dim, eps=eps)
    cosine = (p_adapt * p_source).sum(dim=class_dim)
    return (1.0 - cosine).mean()


class TemporalSAFSGate:
    """Streaming analogue of DAF's batch-level SAFS selector.

    The current score is compared against statistics from previous samples in a
    bounded history window. During warmup, or when the history has effectively
    zero variance, the sample is kept. This mirrors DAF's fallback behavior that
    avoids filtering the entire batch when the selector has no useful spread.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        window: int = 32,
        warmup: int = 8,
        std_eps: float = 1e-8,
    ):
        if alpha < 0:
            raise ValueError(f'SAFS alpha must be non-negative, got {alpha}.')
        if window < 2:
            raise ValueError(f'SAFS window must be >= 2, got {window}.')
        if warmup < 1:
            raise ValueError(f'SAFS warmup must be >= 1, got {warmup}.')
        if warmup > window:
            raise ValueError(
                f'SAFS warmup ({warmup}) cannot exceed window ({window}).'
            )

        self.alpha = float(alpha)
        self.window = int(window)
        self.warmup = int(warmup)
        self.std_eps = float(std_eps)
        self.history: Deque[float] = deque(maxlen=self.window)

    def reset(self):
        self.history.clear()

    def decide(self, score: Union[float, torch.Tensor]) -> Dict[str, Optional[float]]:
        """Record a shift score and return whether the sample should be updated."""
        if isinstance(score, torch.Tensor):
            value = float(score.detach().item())
        else:
            value = float(score)

        history_mean = None
        history_std = None
        threshold = None

        if len(self.history) < self.warmup:
            keep = True
        else:
            stats = torch.tensor(list(self.history), dtype=torch.float64)
            history_mean = float(stats.mean().item())
            history_std = float(stats.std(unbiased=False).item())

            if history_std <= self.std_eps:
                keep = True
            else:
                threshold = history_mean - self.alpha * history_std
                keep = value > threshold

        history_size = len(self.history)
        self.history.append(value)

        return {
            'keep': float(keep),
            'score': value,
            'threshold': threshold,
            'history_mean': history_mean,
            'history_std': history_std,
            'history_size': float(history_size),
        }
