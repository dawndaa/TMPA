"""Consistency losses adapted from DAF for TMPA CTTA.

DAF source consistency is transferred as a symmetric KL penalty between the
adapted prediction and a frozen source model prediction. DAF's original visual
feature consistency is not directly applicable to TMPA because TMPA freezes its
visual encoder and adapts prompt/text state. The TMPA counterpart therefore
regularizes the adapted prompt features against a frozen source prompt anchor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _class_dim(prob_maps: torch.Tensor) -> int:
    if prob_maps.ndim == 3:
        return 0
    if prob_maps.ndim == 4:
        return 1
    raise ValueError(
        f'Expected segmentation probabilities with 3 or 4 dims, got {tuple(prob_maps.shape)}.'
    )


def _normalize_probabilities(prob_maps: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize TMPA post-processed class maps into per-pixel distributions."""
    class_dim = _class_dim(prob_maps)
    probs = prob_maps.float().clamp_min(eps)
    return probs / probs.sum(dim=class_dim, keepdim=True).clamp_min(eps)


def source_prediction_consistency(
    adapted_prob_maps: torch.Tensor,
    source_prob_maps: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """DAF-style symmetric KL consistency on TMPA class probability maps.

    DAF applies symmetric KL to source/adapted segmentation logits after
    softmax. TMPA exposes post-processed class probability maps, so they are
    re-normalized per pixel before computing the same symmetric KL objective.
    Gradients flow only through the adapted prediction.
    """
    class_dim = _class_dim(adapted_prob_maps)
    p = _normalize_probabilities(adapted_prob_maps, eps=eps)
    q = _normalize_probabilities(source_prob_maps.detach(), eps=eps)

    log_p = p.clamp_min(eps).log()
    log_q = q.clamp_min(eps).log()
    kl_pq = (p * (log_p - log_q)).sum(dim=class_dim)
    kl_qp = (q * (log_q - log_p)).sum(dim=class_dim)
    return 0.5 * (kl_pq + kl_qp).mean()


def extract_prompt_features(model, args, test: bool = False) -> torch.Tensor:
    """Return the normalized prompt/text features currently used by TMPA."""
    text_features = model.get_text_features(test=test)
    if model.text_shift:
        if args.with_templates:
            text_features = model.text_shifter(text_features, test)
        else:
            text_features = model.text_shifter(text_features.squeeze(1), test)
    return F.normalize(text_features, dim=-1)


def prompt_feature_consistency(
    adapted_model,
    source_model,
    args,
    loss_type: str = 'cosine',
) -> torch.Tensor:
    """Feature consistency in the parameter space TMPA actually adapts.

    TMPA keeps visual features frozen, so DAF's visual feature consistency would
    be identically zero. This counterpart anchors the adapted prompt/text
    features to those of the frozen source model.
    """
    adapted_features = extract_prompt_features(adapted_model, args).float()
    with torch.no_grad():
        source_features = extract_prompt_features(source_model, args).float()

    if loss_type == 'l2':
        return ((adapted_features - source_features) ** 2).mean()
    if loss_type == 'cosine':
        return (1.0 - F.cosine_similarity(adapted_features, source_features, dim=-1)).mean()
    raise ValueError(f'Unknown prompt feature consistency type: {loss_type}')
