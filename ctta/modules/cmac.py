"""CMAC-style directional source anchoring for TMPA CTTA.

DAF CMAC operates on visual-feature/text similarities: source features assign an
anchor class to each patch, then adaptation is penalized if it moves away from
the anchor or toward non-anchor classes. TMPA freezes its visual encoder and
adapts prompt/text state, so copying the original feature-space implementation
would not reflect the parameter that actually drifts.

This module transfers the same directional hinge principle to TMPA's pixel-wise
class probabilities. The frozen source prediction supplies the anchor class.
Only two harmful directions are penalized:
  1) adapted probability of the source-assigned class decreases;
  2) adapted probability of any non-assigned class increases.

Unlike symmetric source consistency, beneficial motion in the opposite
 directions is not penalized.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .consistency import _class_dim, _normalize_probabilities


def cmac_source_anchor_loss(
    adapted_prob_maps: torch.Tensor,
    source_prob_maps: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Directional source-anchor hinge loss adapted from DAF CMAC.

    Args:
        adapted_prob_maps: TMPA adapted class probability maps, [C,H,W] or [B,C,H,W].
        source_prob_maps: frozen-source class probability maps with the same shape.
        eps: numerical floor used before probability normalization.

    Returns:
        Scalar loss = loss_away + loss_toward.
    """
    if adapted_prob_maps.shape != source_prob_maps.shape:
        raise ValueError(
            'CMAC requires adapted/source probability maps with identical shapes, '
            f'got {tuple(adapted_prob_maps.shape)} and {tuple(source_prob_maps.shape)}.'
        )

    class_dim = _class_dim(adapted_prob_maps)
    p_adapt = _normalize_probabilities(adapted_prob_maps, eps=eps)
    p_source = _normalize_probabilities(source_prob_maps.detach(), eps=eps)

    assigned = p_source.argmax(dim=class_dim, keepdim=True)
    adapt_assigned = p_adapt.gather(class_dim, assigned).squeeze(class_dim)
    source_assigned = p_source.gather(class_dim, assigned).squeeze(class_dim)

    # DAF CMAC loss_away: do not move away from the source-assigned class.
    loss_away = F.relu(source_assigned - adapt_assigned).mean()

    # DAF CMAC loss_toward: do not move toward source-unassigned classes.
    toward = F.relu(p_adapt - p_source)
    assigned_mask = torch.zeros_like(p_adapt, dtype=torch.bool)
    assigned_mask.scatter_(class_dim, assigned, True)
    toward = toward.masked_fill(assigned_mask, 0.0)

    num_classes = p_adapt.shape[class_dim]
    if num_classes <= 1:
        loss_toward = toward.new_zeros(())
    else:
        loss_toward = toward.sum(dim=class_dim).div(num_classes - 1).mean()

    return loss_away + loss_toward
