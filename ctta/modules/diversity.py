"""DAF-style class diversity regularization for TMPA CTTA."""

from __future__ import annotations

import torch


def _class_dim(prob_maps: torch.Tensor) -> int:
    if prob_maps.ndim == 3:
        return 0
    if prob_maps.ndim == 4:
        return 1
    raise ValueError(
        f'Expected segmentation probabilities with 3 or 4 dims, got {tuple(prob_maps.shape)}.'
    )


def diversity_loss(prob_maps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """DAF-style class-wise diversity loss on TMPA probability maps.

    DAF computes ``-H(E[p(y|x)])`` from segmentation logits after softmax so
    that minimizing the loss maximizes the entropy of the marginal class
    distribution and discourages collapse to a small set of classes.

    TMPA exposes post-processed class probability maps rather than raw logits,
    so this implementation first re-normalizes them along the class dimension
    and then applies the same marginal-entropy objective across batch/spatial
    dimensions.

    Args:
        prob_maps: ``[C,H,W]`` or ``[B,C,H,W]`` TMPA class probability maps.
        eps: Numerical stability constant.

    Returns:
        Scalar negative marginal entropy. Minimizing it encourages diversity.
    """
    class_dim = _class_dim(prob_maps)
    probs = prob_maps.float().clamp_min(eps)
    probs = probs / probs.sum(dim=class_dim, keepdim=True).clamp_min(eps)

    reduce_dims = tuple(dim for dim in range(probs.ndim) if dim != class_dim)
    marginal = probs.mean(dim=reduce_dims)
    marginal = marginal / marginal.sum().clamp_min(eps)
    entropy = -(marginal * marginal.clamp_min(eps).log()).sum()
    return -entropy
