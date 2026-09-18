"""Self-contained DAF corruption integration for TMPA CTTA.

The DAF/ImageNet-C corruption backend is vendored under
``ctta/vendor/daf_imagecorruptions``. No external DAF checkout is required.
"""

from __future__ import annotations

from typing import Iterable, List

import cv2
import numpy as np

from ctta.vendor import daf_imagecorruptions as backend


COMMON_CORRUPTIONS = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'glass_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression',
]


def expand_corruptions(corruptions: Iterable[str]) -> List[str]:
    """Expand convenience aliases while preserving the requested stream order."""
    expanded: List[str] = []
    for name in corruptions:
        normalized = name.strip().lower()
        if normalized == 'common':
            expanded.extend(COMMON_CORRUPTIONS)
        else:
            expanded.append(normalized)
    if not expanded:
        raise ValueError('At least one corruption domain is required.')
    return expanded


def _procedural_frost(image: np.ndarray, severity: int) -> np.ndarray:
    """Standalone frost texture used when external DAF image assets are absent.

    DAF's frost formula blends the input with a randomly selected icy texture.
    To keep this repository self-contained without binary third-party assets,
    we synthesize the icy texture from the same NumPy RNG that is already seeded
    per sample. The DAF severity blend coefficients are preserved.
    """
    blend = [(1.0, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
    h, w = image.shape[:2]

    noise = np.random.normal(loc=0.55, scale=0.22, size=(h, w)).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    angle = np.random.uniform(0.0, np.pi)
    phase = np.random.uniform(0.0, 2.0 * np.pi)
    streak = np.sin((xx * np.cos(angle) + yy * np.sin(angle)) * 0.055 + phase)
    ice = np.clip(noise + 0.18 * streak, 0.0, 1.0)

    texture = np.stack(
        [0.65 * ice + 0.25, 0.78 * ice + 0.18, 0.95 * ice + 0.05],
        axis=-1,
    )
    texture = np.uint8(np.clip(texture, 0.0, 1.0) * 255.0)
    sigma = max(1.0, min(h, w) / 160.0)
    texture = cv2.GaussianBlur(texture, (0, 0), sigmaX=sigma, sigmaY=sigma)

    return np.clip(blend[0] * image.astype(np.float32) + blend[1] * texture, 0, 255).astype(np.uint8)


def apply_corruption(
    image: np.ndarray,
    corruption_name: str,
    severity: int,
    sample_idx: int,
    daf_root: str | None = None,
) -> np.ndarray:
    """Apply one vendored DAF corruption deterministically to a uint8 RGB image.

    ``daf_root`` is retained only for CLI/backward compatibility and is ignored.
    """
    del daf_root
    corruption_name = corruption_name.lower()
    if corruption_name == 'original':
        return image

    if severity not in (1, 2, 3, 4, 5):
        raise ValueError(f'corruption severity must be in [1, 5], got {severity}')
    if image.dtype != np.uint8:
        raise TypeError(f'DAF corruptions expect uint8 input, got {image.dtype}')

    valid_names = backend.get_corruption_names('common')
    if corruption_name not in valid_names:
        raise ValueError(
            f"Unknown DAF common corruption '{corruption_name}'. Valid names: {valid_names}"
        )

    rng_state = np.random.get_state()
    try:
        np.random.seed(int(sample_idx))
        if corruption_name == 'frost':
            corrupted = _procedural_frost(image, severity)
        else:
            corrupted = backend.corrupt(
                image,
                severity=severity,
                corruption_name=corruption_name,
            )
    finally:
        np.random.set_state(rng_state)

    return np.asarray(corrupted, dtype=np.uint8)
