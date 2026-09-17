"""DAF corruption backend integration.

This module deliberately reuses DAF's ``utils/imagecorruptions`` implementation
instead of reimplementing the corruption formulas. That keeps the corruption
parameters and frost assets identical to the DAF evaluation protocol.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

import numpy as np


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


def _resolve_daf_root(daf_root: str | None) -> Path:
    candidates = []
    if daf_root:
        candidates.append(Path(daf_root))
    if os.environ.get('DAF_ROOT'):
        candidates.append(Path(os.environ['DAF_ROOT']))

    candidates.append(Path(__file__).resolve().parents[2] / 'DAF')
    candidates.append(Path.cwd().parent / 'DAF')

    for candidate in candidates:
        init_file = candidate.expanduser().resolve() / 'utils' / 'imagecorruptions' / '__init__.py'
        if init_file.is_file():
            return candidate.expanduser().resolve()

    searched = ', '.join(str(p) for p in candidates)
    raise FileNotFoundError(
        'Cannot locate the DAF repository. Pass --daf_root /path/to/DAF or set DAF_ROOT. '
        f'Searched: {searched}'
    )


@lru_cache(maxsize=4)
def _load_daf_imagecorruptions(daf_root: str | None):
    root = _resolve_daf_root(daf_root)
    package_dir = root / 'utils' / 'imagecorruptions'
    init_file = package_dir / '__init__.py'

    package_name = '_tmpa_daf_imagecorruptions'
    cached = sys.modules.get(package_name)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to load DAF imagecorruptions from {init_file}')

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def apply_corruption(
    image: np.ndarray,
    corruption_name: str,
    severity: int,
    sample_idx: int,
    daf_root: str | None,
) -> np.ndarray:
    """Apply one DAF corruption deterministically to a uint8 RGB image."""
    corruption_name = corruption_name.lower()
    if corruption_name == 'original':
        return image

    if severity not in (1, 2, 3, 4, 5):
        raise ValueError(f'corruption severity must be in [1, 5], got {severity}')
    if image.dtype != np.uint8:
        raise TypeError(f'DAF corruptions expect uint8 input, got {image.dtype}')

    backend = _load_daf_imagecorruptions(daf_root)
    valid_names = backend.get_corruption_names('common')
    if corruption_name not in valid_names:
        raise ValueError(
            f"Unknown DAF common corruption '{corruption_name}'. Valid names: {valid_names}"
        )

    rng_state = np.random.get_state()
    try:
        np.random.seed(int(sample_idx))
        corrupted = backend.corrupt(
            image,
            severity=severity,
            corruption_name=corruption_name,
        )
    finally:
        np.random.set_state(rng_state)

    return np.asarray(corrupted, dtype=np.uint8)
