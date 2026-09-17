"""Vendored DAF image corruption backend.

Derived from ``dawndaa/DAF/utils/imagecorruptions`` (MIT licence). The public
interface is intentionally kept compatible with DAF so CTTA code can call
``corrupt`` and ``get_corruption_names`` without an external repository.
"""

import numpy as np
from PIL import Image

from .corruptions import (
    brightness,
    contrast,
    defocus_blur,
    elastic_transform,
    fog,
    frost,
    gaussian_blur,
    gaussian_noise,
    glass_blur,
    impulse_noise,
    jpeg_compression,
    motion_blur,
    pixelate,
    saturate,
    shot_noise,
    snow,
    spatter,
    speckle_noise,
    zoom_blur,
)

corruption_tuple = (
    gaussian_noise,
    shot_noise,
    impulse_noise,
    defocus_blur,
    glass_blur,
    motion_blur,
    zoom_blur,
    snow,
    frost,
    fog,
    brightness,
    contrast,
    elastic_transform,
    pixelate,
    jpeg_compression,
    speckle_noise,
    gaussian_blur,
    spatter,
    saturate,
)
corruption_dict = {func.__name__: func for func in corruption_tuple}


def corrupt(image, severity=1, corruption_name=None, corruption_number=-1):
    if not isinstance(image, np.ndarray):
        raise AttributeError('Expecting type(image) to be numpy.ndarray')
    if image.dtype.type is not np.uint8:
        raise AttributeError('Expecting image.dtype.type to be numpy.uint8')
    if image.ndim not in (2, 3):
        raise AttributeError('Expecting image.shape to be HxW or HxWxC')
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    height, width, channels = image.shape
    if height < 32 or width < 32:
        raise AttributeError('Image width and height must be at least 32 pixels')
    if channels not in (1, 3):
        raise AttributeError('Expecting image to have either 1 or 3 channels')
    if channels == 1:
        image = np.stack((np.squeeze(image),) * 3, axis=-1)
    if severity not in (1, 2, 3, 4, 5):
        raise AttributeError('Severity must be an integer in [1, 5]')

    if corruption_name is not None:
        func = corruption_dict[corruption_name]
    elif corruption_number != -1:
        func = corruption_tuple[corruption_number]
    else:
        raise ValueError('Either corruption_name or corruption_number must be passed')
    return np.uint8(func(Image.fromarray(image), severity))


def get_corruption_names(subset='common'):
    if subset == 'common':
        return [f.__name__ for f in corruption_tuple[:15]]
    if subset == 'validation':
        return [f.__name__ for f in corruption_tuple[15:]]
    if subset == 'all':
        return [f.__name__ for f in corruption_tuple]
    if subset == 'noise':
        return [f.__name__ for f in corruption_tuple[:3]]
    if subset == 'blur':
        return [f.__name__ for f in corruption_tuple[3:7]]
    if subset == 'weather':
        return [f.__name__ for f in corruption_tuple[7:11]]
    if subset == 'digital':
        return [f.__name__ for f in corruption_tuple[11:15]]
    raise ValueError("subset must be one of ['common', 'validation', 'all']")
