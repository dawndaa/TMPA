# -*- coding: utf-8 -*-
"""DAF/ImageNet-C corruption functions vendored for standalone TMPA CTTA.

The common corruption formulas and severity tables follow DAF's
``utils/imagecorruptions/corruptions.py``. The implementation is packaged here
so no external DAF checkout is required.
"""

from io import BytesIO
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import skimage as sk
from skimage.filters import gaussian
from scipy.ndimage import map_coordinates
from scipy.ndimage import zoom as scizoom


def _gaussian(image, sigma, **kwargs):
    """Compatibility wrapper for old/new scikit-image channel arguments."""
    try:
        return gaussian(image, sigma=sigma, channel_axis=kwargs.pop('channel_axis', None), **kwargs)
    except TypeError:
        multichannel = image.ndim == 3
        return gaussian(image, sigma=sigma, multichannel=multichannel, **kwargs)


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        coords = np.arange(-8, 9)
        ksize = (3, 3)
    else:
        coords = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    x, y = np.meshgrid(coords, coords)
    kernel = np.array((x ** 2 + y ** 2) <= radius ** 2, dtype=dtype)
    kernel /= np.sum(kernel)
    return cv2.GaussianBlur(kernel, ksize=ksize, sigmaX=alias_blur)


def plasma_fractal(mapsize=256, wibbledecay=3):
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, -1, axis=0)
        squareaccum += np.roll(squareaccum, -1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        drgrid = maparray[stepsize // 2:mapsize:stepsize,
                          stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        maparray[0:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize] = wibbledmean(ldrsum + lulsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        maparray[stepsize // 2:mapsize:stepsize,
                 0:mapsize:stepsize] = wibbledmean(tdrsum + tulsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay
    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
    top0 = (img.shape[0] - ch0) // 2
    ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
    top1 = (img.shape[1] - ch1) // 2
    img = scizoom(
        img[top0:top0 + ch0, top1:top1 + ch1],
        (zoom_factor, zoom_factor, 1),
        order=1,
    )
    return img


def _motion_blur(x, radius, sigma, angle):
    width = radius * 2 + 1
    k = np.exp(-(np.arange(width) ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    k /= np.sum(k)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])
    blurred = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(((i * point[0]) / hypot) - 0.5)
        dx = -math.ceil(((i * point[1]) / hypot) - 0.5)
        if abs(dy) >= x.shape[0] or abs(dx) >= x.shape[1]:
            break
        shifted = np.roll(x, shift=dx, axis=1)
        if dx < 0:
            shifted[:, dx:] = shifted[:, dx - 1:dx]
        elif dx > 0:
            shifted[:, :dx] = shifted[:, dx:dx + 1]
        shifted = np.roll(shifted, shift=dy, axis=0)
        if dy < 0:
            shifted[dy:, :] = shifted[dy - 1:dy, :]
        elif dy > 0:
            shifted[:dy, :] = shifted[dy:dy + 1, :]
        blurred += k[i] * shifted
    return blurred


def gaussian_noise(x, severity=1):
    c = [.08, .12, .18, .26, .38][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.03, .06, .09, .17, .27][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.0
    out = np.copy(x)
    flipped = np.random.rand(*x.shape) <= c
    salted = np.random.rand(*x.shape) <= .5
    out[flipped & salted] = 1
    out[flipped & ~salted] = 0
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def speckle_noise(x, severity=1):
    c = [.15, .2, .35, .45, .6][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]
    x = _gaussian(np.array(x) / 255.0, c, channel_axis=-1)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1):
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
    x = np.uint8(_gaussian(np.array(x) / 255.0, c[0], channel_axis=-1) * 255)
    shape = x.shape
    for _ in range(c[2]):
        for h in range(shape[0] - c[1], c[1], -1):
            for w in range(shape[1] - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                hp, wp = h + dy, w + dx
                x[h, w], x[hp, wp] = x[hp, wp], x[h, w]
    return np.clip(_gaussian(x / 255.0, c[0], channel_axis=-1), 0, 1) * 255


def defocus_blur(x, severity=1):
    c = [(3, .1), (4, .5), (6, .5), (8, .5), (10, .5)][severity - 1]
    x = np.array(x) / 255.0
    kernel = disk(c[0], c[1])
    channels = [cv2.filter2D(x[:, :, d], -1, kernel) for d in range(3)]
    return np.clip(np.array(channels).transpose((1, 2, 0)), 0, 1) * 255


def motion_blur(x, severity=1):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    x = np.array(x)
    return np.clip(_motion_blur(x, c[0], c[1], np.random.uniform(-45, 45)), 0, 255)


def zoom_blur(x, severity=1):
    c = [
        np.arange(1, 1.11, .01), np.arange(1, 1.16, .01),
        np.arange(1, 1.21, .02), np.arange(1, 1.26, .02),
        np.arange(1, 1.31, .03),
    ][severity - 1]
    x = (np.array(x) / 255.0).astype(np.float32)
    out = np.zeros_like(x)
    for factor in c:
        z = clipped_zoom(x, factor)[:x.shape[0], :x.shape[1], :]
        out[:z.shape[0], :z.shape[1]] += z
    return np.clip((x + out) / (len(c) + 1), 0, 1) * 255


def snow(x, severity=1):
    c = [
        (.1, .3, 3, .5, 10, 4, .8), (.2, .3, 2, .5, 12, 4, .7),
        (.55, .3, 4, .9, 12, 8, .7), (.55, .3, 4.5, .85, 12, 8, .65),
        (.55, .3, 2.5, .85, 12, 12, .55),
    ][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.0
    layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])
    layer = clipped_zoom(layer[..., None], c[2])
    layer[layer < c[3]] = 0
    layer = np.clip(layer.squeeze(), 0, 1)
    layer = _motion_blur(layer, c[4], c[5], np.random.uniform(-135, -45))
    layer = np.round(layer * 255).astype(np.uint8) / 255.0
    layer = layer[:x.shape[0], :x.shape[1]][..., None]
    gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(x.shape[0], x.shape[1], 1)
    x = c[6] * x + (1 - c[6]) * np.maximum(x, gray * 1.5 + .5)
    return np.clip(x + layer + np.rot90(layer, k=2), 0, 1) * 255


def frost(x, severity=1):
    c = [(1, .4), (.8, .6), (.7, .7), (.65, .7), (.6, .75)][severity - 1]
    frost_dir = Path(__file__).resolve().parent / 'frost'
    files = [
        frost_dir / 'frost1.png', frost_dir / 'frost2.png', frost_dir / 'frost3.png',
        frost_dir / 'frost4.jpg', frost_dir / 'frost5.jpg', frost_dir / 'frost6.jpg',
    ]
    texture = cv2.imread(str(files[np.random.randint(len(files))]))
    if texture is None:
        raise FileNotFoundError(f'Bundled frost texture missing under {frost_dir}')
    shape = np.array(x).shape
    th, tw = texture.shape[:2]
    scale = max(shape[0] / th, shape[1] / tw, 1.0) * 1.1
    texture = cv2.resize(texture, (int(np.ceil(tw * scale)), int(np.ceil(th * scale))), interpolation=cv2.INTER_CUBIC)
    max_y = max(texture.shape[0] - shape[0], 1)
    max_x = max(texture.shape[1] - shape[1], 1)
    y = np.random.randint(0, max_y)
    xx = np.random.randint(0, max_x)
    texture = texture[y:y + shape[0], xx:xx + shape[1]][..., [2, 1, 0]]
    return np.clip(c[0] * np.array(x) + c[1] * texture, 0, 255)


def fog(x, severity=1):
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
    shape = np.array(x).shape
    mapsize = 1 if max(shape) == 0 else 2 ** (int(max(shape)) - 1).bit_length()
    x = np.array(x) / 255.0
    max_val = x.max()
    x += c[0] * plasma_fractal(mapsize=mapsize, wibbledecay=c[1])[:shape[0], :shape[1]][..., None]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]
    x = sk.color.rgb2hsv(np.array(x) / 255.0)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    return np.clip(sk.color.hsv2rgb(x), 0, 1) * 255


def contrast(x, severity=1):
    c = [.4, .3, .2, .1, .05][severity - 1]
    x = np.array(x) / 255.0
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]
    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    return Image.open(output).convert('RGB')


def pixelate(x, severity=1):
    c = [.6, .5, .4, .3, .25][severity - 1]
    shape = np.array(x).shape
    x = x.resize((int(shape[1] * c), int(shape[0] * c)), Image.BOX)
    return x.resize((shape[1], shape[0]), Image.NEAREST)


def elastic_transform(image, severity=1):
    image = np.array(image, dtype=np.float32) / 255.0
    shape = image.shape
    sigma = np.array(shape[:2]) * .01
    alpha = [250 * .05, 250 * .065, 250 * .085, 250 * .1, 250 * .12][severity - 1]
    max_dx = shape[0] * .005
    max_dy = shape[0] * .005
    dx = (_gaussian(np.random.uniform(-max_dx, max_dx, size=shape[:2]), sigma, mode='reflect', truncate=3) * alpha).astype(np.float32)
    dy = (_gaussian(np.random.uniform(-max_dy, max_dy, size=shape[:2]), sigma, mode='reflect', truncate=3) * alpha).astype(np.float32)
    dx, dy = dx[..., None], dy[..., None]
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = (
        np.reshape(y + dy, (-1, 1)),
        np.reshape(x + dx, (-1, 1)),
        np.reshape(z, (-1, 1)),
    )
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


def spatter(x, severity=1):
    # Validation-only corruption. Kept lightweight because CTTA common uses the first 15.
    c = [.55, .60, .65, .70, .75][severity - 1]
    arr = np.array(x, dtype=np.float32) / 255.0
    mask = _gaussian((np.random.rand(*arr.shape[:2]) > c).astype(np.float32), 1.5)
    color = np.array([63, 42, 20], dtype=np.float32) / 255.0
    return np.clip(arr * (1 - mask[..., None]) + color * mask[..., None], 0, 1) * 255


def saturate(x, severity=1):
    c = [(.3, 0), (.1, 0), (2, 0), (5, .1), (20, .2)][severity - 1]
    arr = sk.color.rgb2hsv(np.array(x) / 255.0)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * c[0] + c[1], 0, 1)
    return np.clip(sk.color.hsv2rgb(arr), 0, 1) * 255
