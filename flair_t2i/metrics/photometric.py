"""Colour, size, and lighting metrics (spec section 3.4).

Each function returns a value in [0, 1]. The absolute forms land on the
universes declared in ``flair_t2i.fuzzy.membership``; the delta forms are
what BASM calibration measures.

Two families, deliberately distinct:

* ``*_delta(a, b, ...)``  -- how far an attribute MOVED between two images.
  This is what calibration measures when a prompt is swapped at one block.
* ``*_absolute(image, ...)`` -- where an attribute SITS on its universe.
  This is what the coherence guard and the evaluation compare against a
  fuzzy region.

Conflating them would silently produce a meaningless BASM.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage import color as skcolor

from .masking import mask_area_ratio, masked_mean_rgb

#: CIELAB dE beyond this counts as a total colour change.
DELTA_E_CEILING = 100.0


def _lab(rgb: np.ndarray) -> np.ndarray:
    return skcolor.rgb2lab((np.asarray(rgb, dtype=np.float64) / 255.0).reshape(1, 1, 3)).reshape(3)


def size_absolute(image: Image.Image, mask: np.ndarray) -> float:
    """Object mask area ratio -- the SIZE universe."""
    return mask_area_ratio(mask)


def size_delta(
    image_a: Image.Image,
    image_b: Image.Image,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    """Change in how much of the frame the object occupies.

    Takes two masks, not one: size is only observable by re-segmenting the
    changed image.
    """
    return float(abs(mask_area_ratio(mask_a) - mask_area_ratio(mask_b)))


def warmth_absolute(image: Image.Image) -> float:
    """Warm/cool balance of the whole frame, 0 cool to 1 warm.

    Lighting is a scene property, so this is deliberately unmasked.
    """
    pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
    red, blue = pixels[..., 0].mean(), pixels[..., 2].mean()
    total = red + blue
    if total <= 0.0:
        return 0.5
    return float(np.clip(((red - blue) / total + 1.0) / 2.0, 0.0, 1.0))


def lighting_delta(
    image_a: Image.Image, image_b: Image.Image, mask: np.ndarray | None = None
) -> float:
    """Shift in warm/cool balance. ``mask`` is accepted and ignored."""
    return float(abs(warmth_absolute(image_a) - warmth_absolute(image_b)))


def color_delta(image_a: Image.Image, image_b: Image.Image, mask: np.ndarray) -> float:
    """Masked CIELAB dE between two images, normalised to [0, 1]."""
    mean_a = masked_mean_rgb(image_a, mask)
    mean_b = masked_mean_rgb(image_b, mask)
    if mean_a is None or mean_b is None:
        return 0.0
    distance = float(np.linalg.norm(_lab(mean_a) - _lab(mean_b)))
    return float(np.clip(distance / DELTA_E_CEILING, 0.0, 1.0))


def color_absolute(
    image: Image.Image, mask: np.ndarray, target_rgb: tuple[int, int, int]
) -> float:
    """1 - normalised dE to ``target_rgb`` -- the COLOR universe."""
    mean = masked_mean_rgb(image, mask)
    if mean is None:
        return 0.0
    distance = float(np.linalg.norm(_lab(mean) - _lab(target_rgb)))
    return float(np.clip(1.0 - distance / DELTA_E_CEILING, 0.0, 1.0))
