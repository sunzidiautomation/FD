"""Gram-matrix texture distance (spec section 3.4).

Classical Gram-matrix style statistics computed over multi-scale gradient
responses, in NumPy, so calibration needs no extra model weights. The
signature matches the other delta metrics, so a DISTS implementation can
replace this for the paper's final numbers without touching the harness.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .masking import MIN_MASK_PIXELS

_SCALES = (1, 2, 4)


def _responses(gray: np.ndarray, scale: int) -> np.ndarray:
    """Horizontal, vertical, and diagonal gradient responses at one scale."""
    shifted_x = np.roll(gray, scale, axis=1)
    shifted_y = np.roll(gray, scale, axis=0)
    shifted_d = np.roll(shifted_x, scale, axis=0)
    return np.stack([gray - shifted_x, gray - shifted_y, gray - shifted_d], axis=0)


def _gram(image: Image.Image, mask: np.ndarray) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    selected = mask > 0.5

    grams = []
    for scale in _SCALES:
        responses = _responses(gray, scale)
        flat = responses[:, selected]
        grams.append(flat @ flat.T / max(flat.shape[1], 1))
    return np.stack(grams)


def gram_texture_delta(
    image_a: Image.Image, image_b: Image.Image, mask: np.ndarray
) -> float:
    """Normalised Gram-matrix distance inside the mask, in [0, 1]."""
    if (mask > 0.5).sum() < MIN_MASK_PIXELS:
        return 0.0

    gram_a, gram_b = _gram(image_a, mask), _gram(image_b, mask)
    distance = float(np.linalg.norm(gram_a - gram_b))
    scale = float(np.linalg.norm(gram_a) + np.linalg.norm(gram_b))
    if scale == 0.0:
        return 0.0
    return float(np.clip(distance / scale, 0.0, 1.0))
