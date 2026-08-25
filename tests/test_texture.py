import numpy as np
import pytest
from PIL import Image

from flair_t2i.metrics.texture import gram_texture_delta

FULL = np.ones((64, 64), dtype=np.float32)


def _solid(value=128):
    return Image.new("RGB", (64, 64), (value, value, value))


def _noise(seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))


def _stripes(period=4):
    array = np.zeros((64, 64, 3), dtype=np.uint8)
    array[:, ::period] = 255
    return Image.fromarray(array)


def test_identical_images_have_zero_texture_delta():
    image = _noise(1)
    assert gram_texture_delta(image, image, FULL) == pytest.approx(0.0, abs=1e-9)


def test_smooth_versus_noisy_registers_a_large_delta():
    assert gram_texture_delta(_solid(), _noise(2), FULL) > 0.1


def test_two_smooth_images_register_a_small_delta():
    assert gram_texture_delta(_solid(120), _solid(140), FULL) < 0.05


def test_delta_is_symmetric():
    a, b = _solid(), _stripes()
    assert gram_texture_delta(a, b, FULL) == pytest.approx(
        gram_texture_delta(b, a, FULL)
    )


def test_delta_stays_within_the_unit_interval():
    assert 0.0 <= gram_texture_delta(_noise(3), _stripes(), FULL) <= 1.0


def test_delta_is_zero_when_the_mask_is_empty():
    assert gram_texture_delta(_solid(), _noise(4), np.zeros((64, 64))) == 0.0
