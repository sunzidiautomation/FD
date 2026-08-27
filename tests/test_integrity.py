import numpy as np
import pytest
from PIL import Image

from flair_t2i.metrics.integrity import (
    IntegrityGate,
    colour_spread,
    structural_change,
)


def _rich(seed: int = 0, size: int = 64) -> Image.Image:
    """An image with real colour variation and structure."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    return Image.fromarray(base).resize((size, size), Image.NEAREST)


def _flat(rgb=(255, 200, 0), size: int = 64) -> Image.Image:
    return Image.new("RGB", (size, size), rgb)


def _scrambled(image: Image.Image) -> Image.Image:
    """Same pixels, structure destroyed -- colour spread is untouched."""
    a = np.asarray(image).reshape(-1, 3)
    rng = np.random.default_rng(1)
    return Image.fromarray(rng.permutation(a).reshape(np.asarray(image).shape))


# --------------------------------------------------------------- primitives


def test_colour_spread_of_a_solid_image_is_zero():
    assert colour_spread(_flat()) == pytest.approx(0.0)


def test_colour_spread_grows_with_variation():
    assert colour_spread(_rich()) > colour_spread(_flat())


def test_structural_change_of_an_image_with_itself_is_zero():
    image = _rich()
    assert structural_change(image, image) == pytest.approx(0.0, abs=1e-6)


def test_structural_change_is_large_for_scrambled_pixels():
    image = _rich()
    assert structural_change(image, _scrambled(image)) > 0.5


# --------------------------------------------------------------- the gate


def test_identical_images_pass():
    image = _rich()
    assert IntegrityGate().check(image, image).ok


def test_colour_collapse_is_rejected():
    """A frame that has fallen into one saturated hue is not a measurement.

    This is the failure this gate exists for: a destroyed generation
    maximises every 'how much did X change' metric, so without this it
    outranks every genuine result.
    """
    verdict = IntegrityGate().check(_rich(), _flat())

    assert not verdict.ok
    assert "colour" in verdict.reason
    assert verdict.colour_ratio == pytest.approx(0.0)


def test_structural_collapse_is_rejected_even_with_healthy_colour():
    """Scrambling preserves the colour histogram exactly, so only the
    structural half of the gate can catch it."""
    image = _rich()
    verdict = IntegrityGate().check(image, _scrambled(image))

    assert not verdict.ok
    assert "structure" in verdict.reason
    assert verdict.colour_ratio == pytest.approx(1.0, abs=0.05)


def test_verdict_reports_the_measured_numbers():
    verdict = IntegrityGate().check(_rich(), _rich())
    assert 0.0 <= verdict.structural_change <= 2.0
    assert verdict.colour_ratio > 0.0
    assert verdict.reason is None


def test_thresholds_are_configurable():
    image = _rich()
    scrambled = _scrambled(image)

    assert not IntegrityGate(max_structural_change=0.1).check(image, scrambled).ok
    assert IntegrityGate(
        max_structural_change=1.5, min_colour_ratio=0.0
    ).check(image, scrambled).ok


def test_a_black_baseline_does_not_divide_by_zero():
    """colour_ratio is relative to the baseline; a degenerate baseline
    must not raise."""
    verdict = IntegrityGate().check(_flat((0, 0, 0)), _rich())
    assert isinstance(verdict.ok, bool)
