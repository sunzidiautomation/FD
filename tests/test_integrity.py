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
        max_structural_change=1.5, min_colour_ratio=0.0, max_speckle_ratio=1e9
    ).check(image, scrambled).ok


def test_a_black_baseline_does_not_divide_by_zero():
    """colour_ratio is relative to the baseline; a degenerate baseline
    must not raise."""
    verdict = IntegrityGate().check(_flat((0, 0, 0)), _rich())
    assert isinstance(verdict.ok, bool)


# --------------------------------------------------------------- speckle
#
# The third failure mode, found in a real action sweep: the top-ranked
# frame was a car buried in mottled multicoloured specks. It passed both
# original signals -- the specks are colourful, so spread survived at
# 0.91, and the layout was untouched, so 1-SSIM stayed at 0.30. Neither
# signal looks at anything smaller than the whole frame.

from flair_t2i.metrics.integrity import speckle  # noqa: E402


def _scene(side=128, texture=0.0, seed=7):
    """A gradient with solid blocks, optionally over fine grain.

    ``texture`` stands in for the grain any real photograph carries. With
    none, the speckle ratio against a noised copy is absurdly large, which
    would prove the mechanism works but say nothing about whether the
    default threshold is set anywhere near the right place.
    """
    rng = np.random.default_rng(seed)
    row = np.linspace(20, 200, side)
    plane = np.repeat(row[None, :], side, axis=0)
    array = np.stack([plane, plane * 0.7, plane * 0.4], axis=-1)
    array[20:60, 20:60] = (200, 40, 40)
    array[70:110, 60:120] = (30, 80, 160)
    if texture:
        array = array + rng.normal(0.0, texture, array.shape)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")


def _specked(image, fraction=0.04, seed=0):
    """Salt-and-pepper in colour: isolated pixels, layout untouched."""
    rng = np.random.default_rng(seed)
    array = np.asarray(image.convert("RGB")).copy()
    hit = rng.random(array.shape[:2]) < fraction
    array[hit] = rng.integers(0, 256, size=(int(hit.sum()), 3), dtype=np.uint8)
    return Image.fromarray(array, "RGB")


def test_speckle_is_near_zero_for_a_smooth_image():
    assert speckle(_scene()) < 0.01


def test_speckle_rises_with_isolated_noisy_pixels():
    smooth = _scene()
    assert speckle(_specked(smooth)) > 5 * speckle(smooth)


def test_a_genuine_sharp_edge_is_not_speckle():
    """The discriminating property.

    Removing motion blur -- a legitimate outcome for an action swap --
    raises high-frequency energy, so plain sharpness cannot be the signal.
    A median filter erases isolated pixels and preserves edges, so the
    deviation from it separates the two.
    """
    edged = np.zeros((128, 128, 3), dtype=np.uint8)
    edged[:, 64:] = 255
    assert speckle(Image.fromarray(edged, "RGB")) < 0.01


def test_gate_rejects_speckle_that_both_original_signals_miss():
    """The exact shape of the bug, at a realistic magnitude.

    Grain in the baseline keeps the ratio near what the real confetti
    frame scored (1.92), so this pins the default threshold and not just
    the mechanism.
    """
    baseline = _scene(texture=6.0)
    candidate = _specked(baseline)

    loose = IntegrityGate(max_speckle_ratio=1e9).check(baseline, candidate)
    assert loose.ok, "premise: the two original signals both pass this frame"
    assert loose.colour_ratio >= 0.75
    assert loose.structural_change <= 0.60
    assert 1.5 < loose.speckle_ratio < 3.0, "premise: a realistic speckle ratio"

    verdict = IntegrityGate().check(baseline, candidate)
    assert not verdict.ok
    assert "speckle" in verdict.reason


def test_gate_accepts_a_change_that_adds_no_speckle():
    baseline = _scene(texture=6.0)
    shifted = Image.fromarray(
        np.clip(np.asarray(baseline, dtype=np.int16) + 25, 0, 255).astype(np.uint8),
        "RGB",
    )
    assert IntegrityGate().check(baseline, shifted).ok


def test_verdict_reports_the_speckle_ratio():
    scene = _scene(texture=6.0)
    assert IntegrityGate().check(scene, scene).speckle_ratio == pytest.approx(1.0)


def test_a_speckle_free_baseline_does_not_divide_by_zero():
    flat = _flat((10, 10, 10), size=128)
    assert isinstance(IntegrityGate().check(flat, _scene(texture=6.0)).ok, bool)
