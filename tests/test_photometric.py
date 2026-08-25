import numpy as np
import pytest
from PIL import Image

pytest.importorskip("skimage")

from flair_t2i.metrics.masking import RectMasker
from flair_t2i.metrics.photometric import (
    color_absolute,
    color_delta,
    lighting_delta,
    size_absolute,
    size_delta,
    warmth_absolute,
)

FULL = np.ones((64, 64), dtype=np.float32)


def _solid(rgb):
    return Image.new("RGB", (64, 64), rgb)


def test_size_absolute_is_the_area_ratio():
    mask = RectMasker((0.0, 0.0, 0.5, 1.0))(_solid((0, 0, 0)), "x")
    assert size_absolute(_solid((0, 0, 0)), mask) == pytest.approx(0.5)


def test_size_delta_is_the_area_difference():
    small = RectMasker((0.0, 0.0, 0.5, 0.5))(_solid((0, 0, 0)), "x")
    large = RectMasker((0.0, 0.0, 1.0, 0.5))(_solid((0, 0, 0)), "x")
    assert size_delta(
        _solid((0, 0, 0)), _solid((0, 0, 0)), small, large
    ) == pytest.approx(0.25)


def test_size_delta_of_identical_masks_is_zero():
    assert size_delta(_solid((0, 0, 0)), _solid((0, 0, 0)), FULL, FULL) == 0.0


def test_warmth_is_high_for_orange_and_low_for_blue():
    assert warmth_absolute(_solid((255, 180, 80))) == pytest.approx(0.7612, abs=1e-3)
    assert warmth_absolute(_solid((80, 180, 255))) == pytest.approx(0.2388, abs=1e-3)


def test_warmth_of_grey_is_neutral():
    assert warmth_absolute(_solid((128, 128, 128))) == pytest.approx(0.5)


def test_lighting_delta_spans_warm_to_cool():
    delta = lighting_delta(_solid((255, 180, 80)), _solid((80, 180, 255)))
    assert delta == pytest.approx(0.5224, abs=1e-3)


def test_color_delta_of_identical_images_is_zero():
    assert color_delta(
        _solid((220, 30, 30)), _solid((220, 30, 30)), FULL
    ) == pytest.approx(0.0)


def test_color_delta_of_red_versus_blue_is_large():
    assert color_delta(_solid((220, 30, 30)), _solid((30, 30, 220)), FULL) > 0.3


def test_color_delta_reads_only_inside_the_mask():
    left_red = Image.new("RGB", (64, 64), (0, 0, 0))
    left_red.paste(Image.new("RGB", (32, 64), (220, 30, 30)), (0, 0))
    left_blue = Image.new("RGB", (64, 64), (0, 0, 0))
    left_blue.paste(Image.new("RGB", (32, 64), (30, 30, 220)), (0, 0))

    left = RectMasker((0.0, 0.0, 0.5, 1.0))(left_red, "left")
    right = RectMasker((0.5, 0.0, 1.0, 1.0))(left_red, "right")

    assert color_delta(left_red, left_blue, left) > 0.3
    assert color_delta(left_red, left_blue, right) == pytest.approx(0.0)


def test_color_delta_is_zero_when_the_mask_is_too_small():
    tiny = np.zeros((64, 64), dtype=np.float32)
    tiny[0, :3] = 1.0
    assert color_delta(_solid((220, 30, 30)), _solid((30, 30, 220)), tiny) == 0.0


def test_color_absolute_peaks_on_an_exact_match():
    assert color_absolute(_solid((220, 30, 30)), FULL, (220, 30, 30)) == pytest.approx(
        1.0
    )


def test_color_absolute_drops_for_a_mismatch():
    assert color_absolute(_solid((30, 30, 220)), FULL, (220, 30, 30)) < 0.7


def test_all_metrics_stay_within_the_unit_interval():
    pairs = [((255, 255, 255), (0, 0, 0)), ((220, 30, 30), (30, 220, 30))]
    for a, b in pairs:
        assert 0.0 <= color_delta(_solid(a), _solid(b), FULL) <= 1.0
        assert 0.0 <= lighting_delta(_solid(a), _solid(b)) <= 1.0
