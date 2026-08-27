import numpy as np
import pytest
from PIL import Image

from flair_t2i.metrics.masking import (
    MIN_MASK_PIXELS,
    ClipSegMasker,
    RectMasker,
    mask_area_ratio,
    masked_mean_rgb,
)


def _solid(rgb, size=(64, 64)):
    return Image.new("RGB", size, rgb)


class _RecordingModel:
    """A model that only remembers where it was asked to move."""

    def __init__(self):
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


def test_clipseg_masker_moves_its_model_to_the_device():
    """Inputs are sent to self.device, so the weights must go there too.

    Skipping this leaves the model on CPU while inputs land on CUDA, and
    the failure surfaces only after a 600MB download and a full generation:
    'Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)
    should be the same'.
    """
    model = _RecordingModel()
    ClipSegMasker(model=model, processor=object(), device="cuda")
    assert model.moved_to == "cuda"


def test_clipseg_masker_tolerates_a_model_without_to():
    """Test stubs are plain objects; construction must not require .to."""
    ClipSegMasker(model=object(), processor=object(), device="cuda")


def test_rect_masker_covers_the_requested_fraction():
    masker = RectMasker((0.0, 0.0, 0.5, 1.0))
    mask = masker(_solid((0, 0, 0)), "anything")
    assert mask.shape == (64, 64)
    assert mask_area_ratio(mask) == pytest.approx(0.5)


def test_rect_masker_values_are_binary():
    mask = RectMasker((0.25, 0.25, 0.75, 0.75))(_solid((0, 0, 0)), "x")
    assert set(np.unique(mask)) <= {0.0, 1.0}


def test_area_ratio_of_empty_and_full_masks():
    assert mask_area_ratio(np.zeros((8, 8))) == pytest.approx(0.0)
    assert mask_area_ratio(np.ones((8, 8))) == pytest.approx(1.0)


def test_masked_mean_rgb_reads_only_inside_the_mask():
    image = Image.new("RGB", (64, 64), (0, 0, 255))
    image.paste(Image.new("RGB", (32, 64), (255, 0, 0)), (0, 0))
    mask = RectMasker((0.0, 0.0, 0.5, 1.0))(image, "left half")

    mean = masked_mean_rgb(image, mask)

    np.testing.assert_allclose(mean, [255.0, 0.0, 0.0])


def test_masked_mean_rgb_returns_none_for_a_tiny_mask():
    mask = np.zeros((64, 64))
    mask[0, : MIN_MASK_PIXELS - 1] = 1.0
    assert masked_mean_rgb(_solid((10, 20, 30)), mask) is None


def test_masked_mean_rgb_accepts_a_full_mask():
    mean = masked_mean_rgb(_solid((10, 20, 30)), np.ones((64, 64)))
    np.testing.assert_allclose(mean, [10.0, 20.0, 30.0])
