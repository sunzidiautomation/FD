import numpy as np
import pytest
from PIL import Image

from flair_t2i.metrics.embedding import (
    ClipScorer,
    action_delta,
    clip_norm,
    identity_delta,
    style_delta,
)

FULL = np.ones((64, 64), dtype=np.float32)


class _RecordingModel:
    """A model that only remembers where it was asked to move."""

    def __init__(self):
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


def test_clip_scorer_moves_its_model_to_the_device():
    """Same failure mode as ClipSegMasker: inputs on CUDA, weights on CPU."""
    model = _RecordingModel()
    ClipScorer(model=model, processor=object(), device="cuda")
    assert model.moved_to == "cuda"


def test_clip_scorer_tolerates_a_model_without_to():
    ClipScorer(model=object(), processor=object(), device="cuda")


def _solid(rgb):
    return Image.new("RGB", (64, 64), rgb)


class FakeScorer:
    """Maps an image to a unit vector derived from its mean colour."""

    def image_embedding(self, image):
        mean = np.asarray(image.convert("RGB"), dtype=np.float64).mean(axis=(0, 1))
        norm = np.linalg.norm(mean)
        return mean / norm if norm else mean

    def image_text_similarity(self, image, texts):
        base = float(self.image_embedding(image)[0])
        return np.array([0.15 + 0.2 * base * (i + 1) for i in range(len(texts))])


def test_clip_norm_maps_the_documented_range_onto_the_unit_interval():
    assert clip_norm(0.15) == pytest.approx(0.0)
    assert clip_norm(0.35) == pytest.approx(1.0)
    assert clip_norm(0.25) == pytest.approx(0.5)


def test_clip_norm_clamps_outside_the_range():
    assert clip_norm(-1.0) == 0.0
    assert clip_norm(2.0) == 1.0


def test_identity_delta_of_identical_images_is_zero():
    image = _solid((200, 100, 50))
    assert identity_delta(image, image, FULL, FakeScorer()) == pytest.approx(0.0)


def test_identity_delta_grows_for_different_images():
    delta = identity_delta(
        _solid((250, 10, 10)), _solid((10, 10, 250)), FULL, FakeScorer()
    )
    assert delta > 0.05


def test_identity_delta_stays_within_the_unit_interval():
    delta = identity_delta(_solid((255, 0, 0)), _solid((0, 0, 255)), FULL, FakeScorer())
    assert 0.0 <= delta <= 1.0


def test_style_delta_of_identical_images_is_zero():
    image = _solid((120, 120, 120))
    assert style_delta(image, image, FULL, FakeScorer()) == pytest.approx(0.0)


def test_action_delta_of_identical_images_is_zero():
    image = _solid((120, 120, 120))
    assert action_delta(
        image, image, FULL, FakeScorer(), "a car driving"
    ) == pytest.approx(0.0)


def test_action_delta_responds_to_a_changed_image():
    delta = action_delta(
        _solid((255, 200, 200)),
        _solid((10, 10, 10)),
        FULL,
        FakeScorer(),
        "a car driving",
    )
    assert delta > 0.0


# --------------------------------------------------- identity uses its mask


def _object_on(background, size=64, box=(20, 20, 44, 44)):
    """A fixed bright object drawn on a variable background."""
    image = Image.new("RGB", (size, size), background)
    image.paste(Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (250, 250, 250)), box[:2])
    return image


def _object_mask(size=64, box=(20, 20, 44, 44)):
    mask = np.zeros((size, size), dtype=np.float32)
    mask[box[1] : box[3], box[0] : box[2]] = 1.0
    return mask


def test_identity_ignores_a_background_only_change():
    """Identity is an object-level attribute.

    Its metric is handed the object mask precisely so a repainted sky or a
    recomposed background cannot be scored as the object becoming a
    different thing. Without this the metric degenerates into 'how
    different is this image', which every collapsed or merely degraded
    frame maximises.
    """
    mask = _object_mask()
    same_object = _object_on((30, 30, 30))
    moved_background = _object_on((200, 40, 40))

    assert identity_delta(
        same_object, moved_background, mask, FakeScorer()
    ) == pytest.approx(0.0, abs=1e-6)


def test_identity_still_sees_a_change_inside_the_mask():
    mask = _object_mask()
    base = _object_on((30, 30, 30))
    swapped = _object_on((30, 30, 30))
    swapped.paste(Image.new("RGB", (24, 24), (10, 10, 240)), (20, 20))

    assert identity_delta(base, swapped, mask, FakeScorer()) > 0.05


def test_identity_falls_back_to_the_whole_image_without_a_mask():
    base = _object_on((30, 30, 30))
    other = _object_on((200, 40, 40))
    assert identity_delta(base, other, None, FakeScorer()) > 0.0


def test_identity_tolerates_an_empty_mask():
    empty = np.zeros((64, 64), dtype=np.float32)
    base = _object_on((30, 30, 30))
    assert identity_delta(base, base, empty, FakeScorer()) == pytest.approx(0.0)
