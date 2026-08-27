import numpy as np
import pytest
from PIL import Image

from flair_t2i.attributes import AttributeClass
from flair_t2i.metrics.registry import DELTA_METRICS, delta_for

FULL = np.ones((64, 64), dtype=np.float32)


class FakeScorer:
    def image_embedding(self, image):
        mean = np.asarray(image.convert("RGB"), dtype=np.float64).mean(axis=(0, 1))
        norm = np.linalg.norm(mean)
        return mean / norm if norm else mean

    def image_text_similarity(self, image, texts):
        base = float(self.image_embedding(image)[0])
        return np.array([0.15 + 0.2 * base * (i + 1) for i in range(len(texts))])


def _solid(rgb):
    return Image.new("RGB", (64, 64), rgb)


def test_every_attribute_has_a_delta_metric():
    assert set(DELTA_METRICS) == set(AttributeClass)


def test_photometric_metrics_need_no_scorer():
    metric = delta_for(AttributeClass.COLOR)
    assert metric(_solid((220, 30, 30)), _solid((30, 30, 220)), FULL) > 0.3


def test_embedding_metrics_require_a_scorer():
    with pytest.raises(ValueError, match="scorer"):
        delta_for(AttributeClass.IDENTITY)


def test_action_metric_requires_a_phrase():
    with pytest.raises(ValueError, match="phrase"):
        delta_for(AttributeClass.ACTION, scorer=FakeScorer())


def test_every_metric_is_callable_with_the_common_signature():
    for attr in AttributeClass:
        metric = delta_for(attr, scorer=FakeScorer(), phrase="a car driving")
        value = metric(_solid((200, 100, 50)), _solid((50, 100, 200)), FULL)
        assert 0.0 <= value <= 1.0, attr


def test_size_metric_uses_the_shared_mask_for_both_images():
    metric = delta_for(AttributeClass.SIZE)
    assert metric(_solid((0, 0, 0)), _solid((0, 0, 0)), FULL) == pytest.approx(0.0)


# ------------------------------------------------- damage-proof target form


class _TargetScorer:
    """Similarity to any phrase falls with mean brightness -- a stand-in for
    a frame that resembles less and less as it degrades."""

    def image_text_similarity(self, image, texts):
        import numpy as np

        mean = np.asarray(image.convert("RGB"), dtype=np.float64).mean()
        return np.array([mean / 255.0 * 0.3] * len(texts))

    def image_embedding(self, image):
        import numpy as np

        return np.asarray(image.convert("RGB"), dtype=np.float64).mean(axis=(0, 1))


def test_action_accepts_a_target_instead_of_a_phrase():
    """action_delta compares against the BASE action, which the real swap and
    the damage both reduce -- signal and artefact are the same direction. A
    target lets it ask 'more like parked?' instead."""
    metric = delta_for(
        AttributeClass.ACTION,
        scorer=_TargetScorer(),
        target="a sports car parked along a road",
    )
    assert callable(metric)


def test_action_still_requires_one_of_phrase_or_target():
    with pytest.raises(ValueError, match="phrase or a target"):
        delta_for(AttributeClass.ACTION, scorer=_TargetScorer())


def test_identity_target_overrides_the_distance_form():
    from PIL import Image

    scorer = _TargetScorer()
    metric = delta_for(
        AttributeClass.IDENTITY, scorer=scorer, target="a tractor parked on a road"
    )
    dark = Image.new("RGB", (32, 32), (10, 10, 10))  # degraded: resembles less
    assert metric(Image.new("RGB", (32, 32), (200, 200, 200)), dark, None) == 0.0
