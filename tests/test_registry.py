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


def test_color_with_a_target_uses_the_directed_metric():
    """Without a target colour, colour is measured as distance FROM the
    baseline -- which a tone shift maximises without recolouring anything."""
    mask = np.ones((32, 32), dtype=np.float32)
    red, blue = Image.new("RGB", (32, 32), (220, 30, 30)), Image.new(
        "RGB", (32, 32), (0, 0, 255)
    )
    warm = Image.new("RGB", (32, 32), (200, 90, 40))

    directed = delta_for(AttributeClass.COLOR, target="blue")
    assert directed(red, blue, mask) == pytest.approx(1.0)
    assert directed(red, warm, mask) == pytest.approx(0.0)

    undirected = delta_for(AttributeClass.COLOR)
    assert undirected(red, warm, mask) > 0.0, "the old metric rewards the wash"


def test_color_rejects_a_target_that_is_not_a_colour_name():
    with pytest.raises(ValueError, match="not a known colour"):
        delta_for(AttributeClass.COLOR, target="tractor")


# ------------------------------------- what a pair says the swap injected


def _pair(base, changed, label="sports car", phrase=None):
    from flair_t2i.calibration.corpus import ContrastivePair

    return ContrastivePair(
        base=base, changed=changed, object_label=label, phrase=phrase
    )


def test_target_for_colour_is_the_colour_word():
    from flair_t2i.metrics.registry import target_for

    pair = _pair("a red sports car on a road", "a blue sports car on a road")
    assert target_for(AttributeClass.COLOR, pair) == "blue"


def test_target_for_identity_is_the_whole_changed_phrase():
    from flair_t2i.metrics.registry import target_for

    pair = _pair("a sedan on a road", "a tractor on a road")
    assert target_for(AttributeClass.IDENTITY, pair) == "a tractor on a road"


def test_target_for_colour_is_none_when_the_word_is_not_a_colour():
    """Falls back to the undirected metric rather than raising mid-sweep."""
    from flair_t2i.metrics.registry import target_for

    pair = _pair("a small sports car", "a large sports car")
    assert target_for(AttributeClass.COLOR, pair) is None


def test_target_for_a_photometric_attribute_is_none():
    from flair_t2i.metrics.registry import target_for

    pair = _pair("a car in warm light", "a car in cool light")
    assert target_for(AttributeClass.LIGHTING, pair) is None
