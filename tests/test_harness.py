import json

import numpy as np
import pytest
from PIL import Image

from flair_t2i.attributes import AttributeClass
from flair_t2i.calibration.corpus import ContrastivePair
from flair_t2i.calibration.harness import SwapSpec, calibrate
from flair_t2i.metrics.masking import RectMasker

VITAL = (0, 1, 2)
SEEDS = [0]
#: The fake model routes the attribute through block 1 and nothing else.
LIVE_BLOCK = 1

FULL_MASKER = RectMasker((0.0, 0.0, 1.0, 1.0))


def _corpus(attr=AttributeClass.COLOR, phrase=None):
    pairs = [
        ContrastivePair("a red car on a road", "a blue car on a road", "car", phrase),
        ContrastivePair(
            "a red vase on a table", "a blue vase on a table", "vase", phrase
        ),
    ]
    return {attr: pairs}


def fake_generate(prompt: str, seed: int, swap: SwapSpec | None) -> Image.Image:
    """Only a swap at LIVE_BLOCK actually changes the image."""
    if swap is not None and swap.block_id == LIVE_BLOCK and "blue" in swap.prompt:
        return Image.new("RGB", (64, 64), (30, 30, 220))
    return Image.new("RGB", (64, 64), (220, 30, 30))


class FakeScorer:
    def image_embedding(self, image):
        mean = np.asarray(image.convert("RGB"), dtype=np.float64).mean(axis=(0, 1))
        norm = np.linalg.norm(mean)
        return mean / norm if norm else mean

    def image_text_similarity(self, image, texts):
        base = float(self.image_embedding(image)[0])
        return np.array([0.15 + 0.2 * base * (i + 1) for i in range(len(texts))])


def _calibrate(**kw):
    kw.setdefault("generate_fn", fake_generate)
    kw.setdefault("corpus", _corpus())
    kw.setdefault("vital_blocks", VITAL)
    kw.setdefault("masker", FULL_MASKER)
    kw.setdefault("seeds", SEEDS)
    return calibrate(**kw)


# --------------------------------------------------------------- core sweep


def test_calibrate_returns_a_basm_over_the_vital_blocks():
    basm = _calibrate()
    assert basm.block_ids == VITAL
    assert basm.attributes == (AttributeClass.COLOR,)


def test_the_sensitive_block_scores_highest():
    basm = _calibrate()
    assert basm.top_k(AttributeClass.COLOR, 1) == [(LIVE_BLOCK, pytest.approx(1.0))]


def test_insensitive_blocks_score_zero_after_normalisation():
    basm = _calibrate()
    assert basm.score(0, AttributeClass.COLOR) == pytest.approx(0.0)
    assert basm.score(2, AttributeClass.COLOR) == pytest.approx(0.0)


def test_all_scores_land_in_the_unit_interval():
    basm = _calibrate()
    assert basm.matrix.min() >= 0.0 and basm.matrix.max() <= 1.0


def test_a_flat_response_normalises_to_zero_rather_than_dividing_by_zero():
    flat = lambda prompt, seed, swap: Image.new("RGB", (64, 64), (128, 128, 128))
    basm = _calibrate(generate_fn=flat)
    assert basm.matrix.max() == pytest.approx(0.0)


def test_progress_callback_reports_every_attribute_block_pair():
    seen = []
    _calibrate(progress=lambda attr, block, value: seen.append((attr, block)))
    assert seen == [(AttributeClass.COLOR, b) for b in VITAL]


def test_action_attribute_binds_its_phrase():
    basm = _calibrate(
        corpus=_corpus(AttributeClass.ACTION, phrase="a car driving"),
        scorer=FakeScorer(),
    )
    assert basm.attributes == (AttributeClass.ACTION,)


class GrowMasker:
    """A bigger mask for brighter images, so size change is observable."""

    def __call__(self, image, label):
        bright = image.getpixel((0, 0))[2] > 150  # blue channel
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[: int(64 * (0.75 if bright else 0.25)), :] = 1.0
        return mask


def test_size_remasks_the_swapped_image():
    """Bound to one shared mask, size always reads 0 -- see registry.py."""
    basm = _calibrate(corpus=_corpus(AttributeClass.SIZE), masker=GrowMasker())
    assert basm.score(LIVE_BLOCK, AttributeClass.SIZE) == pytest.approx(1.0)
    assert basm.score(0, AttributeClass.SIZE) == pytest.approx(0.0)


# ------------------------------------------------------------ checkpointing


def _counting_generate(counter):
    def generate(prompt, seed, swap):
        counter["calls"] += 1
        return fake_generate(prompt, seed, swap)

    return generate


def test_checkpoint_writes_one_file_per_cell(tmp_path):
    _calibrate(checkpoint_dir=tmp_path)
    cells = sorted(p.name for p in (tmp_path / "cells").glob("*.json"))
    assert cells == ["color_0.json", "color_1.json", "color_2.json"]


def test_resuming_does_not_regenerate_completed_cells(tmp_path):
    first = {"calls": 0}
    _calibrate(generate_fn=_counting_generate(first), checkpoint_dir=tmp_path)
    assert first["calls"] > 0

    second = {"calls": 0}
    _calibrate(generate_fn=_counting_generate(second), checkpoint_dir=tmp_path)
    assert second["calls"] == 0


def test_a_resumed_run_produces_the_same_matrix(tmp_path):
    original = _calibrate(checkpoint_dir=tmp_path)
    resumed = _calibrate(checkpoint_dir=tmp_path)
    np.testing.assert_allclose(resumed.matrix, original.matrix)


def test_a_partial_checkpoint_only_recomputes_missing_cells(tmp_path):
    _calibrate(checkpoint_dir=tmp_path)
    (tmp_path / "cells" / f"color_{LIVE_BLOCK}.json").unlink()

    counter = {"calls": 0}
    resumed = _calibrate(
        generate_fn=_counting_generate(counter), checkpoint_dir=tmp_path
    )

    # two pairs x one seed x (baseline + swap) for the one missing cell
    assert counter["calls"] == 4
    assert resumed.top_k(AttributeClass.COLOR, 1) == [(LIVE_BLOCK, pytest.approx(1.0))]


def test_a_corrupt_cell_file_is_recomputed(tmp_path):
    _calibrate(checkpoint_dir=tmp_path)
    (tmp_path / "cells" / "color_0.json").write_text('{"raw": ')  # truncated

    counter = {"calls": 0}
    resumed = _calibrate(
        generate_fn=_counting_generate(counter), checkpoint_dir=tmp_path
    )

    assert counter["calls"] == 4
    assert resumed.score(0, AttributeClass.COLOR) == pytest.approx(0.0)


def test_checkpoint_records_the_raw_value_not_the_normalised_one(tmp_path):
    _calibrate(checkpoint_dir=tmp_path)
    cell = json.loads((tmp_path / "cells" / f"color_{LIVE_BLOCK}.json").read_text())

    assert cell["attribute"] == "color"
    assert cell["block"] == LIVE_BLOCK
    assert 0.0 < cell["raw"] <= 1.0
    assert cell["samples"] == 2
