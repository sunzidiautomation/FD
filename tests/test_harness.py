import json

import numpy as np
import pytest
from PIL import Image

from flair_t2i.attributes import AttributeClass
from flair_t2i.calibration.corpus import ContrastivePair
from flair_t2i.calibration.harness import SwapSpec, calibrate
from flair_t2i.heads import HeadUnit
from flair_t2i.metrics.masking import RectMasker

BLOCK_IDS = (0, 1, 2)
HEAD_IDS = (0, 1)
SEEDS = [0]
#: The fake model routes the attribute through exactly one head.
LIVE_UNIT = HeadUnit(block=1, head=1)

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
    """Only a swap at LIVE_UNIT actually changes the image."""
    if swap is not None and swap.units == (LIVE_UNIT,) and "blue" in swap.prompt:
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
    kw.setdefault("block_ids", BLOCK_IDS)
    kw.setdefault("head_ids", HEAD_IDS)
    kw.setdefault("masker", FULL_MASKER)
    kw.setdefault("seeds", SEEDS)
    return calibrate(**kw)


# --------------------------------------------------------------- core sweep


def test_calibrate_returns_a_basm_over_the_vital_blocks():
    hasm = _calibrate()
    assert hasm.block_ids == BLOCK_IDS
    assert hasm.head_ids == HEAD_IDS
    assert hasm.attributes == (AttributeClass.COLOR,)


def test_the_sensitive_block_scores_highest():
    hasm = _calibrate()
    assert hasm.top_k(AttributeClass.COLOR, 1) == [(LIVE_UNIT, pytest.approx(1.0))]


def test_insensitive_blocks_score_zero_after_normalisation():
    hasm = _calibrate()
    assert hasm.score(HeadUnit(0, 0), AttributeClass.COLOR) == pytest.approx(0.0)
    assert hasm.score(HeadUnit(2, 0), AttributeClass.COLOR) == pytest.approx(0.0)


def test_all_scores_land_in_the_unit_interval():
    hasm = _calibrate()
    assert hasm.tensor.min() >= 0.0 and hasm.tensor.max() <= 1.0


def test_a_flat_response_normalises_to_zero_rather_than_dividing_by_zero():
    def flat(prompt, seed, swap):
        return Image.new("RGB", (64, 64), (128, 128, 128))

    hasm = _calibrate(generate_fn=flat)
    assert hasm.tensor.max() == pytest.approx(0.0)


def test_progress_callback_reports_every_attribute_block_pair():
    seen = []
    _calibrate(progress=lambda attr, unit, value: seen.append((attr, unit)))
    assert seen == [
        (AttributeClass.COLOR, HeadUnit(b, h)) for b in BLOCK_IDS for h in HEAD_IDS
    ]


def test_action_attribute_binds_its_phrase():
    hasm = _calibrate(
        corpus=_corpus(AttributeClass.ACTION, phrase="a car driving"),
        scorer=FakeScorer(),
    )
    assert hasm.attributes == (AttributeClass.ACTION,)


class GrowMasker:
    """A bigger mask for brighter images, so size change is observable."""

    def __call__(self, image, label):
        bright = image.getpixel((0, 0))[2] > 150  # blue channel
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[: int(64 * (0.75 if bright else 0.25)), :] = 1.0
        return mask


def test_size_remasks_the_swapped_image():
    """Bound to one shared mask, size always reads 0 -- see registry.py."""
    hasm = _calibrate(corpus=_corpus(AttributeClass.SIZE), masker=GrowMasker())
    assert hasm.score(LIVE_UNIT, AttributeClass.SIZE) == pytest.approx(1.0)
    assert hasm.score(HeadUnit(0, 0), AttributeClass.SIZE) == pytest.approx(0.0)


# ------------------------------------------------------------ checkpointing


def _counting_generate(counter):
    def generate(prompt, seed, swap):
        counter["calls"] += 1
        return fake_generate(prompt, seed, swap)

    return generate


def test_checkpoint_writes_one_file_per_cell(tmp_path):
    _calibrate(checkpoint_dir=tmp_path)
    cells = sorted(p.name for p in (tmp_path / "cells").glob("*.json"))
    expected = sorted(f"color_{b}_{h}.json" for b in BLOCK_IDS for h in HEAD_IDS)
    assert cells == expected


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
    np.testing.assert_allclose(resumed.tensor, original.tensor)


def test_a_partial_checkpoint_only_recomputes_missing_cells(tmp_path):
    _calibrate(checkpoint_dir=tmp_path)
    (tmp_path / "cells" / f"color_{LIVE_UNIT.block}_{LIVE_UNIT.head}.json").unlink()

    counter = {"calls": 0}
    resumed = _calibrate(
        generate_fn=_counting_generate(counter), checkpoint_dir=tmp_path
    )

    # two pairs x one seed x (baseline + swap) for the one missing cell
    assert counter["calls"] == 4
    assert resumed.top_k(AttributeClass.COLOR, 1) == [(LIVE_UNIT, pytest.approx(1.0))]


def test_a_corrupt_cell_file_is_recomputed(tmp_path):
    _calibrate(checkpoint_dir=tmp_path)
    (tmp_path / "cells" / "color_0_0.json").write_text('{"raw": ')  # truncated

    counter = {"calls": 0}
    resumed = _calibrate(
        generate_fn=_counting_generate(counter), checkpoint_dir=tmp_path
    )

    assert counter["calls"] == 4
    assert resumed.score(HeadUnit(0, 0), AttributeClass.COLOR) == pytest.approx(0.0)


def test_checkpoint_records_the_raw_value_not_the_normalised_one(tmp_path):
    _calibrate(checkpoint_dir=tmp_path)
    cell = json.loads(
        (tmp_path / "cells" / f"color_{LIVE_UNIT.block}_{LIVE_UNIT.head}.json").read_text()
    )

    assert cell["attribute"] == "color"
    assert cell["block"] == LIVE_UNIT.block
    assert cell["head"] == LIVE_UNIT.head
    assert 0.0 < cell["raw"] <= 1.0
    assert cell["samples"] == 2
