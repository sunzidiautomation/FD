"""The offline rescoring path, which every demo bundle goes through.

``rescore`` recomputes a bundle's scores from its saved images. Two things
it must get right, both of which ``harness._measure_cell`` already gets
right and which a second implementation is free to drift from:

* SIZE is only visible by re-segmenting the candidate.
* The integrity gate must judge the region its attribute's metric reads.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from flair_t2i.attributes import AttributeClass
from flair_t2i.demo.sweep import DemoPaths
from flair_t2i.heads import HeadUnit

_SPEC = importlib.util.spec_from_file_location(
    "rescore", Path(__file__).resolve().parent.parent / "scripts" / "rescore.py"
)
rescore_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rescore_mod)

SIDE = 64
BOX = (20, 20, 44, 44)
GROWN = (10, 10, 54, 54)

BASE_BG = (30, 30, 30)
GROWN_BG = (60, 60, 60)


def _image(background, box=BOX):
    image = Image.new("RGB", (SIDE, SIDE), background)
    image.paste(Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (250, 250, 250)), box[:2])
    return image


def _mask(box):
    mask = np.zeros((SIDE, SIDE), dtype=np.float32)
    mask[box[1] : box[3], box[0] : box[2]] = 1.0
    return mask


def _masker(image, label):
    """Segments the grown object only for the image marked with GROWN_BG."""
    return _mask(GROWN if image.getpixel((0, 0)) == GROWN_BG else BOX)


class RecordingGate:
    """Passes everything, and remembers the region it was asked to judge."""

    def __init__(self):
        self.seen = []

    def check(self, baseline, candidate):
        self.seen.append((baseline.size, candidate.size))
        return type("Verdict", (), {"ok": True, "reason": ""})()


class FakeScorer:
    def image_embedding(self, image):
        return np.asarray(image.convert("RGB"), dtype=np.float64).mean(axis=(0, 1))

    def image_text_similarity(self, image, texts):
        mean = float(np.asarray(image.convert("RGB"), dtype=np.float64).mean())
        return np.array([0.2 + mean / 10_000.0] * len(texts))


def _bundle(tmp_path, attr, candidates):
    """candidates: {unit: background colour}"""
    paths = DemoPaths(tmp_path / "bundle")
    _image(BASE_BG).save(paths.baseline_image(attr))
    for unit, background in candidates.items():
        _image(background).save(paths.head_image(attr, unit))
    return paths


# ----------------------------------------------------------------- size


def test_size_re_segments_the_candidate(tmp_path):
    """Against one shared mask ``size_delta`` is identically zero.

    The harness re-segments the swapped image; rescoring reads the same
    metric from saved images and must do the same. Otherwise every cell
    reads 0, min-max normalises the plane to zeros, and the report names an
    arbitrary tie-broken unit as the peak -- silently.
    """
    grew, unchanged = HeadUnit(0, 0), HeadUnit(0, 1)
    paths = _bundle(
        tmp_path,
        AttributeClass.SIZE,
        {grew: GROWN_BG, unchanged: BASE_BG},
    )

    raw, rejected, _ = rescore_mod.rescore(
        paths, AttributeClass.SIZE, RecordingGate(), masker=_masker
    )

    assert raw[grew] > 0.0, "a grown object must register as a size change"
    assert raw[unchanged] == pytest.approx(0.0)
    assert not rejected


def test_size_needs_a_masker(tmp_path):
    paths = _bundle(tmp_path, AttributeClass.SIZE, {HeadUnit(0, 0): BASE_BG})
    with pytest.raises(SystemExit, match="re-segment"):
        rescore_mod.rescore(paths, AttributeClass.SIZE, RecordingGate(), masker=None)


# ------------------------------------------------------- the gate's region


def _region(tmp_path, attr):
    unit = HeadUnit(0, 0)
    paths = _bundle(tmp_path, attr, {unit: BASE_BG})
    gate = RecordingGate()
    rescore_mod.rescore(paths, attr, gate, scorer=FakeScorer(), masker=_masker)
    return gate.seen[0]


def test_object_level_attributes_are_gated_on_the_object_crop(tmp_path):
    """COLOR reads the masked mean, so the gate must judge the same pixels."""
    assert _region(tmp_path, AttributeClass.COLOR) == ((24, 24), (24, 24))


def test_scene_level_attributes_are_gated_on_the_whole_frame(tmp_path):
    """LIGHTING ignores its mask and reads warm/cool over the whole frame.

    Gating the object crop instead would let a frame whose background
    collapsed -- which is most of what lighting measures -- pass the gate
    and then max out the metric. This is the mirror image of the bug that
    let a confetti-covered car win identity.
    """
    assert _region(tmp_path, AttributeClass.LIGHTING) == ((SIDE, SIDE), (SIDE, SIDE))


def test_size_is_gated_on_the_whole_frame(tmp_path):
    """Size has no fixed region: its object legitimately changes extent.

    Cropping both images to the baseline's box and demanding structural
    similarity would reject exactly the successful swaps.
    """
    assert _region(tmp_path, AttributeClass.SIZE) == ((SIDE, SIDE), (SIDE, SIDE))
