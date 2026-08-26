import pytest
from PIL import Image

from flair_t2i.attributes import AttributeClass
from flair_t2i.calibration.corpus import ContrastivePair
from flair_t2i.calibration.harness import SwapSpec
from flair_t2i.demo.sweep import DemoPaths, run_demo_sweep
from flair_t2i.heads import HeadUnit
from flair_t2i.metrics.masking import RectMasker

BLOCK_IDS = (0, 1)
HEAD_IDS = (0, 1)
LIVE_UNIT = HeadUnit(block=1, head=1)
FULL_MASKER = RectMasker((0.0, 0.0, 1.0, 1.0))


def _corpus():
    return {
        AttributeClass.COLOR: [
            ContrastivePair("a red car on a road", "a blue car on a road", "car", None)
        ]
    }


def _generate(prompt: str, seed: int, swap: SwapSpec | None) -> Image.Image:
    if swap is not None and swap.unit == LIVE_UNIT and "blue" in swap.prompt:
        return Image.new("RGB", (16, 16), (30, 30, 220))
    return Image.new("RGB", (16, 16), (220, 30, 30))


def _sweep(tmp_path):
    return run_demo_sweep(
        generate_fn=_generate,
        corpus=_corpus(),
        block_ids=BLOCK_IDS,
        head_ids=HEAD_IDS,
        masker=FULL_MASKER,
        paths=DemoPaths(tmp_path),
        seeds=[0],
    )


def test_writes_one_image_per_head_unit(tmp_path):
    _sweep(tmp_path)
    written = sorted(p.name for p in (tmp_path / "heads").glob("*.png"))
    assert written == [
        "color_b0_h0.png",
        "color_b0_h1.png",
        "color_b1_h0.png",
        "color_b1_h1.png",
    ]


def test_writes_one_image_per_block(tmp_path):
    _sweep(tmp_path)
    written = sorted(p.name for p in (tmp_path / "blocks").glob("*.png"))
    assert written == ["color_b0.png", "color_b1.png"]


def test_writes_a_baseline_per_attribute(tmp_path):
    _sweep(tmp_path)
    assert (tmp_path / "baselines" / "color.png").exists()


def test_returns_a_hasm_scoring_the_live_unit_highest(tmp_path):
    hasm = _sweep(tmp_path)
    assert hasm.top_k(AttributeClass.COLOR, 1) == [(LIVE_UNIT, pytest.approx(1.0))]


def test_scores_are_saved_alongside_the_images(tmp_path):
    _sweep(tmp_path)
    assert (tmp_path / "hasm.npz").exists()


def test_paths_are_created_on_construction(tmp_path):
    paths = DemoPaths(tmp_path / "bundle")
    for directory in (paths.heads, paths.blocks, paths.latents, paths.baselines):
        assert directory.is_dir()


def test_head_image_path_is_stable(tmp_path):
    paths = DemoPaths(tmp_path)
    got = paths.head_image(AttributeClass.COLOR, HeadUnit(3, 7))
    assert got == tmp_path / "heads" / "color_b3_h7.png"
