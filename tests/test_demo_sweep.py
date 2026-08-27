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


def _make_generate(calls: list | None = None):
    """A fake model in which only LIVE_UNIT carries the attribute."""

    def generate(prompt: str, seed: int, swap: SwapSpec | None) -> Image.Image:
        if swap is None:
            return Image.new("RGB", (16, 16), (220, 30, 30))
        if calls is not None:
            calls.append(swap.units)
        if LIVE_UNIT in swap.units and "blue" in swap.prompt:
            return Image.new("RGB", (16, 16), (30, 30, 220))
        return Image.new("RGB", (16, 16), (220, 30, 30))

    return generate


_generate = _make_generate()


def _sweep(tmp_path, calls: list | None = None):
    return run_demo_sweep(
        generate_fn=_make_generate(calls),
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


def test_block_image_routes_to_every_head_not_just_the_first(tmp_path):
    """The block image is the head image unless it selects the whole block.

    LIVE_UNIT is (block 1, head 1), so a block-1 swap that covers every head
    picks up the change while the block-1/head-0 swap does not. If these two
    files match, the block loop is routing to a single head and the demo's
    whole granularity comparison is two copies of the same picture.
    """
    _sweep(tmp_path)
    block = (tmp_path / "blocks" / "color_b1.png").read_bytes()
    head0 = (tmp_path / "heads" / "color_b1_h0.png").read_bytes()
    assert block != head0


def test_block_swap_passes_every_head_of_that_block(tmp_path):
    calls: list = []
    _sweep(tmp_path, calls)

    multi = [units for units in calls if len(units) > 1]
    assert multi, "no swap ever routed to more than one head"
    assert set(multi[-1]) == {HeadUnit(1, 0), HeadUnit(1, 1)}


def test_head_swap_passes_exactly_one_unit(tmp_path):
    calls: list = []
    _sweep(tmp_path, calls)

    per_head = calls[: len(BLOCK_IDS) * len(HEAD_IDS)]
    assert all(len(units) == 1 for units in per_head)


def test_head_image_path_is_stable(tmp_path):
    paths = DemoPaths(tmp_path)
    got = paths.head_image(AttributeClass.COLOR, HeadUnit(3, 7))
    assert got == tmp_path / "heads" / "color_b3_h7.png"
