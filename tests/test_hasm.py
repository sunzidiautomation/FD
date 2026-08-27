import numpy as np
import pytest

from flair_t2i.attributes import AttributeClass
from flair_t2i.hasm import HASM
from flair_t2i.heads import HeadUnit

BLOCKS = (3, 7)
HEADS = (0, 1)
ATTRS = (AttributeClass.COLOR, AttributeClass.SIZE)


def _hasm():
    # [block, head, attribute]
    tensor = np.array(
        [
            [[0.20, 0.80], [0.10, 0.40]],  # block 3: head 0, head 1
            [[0.90, 0.30], [0.50, 0.60]],  # block 7: head 0, head 1
        ]
    )
    return HASM(tensor=tensor, block_ids=BLOCKS, head_ids=HEADS, attributes=ATTRS)


def test_score_lookup_by_unit_and_attribute():
    assert _hasm().score(HeadUnit(7, 0), AttributeClass.COLOR) == pytest.approx(0.90)
    assert _hasm().score(HeadUnit(3, 1), AttributeClass.SIZE) == pytest.approx(0.40)


def test_top_k_is_descending_by_score():
    assert _hasm().top_k(AttributeClass.COLOR, 2) == [
        (HeadUnit(7, 0), 0.90),
        (HeadUnit(7, 1), 0.50),
    ]


def test_top_k_clamps_to_available_units():
    assert len(_hasm().top_k(AttributeClass.COLOR, 99)) == 4


def test_ties_break_by_ascending_block_then_head():
    tensor = np.full((2, 2, 1), 0.5)
    hasm = HASM(tensor, (9, 4), (1, 0), (AttributeClass.COLOR,))
    units = [unit for unit, _ in hasm.top_k(AttributeClass.COLOR, 4)]
    assert units == [HeadUnit(4, 0), HeadUnit(4, 1), HeadUnit(9, 0), HeadUnit(9, 1)]


def test_to_basm_max_reduces_over_heads():
    basm = _hasm().to_basm(reduce="max")
    assert basm.block_ids == BLOCKS
    assert basm.attributes == ATTRS
    assert basm.score(3, AttributeClass.COLOR) == pytest.approx(0.20)
    assert basm.score(7, AttributeClass.COLOR) == pytest.approx(0.90)
    assert basm.score(3, AttributeClass.SIZE) == pytest.approx(0.80)


def test_to_basm_mean_reduces_over_heads():
    basm = _hasm().to_basm(reduce="mean")
    assert basm.score(3, AttributeClass.COLOR) == pytest.approx(0.15)
    assert basm.score(7, AttributeClass.SIZE) == pytest.approx(0.45)


def test_to_basm_rejects_unknown_reduction():
    with pytest.raises(ValueError, match="unknown reduction"):
        _hasm().to_basm(reduce="median")


def test_rejects_out_of_range_scores():
    with pytest.raises(ValueError, match="within"):
        HASM(np.full((1, 1, 1), 1.4), (3,), (0,), (AttributeClass.COLOR,))


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        HASM(np.zeros((2, 2, 2)), (3,), (0,), ATTRS)


def test_unknown_attribute_raises():
    with pytest.raises(KeyError, match="lighting"):
        _hasm().score(HeadUnit(3, 0), AttributeClass.LIGHTING)


def test_unknown_head_raises():
    with pytest.raises(KeyError, match="head 9"):
        _hasm().score(HeadUnit(3, 9), AttributeClass.COLOR)


def test_excluding_zeroes_the_named_units():
    hasm = _hasm().excluding({HeadUnit(7, 0)})
    assert hasm.score(HeadUnit(7, 0), AttributeClass.COLOR) == pytest.approx(0.0)


def test_excluding_promotes_the_next_best_survivor():
    """Dropping the peak must lift the runner-up to 1.0, not leave a hole."""
    hasm = _hasm().excluding({HeadUnit(7, 0)})
    top, score = hasm.top_k(AttributeClass.COLOR, 1)[0]
    assert top == HeadUnit(7, 1)  # was 0.50, the best remaining COLOR cell
    assert score == pytest.approx(1.0)


def test_excluding_matches_normalising_the_raw_values_directly():
    """Re-normalising already-normalised scores is exact, not an approximation.

    Min-max is affine, so rescaling the normalised plane over a subset gives
    the same answer as rescaling the original raw plane over that subset.
    That is what lets a contaminated matrix be repaired without recomputing
    a metric whose model (CLIP, ClipSeg) may not even be installed.
    """
    raw = np.array([[[0.20], [0.80]], [[5.00], [0.50]]])  # 5.00 is corrupt
    blocks, heads, attrs = (0, 1), (0, 1), (AttributeClass.COLOR,)

    full = raw[:, :, 0]
    normalised = (full - full.min()) / (full.max() - full.min())
    from_normalised = HASM(
        normalised[:, :, None], blocks, heads, attrs
    ).excluding({HeadUnit(1, 0)})

    survivors = np.array([0.20, 0.80, 0.50])
    lo, hi = survivors.min(), survivors.max()
    expected = {
        HeadUnit(0, 0): (0.20 - lo) / (hi - lo),
        HeadUnit(0, 1): (0.80 - lo) / (hi - lo),
        HeadUnit(1, 1): (0.50 - lo) / (hi - lo),
    }
    for unit, want in expected.items():
        assert from_normalised.score(unit, AttributeClass.COLOR) == pytest.approx(want)


def test_excluding_everything_gives_a_flat_zero_plane():
    every = {HeadUnit(b, h) for b in BLOCKS for h in HEADS}
    hasm = _hasm().excluding(every)
    assert hasm.tensor.max() == pytest.approx(0.0)


def test_excluding_nothing_is_a_no_op_on_an_already_normalised_matrix():
    """The repair path always re-normalises over the survivors. With nothing
    excluded and a matrix that is already on [0, 1], that must change nothing
    -- otherwise repairing a clean bundle would silently rescale it."""
    tensor = np.array([[[0.00], [0.40]], [[1.00], [0.25]]])
    hasm = HASM(tensor, (0, 1), (0, 1), (AttributeClass.COLOR,))
    np.testing.assert_allclose(hasm.excluding(set()).tensor, hasm.tensor)


def test_save_load_round_trip(tmp_path):
    original = _hasm()
    path = tmp_path / "hasm.npz"
    original.save(path)
    restored = HASM.load(path)

    assert restored.block_ids == original.block_ids
    assert restored.head_ids == original.head_ids
    assert restored.attributes == original.attributes
    np.testing.assert_allclose(restored.tensor, original.tensor)


def test_uniform_factory_is_all_half():
    hasm = HASM.uniform((1, 2), (0, 1), ATTRS)
    assert hasm.score(HeadUnit(1, 0), AttributeClass.COLOR) == 0.5


def test_merge_combines_single_attribute_hasms():
    h_color = HASM(
        np.array([[[0.9], [0.1]], [[0.2], [0.3]]]),
        BLOCKS,
        HEADS,
        (AttributeClass.COLOR,),
    )
    h_size = HASM(
        np.array([[[0.4], [0.8]], [[0.6], [0.7]]]),
        BLOCKS,
        HEADS,
        (AttributeClass.SIZE,),
    )

    merged = HASM.merge([h_color, h_size])

    assert merged.block_ids == BLOCKS
    assert merged.head_ids == HEADS
    assert merged.attributes == (AttributeClass.COLOR, AttributeClass.SIZE)
    assert merged.score(HeadUnit(3, 0), AttributeClass.COLOR) == pytest.approx(0.9)
    assert merged.score(HeadUnit(3, 1), AttributeClass.SIZE) == pytest.approx(0.8)


def test_merge_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        HASM.merge([])


def test_merge_rejects_block_head_mismatch():
    h1 = HASM.uniform((1, 2), (0, 1), (AttributeClass.COLOR,))
    h2 = HASM.uniform((1, 3), (0, 1), (AttributeClass.SIZE,))
    with pytest.raises(ValueError, match="identical"):
        HASM.merge([h1, h2])

