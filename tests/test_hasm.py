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
