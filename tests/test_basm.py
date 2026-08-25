import numpy as np
import pytest

from flair_t2i.attributes import AttributeClass
from flair_t2i.basm import BASM

BLOCKS = (3, 7, 11)
ATTRS = (AttributeClass.COLOR, AttributeClass.SIZE)


def _basm():
    matrix = np.array(
        [
            [0.22, 0.81],  # block 3
            [0.93, 0.14],  # block 7
            [0.19, 0.16],  # block 11
        ]
    )
    return BASM(matrix=matrix, block_ids=BLOCKS, attributes=ATTRS)


def test_score_lookup_by_block_and_attribute():
    assert _basm().score(7, AttributeClass.COLOR) == pytest.approx(0.93)
    assert _basm().score(3, AttributeClass.SIZE) == pytest.approx(0.81)


def test_top_k_is_descending_by_score():
    assert _basm().top_k(AttributeClass.COLOR, 2) == [(7, 0.93), (3, 0.22)]
    assert _basm().top_k(AttributeClass.SIZE, 1) == [(3, 0.81)]


def test_top_k_clamps_to_available_blocks():
    assert len(_basm().top_k(AttributeClass.COLOR, 99)) == 3


def test_ties_break_by_ascending_block_id():
    matrix = np.array([[0.5], [0.5]])
    basm = BASM(matrix=matrix, block_ids=(9, 4), attributes=(AttributeClass.COLOR,))
    assert basm.top_k(AttributeClass.COLOR, 2) == [(4, 0.5), (9, 0.5)]


def test_rejects_out_of_range_scores():
    with pytest.raises(ValueError, match="within"):
        BASM(
            matrix=np.array([[1.4]]),
            block_ids=(3,),
            attributes=(AttributeClass.COLOR,),
        )


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        BASM(matrix=np.zeros((2, 2)), block_ids=(3,), attributes=ATTRS)


def test_unknown_attribute_raises():
    with pytest.raises(KeyError, match="lighting"):
        _basm().score(3, AttributeClass.LIGHTING)


def test_save_load_round_trip(tmp_path):
    original = _basm()
    path = tmp_path / "basm.npz"
    original.save(path)
    restored = BASM.load(path)

    assert restored.block_ids == original.block_ids
    assert restored.attributes == original.attributes
    np.testing.assert_allclose(restored.matrix, original.matrix)


def test_uniform_factory_is_all_half():
    basm = BASM.uniform((1, 2), ATTRS)
    assert basm.score(1, AttributeClass.COLOR) == 0.5
