import pytest

torch = pytest.importorskip("torch")

from flair_t2i.heads import HeadUnit, alpha_vector, head_slice

N_HEADS, HEAD_DIM = 3, 4
INNER = N_HEADS * HEAD_DIM


def test_head_slice_is_contiguous_and_non_overlapping():
    assert head_slice(0, HEAD_DIM) == slice(0, 4)
    assert head_slice(2, HEAD_DIM) == slice(8, 12)


def test_head_units_sort_by_block_then_head():
    units = [HeadUnit(7, 2), HeadUnit(3, 5), HeadUnit(3, 1)]
    assert sorted(units) == [HeadUnit(3, 1), HeadUnit(3, 5), HeadUnit(7, 2)]


def test_head_unit_is_frozen_and_hashable():
    unit = HeadUnit(block=3, head=1)
    assert {unit: "ok"}[HeadUnit(3, 1)] == "ok"
    with pytest.raises(Exception):
        unit.block = 9


def test_alpha_vector_fills_only_the_selected_head_slice():
    vector = alpha_vector({1: 0.75}, N_HEADS, HEAD_DIM)

    assert vector.shape == (INNER,)
    assert vector[head_slice(1, HEAD_DIM)].tolist() == [0.75] * HEAD_DIM
    assert vector[head_slice(0, HEAD_DIM)].abs().max().item() == 0.0
    assert vector[head_slice(2, HEAD_DIM)].abs().max().item() == 0.0


def test_alpha_vector_supports_different_alpha_per_head():
    vector = alpha_vector({0: 0.2, 2: 0.9}, N_HEADS, HEAD_DIM)

    assert vector[head_slice(0, HEAD_DIM)].tolist() == pytest.approx([0.2] * HEAD_DIM)
    assert vector[head_slice(1, HEAD_DIM)].abs().max().item() == 0.0
    assert vector[head_slice(2, HEAD_DIM)].tolist() == pytest.approx([0.9] * HEAD_DIM)


def test_alpha_vector_of_every_head_is_uniform():
    vector = alpha_vector({h: 0.5 for h in range(N_HEADS)}, N_HEADS, HEAD_DIM)
    assert vector.tolist() == [0.5] * INNER


def test_alpha_vector_rejects_out_of_range_head():
    with pytest.raises(ValueError, match="out of range"):
        alpha_vector({9: 1.0}, N_HEADS, HEAD_DIM)


def test_alpha_vector_honours_dtype():
    vector = alpha_vector({0: 1.0}, N_HEADS, HEAD_DIM, dtype=torch.float16)
    assert vector.dtype == torch.float16
