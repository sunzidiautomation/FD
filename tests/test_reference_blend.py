import pytest

torch = pytest.importorskip("torch")

from flair_t2i.attributes import AttributeClass
from flair_t2i.components import Component
from flair_t2i.config import FlairConfig

from .reference_blend import ReferenceRouted, reference_blend

SEQ, DIM = 4, 8
CFG = FlairConfig(alpha_0=1.0, t_window=(0.0, 1.0), top_k_default=1)


def _routed(blocks=((7, 1.0),), intensity=1.0, fill=1.0):
    return [
        ReferenceRouted(
            component=Component(id="c_color", text="a red car", attr=AttributeClass.COLOR),
            embedding=torch.full((SEQ, DIM), fill),
            blocks=blocks,
            intensity=intensity,
        )
    ]


def _states(batch=2, fill=0.0):
    return torch.full((batch, SEQ, DIM), fill)


def test_identity_on_untouched_block():
    states = _states()
    assert reference_blend(_routed(), CFG, states, 99, 0.0, slice(1, 2)) is states


def test_moves_conditional_rows_toward_the_component():
    out = reference_blend(_routed(), CFG, _states(), 7, 0.0, slice(1, 2))
    assert out[1].mean().item() == pytest.approx(1.0)


def test_leaves_unconditional_rows_untouched():
    out = reference_blend(_routed(), CFG, _states(), 7, 0.0, slice(1, 2))
    assert out[0].abs().max().item() == pytest.approx(0.0)


def test_does_not_mutate_its_input():
    states = _states()
    reference_blend(_routed(), CFG, states, 7, 0.0, slice(1, 2))
    assert states.abs().max().item() == pytest.approx(0.0)


def test_scales_with_score_and_intensity():
    out = reference_blend(
        _routed(blocks=((7, 0.5),), intensity=0.4), CFG, _states(), 7, 0.0, slice(1, 2)
    )
    assert out[1].mean().item() == pytest.approx(0.2)


def test_honours_alpha_scale():
    out = reference_blend(
        _routed(), CFG, _states(), 7, 0.0, slice(1, 2), alpha_scale=0.5
    )
    assert out[1].mean().item() == pytest.approx(0.5)
