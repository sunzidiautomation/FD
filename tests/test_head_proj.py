import pytest

torch = pytest.importorskip("torch")

from flair_t2i.attributes import AttributeClass
from flair_t2i.components import Component
from flair_t2i.config import FlairConfig
from flair_t2i.head_proj import HeadResidualProj
from flair_t2i.heads import HeadUnit
from flair_t2i.processor import PlanRef
from flair_t2i.routing import RoutedComponent, RoutingPlan

SEQ, DIM = 4, 8
N_HEADS, HEAD_DIM = 3, 4
INNER = N_HEADS * HEAD_DIM
CFG = FlairConfig(alpha_0=1.0, t_window=(0.0, 1.0))

COMPONENT = Component(id="c_color", text="a red car", attr=AttributeClass.COLOR)


def _linear():
    torch.manual_seed(0)
    linear = torch.nn.Linear(DIM, INNER)
    torch.nn.init.normal_(linear.bias)
    return linear


def _ref(units=((HeadUnit(7, 1), 1.0),), plan=True):
    routed = RoutedComponent(
        component=COMPONENT, embedding=torch.ones((SEQ, DIM)), units=units
    )
    return PlanRef(
        plan=RoutingPlan(routed=(routed,), cfg=CFG) if plan else None,
        step=0,
        total_steps=10,
        do_cfg=True,
    )


def test_passthrough_when_no_plan():
    linear = _linear()
    proj = HeadResidualProj(linear, 7, _ref(plan=False), N_HEADS, HEAD_DIM)
    x = torch.randn((2, SEQ, DIM))
    torch.testing.assert_close(proj(x), linear(x))


def test_passthrough_on_unrouted_block():
    linear = _linear()
    proj = HeadResidualProj(linear, 99, _ref(), N_HEADS, HEAD_DIM)
    x = torch.randn((2, SEQ, DIM))
    torch.testing.assert_close(proj(x), linear(x))


def test_unconditional_rows_are_never_written():
    linear = _linear()
    proj = HeadResidualProj(linear, 7, _ref(), N_HEADS, HEAD_DIM)
    x = torch.randn((2, SEQ, DIM))
    torch.testing.assert_close(proj(x)[0], linear(x)[0])


def test_only_the_selected_head_slice_changes():
    linear = _linear()
    proj = HeadResidualProj(linear, 7, _ref(), N_HEADS, HEAD_DIM)
    x = torch.randn((2, SEQ, DIM))

    delta = (proj(x) - linear(x))[1]
    assert delta[..., 0:4].abs().max().item() == pytest.approx(0.0)
    assert delta[..., 4:8].abs().max().item() > 0.0
    assert delta[..., 8:12].abs().max().item() == pytest.approx(0.0)


def test_does_not_mutate_the_input():
    linear = _linear()
    proj = HeadResidualProj(linear, 7, _ref(), N_HEADS, HEAD_DIM)
    x = torch.randn((2, SEQ, DIM))
    before = x.clone()
    proj(x)
    torch.testing.assert_close(x, before)


def test_matches_a_pre_projection_blend_for_all_heads():
    """The wrapper must honour condition 1: weight-only, no bias."""
    linear = _linear()
    units = tuple((HeadUnit(7, h), 1.0) for h in range(N_HEADS))
    proj = HeadResidualProj(linear, 7, _ref(units=units), N_HEADS, HEAD_DIM)

    x = torch.randn((2, SEQ, DIM))
    blended = x.clone()
    blended[1:2] = blended[1:2] + 1.0 * (torch.ones((SEQ, DIM)) - blended[1:2])

    torch.testing.assert_close(proj(x)[1], linear(blended)[1])
