import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flair_t2i.attributes import AttributeClass
from flair_t2i.components import Component
from flair_t2i.config import FlairConfig
from flair_t2i.hasm import HASM
from flair_t2i.heads import HeadUnit
from flair_t2i.routing import RoutedComponent, RoutingPlan, build_routing_plan

from .reference_blend import ReferenceRouted, reference_blend

SEQ, DIM = 4, 8
N_HEADS, HEAD_DIM = 3, 4
INNER = N_HEADS * HEAD_DIM
CFG = FlairConfig(alpha_0=1.0, t_window=(0.0, 1.0), top_k_default=1)

COMPONENT = Component(id="c_color", text="a red car", attr=AttributeClass.COLOR)


def _plan(units, intensity=1.0, fill=1.0):
    return RoutingPlan(
        routed=(
            RoutedComponent(
                component=COMPONENT,
                embedding=torch.full((SEQ, DIM), fill),
                units=units,
                intensity=intensity,
            ),
        ),
        cfg=CFG,
    )


def _residual(plan, block_id, x, weight, step_frac=0.0):
    return plan.head_residual(
        block_id=block_id,
        x=x,
        weight=weight,
        step_frac=step_frac,
        cond_slice=slice(1, 2),
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
    )


def test_untouched_block_returns_none():
    plan = _plan(((HeadUnit(7, 0), 1.0),))
    assert _residual(plan, 99, torch.zeros((2, SEQ, DIM)), torch.zeros((INNER, DIM))) is None


def test_inactive_plan_returns_none():
    plan = _plan(((HeadUnit(7, 0), 1.0),))
    plan.active = False
    assert _residual(plan, 7, torch.zeros((2, SEQ, DIM)), torch.zeros((INNER, DIM))) is None


def test_residual_touches_only_the_selected_head_slice():
    plan = _plan(((HeadUnit(7, 1), 1.0),))
    residual = _residual(plan, 7, torch.zeros((2, SEQ, DIM)), torch.ones((INNER, DIM)))

    assert residual.shape == (1, SEQ, INNER)
    assert residual[..., 0:4].abs().max().item() == pytest.approx(0.0)
    assert residual[..., 4:8].abs().max().item() > 0.0
    assert residual[..., 8:12].abs().max().item() == pytest.approx(0.0)


def test_two_heads_in_one_block_scale_independently():
    plan = _plan(((HeadUnit(7, 0), 0.25), (HeadUnit(7, 2), 1.0)))
    residual = _residual(plan, 7, torch.zeros((2, SEQ, DIM)), torch.ones((INNER, DIM)))

    head0 = residual[..., 0:4].abs().max().item()
    head2 = residual[..., 8:12].abs().max().item()
    assert head2 == pytest.approx(head0 * 4.0)


def test_alpha_scales_with_score_and_intensity():
    plan = _plan(((HeadUnit(7, 0), 0.5),), intensity=0.4)
    assert plan.alpha(plan.routed[0], HeadUnit(7, 0), 0.0) == pytest.approx(0.2)


def test_alpha_respects_guard_backoff():
    plan = _plan(((HeadUnit(7, 0), 1.0),))
    plan.alpha_scale = 0.5
    assert plan.alpha(plan.routed[0], HeadUnit(7, 0), 0.0) == pytest.approx(0.5)


def test_alpha_is_zero_outside_timestep_window():
    plan = _plan(((HeadUnit(7, 0), 1.0),))
    plan.cfg = FlairConfig(alpha_0=1.0, t_window=(0.0, 0.5))
    assert plan.alpha(plan.routed[0], HeadUnit(7, 0), 0.9) == 0.0


def test_alpha_is_zero_for_an_unrouted_unit():
    plan = _plan(((HeadUnit(7, 0), 1.0),))
    assert plan.alpha(plan.routed[0], HeadUnit(7, 2), 0.0) == 0.0


def test_rejects_sequence_length_mismatch():
    plan = _plan(((HeadUnit(7, 0), 1.0),))
    with pytest.raises(ValueError, match="sequence length"):
        _residual(plan, 7, torch.zeros((2, SEQ + 1, DIM)), torch.ones((INNER, DIM)))


def test_blocks_touched_reports_blocks_not_units():
    plan = _plan(((HeadUnit(7, 0), 1.0), (HeadUnit(3, 2), 1.0)))
    assert plan.blocks_touched() == frozenset({3, 7})


# --- the equivalence invariant (spec section 3.3) ------------------------


def test_all_heads_reproduces_the_frozen_block_level_oracle():
    """Selecting every head of a block must equal the shipped block blend.

    The bias on the Linear is deliberately non-zero: the residual must be
    projected weight-only, or this test fails.
    """
    torch.manual_seed(0)
    linear = torch.nn.Linear(DIM, INNER)
    torch.nn.init.normal_(linear.bias)

    x = torch.randn((2, SEQ, DIM))
    embedding = torch.randn((SEQ, DIM))
    score, cond = 0.9, slice(1, 2)

    reference = reference_blend(
        [ReferenceRouted(COMPONENT, embedding, ((7, score),))],
        CFG,
        x,
        block_id=7,
        step_frac=0.0,
        cond_slice=cond,
    )
    expected = linear(reference)[cond]

    plan = RoutingPlan(
        routed=(
            RoutedComponent(
                component=COMPONENT,
                embedding=embedding,
                units=tuple((HeadUnit(7, h), score) for h in range(N_HEADS)),
            ),
        ),
        cfg=CFG,
    )
    got = linear(x)[cond] + _residual(plan, 7, x, linear.weight)

    torch.testing.assert_close(got, expected)


def test_two_components_into_one_block_sum_against_the_original_base():
    torch.manual_seed(1)
    linear = torch.nn.Linear(DIM, INNER)
    torch.nn.init.normal_(linear.bias)

    x = torch.randn((2, SEQ, DIM))
    e1, e2 = torch.randn((SEQ, DIM)), torch.randn((SEQ, DIM))
    other = Component(id="c_size", text="a small car", attr=AttributeClass.SIZE)
    cond = slice(1, 2)

    reference = reference_blend(
        [
            ReferenceRouted(COMPONENT, e1, ((7, 0.6),)),
            ReferenceRouted(other, e2, ((7, 0.3),)),
        ],
        CFG,
        x,
        block_id=7,
        step_frac=0.0,
        cond_slice=cond,
    )
    expected = linear(reference)[cond]

    plan = RoutingPlan(
        routed=(
            RoutedComponent(COMPONENT, e1, tuple((HeadUnit(7, h), 0.6) for h in range(N_HEADS))),
            RoutedComponent(other, e2, tuple((HeadUnit(7, h), 0.3) for h in range(N_HEADS))),
        ),
        cfg=CFG,
    )
    got = linear(x)[cond] + _residual(plan, 7, x, linear.weight)

    torch.testing.assert_close(got, expected)


# --- build_routing_plan --------------------------------------------------


def _hasm():
    # blocks (3, 7) x heads (0, 1) x attrs (COLOR, SIZE)
    tensor = np.array(
        [
            [[0.20, 0.80], [0.10, 0.40]],
            [[0.90, 0.30], [0.50, 0.60]],
        ]
    )
    return HASM(tensor, (3, 7), (0, 1), (AttributeClass.COLOR, AttributeClass.SIZE))


def test_head_granularity_selects_the_single_best_head():
    plan = build_routing_plan(
        [COMPONENT], {"c_color": torch.zeros((SEQ, DIM))}, _hasm(), CFG
    )
    assert plan.routed[0].units == ((HeadUnit(7, 0), 0.90),)


def test_block_granularity_expands_to_every_head_at_the_block_score():
    plan = build_routing_plan(
        [COMPONENT],
        {"c_color": torch.zeros((SEQ, DIM))},
        _hasm(),
        CFG,
        granularity="block",
    )
    # to_basm(max) gives block 7 a COLOR score of 0.90
    assert plan.routed[0].units == ((HeadUnit(7, 0), 0.90), (HeadUnit(7, 1), 0.90))


def test_rejects_unknown_granularity():
    with pytest.raises(ValueError, match="unknown granularity"):
        build_routing_plan(
            [COMPONENT],
            {"c_color": torch.zeros((SEQ, DIM))},
            _hasm(),
            CFG,
            granularity="layer",
        )


def test_skips_uncalibrated_attributes():
    action = Component(id="c_action", text="a car driving", attr=AttributeClass.ACTION)
    plan = build_routing_plan(
        [action], {"c_action": torch.zeros((SEQ, DIM))}, _hasm(), CFG
    )
    assert plan.routed == ()


def test_applies_k_override():
    plan = build_routing_plan(
        [COMPONENT],
        {"c_color": torch.zeros((SEQ, DIM))},
        _hasm(),
        CFG,
        k_overrides={"c_color": 2},
    )
    assert plan.routed[0].units == ((HeadUnit(7, 0), 0.90), (HeadUnit(7, 1), 0.50))
