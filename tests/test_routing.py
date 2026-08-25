import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flair_t2i.attributes import AttributeClass
from flair_t2i.basm import BASM
from flair_t2i.components import Component
from flair_t2i.config import FlairConfig
from flair_t2i.routing import RoutedComponent, RoutingPlan, build_routing_plan

SEQ, DIM = 4, 8
CFG = FlairConfig(alpha_0=1.0, t_window=(0.0, 1.0), top_k_default=1)


def _plan(blocks=((7, 1.0),), intensity=1.0, embedding_fill=1.0):
    component = Component(id="c_color", text="a red car", attr=AttributeClass.COLOR)
    routed = RoutedComponent(
        component=component,
        embedding=torch.full((SEQ, DIM), embedding_fill),
        blocks=blocks,
        intensity=intensity,
    )
    return RoutingPlan(routed=(routed,), cfg=CFG)


def _states(batch=2, fill=0.0):
    return torch.full((batch, SEQ, DIM), fill)


def test_blend_is_identity_on_untouched_blocks():
    plan = _plan()
    states = _states()
    out = plan.blend(states, block_id=99, step_frac=0.0, cond_slice=slice(1, 2))
    assert out is states


def test_blend_is_identity_when_inactive():
    plan = _plan()
    plan.active = False
    states = _states()
    out = plan.blend(states, block_id=7, step_frac=0.0, cond_slice=slice(1, 2))
    assert out is states


def test_blend_moves_conditional_rows_toward_the_component():
    plan = _plan()
    states = _states(fill=0.0)
    out = plan.blend(states, block_id=7, step_frac=0.0, cond_slice=slice(1, 2))

    # alpha = alpha_0(1.0) * S(1.0) * intensity(1.0) * sched(1.0) = 1.0
    # H = 0 + 1.0 * (1 - 0) = 1.0
    assert out[1].mean().item() == pytest.approx(1.0)


def test_blend_leaves_unconditional_rows_untouched():
    plan = _plan()
    states = _states(fill=0.0)
    out = plan.blend(states, block_id=7, step_frac=0.0, cond_slice=slice(1, 2))
    assert out[0].abs().max().item() == pytest.approx(0.0)


def test_blend_does_not_mutate_its_input():
    plan = _plan()
    states = _states(fill=0.0)
    plan.blend(states, block_id=7, step_frac=0.0, cond_slice=slice(1, 2))
    assert states.abs().max().item() == pytest.approx(0.0)


def test_alpha_scales_with_basm_score_and_intensity():
    plan = _plan(blocks=((7, 0.5),), intensity=0.4)
    rc = plan.routed[0]
    # 1.0 * 0.5 * 0.4 * 1.0
    assert plan.alpha(rc, 7, 0.0) == pytest.approx(0.2)


def test_alpha_respects_guard_backoff():
    plan = _plan()
    plan.alpha_scale = 0.5
    assert plan.alpha(plan.routed[0], 7, 0.0) == pytest.approx(0.5)


def test_alpha_is_zero_outside_timestep_window():
    plan = _plan()
    plan.cfg = FlairConfig(alpha_0=1.0, t_window=(0.0, 0.5))
    assert plan.alpha(plan.routed[0], 7, 0.9) == 0.0


def test_blend_rejects_sequence_length_mismatch():
    plan = _plan()
    bad = torch.zeros((2, SEQ + 1, DIM))
    with pytest.raises(ValueError, match="sequence length"):
        plan.blend(bad, block_id=7, step_frac=0.0, cond_slice=slice(1, 2))


def test_build_routing_plan_selects_top_block_per_attribute():
    components = [
        Component(id="c_color", text="a red car", attr=AttributeClass.COLOR),
        Component(id="c_size", text="a small car", attr=AttributeClass.SIZE),
    ]
    basm = BASM(
        matrix=np.array([[0.22, 0.81], [0.93, 0.14]]),
        block_ids=(3, 7),
        attributes=(AttributeClass.COLOR, AttributeClass.SIZE),
    )
    embeddings = {c.id: torch.zeros((SEQ, DIM)) for c in components}

    plan = build_routing_plan(components, embeddings, basm, CFG)
    by_id = {rc.component.id: rc for rc in plan.routed}

    assert by_id["c_color"].blocks == ((7, 0.93),)
    assert by_id["c_size"].blocks == ((3, 0.81),)
    assert plan.blocks_touched() == frozenset({3, 7})


def test_build_routing_plan_skips_uncalibrated_attributes():
    components = [
        Component(id="c_action", text="a car driving", attr=AttributeClass.ACTION)
    ]
    basm = BASM.uniform((3,), (AttributeClass.COLOR,))
    plan = build_routing_plan(
        components, {"c_action": torch.zeros((SEQ, DIM))}, basm, CFG
    )
    assert plan.routed == ()


def test_build_routing_plan_applies_k_override():
    components = [Component(id="c_color", text="a red car", attr=AttributeClass.COLOR)]
    basm = BASM(
        matrix=np.array([[0.4], [0.9], [0.6]]),
        block_ids=(3, 7, 11),
        attributes=(AttributeClass.COLOR,),
    )
    plan = build_routing_plan(
        components,
        {"c_color": torch.zeros((SEQ, DIM))},
        basm,
        CFG,
        k_overrides={"c_color": 2},
    )
    assert plan.routed[0].blocks == ((7, 0.9), (11, 0.6))
