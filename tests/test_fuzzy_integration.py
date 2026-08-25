import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("skfuzzy")

from flair_t2i.attributes import AttributeClass
from flair_t2i.basm import BASM
from flair_t2i.components import Component
from flair_t2i.config import FlairConfig
from flair_t2i.fuzzy.resolve import resolve_components
from flair_t2i.routing import build_routing_plan

SEQ, DIM = 4, 8
CFG = FlairConfig(alpha_0=1.0, t_window=(0.0, 1.0), top_k_default=1)


def _basm():
    return BASM(
        matrix=np.array([[0.4], [0.9], [0.6]]),
        block_ids=(3, 7, 11),
        attributes=(AttributeClass.SIZE,),
    )


def _component(hedge):
    return Component(
        id="c_size", text="a small car", attr=AttributeClass.SIZE, hedge=hedge
    )


def test_unhedged_component_keeps_intensity_one():
    intensities, k_overrides, _ = resolve_components([_component(None)])
    assert intensities["c_size"] == pytest.approx(1.0)
    assert k_overrides["c_size"] == 1


def test_very_raises_intensity_and_keeps_k_at_one():
    intensities, k_overrides, _ = resolve_components([_component("very")])
    assert intensities["c_size"] > 1.0
    assert k_overrides["c_size"] == 1


def test_slightly_lowers_intensity_and_widens_k():
    intensities, k_overrides, _ = resolve_components([_component("slightly")])
    assert intensities["c_size"] < 1.0
    assert k_overrides["c_size"] >= 2


def test_hedge_flows_through_to_routed_blocks():
    components = [_component("slightly")]
    intensities, k_overrides, _ = resolve_components(components)
    embeddings = {"c_size": torch.zeros((SEQ, DIM))}

    plan = build_routing_plan(
        components, embeddings, _basm(), CFG, intensities, k_overrides
    )

    assert len(plan.routed[0].blocks) >= 2  # widened by dilation
    assert plan.routed[0].intensity < 1.0


def test_hedge_changes_the_resulting_alpha():
    embeddings = {"c_size": torch.zeros((SEQ, DIM))}

    plain = build_routing_plan([_component(None)], embeddings, _basm(), CFG)
    hedged_components = [_component("very")]
    intensities, k_overrides, _ = resolve_components(hedged_components)
    hedged = build_routing_plan(
        hedged_components, embeddings, _basm(), CFG, intensities, k_overrides
    )

    a_plain = plain.alpha(plain.routed[0], 7, 0.0)
    a_hedged = hedged.alpha(hedged.routed[0], 7, 0.0)
    assert a_hedged > a_plain


def test_uncalibrated_attribute_is_still_skipped():
    components = [
        Component(id="c_style", text="cyberpunk", attr=AttributeClass.STYLE, hedge="very")
    ]
    intensities, k_overrides, _ = resolve_components(components)
    plan = build_routing_plan(
        components,
        {"c_style": torch.zeros((SEQ, DIM))},
        _basm(),
        CFG,
        intensities,
        k_overrides,
    )
    assert plan.routed == ()
