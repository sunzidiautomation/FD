import pytest

pytest.importorskip("skfuzzy")

from flair_t2i.attributes import AttributeClass
from flair_t2i.config import FlairConfig
from flair_t2i.guard import CoherenceGuard

CFG = FlairConfig(guard_membership_threshold=0.5, guard_backoff=0.5)


def test_no_event_when_measurement_sits_inside_the_fuzzy_region():
    guard = CoherenceGuard(CFG)
    event = guard.check_membership(AttributeClass.SIZE, "small", measured=0.05, step=2)
    assert event is None


def test_event_when_measurement_falls_outside_the_region():
    guard = CoherenceGuard(CFG)
    event = guard.check_membership(AttributeClass.SIZE, "small", measured=0.9, step=2)
    assert event is not None
    assert event.reason == "attribute_membership"
    assert event.step == 2
    assert event.value < 0.5


def test_membership_event_backs_off_alpha_like_the_cosine_check():
    import torch

    from flair_t2i.components import Component
    from flair_t2i.heads import HeadUnit
    from flair_t2i.routing import RoutedComponent, RoutingPlan

    plan = RoutingPlan(
        routed=(
            RoutedComponent(
                component=Component(id="c", text="x", attr=AttributeClass.SIZE),
                embedding=torch.ones((2, 2)),
                units=((HeadUnit(7, 0), 1.0),),
            ),
        ),
        cfg=CFG,
    )
    guard = CoherenceGuard(CFG)
    guard.apply(plan, guard.check_membership(AttributeClass.SIZE, "small", 0.9, step=1))
    assert plan.alpha_scale == pytest.approx(0.5)
