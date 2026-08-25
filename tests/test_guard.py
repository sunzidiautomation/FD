import pytest

torch = pytest.importorskip("torch")

from flair_t2i.attributes import AttributeClass
from flair_t2i.components import Component
from flair_t2i.config import FlairConfig
from flair_t2i.guard import CoherenceGuard
from flair_t2i.routing import RoutedComponent, RoutingPlan

SEQ, DIM = 4, 8
CFG = FlairConfig(guard_cos_threshold=0.55, guard_backoff=0.5)


def _plan(*embeddings):
    routed = tuple(
        RoutedComponent(
            component=Component(id=f"c{i}", text="x", attr=AttributeClass.COLOR),
            embedding=e,
            blocks=((7, 1.0),),
        )
        for i, e in enumerate(embeddings)
    )
    return RoutingPlan(routed=routed, cfg=CFG)


def test_no_event_for_aligned_streams():
    a = torch.ones((SEQ, DIM))
    plan = _plan(a, a.clone())
    assert CoherenceGuard(CFG).check_streams(plan, step=3) is None


def test_event_for_opposed_streams():
    a = torch.ones((SEQ, DIM))
    plan = _plan(a, -a)
    event = CoherenceGuard(CFG).check_streams(plan, step=3)
    assert event is not None
    assert event.step == 3
    assert event.reason == "cross_stream_similarity"
    assert event.value < 0.55


def test_single_stream_never_fires():
    plan = _plan(torch.ones((SEQ, DIM)))
    assert CoherenceGuard(CFG).check_streams(plan, step=0) is None


def test_apply_backs_off_alpha_scale():
    a = torch.ones((SEQ, DIM))
    plan = _plan(a, -a)
    guard = CoherenceGuard(CFG)
    event = guard.check_streams(plan, step=1)
    guard.apply(plan, event)
    assert plan.alpha_scale == pytest.approx(0.5)


def test_apply_is_a_noop_without_an_event():
    plan = _plan(torch.ones((SEQ, DIM)))
    guard = CoherenceGuard(CFG)
    guard.apply(plan, None)
    assert plan.alpha_scale == pytest.approx(1.0)


def test_backoff_compounds_across_steps():
    a = torch.ones((SEQ, DIM))
    plan = _plan(a, -a)
    guard = CoherenceGuard(CFG)
    for step in range(2):
        guard.apply(plan, guard.check_streams(plan, step))
    assert plan.alpha_scale == pytest.approx(0.25)
    assert len(guard.events) == 2
