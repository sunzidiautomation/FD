import pytest

torch = pytest.importorskip("torch")

from flair_t2i.attributes import AttributeClass
from flair_t2i.components import Component
from flair_t2i.config import FlairConfig
from flair_t2i.processor import FlairJointProcessor, PlanRef
from flair_t2i.routing import RoutedComponent, RoutingPlan

SEQ, DIM = 4, 8
CFG = FlairConfig(alpha_0=1.0, t_window=(0.0, 1.0))


class RecordingProcessor:
    """Stands in for diffusers' JointAttnProcessor2_0."""

    def __init__(self):
        self.seen = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, **kwargs):
        self.seen = encoder_hidden_states
        return hidden_states, encoder_hidden_states


def _ref(block_blocks=((7, 1.0),), total_steps=10, step=0):
    routed = RoutedComponent(
        component=Component(id="c_color", text="a red car", attr=AttributeClass.COLOR),
        embedding=torch.ones((SEQ, DIM)),
        blocks=block_blocks,
    )
    return PlanRef(
        plan=RoutingPlan(routed=(routed,), cfg=CFG),
        step=step,
        total_steps=total_steps,
        do_cfg=True,
    )


def test_step_frac_is_step_over_total():
    assert PlanRef(step=5, total_steps=10).step_frac() == pytest.approx(0.5)


def test_step_frac_handles_zero_total():
    assert PlanRef(step=0, total_steps=0).step_frac() == 0.0


def test_cond_slice_is_second_half_under_cfg():
    assert PlanRef(do_cfg=True).cond_slice(4) == slice(2, 4)


def test_cond_slice_is_everything_without_cfg():
    assert PlanRef(do_cfg=False).cond_slice(2) == slice(0, 2)


def test_processor_blends_before_delegating():
    inner = RecordingProcessor()
    proc = FlairJointProcessor(inner, block_id=7, ref=_ref())
    states = torch.zeros((2, SEQ, DIM))

    proc(attn=None, hidden_states=torch.zeros((2, SEQ, DIM)), encoder_hidden_states=states)

    assert inner.seen[1].mean().item() == pytest.approx(1.0)  # conditional row moved
    assert inner.seen[0].abs().max().item() == pytest.approx(0.0)  # uncond untouched


def test_processor_passes_through_on_unrouted_block():
    inner = RecordingProcessor()
    proc = FlairJointProcessor(inner, block_id=99, ref=_ref())
    states = torch.zeros((2, SEQ, DIM))

    proc(attn=None, hidden_states=torch.zeros((2, SEQ, DIM)), encoder_hidden_states=states)

    assert inner.seen is states


def test_processor_passes_through_when_plan_is_none():
    inner = RecordingProcessor()
    proc = FlairJointProcessor(inner, block_id=7, ref=PlanRef(plan=None))
    states = torch.zeros((2, SEQ, DIM))

    proc(attn=None, hidden_states=torch.zeros((2, SEQ, DIM)), encoder_hidden_states=states)

    assert inner.seen is states


def test_processor_passes_through_when_encoder_states_absent():
    inner = RecordingProcessor()
    proc = FlairJointProcessor(inner, block_id=7, ref=_ref())
    proc(attn=None, hidden_states=torch.zeros((2, SEQ, DIM)), encoder_hidden_states=None)
    assert inner.seen is None
