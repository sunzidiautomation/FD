import pytest

from flair_t2i.processor import PlanRef


def test_step_frac_is_step_over_total():
    assert PlanRef(step=5, total_steps=10).step_frac() == pytest.approx(0.5)


def test_step_frac_handles_zero_total():
    assert PlanRef(step=0, total_steps=0).step_frac() == 0.0


def test_cond_slice_is_second_half_under_cfg():
    assert PlanRef(do_cfg=True).cond_slice(4) == slice(2, 4)


def test_cond_slice_is_everything_without_cfg():
    assert PlanRef(do_cfg=False).cond_slice(2) == slice(0, 2)
