import pytest

from flair_t2i.schedule import timestep_scale

WINDOW = (0.0, 0.6)


def test_full_strength_at_window_start():
    assert timestep_scale(0.0, WINDOW) == pytest.approx(1.0)


def test_decays_linearly_to_zero_at_window_end():
    assert timestep_scale(0.3, WINDOW) == pytest.approx(0.5)
    assert timestep_scale(0.6, WINDOW) == pytest.approx(0.0)


def test_zero_outside_window():
    assert timestep_scale(0.75, WINDOW) == 0.0
    assert timestep_scale(0.2, (0.4, 0.8)) == 0.0


def test_offset_window_starts_at_full_strength():
    assert timestep_scale(0.4, (0.4, 0.8)) == pytest.approx(1.0)


def test_degenerate_window_is_a_point():
    assert timestep_scale(0.5, (0.5, 0.5)) == pytest.approx(1.0)
    assert timestep_scale(0.6, (0.5, 0.5)) == 0.0
