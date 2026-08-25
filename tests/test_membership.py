import numpy as np
import pytest

pytest.importorskip("skfuzzy")

from flair_t2i.attributes import AttributeClass
from flair_t2i.fuzzy.membership import (
    UNIVERSE,
    UNIVERSES,
    default_label,
    membership_at,
    membership_curve,
)


def test_every_attribute_has_a_universe():
    assert set(UNIVERSES) == set(AttributeClass)


def test_universe_is_the_unit_interval():
    assert UNIVERSE[0] == pytest.approx(0.0)
    assert UNIVERSE[-1] == pytest.approx(1.0)
    assert len(UNIVERSE) == 201


def test_curves_are_valid_memberships():
    for attr, universe in UNIVERSES.items():
        for label, curve in universe.labels.items():
            assert curve.shape == UNIVERSE.shape, (attr, label)
            assert curve.min() >= 0.0 and curve.max() <= 1.0, (attr, label)
            assert curve.max() == pytest.approx(1.0), (attr, label)


def test_size_labels_peak_where_expected():
    # 'small' peaks at low area ratio, 'large' at high
    assert membership_at(AttributeClass.SIZE, "small", 0.05) == pytest.approx(1.0)
    assert membership_at(AttributeClass.SIZE, "small", 0.9) == pytest.approx(0.0)
    assert membership_at(AttributeClass.SIZE, "large", 0.9) == pytest.approx(1.0)


def test_color_membership_rises_toward_target():
    low = membership_at(AttributeClass.COLOR, "match", 0.2)
    high = membership_at(AttributeClass.COLOR, "match", 0.95)
    assert high > low


def test_default_label_is_defined_for_every_attribute():
    for attr in AttributeClass:
        label = default_label(attr)
        assert label in UNIVERSES[attr].labels


def test_unknown_label_raises():
    with pytest.raises(KeyError, match="enormous"):
        membership_curve(AttributeClass.SIZE, "enormous")


def test_membership_at_clamps_out_of_range_inputs():
    assert membership_at(AttributeClass.SIZE, "small", -5.0) == pytest.approx(1.0)
    assert membership_at(AttributeClass.SIZE, "small", 5.0) == pytest.approx(0.0)
