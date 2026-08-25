import numpy as np
import pytest

pytest.importorskip("skfuzzy")

from flair_t2i.attributes import AttributeClass
from flair_t2i.fuzzy.hedges import (
    HEDGE_KINDS,
    HedgeKind,
    apply_hedge,
    resolve_hedge,
    specificity,
)
from flair_t2i.fuzzy.membership import membership_curve
from flair_t2i.parsing import HEDGE_WORDS

SIZE_SMALL = membership_curve(AttributeClass.SIZE, "small")


def test_every_parser_hedge_word_has_a_kind():
    assert set(HEDGE_WORDS) <= set(HEDGE_KINDS)


def test_concentration_narrows_and_dilation_widens():
    con = specificity(apply_hedge(SIZE_SMALL, HedgeKind.CONCENTRATE))
    base = specificity(apply_hedge(SIZE_SMALL, HedgeKind.NONE))
    dil = specificity(apply_hedge(SIZE_SMALL, HedgeKind.DILATE))
    assert con > base > dil


def test_complement_inverts_the_curve():
    comp = apply_hedge(SIZE_SMALL, HedgeKind.COMPLEMENT)
    np.testing.assert_allclose(comp, 1.0 - SIZE_SMALL)


def test_no_hedge_leaves_the_curve_untouched():
    np.testing.assert_allclose(apply_hedge(SIZE_SMALL, HedgeKind.NONE), SIZE_SMALL)


def test_specificity_of_full_set_is_zero():
    assert specificity(np.ones(201)) == pytest.approx(0.0)


def test_specificity_of_empty_set_is_one():
    assert specificity(np.zeros(201)) == pytest.approx(1.0)


def test_unhedged_intensity_is_exactly_one():
    result = resolve_hedge(AttributeClass.SIZE, "small", None)
    assert result.intensity == pytest.approx(1.0)
    assert result.k == 1
    assert result.kind is HedgeKind.NONE


def test_very_strengthens_and_narrows():
    result = resolve_hedge(AttributeClass.SIZE, "small", "very")
    assert result.intensity > 1.0
    assert result.k == 1


def test_slightly_weakens_and_widens():
    result = resolve_hedge(AttributeClass.SIZE, "small", "slightly")
    assert result.intensity < 1.0
    assert result.k >= 2


def test_negation_is_weak_and_diffuse():
    result = resolve_hedge(AttributeClass.SIZE, "small", "not")
    assert result.kind is HedgeKind.COMPLEMENT
    assert result.intensity < 0.6
    assert result.k == 3


def test_intensity_is_clipped_to_the_documented_range():
    for word in HEDGE_KINDS:
        result = resolve_hedge(AttributeClass.SIZE, "small", word)
        assert 0.3 <= result.intensity <= 1.6


def test_unknown_hedge_word_is_treated_as_no_hedge():
    result = resolve_hedge(AttributeClass.SIZE, "small", "purple")
    assert result.kind is HedgeKind.NONE
    assert result.intensity == pytest.approx(1.0)


def test_monotonic_across_the_intensity_ladder():
    ladder = ["slightly", "quite", "very"]
    values = [resolve_hedge(AttributeClass.SIZE, "small", w).intensity for w in ladder]
    assert values == sorted(values)
