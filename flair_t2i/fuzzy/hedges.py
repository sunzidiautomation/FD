"""Zadeh linguistic hedge operators (spec section 3.3-B).

Spec section 3.3-B as written applies the operators to a scalar mu_base of
1.0, where mu**2 == mu**0.5 == 1.0 and every hedge collapses to the same
value. Zadeh's operators act on a membership CURVE, so they are applied
pointwise here and the scalar is derived from the resulting set:

    specificity(mu) = 1 - mean(mu)
    intensity       = clip(spec(hedged) / spec(unhedged), 0.3, 1.6)

Concentration ("very") narrows the set, raising specificity and pushing
harder through fewer blocks. Dilation ("slightly") widens it, giving a
weaker, more diffuse push. An unhedged label yields exactly 1.0, so the
crisp pipeline's behaviour is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..attributes import AttributeClass
from .membership import membership_curve


class HedgeKind(str, Enum):
    NONE = "none"
    CONCENTRATE = "concentrate"
    DILATE = "dilate"
    COMPLEMENT = "complement"


HEDGE_KINDS: dict[str, HedgeKind] = {
    # concentration -- stronger, narrower
    "very": HedgeKind.CONCENTRATE,
    "extremely": HedgeKind.CONCENTRATE,
    "super": HedgeKind.CONCENTRATE,
    "highly": HedgeKind.CONCENTRATE,
    "intensely": HedgeKind.CONCENTRATE,
    # mild concentration -- treated as no-op so the ladder stays ordered
    "quite": HedgeKind.NONE,
    "fairly": HedgeKind.NONE,
    "moderately": HedgeKind.NONE,
    "rather": HedgeKind.NONE,
    # dilation -- weaker, wider
    "slightly": HedgeKind.DILATE,
    "somewhat": HedgeKind.DILATE,
    "mildly": HedgeKind.DILATE,
    "faintly": HedgeKind.DILATE,
    "barely": HedgeKind.DILATE,
    # negation
    "not": HedgeKind.COMPLEMENT,
}

_INTENSITY_MIN, _INTENSITY_MAX = 0.3, 1.6


@dataclass(frozen=True)
class HedgeResult:
    kind: HedgeKind
    intensity: float
    k: int
    curve: np.ndarray


def apply_hedge(curve: np.ndarray, kind: HedgeKind) -> np.ndarray:
    if kind is HedgeKind.CONCENTRATE:
        return curve**2
    if kind is HedgeKind.DILATE:
        return curve**0.5
    if kind is HedgeKind.COMPLEMENT:
        return 1.0 - curve
    return curve


def specificity(curve: np.ndarray) -> float:
    """How selective a fuzzy set is: 1 for empty, 0 for the whole universe."""
    return float(1.0 - np.mean(curve))


def _breadth(intensity: float) -> int:
    if intensity >= 1.0:
        return 1
    if intensity >= 0.6:
        return 2
    return 3


def resolve_hedge(
    attr: AttributeClass, label: str, hedge_word: str | None
) -> HedgeResult:
    """Turn a hedge word into an injection intensity and a routing breadth."""
    base = membership_curve(attr, label)
    kind = HEDGE_KINDS.get((hedge_word or "").lower(), HedgeKind.NONE)
    hedged = apply_hedge(base, kind)

    base_spec = specificity(base)
    ratio = 1.0 if base_spec <= 0.0 else specificity(hedged) / base_spec
    intensity = float(np.clip(ratio, _INTENSITY_MIN, _INTENSITY_MAX))

    return HedgeResult(kind=kind, intensity=intensity, k=_breadth(intensity), curve=hedged)
