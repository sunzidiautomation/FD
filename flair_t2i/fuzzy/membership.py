"""Fuzzy membership over attribute metric universes (spec section 3.3-A).

Every attribute's metric is normalised onto [0, 1] so one shared grid
serves all seven. The four core attributes carry real linguistic labels;
the remaining three carry a single 'match' label -- their metrics
(Gram/DISTS distance, CLIP action score) are similarity scores without a
natural linguistic scale.

These curves do two jobs: they give BASM calibration a graded target
(section 3.4) and they replace the coherence guard's crisp percentile band
(section 3.6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import skfuzzy as fuzz

from ..attributes import AttributeClass

#: Shared normalised grid. Each attribute maps its own metric onto [0, 1].
UNIVERSE: np.ndarray = np.linspace(0.0, 1.0, 201)


@dataclass(frozen=True)
class AttributeUniverse:
    attr: AttributeClass
    metric: str
    labels: dict[str, np.ndarray]


def _rising() -> np.ndarray:
    """A shoulder that saturates as the metric approaches its target."""
    return fuzz.trapmf(UNIVERSE, [0.35, 0.85, 1.0, 1.0])


UNIVERSES: dict[AttributeClass, AttributeUniverse] = {
    AttributeClass.SIZE: AttributeUniverse(
        attr=AttributeClass.SIZE,
        metric="object mask area ratio",
        labels={
            "small": fuzz.trapmf(UNIVERSE, [0.0, 0.0, 0.12, 0.28]),
            "medium": fuzz.trimf(UNIVERSE, [0.18, 0.38, 0.58]),
            "large": fuzz.trapmf(UNIVERSE, [0.48, 0.70, 1.0, 1.0]),
        },
    ),
    AttributeClass.LIGHTING: AttributeUniverse(
        attr=AttributeClass.LIGHTING,
        metric="normalised colour temperature (0 cool, 1 warm)",
        labels={
            "cool": fuzz.trapmf(UNIVERSE, [0.0, 0.0, 0.20, 0.42]),
            "neutral": fuzz.trimf(UNIVERSE, [0.32, 0.50, 0.68]),
            "warm": fuzz.trapmf(UNIVERSE, [0.58, 0.80, 1.0, 1.0]),
        },
    ),
    AttributeClass.COLOR: AttributeUniverse(
        attr=AttributeClass.COLOR,
        metric="1 - normalised CIELAB dE to target hue",
        labels={"match": _rising()},
    ),
    AttributeClass.IDENTITY: AttributeUniverse(
        attr=AttributeClass.IDENTITY,
        metric="CLIP similarity to identity anchors",
        labels={"match": _rising()},
    ),
    AttributeClass.TEXTURE: AttributeUniverse(
        attr=AttributeClass.TEXTURE,
        metric="1 - normalised Gram/DISTS distance",
        labels={"match": _rising()},
    ),
    AttributeClass.STYLE: AttributeUniverse(
        attr=AttributeClass.STYLE,
        metric="CLIP style similarity",
        labels={"match": _rising()},
    ),
    AttributeClass.ACTION: AttributeUniverse(
        attr=AttributeClass.ACTION,
        metric="CLIP score for the action phrase",
        labels={"match": _rising()},
    ),
}

_DEFAULT_LABELS: dict[AttributeClass, str] = {
    AttributeClass.SIZE: "small",
    AttributeClass.LIGHTING: "warm",
}


def default_label(attr: AttributeClass) -> str:
    """The label used when a prompt names no specific linguistic value."""
    return _DEFAULT_LABELS.get(attr, "match")


def membership_curve(attr: AttributeClass, label: str) -> np.ndarray:
    labels = UNIVERSES[attr].labels
    if label not in labels:
        raise KeyError(f"{label!r} is not a label of {attr.value}: {sorted(labels)}")
    return labels[label]


def membership_at(attr: AttributeClass, label: str, x: float) -> float:
    """Membership of a measured metric value, clamped to the universe."""
    curve = membership_curve(attr, label)
    clamped = float(np.clip(x, UNIVERSE[0], UNIVERSE[-1]))
    return float(np.interp(clamped, UNIVERSE, curve))
