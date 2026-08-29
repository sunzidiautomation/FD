"""One delta metric per attribute, behind a common signature.

Every metric returned by ``delta_for`` has the signature
``(image_a, image_b, mask) -> float`` in [0, 1], so the calibration harness
does not branch on attribute type.

SIZE is the exception the harness must know about: bound here against a
single shared mask it always reads 0, because area change is only visible
by re-segmenting the changed image. Task 16 passes per-image masks for it.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image

from ..attributes import AttributeClass
from .embedding import (
    ImageTextScorer,
    action_delta,
    identity_delta,
    style_delta,
    target_concept_delta,
)
from .photometric import (
    color_delta,
    colour_reference,
    lighting_delta,
    size_delta,
    target_colour_delta,
)
from .texture import gram_texture_delta

DeltaMetric = Callable[[Image.Image, Image.Image, np.ndarray], float]

#: Which metric family each attribute uses.
DELTA_METRICS: dict[AttributeClass, str] = {
    AttributeClass.COLOR: "color",
    AttributeClass.SIZE: "size",
    AttributeClass.LIGHTING: "lighting",
    AttributeClass.TEXTURE: "texture",
    AttributeClass.IDENTITY: "identity",
    AttributeClass.STYLE: "style",
    AttributeClass.ACTION: "action",
}

#: Attributes whose delta metric reads ONLY inside the object mask. The rest
#: read the whole frame, so anything that has to match "the region this
#: metric measures" -- the integrity gate above all -- must branch on this,
#: not on whether a mask happens to be available.
#:
#: SIZE is deliberately absent though it is an object attribute: it compares
#: two segmentations of two whole frames, and its object changes extent by
#: design, so there is no fixed region to confine a check to.
OBJECT_LEVEL: frozenset = frozenset(
    {AttributeClass.COLOR, AttributeClass.TEXTURE, AttributeClass.IDENTITY}
)

_NEEDS_SCORER = {"identity", "style", "action"}


def target_for(attr: AttributeClass, pair) -> str | None:
    """What ``pair`` injected, in the form ``delta_for`` wants for ``attr``.

    One place, so a sweep and a rescore of that sweep cannot disagree
    about what they were measuring toward.
    """
    if attr in (AttributeClass.IDENTITY, AttributeClass.ACTION):
        return pair.changed
    if attr is AttributeClass.COLOR:
        from PIL import ImageColor

        word = pair.changed_word
        # None rather than raising: a corpus that stops naming a hue there
        # should fall back to the undirected metric, not abort the sweep.
        return word if word and word in ImageColor.colormap else None
    return None


def delta_for(
    attr: AttributeClass,
    *,
    scorer: ImageTextScorer | None = None,
    phrase: str | None = None,
    target: str | None = None,
) -> DeltaMetric:
    """Return the delta metric for ``attr``, bound to its dependencies.

    ``target`` is the concept the swap injected. Where it is given, a
    metric switches from "how far from the baseline" -- which damage
    maximises by construction -- to "how close to what was swapped in",
    which damage cannot.

    For colour it is the colour word, naming a fixed point in Lab, so the
    measurement is exact. For identity and action it is the changed
    phrase and CLIP judges the resemblance, which holds only while the
    target names something damage cannot add: "a tractor" qualifies,
    "parked" does not, being largely the absence of a motion cue.
    """
    kind = DELTA_METRICS[attr]

    if kind in _NEEDS_SCORER and scorer is None:
        raise ValueError(f"{attr.value} needs a scorer")
    if kind == "action" and not (phrase or target):
        raise ValueError(f"{attr.value} needs a phrase or a target")

    if kind == "color":
        if target:
            # Bound eagerly so an unusable target fails at wiring time, not
            # 576 cells into a sweep.
            colour_reference(target)
            return lambda a, b, mask: target_colour_delta(a, b, mask, target)
        return color_delta
    if kind == "lighting":
        return lighting_delta
    if kind == "texture":
        return gram_texture_delta
    if kind == "size":
        # See the module docstring: reads 0 against one shared mask.
        return lambda a, b, mask: size_delta(a, b, mask, mask)
    if kind == "identity":
        if target:
            return lambda a, b, mask: target_concept_delta(a, b, mask, scorer, target)
        return lambda a, b, mask: identity_delta(a, b, mask, scorer)
    if kind == "style":
        return lambda a, b, mask: style_delta(a, b, mask, scorer)
    if target:
        # Action reads from the whole frame, so no crop.
        return lambda a, b, mask: target_concept_delta(a, b, None, scorer, target)
    return lambda a, b, mask: action_delta(a, b, mask, scorer, phrase)
