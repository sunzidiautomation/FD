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
from .photometric import color_delta, lighting_delta, size_delta
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

_NEEDS_SCORER = {"identity", "style", "action"}


def delta_for(
    attr: AttributeClass,
    *,
    scorer: ImageTextScorer | None = None,
    phrase: str | None = None,
    target: str | None = None,
) -> DeltaMetric:
    """Return the delta metric for ``attr``, bound to its dependencies.

    ``target`` is the concept the swap injected. Where it is given, identity
    and action switch from "how far from the baseline" -- which damage
    maximises by construction -- to "how close to what was swapped in",
    which damage cannot. See ``embedding.target_concept_delta``.
    """
    kind = DELTA_METRICS[attr]

    if kind in _NEEDS_SCORER and scorer is None:
        raise ValueError(f"{attr.value} needs a scorer")
    if kind == "action" and not (phrase or target):
        raise ValueError(f"{attr.value} needs a phrase or a target")

    if kind == "color":
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
