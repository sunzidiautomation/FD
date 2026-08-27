"""CLIP-backed identity, style, and action metrics (spec section 3.4).

The scorer is injected so calibration logic can be tested without CLIP
weights. Raw CLIP image-text cosine similarities occupy roughly
[0.15, 0.35]; ``clip_norm`` maps that band onto [0, 1].
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image

CLIP_SIM_FLOOR = 0.15
CLIP_SIM_SPAN = 0.20

_STYLE_DESCRIPTORS = [
    "a photorealistic photograph",
    "a digital illustration",
    "an oil painting",
    "a pencil sketch",
    "a 3d render",
]


class ImageTextScorer(Protocol):
    def image_embedding(self, image: Image.Image) -> np.ndarray: ...

    def image_text_similarity(
        self, image: Image.Image, texts: list[str]
    ) -> np.ndarray: ...


class ClipScorer:
    """CLIP-backed scorer (``openai/clip-vit-base-patch32``)."""

    def __init__(self, model=None, processor=None, device: str = "cpu") -> None:
        if model is None or processor is None:  # pragma: no cover - needs weights
            from transformers import CLIPModel, CLIPProcessor

            name = "openai/clip-vit-base-patch32"
            processor = processor or CLIPProcessor.from_pretrained(name)
            model = model or CLIPModel.from_pretrained(name).eval()

        self.model = model
        self.processor = processor
        self.device = device
        if hasattr(self.model, "to"):
            self.model.to(self.device)

    def image_embedding(self, image: Image.Image) -> np.ndarray:
        import torch

        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.get_image_features(**inputs)

        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            features = output.image_embeds
        elif hasattr(output, "pooler_output") and output.pooler_output is not None:
            features = output.pooler_output
        elif isinstance(output, torch.Tensor):
            features = output
        else:
            features = output[0]

        features = features.squeeze()
        norm = features.norm()
        if norm > 0:
            features = features / norm
        return features.float().cpu().numpy()

    def image_text_similarity(
        self, image: Image.Image, texts: list[str]
    ) -> np.ndarray:
        import torch

        inputs = self.processor(
            text=texts, images=[image], return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)

        image_embeds = getattr(output, "image_embeds", None)
        text_embeds = getattr(output, "text_embeds", None)
        if image_embeds is None:
            image_embeds = output[0] if isinstance(output, (tuple, list)) else output
        if text_embeds is None:
            text_embeds = (
                output[1]
                if isinstance(output, (tuple, list)) and len(output) > 1
                else output
            )

        if image_embeds.dim() == 1:
            image_embeds = image_embeds.unsqueeze(0)
        elif image_embeds.dim() > 2:
            image_embeds = image_embeds.view(image_embeds.shape[0], -1)

        if text_embeds.dim() == 1:
            text_embeds = text_embeds.unsqueeze(0)
        elif text_embeds.dim() > 2:
            text_embeds = text_embeds.view(text_embeds.shape[0], -1)

        image_features = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_features = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return (image_features @ text_features.T)[0].float().cpu().numpy()


def clip_norm(similarity: float) -> float:
    """Map a raw CLIP cosine similarity onto [0, 1]."""
    return float(np.clip((similarity - CLIP_SIM_FLOOR) / CLIP_SIM_SPAN, 0.0, 1.0))


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = np.asarray(a, dtype=np.float64).ravel()
    b_flat = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0.0:
        return 0.0
    cosine = float(np.dot(a_flat, b_flat) / denom)
    return float(np.clip((1.0 - cosine) / 2.0, 0.0, 1.0))


def crop_to_mask(
    image: Image.Image, mask: np.ndarray | None, pad: float = 0.0
) -> Image.Image:
    """The mask's bounding box. Returns the image unchanged without a mask.

    ``pad`` defaults to none on purpose: every pixel of context is a pixel
    of background, and background leaking in is exactly the contamination
    this crop exists to remove. Raise it only if ClipSeg is under-segmenting
    and clipping the object itself.
    """
    if mask is None:
        return image
    rows, cols = np.nonzero(np.asarray(mask) > 0.5)
    if rows.size == 0:
        return image

    height, width = np.asarray(mask).shape[:2]
    margin_y, margin_x = int(pad * height), int(pad * width)
    box = (
        max(0, int(cols.min()) - margin_x),
        max(0, int(rows.min()) - margin_y),
        min(width, int(cols.max()) + 1 + margin_x),
        min(height, int(rows.max()) + 1 + margin_y),
    )
    return image.crop(box)


def identity_delta(
    image_a: Image.Image,
    image_b: Image.Image,
    mask: np.ndarray,
    scorer: ImageTextScorer,
) -> float:
    """How far the object's identity moved, as CLIP embedding distance.

    Confined to the object's bounding box. Identity is an object-level
    attribute, so embedding the whole frame would score a repainted sky or
    a recomposed background as the object becoming a different thing --
    which degenerates the metric into "how different is this image", a
    question every collapsed or merely degraded frame answers loudly.
    """
    return _cosine_distance(
        scorer.image_embedding(crop_to_mask(image_a, mask)),
        scorer.image_embedding(crop_to_mask(image_b, mask)),
    )


def target_concept_delta(
    image_a: Image.Image,
    image_b: Image.Image,
    mask: np.ndarray | None,
    scorer: ImageTextScorer,
    target_phrase: str,
) -> float:
    """How much more the subject resembles the concept swapped in.

    The damage-proof form of a delta metric, and not specific to identity.
    ``identity_delta`` and ``action_delta`` both measure distance FROM the
    baseline, and damage maximises distance by construction -- a destroyed
    frame is maximally far from every starting point, so no gate can
    separate the two: they are the same axis. Resemblance TO the target
    cannot be won that way, because noise does not look like a tractor or
    like a parked car; a collapsed frame's similarity to every phrase falls
    together, so it scores zero rather than maximum.

    Action is the sharper case. Its metric compares against the BASE action
    ("a car driving"), which the real swap and the damage both reduce --
    signal and artefact point the same way and are not separable at all.

    Deliberately not a difference of differences against the base phrase.
    That form counts losing sedan-ness as gaining tractor-ness, so a frame
    resembling nothing reads as a successful swap -- the failure being
    fixed, reintroduced one level up.

    Pass ``mask=None`` for scene-level attributes; action reads from the
    whole frame, identity from the object crop.

    Clamped at zero: becoming less target-like is not control toward the
    target.
    """
    crop_a = crop_to_mask(image_a, mask)
    crop_b = crop_to_mask(image_b, mask)
    before = float(np.asarray(scorer.image_text_similarity(crop_a, [target_phrase]))[0])
    after = float(np.asarray(scorer.image_text_similarity(crop_b, [target_phrase]))[0])
    return float(np.clip((after - before) / CLIP_SIM_SPAN, 0.0, 1.0))


def style_delta(
    image_a: Image.Image,
    image_b: Image.Image,
    mask: np.ndarray,
    scorer: ImageTextScorer,
) -> float:
    """Shift in the image's style-descriptor profile."""
    profile_a = np.asarray(scorer.image_text_similarity(image_a, _STYLE_DESCRIPTORS))
    profile_b = np.asarray(scorer.image_text_similarity(image_b, _STYLE_DESCRIPTORS))
    diff = float(np.abs(profile_a - profile_b).mean())
    return float(np.clip(diff / CLIP_SIM_SPAN, 0.0, 1.0))


def action_delta(
    image_a: Image.Image,
    image_b: Image.Image,
    mask: np.ndarray,
    scorer: ImageTextScorer,
    phrase: str,
) -> float:
    """Change in how well the image matches an action phrase."""
    score_a = float(np.asarray(scorer.image_text_similarity(image_a, [phrase]))[0])
    score_b = float(np.asarray(scorer.image_text_similarity(image_b, [phrase]))[0])
    return float(abs(clip_norm(score_a) - clip_norm(score_b)))
