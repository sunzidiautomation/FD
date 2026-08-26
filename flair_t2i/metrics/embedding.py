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


def identity_delta(
    image_a: Image.Image,
    image_b: Image.Image,
    mask: np.ndarray,
    scorer: ImageTextScorer,
) -> float:
    """How far the object's identity moved, as CLIP embedding distance."""
    return _cosine_distance(
        scorer.image_embedding(image_a), scorer.image_embedding(image_b)
    )


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
