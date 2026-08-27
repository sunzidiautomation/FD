"""Object masks for attribute measurement (spec section 3.4).

Metrics are measured inside the object mask so that a change in, say, the
car's colour is not diluted by the background. CLIPSeg supplies the mask at
run time; ``RectMasker`` supplies a deterministic one in tests.

The model is injected rather than imported at the call site, so every
calibration test runs on CPU with no weights downloaded.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image

#: Below this many selected pixels a masked statistic is not trustworthy.
MIN_MASK_PIXELS = 50


class Masker(Protocol):
    def __call__(self, image: Image.Image, label: str) -> np.ndarray:
        """Return a binary [H, W] float mask selecting ``label`` in ``image``."""
        ...


class RectMasker:
    """A fixed rectangular mask. Deterministic stand-in for CLIPSeg in tests."""

    def __init__(self, box: tuple[float, float, float, float]) -> None:
        self.box = box

    def __call__(self, image: Image.Image, label: str) -> np.ndarray:
        width, height = image.size
        x0, y0, x1, y1 = self.box
        mask = np.zeros((height, width), dtype=np.float32)
        mask[
            int(round(y0 * height)) : int(round(y1 * height)),
            int(round(x0 * width)) : int(round(x1 * width)),
        ] = 1.0
        return mask


class ClipSegMasker:
    """CLIPSeg-backed masker (``CIDAS/clipseg-rd64-refined``)."""

    def __init__(
        self,
        model=None,
        processor=None,
        threshold: float = 0.4,
        device: str = "cpu",
    ) -> None:
        if model is None or processor is None:  # pragma: no cover - needs weights
            from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

            name = "CIDAS/clipseg-rd64-refined"
            processor = processor or CLIPSegProcessor.from_pretrained(name)
            model = model or CLIPSegForImageSegmentation.from_pretrained(name).eval()

        self.model = model
        self.processor = processor
        self.threshold = threshold
        self.device = device
        if hasattr(self.model, "to"):
            self.model.to(self.device)

    def __call__(self, image: Image.Image, label: str) -> np.ndarray:
        import torch

        inputs = self.processor(text=[label], images=[image], return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        if logits.dim() == 2:
            logits = logits[None]

        probs = torch.sigmoid(logits)[0].float().cpu().numpy()
        resized = (
            np.asarray(
                Image.fromarray((probs * 255).astype(np.uint8)).resize(image.size)
            )
            / 255.0
        )
        return (resized > self.threshold).astype(np.float32)


def mask_area_ratio(mask: np.ndarray) -> float:
    """Fraction of the frame the mask selects, in [0, 1]."""
    return float(np.clip(mask.mean(), 0.0, 1.0))


def masked_mean_rgb(image: Image.Image, mask: np.ndarray) -> np.ndarray | None:
    """Mean RGB inside the mask, or None when the mask is too small."""
    selected = mask > 0.5
    if selected.sum() < MIN_MASK_PIXELS:
        return None
    pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
    return pixels[selected].mean(axis=0)
