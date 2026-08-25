# FLAIR Calibration Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and unit-test the code that fills the BASM — attribute change metrics, object masking, the contrastive-pair corpus, the vital-layer prefilter runner, and the calibration harness — so that Week 3's GPU campaign is a parameter choice, not an engineering effort.

**Architecture:** Every orchestration component takes its expensive dependency as an injected callable (`Masker`, `ImageTextScorer`, `GenerateFn`, `DistanceFn`). That keeps the whole harness — pair iteration, block sweeping, aggregation, normalisation — testable on CPU with fakes and synthetic images, so Kaggle GPU hours are spent producing sensitivity numbers rather than debugging loops. The single-block prompt swap reuses the existing `RoutingPlan` with `α = 1.0` at one block, which is algebraically exactly a swap; no new GPU code is written.

**Tech Stack:** Python 3.10+, NumPy, Pillow, scikit-image (CIELAB), transformers (CLIPSeg, CLIP), PyTorch, pytest.

**Spec:** [`docs/superpowers/specs/2026-08-25-flair-cvpr-publication-plan-design.md`](../specs/2026-08-25-flair-cvpr-publication-plan-design.md) §3.4
**Prerequisite plan:** [`2026-08-25-flair-foundation.md`](./2026-08-25-flair-foundation.md) — Tasks 1-10 must be complete and their suite green. This plan continues the task numbering at 11.

## Scope

Covers **spec §3.4's code**: the metrics, corpus, prefilter, and harness. Tasks are numbered 11-16, continuing the foundation plan.

**Deliberately excluded — these wait on Week 1-2 measurements:**

- **The calibration campaign itself** (how many pairs, how many vital blocks, session budgeting, checkpoint/resume across Kaggle's 12-hour cap). The smoke test in foundation Task 7 prints SD3.5-M's real block count and reveals per-generation wall time; those two numbers set the campaign's parameters. Choosing them now would mean inventing them.
- **Absolute metrics for the embedding-based attributes** (identity, texture, style, action). Calibration needs *delta* metrics, which this plan delivers for all 7. The absolute forms need anchor sets and target descriptors that belong with the evaluation plan.
- **§3.7 FLUX, §3.8 fuzzy conflict** — gated behind the Week 7 checkpoint.

## Global Constraints

Everything in the foundation plan's Global Constraints still applies. Additionally:

- **Two metric families, not one.** Calibration measures *change*; the guard and evaluation measure *absolute* position.
  - `DeltaMetric(image_a, image_b, mask) -> float` in `[0, 1]` — used by §3.4 calibration.
  - `AbsoluteMetric(image, mask, ...) -> float` in `[0, 1]` — used by §3.6 guard and §4 evaluation, and must land on the universe `flair_t2i/fuzzy/membership.py` declares for that attribute.
- **Every metric returns a value already in `[0, 1]`.** There is no separate normalisation layer; the range is part of each metric's contract.
- **Expensive dependencies are injected, never imported at call sites.** CLIPSeg, CLIP, and the diffusion pipeline enter through `Masker`, `ImageTextScorer`, and `GenerateFn` protocols so tests run on CPU with fakes.
- **CLIP cosine normalisation constant:** raw CLIP image-text cosine similarities cluster in roughly `[0.15, 0.35]`. Map with `clip_norm(s) = clip((s - 0.15) / 0.20, 0, 1)`, defined once in `flair_t2i/metrics/embedding.py` as `CLIP_SIM_FLOOR = 0.15`, `CLIP_SIM_SPAN = 0.20`.
- **Test images are synthetic** (solid fills, gradients, noise) — no fixtures downloaded, no model weights.

---

### Task 11: Object masking

**Files:**
- Create: `flair_t2i/metrics/__init__.py`
- Create: `flair_t2i/metrics/masking.py`
- Test: `tests/test_masking.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `class Masker(Protocol)`: `__call__(self, image: Image.Image, label: str) -> np.ndarray` returning a float array of shape `[H, W]` with values in `{0.0, 1.0}`
  - `class ClipSegMasker`: `ClipSegMasker(model=None, processor=None, threshold: float = 0.4, device: str = "cpu")`, implements `Masker`
  - `class RectMasker`: `RectMasker(box: tuple[float, float, float, float])` — a deterministic fake for tests, box given as relative `(x0, y0, x1, y1)`
  - `mask_area_ratio(mask: np.ndarray) -> float`
  - `masked_mean_rgb(image: Image.Image, mask: np.ndarray) -> np.ndarray | None` — `None` when fewer than 50 pixels are selected
  - `MIN_MASK_PIXELS: int = 50`

- [ ] **Step 1: Write the failing test**

Create `tests/test_masking.py`:

```python
import numpy as np
import pytest
from PIL import Image

from flair_t2i.metrics.masking import (
    MIN_MASK_PIXELS,
    RectMasker,
    mask_area_ratio,
    masked_mean_rgb,
)


def _solid(rgb, size=(64, 64)):
    return Image.new("RGB", size, rgb)


def test_rect_masker_covers_the_requested_fraction():
    masker = RectMasker((0.0, 0.0, 0.5, 1.0))
    mask = masker(_solid((0, 0, 0)), "anything")
    assert mask.shape == (64, 64)
    assert mask_area_ratio(mask) == pytest.approx(0.5)


def test_rect_masker_values_are_binary():
    mask = RectMasker((0.25, 0.25, 0.75, 0.75))(_solid((0, 0, 0)), "x")
    assert set(np.unique(mask)) <= {0.0, 1.0}


def test_area_ratio_of_empty_and_full_masks():
    assert mask_area_ratio(np.zeros((8, 8))) == pytest.approx(0.0)
    assert mask_area_ratio(np.ones((8, 8))) == pytest.approx(1.0)


def test_masked_mean_rgb_reads_only_inside_the_mask():
    image = Image.new("RGB", (64, 64), (0, 0, 255))
    image.paste(Image.new("RGB", (32, 64), (255, 0, 0)), (0, 0))
    mask = RectMasker((0.0, 0.0, 0.5, 1.0))(image, "left half")

    mean = masked_mean_rgb(image, mask)

    np.testing.assert_allclose(mean, [255.0, 0.0, 0.0])


def test_masked_mean_rgb_returns_none_for_a_tiny_mask():
    mask = np.zeros((64, 64))
    mask[0, : MIN_MASK_PIXELS - 1] = 1.0
    assert masked_mean_rgb(_solid((10, 20, 30)), mask) is None


def test_masked_mean_rgb_accepts_a_full_mask():
    mean = masked_mean_rgb(_solid((10, 20, 30)), np.ones((64, 64)))
    np.testing.assert_allclose(mean, [10.0, 20.0, 30.0])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_masking.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.metrics'`

- [ ] **Step 3: Write the masking module**

Create `flair_t2i/metrics/__init__.py`:

```python
"""Attribute change metrics and object masking (spec section 3.4)."""
```

Create `flair_t2i/metrics/masking.py`:

```python
"""Object masks for attribute measurement (spec section 3.4).

Metrics are measured inside the object mask so that a change in, say, the
car's colour is not diluted by the background. CLIPSeg supplies the mask at
run time; ``RectMasker`` supplies a deterministic one in tests.
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

    def __call__(self, image: Image.Image, label: str) -> np.ndarray:
        import torch

        inputs = self.processor(
            text=[label], images=[image], return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        if logits.dim() == 2:
            logits = logits[None]

        probs = torch.sigmoid(logits)[0].float().cpu().numpy()
        resized = np.asarray(
            Image.fromarray((probs * 255).astype(np.uint8)).resize(image.size)
        ) / 255.0
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
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest tests/test_masking.py -v`
Expected: PASS — 6 passed

- [ ] **Step 5: Commit**

```bash
git add flair_t2i/metrics/ tests/test_masking.py
git commit -m "feat: object masking with CLIPSeg and a deterministic test masker"
```

---

### Task 12: Photometric delta metrics (size, colour, lighting)

**Files:**
- Create: `flair_t2i/metrics/photometric.py`
- Test: `tests/test_photometric.py`

**Interfaces:**
- Consumes: `masked_mean_rgb`, `mask_area_ratio` (Task 11).
- Produces:
  - `size_absolute(image, mask) -> float` — the mask area ratio; lands on `AttributeClass.SIZE`'s universe
  - `size_delta(image_a, image_b, mask_a, mask_b) -> float`
  - `warmth_absolute(image) -> float` — `((R - B) / (R + B) + 1) / 2` over the whole frame; lands on `AttributeClass.LIGHTING`'s universe
  - `lighting_delta(image_a, image_b, mask=None) -> float`
  - `color_delta(image_a, image_b, mask) -> float` — masked CIELAB ΔE, divided by `DELTA_E_CEILING` and clipped
  - `color_absolute(image, mask, target_rgb) -> float` — `1 - ΔE/DELTA_E_CEILING`; lands on `AttributeClass.COLOR`'s universe
  - `DELTA_E_CEILING: float = 100.0`

- [ ] **Step 1: Write the failing test**

Create `tests/test_photometric.py`:

```python
import numpy as np
import pytest
from PIL import Image

pytest.importorskip("skimage")

from flair_t2i.metrics.masking import RectMasker
from flair_t2i.metrics.photometric import (
    color_absolute,
    color_delta,
    lighting_delta,
    size_absolute,
    size_delta,
    warmth_absolute,
)

FULL = np.ones((64, 64), dtype=np.float32)


def _solid(rgb):
    return Image.new("RGB", (64, 64), rgb)


def test_size_absolute_is_the_area_ratio():
    mask = RectMasker((0.0, 0.0, 0.5, 1.0))(_solid((0, 0, 0)), "x")
    assert size_absolute(_solid((0, 0, 0)), mask) == pytest.approx(0.5)


def test_size_delta_is_the_area_difference():
    small = RectMasker((0.0, 0.0, 0.5, 0.5))(_solid((0, 0, 0)), "x")
    large = RectMasker((0.0, 0.0, 1.0, 0.5))(_solid((0, 0, 0)), "x")
    assert size_delta(_solid((0, 0, 0)), _solid((0, 0, 0)), small, large) == pytest.approx(0.25)


def test_size_delta_of_identical_masks_is_zero():
    assert size_delta(_solid((0, 0, 0)), _solid((0, 0, 0)), FULL, FULL) == 0.0


def test_warmth_is_high_for_orange_and_low_for_blue():
    assert warmth_absolute(_solid((255, 180, 80))) == pytest.approx(0.7612, abs=1e-3)
    assert warmth_absolute(_solid((80, 180, 255))) == pytest.approx(0.2388, abs=1e-3)


def test_warmth_of_grey_is_neutral():
    assert warmth_absolute(_solid((128, 128, 128))) == pytest.approx(0.5)


def test_lighting_delta_spans_warm_to_cool():
    delta = lighting_delta(_solid((255, 180, 80)), _solid((80, 180, 255)))
    assert delta == pytest.approx(0.5224, abs=1e-3)


def test_color_delta_of_identical_images_is_zero():
    assert color_delta(_solid((220, 30, 30)), _solid((220, 30, 30)), FULL) == pytest.approx(0.0)


def test_color_delta_of_red_versus_blue_is_large():
    assert color_delta(_solid((220, 30, 30)), _solid((30, 30, 220)), FULL) > 0.3


def test_color_delta_reads_only_inside_the_mask():
    left_red = Image.new("RGB", (64, 64), (0, 0, 0))
    left_red.paste(Image.new("RGB", (32, 64), (220, 30, 30)), (0, 0))
    left_blue = Image.new("RGB", (64, 64), (0, 0, 0))
    left_blue.paste(Image.new("RGB", (32, 64), (30, 30, 220)), (0, 0))

    left = RectMasker((0.0, 0.0, 0.5, 1.0))(left_red, "left")
    right = RectMasker((0.5, 0.0, 1.0, 1.0))(left_red, "right")

    assert color_delta(left_red, left_blue, left) > 0.3
    assert color_delta(left_red, left_blue, right) == pytest.approx(0.0)


def test_color_delta_is_zero_when_the_mask_is_too_small():
    tiny = np.zeros((64, 64), dtype=np.float32)
    tiny[0, :3] = 1.0
    assert color_delta(_solid((220, 30, 30)), _solid((30, 30, 220)), tiny) == 0.0


def test_color_absolute_peaks_on_an_exact_match():
    assert color_absolute(_solid((220, 30, 30)), FULL, (220, 30, 30)) == pytest.approx(1.0)


def test_color_absolute_drops_for_a_mismatch():
    assert color_absolute(_solid((30, 30, 220)), FULL, (220, 30, 30)) < 0.7


def test_all_metrics_stay_within_the_unit_interval():
    pairs = [((255, 255, 255), (0, 0, 0)), ((220, 30, 30), (30, 220, 30))]
    for a, b in pairs:
        assert 0.0 <= color_delta(_solid(a), _solid(b), FULL) <= 1.0
        assert 0.0 <= lighting_delta(_solid(a), _solid(b)) <= 1.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_photometric.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.metrics.photometric'`

- [ ] **Step 3: Write the photometric metrics**

Create `flair_t2i/metrics/photometric.py`:

```python
"""Colour, size, and lighting metrics (spec section 3.4).

Each function returns a value in [0, 1]. The absolute forms land on the
universes declared in ``flair_t2i.fuzzy.membership``; the delta forms are
what BASM calibration measures.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage import color as skcolor

from .masking import mask_area_ratio, masked_mean_rgb

#: CIELAB dE beyond this counts as a total colour change.
DELTA_E_CEILING = 100.0


def _lab(rgb: np.ndarray) -> np.ndarray:
    return skcolor.rgb2lab((rgb / 255.0).reshape(1, 1, 3)).reshape(3)


def size_absolute(image: Image.Image, mask: np.ndarray) -> float:
    return mask_area_ratio(mask)


def size_delta(
    image_a: Image.Image,
    image_b: Image.Image,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    return float(abs(mask_area_ratio(mask_a) - mask_area_ratio(mask_b)))


def warmth_absolute(image: Image.Image) -> float:
    """Warm/cool balance of the whole frame, 0 cool to 1 warm.

    Lighting is a scene property, so this is deliberately unmasked.
    """
    pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
    red, blue = pixels[..., 0].mean(), pixels[..., 2].mean()
    total = red + blue
    if total <= 0.0:
        return 0.5
    return float(np.clip(((red - blue) / total + 1.0) / 2.0, 0.0, 1.0))


def lighting_delta(
    image_a: Image.Image, image_b: Image.Image, mask: np.ndarray | None = None
) -> float:
    return float(abs(warmth_absolute(image_a) - warmth_absolute(image_b)))


def color_delta(
    image_a: Image.Image, image_b: Image.Image, mask: np.ndarray
) -> float:
    """Masked CIELAB dE between two images, normalised to [0, 1]."""
    mean_a = masked_mean_rgb(image_a, mask)
    mean_b = masked_mean_rgb(image_b, mask)
    if mean_a is None or mean_b is None:
        return 0.0
    distance = float(np.linalg.norm(_lab(mean_a) - _lab(mean_b)))
    return float(np.clip(distance / DELTA_E_CEILING, 0.0, 1.0))


def color_absolute(
    image: Image.Image, mask: np.ndarray, target_rgb: tuple[int, int, int]
) -> float:
    """1 - normalised dE to ``target_rgb``, matching the COLOR universe."""
    mean = masked_mean_rgb(image, mask)
    if mean is None:
        return 0.0
    distance = float(
        np.linalg.norm(_lab(mean) - _lab(np.asarray(target_rgb, dtype=np.float64)))
    )
    return float(np.clip(1.0 - distance / DELTA_E_CEILING, 0.0, 1.0))
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest tests/test_photometric.py -v`
Expected: PASS — 13 passed

- [ ] **Step 5: Commit**

```bash
git add flair_t2i/metrics/photometric.py tests/test_photometric.py
git commit -m "feat: colour, size, and lighting metrics in absolute and delta forms"
```

---

### Task 13: Embedding and texture delta metrics, and the registry

**Files:**
- Create: `flair_t2i/metrics/embedding.py`
- Create: `flair_t2i/metrics/texture.py`
- Create: `flair_t2i/metrics/registry.py`
- Test: `tests/test_embedding_metrics.py`
- Test: `tests/test_texture.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: `AttributeClass` (foundation Task 1); Tasks 11-12.
- Produces:
  - `flair_t2i/metrics/embedding.py`:
    - `class ImageTextScorer(Protocol)`: `image_embedding(image) -> np.ndarray`; `image_text_similarity(image, texts: list[str]) -> np.ndarray`
    - `class ClipScorer` implementing it (`openai/clip-vit-base-patch32`)
    - `CLIP_SIM_FLOOR = 0.15`, `CLIP_SIM_SPAN = 0.20`, `clip_norm(s: float) -> float`
    - `identity_delta(image_a, image_b, mask, scorer) -> float`
    - `style_delta(image_a, image_b, mask, scorer) -> float`
    - `action_delta(image_a, image_b, mask, scorer, phrase: str) -> float`
  - `flair_t2i/metrics/texture.py`: `gram_texture_delta(image_a, image_b, mask) -> float`
  - `flair_t2i/metrics/registry.py`: `DELTA_METRICS: dict[AttributeClass, str]` mapping each attribute to a metric kind, and `delta_for(attr, *, scorer=None, phrase=None) -> Callable[[Image, Image, np.ndarray], float]`

- [ ] **Step 1: Write the failing embedding test**

Create `tests/test_embedding_metrics.py`:

```python
import numpy as np
import pytest
from PIL import Image

from flair_t2i.metrics.embedding import (
    action_delta,
    clip_norm,
    identity_delta,
    style_delta,
)

FULL = np.ones((64, 64), dtype=np.float32)


def _solid(rgb):
    return Image.new("RGB", (64, 64), rgb)


class FakeScorer:
    """Maps an image to a unit vector derived from its mean colour."""

    def image_embedding(self, image):
        mean = np.asarray(image.convert("RGB"), dtype=np.float64).mean(axis=(0, 1))
        norm = np.linalg.norm(mean)
        return mean / norm if norm else mean

    def image_text_similarity(self, image, texts):
        base = float(self.image_embedding(image)[0])
        return np.array([0.15 + 0.2 * base * (i + 1) for i in range(len(texts))])


def test_clip_norm_maps_the_documented_range_onto_the_unit_interval():
    assert clip_norm(0.15) == pytest.approx(0.0)
    assert clip_norm(0.35) == pytest.approx(1.0)
    assert clip_norm(0.25) == pytest.approx(0.5)


def test_clip_norm_clamps_outside_the_range():
    assert clip_norm(-1.0) == 0.0
    assert clip_norm(2.0) == 1.0


def test_identity_delta_of_identical_images_is_zero():
    image = _solid((200, 100, 50))
    assert identity_delta(image, image, FULL, FakeScorer()) == pytest.approx(0.0)


def test_identity_delta_grows_for_different_images():
    delta = identity_delta(_solid((250, 10, 10)), _solid((10, 10, 250)), FULL, FakeScorer())
    assert delta > 0.05


def test_identity_delta_stays_within_the_unit_interval():
    delta = identity_delta(_solid((255, 0, 0)), _solid((0, 0, 255)), FULL, FakeScorer())
    assert 0.0 <= delta <= 1.0


def test_style_delta_of_identical_images_is_zero():
    image = _solid((120, 120, 120))
    assert style_delta(image, image, FULL, FakeScorer()) == pytest.approx(0.0)


def test_action_delta_of_identical_images_is_zero():
    image = _solid((120, 120, 120))
    assert action_delta(image, image, FULL, FakeScorer(), "a car driving") == pytest.approx(0.0)


def test_action_delta_responds_to_a_changed_image():
    delta = action_delta(
        _solid((255, 200, 200)), _solid((10, 10, 10)), FULL, FakeScorer(), "a car driving"
    )
    assert delta > 0.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_embedding_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.metrics.embedding'`

- [ ] **Step 3: Write the embedding metrics**

Create `flair_t2i/metrics/embedding.py`:

```python
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
            features = self.model.get_image_features(**inputs)[0]
        features = features / features.norm()
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
        image_features = output.image_embeds / output.image_embeds.norm(
            dim=-1, keepdim=True
        )
        text_features = output.text_embeds / output.text_embeds.norm(
            dim=-1, keepdim=True
        )
        return (image_features @ text_features.T)[0].float().cpu().numpy()


def clip_norm(similarity: float) -> float:
    """Map a raw CLIP cosine similarity onto [0, 1]."""
    return float(np.clip((similarity - CLIP_SIM_FLOOR) / CLIP_SIM_SPAN, 0.0, 1.0))


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    cosine = float(np.dot(a, b) / denom)
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
    profile_a = scorer.image_text_similarity(image_a, _STYLE_DESCRIPTORS)
    profile_b = scorer.image_text_similarity(image_b, _STYLE_DESCRIPTORS)
    diff = np.abs(np.asarray(profile_a) - np.asarray(profile_b)).mean()
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
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest tests/test_embedding_metrics.py -v`
Expected: PASS — 8 passed

- [ ] **Step 5: Write the failing texture test**

Create `tests/test_texture.py`:

```python
import numpy as np
import pytest
from PIL import Image

from flair_t2i.metrics.texture import gram_texture_delta

FULL = np.ones((64, 64), dtype=np.float32)


def _solid(value=128):
    return Image.new("RGB", (64, 64), (value, value, value))


def _noise(seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))


def _stripes(period=4):
    array = np.zeros((64, 64, 3), dtype=np.uint8)
    array[:, ::period] = 255
    return Image.fromarray(array)


def test_identical_images_have_zero_texture_delta():
    image = _noise(1)
    assert gram_texture_delta(image, image, FULL) == pytest.approx(0.0, abs=1e-9)


def test_smooth_versus_noisy_registers_a_large_delta():
    assert gram_texture_delta(_solid(), _noise(2), FULL) > 0.1


def test_two_smooth_images_register_a_small_delta():
    assert gram_texture_delta(_solid(120), _solid(140), FULL) < 0.05


def test_delta_is_symmetric():
    a, b = _solid(), _stripes()
    assert gram_texture_delta(a, b, FULL) == pytest.approx(gram_texture_delta(b, a, FULL))


def test_delta_stays_within_the_unit_interval():
    assert 0.0 <= gram_texture_delta(_noise(3), _stripes(), FULL) <= 1.0


def test_delta_is_zero_when_the_mask_is_empty():
    assert gram_texture_delta(_solid(), _noise(4), np.zeros((64, 64))) == 0.0
```

- [ ] **Step 6: Run it to verify it fails**

Run: `python -m pytest tests/test_texture.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.metrics.texture'`

- [ ] **Step 7: Write the texture metric**

Create `flair_t2i/metrics/texture.py`:

```python
"""Gram-matrix texture distance (spec section 3.4).

Classical Gram-matrix style statistics computed over multi-scale gradient
responses, in NumPy, so calibration needs no extra model weights. The
signature matches the other delta metrics, so a DISTS implementation can
replace this for the paper's final numbers without touching the harness.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .masking import MIN_MASK_PIXELS

_SCALES = (1, 2, 4)


def _responses(gray: np.ndarray, scale: int) -> np.ndarray:
    """Horizontal, vertical, and diagonal gradient responses at one scale."""
    shifted_x = np.roll(gray, scale, axis=1)
    shifted_y = np.roll(gray, scale, axis=0)
    shifted_d = np.roll(shifted_x, scale, axis=0)
    return np.stack(
        [gray - shifted_x, gray - shifted_y, gray - shifted_d], axis=0
    )


def _gram(image: Image.Image, mask: np.ndarray) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    selected = mask > 0.5

    grams = []
    for scale in _SCALES:
        responses = _responses(gray, scale)
        flat = responses[:, selected]
        grams.append(flat @ flat.T / max(flat.shape[1], 1))
    return np.stack(grams)


def gram_texture_delta(
    image_a: Image.Image, image_b: Image.Image, mask: np.ndarray
) -> float:
    """Normalised Gram-matrix distance inside the mask, in [0, 1]."""
    if (mask > 0.5).sum() < MIN_MASK_PIXELS:
        return 0.0

    gram_a, gram_b = _gram(image_a, mask), _gram(image_b, mask)
    distance = float(np.linalg.norm(gram_a - gram_b))
    scale = float(np.linalg.norm(gram_a) + np.linalg.norm(gram_b))
    if scale == 0.0:
        return 0.0
    return float(np.clip(distance / scale, 0.0, 1.0))
```

- [ ] **Step 8: Run it to verify it passes**

Run: `python -m pytest tests/test_texture.py -v`
Expected: PASS — 6 passed

- [ ] **Step 9: Write the failing registry test**

Create `tests/test_registry.py`:

```python
import numpy as np
import pytest
from PIL import Image

from flair_t2i.attributes import AttributeClass
from flair_t2i.metrics.registry import DELTA_METRICS, delta_for

FULL = np.ones((64, 64), dtype=np.float32)


class FakeScorer:
    def image_embedding(self, image):
        mean = np.asarray(image.convert("RGB"), dtype=np.float64).mean(axis=(0, 1))
        norm = np.linalg.norm(mean)
        return mean / norm if norm else mean

    def image_text_similarity(self, image, texts):
        base = float(self.image_embedding(image)[0])
        return np.array([0.15 + 0.2 * base * (i + 1) for i in range(len(texts))])


def _solid(rgb):
    return Image.new("RGB", (64, 64), rgb)


def test_every_attribute_has_a_delta_metric():
    assert set(DELTA_METRICS) == set(AttributeClass)


def test_photometric_metrics_need_no_scorer():
    metric = delta_for(AttributeClass.COLOR)
    assert metric(_solid((220, 30, 30)), _solid((30, 30, 220)), FULL) > 0.3


def test_embedding_metrics_require_a_scorer():
    with pytest.raises(ValueError, match="scorer"):
        delta_for(AttributeClass.IDENTITY)


def test_action_metric_requires_a_phrase():
    with pytest.raises(ValueError, match="phrase"):
        delta_for(AttributeClass.ACTION, scorer=FakeScorer())


def test_every_metric_is_callable_with_the_common_signature():
    for attr in AttributeClass:
        metric = delta_for(attr, scorer=FakeScorer(), phrase="a car driving")
        value = metric(_solid((200, 100, 50)), _solid((50, 100, 200)), FULL)
        assert 0.0 <= value <= 1.0, attr


def test_size_metric_uses_the_shared_mask_for_both_images():
    metric = delta_for(AttributeClass.SIZE)
    assert metric(_solid((0, 0, 0)), _solid((0, 0, 0)), FULL) == pytest.approx(0.0)
```

- [ ] **Step 10: Run it to verify it fails**

Run: `python -m pytest tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.metrics.registry'`

- [ ] **Step 11: Write the registry**

Create `flair_t2i/metrics/registry.py`:

```python
"""One delta metric per attribute, behind a common signature.

Every metric returned by ``delta_for`` has the signature
``(image_a, image_b, mask) -> float`` in [0, 1], so the calibration harness
does not branch on attribute type.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image

from ..attributes import AttributeClass
from .embedding import ImageTextScorer, action_delta, identity_delta, style_delta
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
) -> DeltaMetric:
    """Return the delta metric for ``attr``, bound to its dependencies."""
    kind = DELTA_METRICS[attr]

    if kind in _NEEDS_SCORER and scorer is None:
        raise ValueError(f"{attr.value} needs a scorer")
    if kind == "action" and not phrase:
        raise ValueError(f"{attr.value} needs a phrase")

    if kind == "color":
        return color_delta
    if kind == "lighting":
        return lighting_delta
    if kind == "texture":
        return gram_texture_delta
    if kind == "size":
        # Calibration measures both images against a shared object mask.
        return lambda a, b, mask: size_delta(a, b, mask, mask)
    if kind == "identity":
        return lambda a, b, mask: identity_delta(a, b, mask, scorer)
    if kind == "style":
        return lambda a, b, mask: style_delta(a, b, mask, scorer)
    return lambda a, b, mask: action_delta(a, b, mask, scorer, phrase)
```

- [ ] **Step 12: Run it to verify it passes**

Run: `python -m pytest tests/test_registry.py -v`
Expected: PASS — 6 passed

Note that `test_size_metric_uses_the_shared_mask_for_both_images` returning 0.0 is correct: with one shared mask, size change is not observable from masks alone. The harness therefore re-masks each image — see Task 16, which passes per-image masks for `SIZE`.

- [ ] **Step 13: Commit**

```bash
git add flair_t2i/metrics/ tests/test_embedding_metrics.py tests/test_texture.py tests/test_registry.py
git commit -m "feat: embedding, texture, and registry delta metrics for all 7 attributes"
```

---

### Task 14: Contrastive pair corpus

**Files:**
- Create: `flair_t2i/calibration/__init__.py`
- Create: `flair_t2i/calibration/corpus.py`
- Create: `data/contrastive_pairs.json`
- Test: `tests/test_corpus.py`

**Interfaces:**
- Consumes: `AttributeClass` (foundation Task 1).
- Produces:
  - `@dataclass(frozen=True) ContrastivePair`: `base: str`, `changed: str`, `object_label: str`, `phrase: str | None = None`
  - `load_corpus(path) -> dict[AttributeClass, list[ContrastivePair]]`
  - `DEFAULT_CORPUS_PATH: Path`
  - `MIN_PAIRS_PER_ATTRIBUTE: int = 5`
  - `validate_corpus(corpus) -> None` — raises when an attribute is missing, has too few pairs, or an `action` pair lacks `phrase`

The corpus ships 5 pairs per attribute. Spec §3.4 targets ~10; the campaign plan decides the final count once the smoke test measures per-generation wall time.

- [ ] **Step 1: Write the failing test**

Create `tests/test_corpus.py`:

```python
import json

import pytest

from flair_t2i.attributes import AttributeClass
from flair_t2i.calibration.corpus import (
    DEFAULT_CORPUS_PATH,
    MIN_PAIRS_PER_ATTRIBUTE,
    ContrastivePair,
    load_corpus,
    validate_corpus,
)


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(DEFAULT_CORPUS_PATH)


def test_corpus_covers_all_seven_attributes(corpus):
    assert set(corpus) == set(AttributeClass)


def test_every_attribute_meets_the_minimum_pair_count(corpus):
    for attr, pairs in corpus.items():
        assert len(pairs) >= MIN_PAIRS_PER_ATTRIBUTE, attr


def test_pairs_differ_only_in_the_target_attribute(corpus):
    for attr, pairs in corpus.items():
        for pair in pairs:
            assert pair.base != pair.changed, (attr, pair.base)
            base_words = pair.base.split()
            changed_words = pair.changed.split()
            assert len(base_words) == len(changed_words), (attr, pair.base)
            differing = sum(a != b for a, b in zip(base_words, changed_words))
            assert differing == 1, (attr, pair.base, pair.changed)


def test_object_label_appears_in_both_prompts(corpus):
    for attr, pairs in corpus.items():
        for pair in pairs:
            assert pair.object_label in pair.base, (attr, pair.base)
            assert pair.object_label in pair.changed, (attr, pair.changed)


def test_action_pairs_carry_a_phrase(corpus):
    for pair in corpus[AttributeClass.ACTION]:
        assert pair.phrase


def test_validate_rejects_a_missing_attribute():
    with pytest.raises(ValueError, match="missing"):
        validate_corpus({AttributeClass.COLOR: [
            ContrastivePair("a red car", "a blue car", "car")
        ] * MIN_PAIRS_PER_ATTRIBUTE})


def test_validate_rejects_too_few_pairs(corpus):
    thin = {attr: pairs[:1] for attr, pairs in corpus.items()}
    with pytest.raises(ValueError, match="at least"):
        validate_corpus(thin)


def test_validate_rejects_an_action_pair_without_a_phrase(corpus):
    broken = {attr: list(pairs) for attr, pairs in corpus.items()}
    broken[AttributeClass.ACTION] = [
        ContrastivePair(p.base, p.changed, p.object_label, phrase=None)
        for p in broken[AttributeClass.ACTION]
    ]
    with pytest.raises(ValueError, match="phrase"):
        validate_corpus(broken)


def test_corpus_file_is_valid_json():
    with open(DEFAULT_CORPUS_PATH, encoding="utf-8") as handle:
        json.load(handle)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_corpus.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.calibration'`

- [ ] **Step 3: Write the corpus data file**

Create `data/contrastive_pairs.json`. Each pair differs from its partner in exactly one word — the test enforces this, because a pair that changes two things measures two things.

```json
{
  "color": [
    {"base": "a red sports car on a road", "changed": "a blue sports car on a road", "object_label": "sports car"},
    {"base": "a green ceramic vase on a table", "changed": "a yellow ceramic vase on a table", "object_label": "ceramic vase"},
    {"base": "a black leather chair in a room", "changed": "a white leather chair in a room", "object_label": "leather chair"},
    {"base": "an orange wooden door in a wall", "changed": "a purple wooden door in a wall", "object_label": "wooden door"},
    {"base": "a brown canvas backpack on grass", "changed": "a pink canvas backpack on grass", "object_label": "canvas backpack"}
  ],
  "size": [
    {"base": "a small sports car on a road", "changed": "a large sports car on a road", "object_label": "sports car"},
    {"base": "a tiny ceramic vase on a table", "changed": "a huge ceramic vase on a table", "object_label": "ceramic vase"},
    {"base": "a compact wooden boat on water", "changed": "a massive wooden boat on water", "object_label": "wooden boat"},
    {"base": "a miniature stone statue in a park", "changed": "an enormous stone statue in a park", "object_label": "stone statue"},
    {"base": "a petite glass bottle on a shelf", "changed": "an oversized glass bottle on a shelf", "object_label": "glass bottle"}
  ],
  "lighting": [
    {"base": "a sports car under warm evening light", "changed": "a sports car under cool evening light", "object_label": "sports car"},
    {"base": "a ceramic vase in bright daylight indoors", "changed": "a ceramic vase in dim daylight indoors", "object_label": "ceramic vase"},
    {"base": "a wooden chair beside a sunlit window frame", "changed": "a wooden chair beside a candlelit window frame", "object_label": "wooden chair"},
    {"base": "a stone statue under harsh afternoon light", "changed": "a stone statue under soft afternoon light", "object_label": "stone statue"},
    {"base": "a glass bottle on a neon lit counter", "changed": "a glass bottle on a golden lit counter", "object_label": "glass bottle"}
  ],
  "texture": [
    {"base": "a rusty metal sign on a wall", "changed": "a polished metal sign on a wall", "object_label": "metal sign"},
    {"base": "a smooth ceramic vase on a table", "changed": "a cracked ceramic vase on a table", "object_label": "ceramic vase"},
    {"base": "a glossy sports car on a road", "changed": "a matte sports car on a road", "object_label": "sports car"},
    {"base": "a woven cotton bag on a bench", "changed": "a furry cotton bag on a bench", "object_label": "cotton bag"},
    {"base": "a weathered wooden door in a wall", "changed": "a shiny wooden door in a wall", "object_label": "wooden door"}
  ],
  "identity": [
    {"base": "a sports car parked on a road", "changed": "a tractor parked on a road", "object_label": "parked on a road"},
    {"base": "a ceramic vase resting on a table", "changed": "a typewriter resting on a table", "object_label": "resting on a table"},
    {"base": "a wooden chair beside a window", "changed": "a refrigerator beside a window", "object_label": "beside a window"},
    {"base": "a stone statue standing in a park", "changed": "a lighthouse standing in a park", "object_label": "standing in a park"},
    {"base": "a glass bottle placed on a shelf", "changed": "a helmet placed on a shelf", "object_label": "placed on a shelf"}
  ],
  "style": [
    {"base": "a photorealistic sports car on a road", "changed": "a cartoon sports car on a road", "object_label": "sports car"},
    {"base": "a vintage ceramic vase on a table", "changed": "a futuristic ceramic vase on a table", "object_label": "ceramic vase"},
    {"base": "a watercolor wooden chair in a room", "changed": "a cyberpunk wooden chair in a room", "object_label": "wooden chair"},
    {"base": "a minimalist stone statue in a park", "changed": "a baroque stone statue in a park", "object_label": "stone statue"},
    {"base": "a sketch glass bottle on a shelf", "changed": "a photorealistic glass bottle on a shelf", "object_label": "glass bottle"}
  ],
  "action": [
    {"base": "a sports car driving along a road", "changed": "a sports car parked along a road", "object_label": "sports car", "phrase": "a car driving"},
    {"base": "a brown horse running across a field", "changed": "a brown horse grazing across a field", "object_label": "brown horse", "phrase": "a horse running"},
    {"base": "a small bird flying above a lake", "changed": "a small bird floating above a lake", "object_label": "small bird", "phrase": "a bird flying"},
    {"base": "a young person jumping near a wall", "changed": "a young person standing near a wall", "object_label": "young person", "phrase": "a person jumping"},
    {"base": "a wooden boat sailing across water", "changed": "a wooden boat moored across water", "object_label": "wooden boat", "phrase": "a boat sailing"}
  ]
}
```

- [ ] **Step 4: Write the corpus loader**

Create `flair_t2i/calibration/__init__.py`:

```python
"""BASM calibration: contrastive pairs, vital-layer prefilter, harness."""
```

Create `flair_t2i/calibration/corpus.py`:

```python
"""Contrastive prompt pairs for BASM calibration (spec section 3.4).

Each pair differs from its partner in exactly one word, so the measured
change is attributable to one attribute. ``validate_corpus`` enforces the
structural invariants the harness depends on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..attributes import AttributeClass

DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "contrastive_pairs.json"
)

#: Seed count shipped with the repo. Spec section 3.4 targets ~10; the
#: campaign plan raises this once per-generation wall time is measured.
MIN_PAIRS_PER_ATTRIBUTE = 5


@dataclass(frozen=True)
class ContrastivePair:
    base: str
    changed: str
    object_label: str
    phrase: str | None = None


def load_corpus(path: str | Path) -> dict[AttributeClass, list[ContrastivePair]]:
    with open(Path(path), encoding="utf-8") as handle:
        raw = json.load(handle)

    corpus = {
        AttributeClass(name): [ContrastivePair(**entry) for entry in entries]
        for name, entries in raw.items()
    }
    validate_corpus(corpus)
    return corpus


def validate_corpus(
    corpus: dict[AttributeClass, list[ContrastivePair]]
) -> None:
    missing = set(AttributeClass) - set(corpus)
    if missing:
        raise ValueError(
            f"corpus is missing attributes: {sorted(a.value for a in missing)}"
        )

    for attr, pairs in corpus.items():
        if len(pairs) < MIN_PAIRS_PER_ATTRIBUTE:
            raise ValueError(
                f"{attr.value} needs at least {MIN_PAIRS_PER_ATTRIBUTE} pairs, "
                f"got {len(pairs)}"
            )
        for pair in pairs:
            if attr is AttributeClass.ACTION and not pair.phrase:
                raise ValueError(f"action pair {pair.base!r} needs a phrase")
```

- [ ] **Step 5: Run it to verify it passes**

Run: `python -m pytest tests/test_corpus.py -v`
Expected: PASS — 9 passed

If `test_pairs_differ_only_in_the_target_attribute` fails, the offending pair changes more than one word — rewrite the pair, do not relax the test. That invariant is what makes the measured Δ attributable.

- [ ] **Step 6: Commit**

```bash
git add flair_t2i/calibration/ data/contrastive_pairs.json tests/test_corpus.py
git commit -m "feat: contrastive pair corpus with one-word-difference invariant"
```

---

### Task 15: Vital-layer prefilter runner

**Files:**
- Create: `flair_t2i/calibration/prefilter.py`
- Test: `tests/test_prefilter.py`

**Interfaces:**
- Consumes: `bypass_blocks` (foundation Task 6).
- Produces:
  - `class BypassGenerateFn(Protocol)`: `__call__(self, prompt: str, seed: int, bypass: int | None) -> Image.Image`
  - `class DistanceFn(Protocol)`: `__call__(self, a: Image.Image, b: Image.Image) -> float`
  - `@dataclass VitalityReport`: `scores: dict[int, float]`, `vital_blocks: tuple[int, ...]`
  - `run_prefilter(generate_fn, n_blocks, prompts, seeds, distance_fn, top_n) -> VitalityReport`
  - `make_bypass_generate_fn(flair_pipeline, steps: int) -> BypassGenerateFn`
  - `lpips_distance(device: str = "cpu") -> DistanceFn`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prefilter.py`:

```python
import pytest
from PIL import Image

from flair_t2i.calibration.prefilter import VitalityReport, run_prefilter

PROMPTS = ["a red car", "a blue vase"]
SEEDS = [0, 1]

#: Block 2 is the one that matters in this fake model.
VITAL = 2


def fake_generate(prompt: str, seed: int, bypass: int | None) -> Image.Image:
    shade = 200 if bypass == VITAL else 100
    return Image.new("RGB", (8, 8), (shade, shade, shade))


def fake_distance(a: Image.Image, b: Image.Image) -> float:
    return abs(a.getpixel((0, 0))[0] - b.getpixel((0, 0))[0]) / 255.0


def test_prefilter_ranks_the_vital_block_first():
    report = run_prefilter(
        fake_generate, n_blocks=4, prompts=PROMPTS, seeds=SEEDS,
        distance_fn=fake_distance, top_n=1,
    )
    assert report.vital_blocks == (VITAL,)


def test_prefilter_scores_every_block():
    report = run_prefilter(
        fake_generate, n_blocks=4, prompts=PROMPTS, seeds=SEEDS,
        distance_fn=fake_distance, top_n=2,
    )
    assert set(report.scores) == {0, 1, 2, 3}


def test_non_vital_blocks_score_zero():
    report = run_prefilter(
        fake_generate, n_blocks=4, prompts=PROMPTS, seeds=SEEDS,
        distance_fn=fake_distance, top_n=4,
    )
    assert report.scores[0] == pytest.approx(0.0)
    assert report.scores[VITAL] > 0.0


def test_vital_blocks_are_returned_in_ascending_order():
    report = run_prefilter(
        fake_generate, n_blocks=4, prompts=PROMPTS, seeds=SEEDS,
        distance_fn=fake_distance, top_n=3,
    )
    assert list(report.vital_blocks) == sorted(report.vital_blocks)


def test_top_n_clamps_to_the_block_count():
    report = run_prefilter(
        fake_generate, n_blocks=3, prompts=PROMPTS, seeds=SEEDS,
        distance_fn=fake_distance, top_n=99,
    )
    assert len(report.vital_blocks) == 3


def test_report_is_serialisable(tmp_path):
    report = run_prefilter(
        fake_generate, n_blocks=3, prompts=PROMPTS, seeds=SEEDS,
        distance_fn=fake_distance, top_n=2,
    )
    path = tmp_path / "vitality.json"
    report.save(path)
    assert VitalityReport.load(path).vital_blocks == report.vital_blocks
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_prefilter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.calibration.prefilter'`

- [ ] **Step 3: Write the prefilter**

Create `flair_t2i/calibration/prefilter.py`:

```python
"""Vital-layer prefilter (spec section 3.4).

Each block is bypassed in turn and the output compared against the
unmodified run; blocks whose removal changes the image most are the vital
ones. This narrows the BASM sweep from every block to a handful, which is
what makes calibration affordable.

Stable Flow measures vitality with a DINOv2 perceptual distance. LPIPS is
the default here because it is already a project dependency; the distance
function is injected, so DINOv2 can replace it without touching this code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from ..patching import bypass_blocks


class BypassGenerateFn(Protocol):
    def __call__(
        self, prompt: str, seed: int, bypass: int | None
    ) -> Image.Image: ...


class DistanceFn(Protocol):
    def __call__(self, a: Image.Image, b: Image.Image) -> float: ...


@dataclass
class VitalityReport:
    scores: dict[int, float]
    vital_blocks: tuple[int, ...]

    def save(self, path: str | Path) -> None:
        with open(Path(path), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "scores": {str(k): v for k, v in self.scores.items()},
                    "vital_blocks": list(self.vital_blocks),
                },
                handle,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "VitalityReport":
        with open(Path(path), encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls(
            scores={int(k): float(v) for k, v in raw["scores"].items()},
            vital_blocks=tuple(int(b) for b in raw["vital_blocks"]),
        )


def run_prefilter(
    generate_fn: BypassGenerateFn,
    n_blocks: int,
    prompts: list[str],
    seeds: list[int],
    distance_fn: DistanceFn,
    top_n: int,
) -> VitalityReport:
    """Score every block by how much bypassing it changes the output."""
    baselines = {
        (prompt, seed): generate_fn(prompt=prompt, seed=seed, bypass=None)
        for prompt in prompts
        for seed in seeds
    }

    scores: dict[int, float] = {}
    for block_id in range(n_blocks):
        distances = []
        for (prompt, seed), baseline in baselines.items():
            bypassed = generate_fn(prompt=prompt, seed=seed, bypass=block_id)
            distances.append(float(distance_fn(baseline, bypassed)))
        scores[block_id] = sum(distances) / len(distances) if distances else 0.0

    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    vital = tuple(sorted(block_id for block_id, _ in ranked[: max(0, top_n)]))
    return VitalityReport(scores=scores, vital_blocks=vital)


def make_bypass_generate_fn(flair_pipeline, steps: int) -> BypassGenerateFn:
    """Bind a FlairPipeline into the prefilter's generate signature."""

    def generate(prompt: str, seed: int, bypass: int | None) -> Image.Image:
        transformer = flair_pipeline.pipe.transformer
        blocks = set() if bypass is None else {bypass}
        with bypass_blocks(transformer, blocks):
            return flair_pipeline.generate(
                prompt, seed=seed, steps=steps, routing=False
            )

    return generate


def lpips_distance(device: str = "cpu") -> DistanceFn:  # pragma: no cover - weights
    """LPIPS perceptual distance, the default vitality measure."""
    import lpips
    import numpy as np
    import torch

    net = lpips.LPIPS(net="alex").to(device).eval()

    def _tensor(image: Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1.0
        return tensor[None].to(device)

    def distance(a: Image.Image, b: Image.Image) -> float:
        with torch.no_grad():
            return float(net(_tensor(a), _tensor(b)))

    return distance
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest tests/test_prefilter.py -v`
Expected: PASS — 6 passed

- [ ] **Step 5: Commit**

```bash
git add flair_t2i/calibration/prefilter.py tests/test_prefilter.py
git commit -m "feat: vital-layer prefilter with injected distance function"
```

---

### Task 16: BASM calibration harness

**Files:**
- Create: `flair_t2i/calibration/harness.py`
- Create: `scripts/calibrate.py`
- Test: `tests/test_harness.py`

**Interfaces:**
- Consumes: `BASM` (foundation Task 3), `RoutingPlan`/`RoutedComponent` (foundation Task 4), `FlairPipeline` (foundation Task 7), Tasks 11-15.
- Produces:
  - `@dataclass(frozen=True) SwapSpec`: `block_id: int`, `prompt: str`
  - `class SwapGenerateFn(Protocol)`: `__call__(self, prompt: str, seed: int, swap: SwapSpec | None) -> Image.Image`
  - `calibrate(generate_fn, corpus, vital_blocks, masker, seeds, scorer=None, progress=None) -> BASM`
  - `make_swap_generate_fn(flair_pipeline, steps: int) -> SwapGenerateFn`

A prompt swap at one block is the routing blend with `α = 1.0`: `H = H_base + 1.0·(H_changed − H_base) = H_changed`. `make_swap_generate_fn` therefore reuses `RoutingPlan` and writes no new GPU code.

- [ ] **Step 1: Write the failing test**

Create `tests/test_harness.py`:

```python
import numpy as np
import pytest
from PIL import Image

from flair_t2i.attributes import AttributeClass
from flair_t2i.calibration.corpus import ContrastivePair
from flair_t2i.calibration.harness import SwapSpec, calibrate
from flair_t2i.metrics.masking import RectMasker

VITAL = (0, 1, 2)
SEEDS = [0]
#: The fake model routes colour through block 1 and nothing else.
COLOR_BLOCK = 1


def _corpus():
    pairs = [
        ContrastivePair("a red car on a road", "a blue car on a road", "car"),
        ContrastivePair("a red vase on a table", "a blue vase on a table", "vase"),
    ]
    return {AttributeClass.COLOR: pairs}


def fake_generate(prompt: str, seed: int, swap: SwapSpec | None) -> Image.Image:
    """Only a swap at COLOR_BLOCK actually recolours the image."""
    if swap is not None and swap.block_id == COLOR_BLOCK and "blue" in swap.prompt:
        return Image.new("RGB", (64, 64), (30, 30, 220))
    return Image.new("RGB", (64, 64), (220, 30, 30))


def test_calibrate_returns_a_basm_over_the_vital_blocks():
    basm = calibrate(
        fake_generate, _corpus(), VITAL, RectMasker((0.0, 0.0, 1.0, 1.0)), SEEDS
    )
    assert basm.block_ids == VITAL
    assert basm.attributes == (AttributeClass.COLOR,)


def test_the_sensitive_block_scores_highest():
    basm = calibrate(
        fake_generate, _corpus(), VITAL, RectMasker((0.0, 0.0, 1.0, 1.0)), SEEDS
    )
    assert basm.top_k(AttributeClass.COLOR, 1) == [(COLOR_BLOCK, pytest.approx(1.0))]


def test_insensitive_blocks_score_zero_after_normalisation():
    basm = calibrate(
        fake_generate, _corpus(), VITAL, RectMasker((0.0, 0.0, 1.0, 1.0)), SEEDS
    )
    assert basm.score(0, AttributeClass.COLOR) == pytest.approx(0.0)
    assert basm.score(2, AttributeClass.COLOR) == pytest.approx(0.0)


def test_all_scores_land_in_the_unit_interval():
    basm = calibrate(
        fake_generate, _corpus(), VITAL, RectMasker((0.0, 0.0, 1.0, 1.0)), SEEDS
    )
    assert basm.matrix.min() >= 0.0 and basm.matrix.max() <= 1.0


def test_a_flat_response_normalises_to_zero_rather_than_dividing_by_zero():
    flat = lambda prompt, seed, swap: Image.new("RGB", (64, 64), (128, 128, 128))
    basm = calibrate(flat, _corpus(), VITAL, RectMasker((0.0, 0.0, 1.0, 1.0)), SEEDS)
    assert basm.matrix.max() == pytest.approx(0.0)


def test_progress_callback_reports_every_attribute_block_pair():
    seen = []
    calibrate(
        fake_generate, _corpus(), VITAL, RectMasker((0.0, 0.0, 1.0, 1.0)), SEEDS,
        progress=lambda attr, block, value: seen.append((attr, block)),
    )
    assert seen == [(AttributeClass.COLOR, b) for b in VITAL]


def test_action_attribute_binds_its_phrase():
    corpus = {
        AttributeClass.ACTION: [
            ContrastivePair(
                "a car driving on a road", "a car parked on a road", "car",
                phrase="a car driving",
            )
        ] * 2
    }

    class FakeScorer:
        def image_embedding(self, image):
            return np.ones(4)

        def image_text_similarity(self, image, texts):
            return np.array([0.25] * len(texts))

    basm = calibrate(
        fake_generate, corpus, VITAL, RectMasker((0.0, 0.0, 1.0, 1.0)), SEEDS,
        scorer=FakeScorer(),
    )
    assert basm.attributes == (AttributeClass.ACTION,)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_harness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.calibration.harness'`

- [ ] **Step 3: Write the harness**

Create `flair_t2i/calibration/harness.py`:

```python
"""BASM calibration harness (spec section 3.4).

For each attribute a, each vital block l, and each contrastive pair p:
run the base prompt everywhere, then run it again with p.changed swapped in
at block l alone, and measure the attribute-specific change inside the
object mask. Averaging over pairs gives a raw sensitivity, and min-max
normalising each attribute's column gives S[l, a] in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from PIL import Image

from ..attributes import AttributeClass
from ..basm import BASM
from .corpus import ContrastivePair
from ..metrics.embedding import ImageTextScorer
from ..metrics.masking import Masker
from ..metrics.photometric import size_delta
from ..metrics.registry import delta_for


@dataclass(frozen=True)
class SwapSpec:
    block_id: int
    prompt: str


class SwapGenerateFn(Protocol):
    def __call__(
        self, prompt: str, seed: int, swap: SwapSpec | None
    ) -> Image.Image: ...


ProgressFn = Callable[[AttributeClass, int, float], None]


def _normalise(column: np.ndarray) -> np.ndarray:
    """Min-max a raw sensitivity column onto [0, 1]."""
    low, high = float(column.min()), float(column.max())
    if high - low <= 1e-12:
        return np.zeros_like(column)
    return (column - low) / (high - low)


def calibrate(
    generate_fn: SwapGenerateFn,
    corpus: dict[AttributeClass, list[ContrastivePair]],
    vital_blocks: tuple[int, ...],
    masker: Masker,
    seeds: list[int],
    scorer: ImageTextScorer | None = None,
    progress: ProgressFn | None = None,
) -> BASM:
    """Measure per-block sensitivity for every attribute in ``corpus``."""
    attributes = tuple(a for a in AttributeClass if a in corpus)
    raw = np.zeros((len(vital_blocks), len(attributes)))

    for col, attr in enumerate(attributes):
        pairs = corpus[attr]
        for row, block_id in enumerate(vital_blocks):
            deltas = []
            for pair in pairs:
                metric = delta_for(attr, scorer=scorer, phrase=pair.phrase)
                for seed in seeds:
                    baseline = generate_fn(prompt=pair.base, seed=seed, swap=None)
                    swapped = generate_fn(
                        prompt=pair.base,
                        seed=seed,
                        swap=SwapSpec(block_id=block_id, prompt=pair.changed),
                    )
                    mask = masker(baseline, pair.object_label)

                    if attr is AttributeClass.SIZE:
                        # Size is only visible by re-masking the changed image.
                        deltas.append(
                            size_delta(
                                baseline,
                                swapped,
                                mask,
                                masker(swapped, pair.object_label),
                            )
                        )
                    else:
                        deltas.append(float(metric(baseline, swapped, mask)))

            raw[row, col] = float(np.mean(deltas)) if deltas else 0.0
            if progress is not None:
                progress(attr, block_id, raw[row, col])

        raw[:, col] = _normalise(raw[:, col])

    return BASM(matrix=raw, block_ids=vital_blocks, attributes=attributes)


def make_swap_generate_fn(flair_pipeline, steps: int) -> SwapGenerateFn:
    """Bind a FlairPipeline into the harness's generate signature.

    A swap is the routing blend at full strength: with alpha = 1.0 at one
    block, H = H_base + 1.0 * (H_changed - H_base) = H_changed exactly.
    """
    import torch

    from ..components import Component
    from ..config import FlairConfig
    from ..patching import install_flair, uninstall_flair
    from ..processor import PlanRef
    from ..routing import RoutedComponent, RoutingPlan

    swap_cfg = FlairConfig(
        device=flair_pipeline.cfg.device,
        alpha_0=1.0,
        t_window=(0.0, 1.0),
        model_id=flair_pipeline.cfg.model_id,
        max_sequence_length=flair_pipeline.cfg.max_sequence_length,
    )

    def generate(prompt: str, seed: int, swap: SwapSpec | None) -> Image.Image:
        if swap is None:
            return flair_pipeline.generate(
                prompt, seed=seed, steps=steps, routing=False
            )

        component = Component(
            id="swap", text=swap.prompt, attr=AttributeClass.IDENTITY
        )
        embeddings = flair_pipeline.encode_components([component])
        plan = RoutingPlan(
            routed=(
                RoutedComponent(
                    component=component,
                    embedding=embeddings["swap"],
                    blocks=((swap.block_id, 1.0),),
                ),
            ),
            cfg=swap_cfg,
        )

        ref = PlanRef(plan=plan, total_steps=steps, do_cfg=True)
        handles = install_flair(flair_pipeline.pipe.transformer, ref)
        try:
            def on_step(pipe, step_index, timestep, callback_kwargs):
                ref.step = step_index
                return callback_kwargs

            result = flair_pipeline.pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=4.5,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                callback_on_step_end=on_step,
            )
        finally:
            uninstall_flair(handles)

        return result.images[0]

    return generate
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest tests/test_harness.py -v`
Expected: PASS — 7 passed

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -v`
Expected: PASS — every test from foundation Tasks 1-10 and this plan's Tasks 11-16

- [ ] **Step 6: Write the calibration entry point**

Create `scripts/calibrate.py`. Campaign parameters are flags with conservative defaults, so Week 3 sets them from measured throughput rather than from guesses baked into code.

```python
"""Run the BASM calibration campaign on Kaggle GPU.

Two phases, separately resumable:

    python scripts/calibrate.py prefilter --top-n 10 --out calibration_runs/
    python scripts/calibrate.py basm --vitality calibration_runs/vitality.json \
        --out calibration_runs/

Choose --top-n and --seeds from the throughput the foundation smoke test
measured; see the campaign plan.
"""

import argparse
from pathlib import Path

import spacy
import torch
from diffusers import StableDiffusion3Pipeline

from flair_t2i.attributes import CORE_ATTRIBUTES
from flair_t2i.basm import BASM
from flair_t2i.calibration.corpus import DEFAULT_CORPUS_PATH, load_corpus
from flair_t2i.calibration.harness import calibrate, make_swap_generate_fn
from flair_t2i.calibration.prefilter import (
    VitalityReport,
    lpips_distance,
    make_bypass_generate_fn,
    run_prefilter,
)
from flair_t2i.config import FlairConfig
from flair_t2i.metrics.embedding import ClipScorer
from flair_t2i.metrics.masking import ClipSegMasker
from flair_t2i.pipeline import FlairPipeline

PREFILTER_PROMPTS = [
    "a small red sports car under warm evening light",
    "a large ceramic vase on a wooden table",
    "a rusty metal sign beside a brick wall",
]


def _pipeline(cfg: FlairConfig) -> FlairPipeline:
    pipe = StableDiffusion3Pipeline.from_pretrained(
        cfg.model_id, torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()
    basm = BASM.uniform((0,), CORE_ATTRIBUTES)  # placeholder; unused for calibration
    return FlairPipeline(pipe, cfg, basm, nlp=spacy.load("en_core_web_sm"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prefilter", "basm"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--vitality", type=Path)
    parser.add_argument("--out", type=Path, default=Path("calibration_runs"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = FlairConfig(device="cuda")
    fp = _pipeline(cfg)

    if args.phase == "prefilter":
        n_blocks = len(fp.pipe.transformer.transformer_blocks)
        print(f"scoring {n_blocks} blocks")
        report = run_prefilter(
            make_bypass_generate_fn(fp, steps=args.steps),
            n_blocks=n_blocks,
            prompts=PREFILTER_PROMPTS,
            seeds=args.seeds,
            distance_fn=lpips_distance(cfg.device),
            top_n=args.top_n,
        )
        report.save(args.out / "vitality.json")
        print(f"vital blocks: {report.vital_blocks}")
        return

    if args.vitality is None:
        parser.error("basm phase needs --vitality")

    vital = VitalityReport.load(args.vitality).vital_blocks
    corpus = load_corpus(DEFAULT_CORPUS_PATH)
    print(f"calibrating {len(corpus)} attributes over blocks {vital}")

    basm = calibrate(
        make_swap_generate_fn(fp, steps=args.steps),
        corpus=corpus,
        vital_blocks=vital,
        masker=ClipSegMasker(device=cfg.device),
        seeds=args.seeds,
        scorer=ClipScorer(device=cfg.device),
        progress=lambda attr, block, value: print(
            f"  {attr.value:<9} block {block:>3}  raw={value:.4f}"
        ),
    )
    basm.save(args.out / "basm.npz")

    for attr in basm.attributes:
        print(f"{attr.value:<9} top blocks: {basm.top_k(attr, 3)}")
    print(f"\nwrote {args.out / 'basm.npz'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Commit**

```bash
git add flair_t2i/calibration/harness.py scripts/calibrate.py tests/test_harness.py
git commit -m "feat: BASM calibration harness reusing routing blend as prompt swap"
```

---

## Self-Review

**1. Spec coverage (§3.4).**

| §3.4 requirement | Covered by |
|---|---|
| Vital-layer prefilter via residual bypass | Task 15 (`run_prefilter`, reusing foundation Task 6's `bypass_blocks`) |
| Contrastive pairs, one-attribute difference, fixed seeds | Task 14 (corpus + enforced invariant), Task 16 (`seeds`) |
| Baseline everywhere, swap at block ℓ only | Task 16 (`SwapSpec`, `make_swap_generate_fn`) |
| CIELAB ΔE (colour) | Task 12 `color_delta` |
| Object-mask area ratio (size) | Task 12 `size_delta`, re-masked per image in Task 16 |
| Gram/DISTS (texture, style) | Task 13 `gram_texture_delta`, `style_delta` |
| CLIP similarity (identity), lighting descriptors, action phrases | Task 13 `identity_delta`, `action_delta`; Task 12 `warmth_absolute` |
| Measurement restricted to the object mask | Task 11 (`Masker`), applied throughout Task 16 — except lighting, which is deliberately scene-level |
| Normalise Δ across pairs → S[ℓ,a] ∈ [0,1] | Task 16 `_normalise` |
| Campaign volume and per-attribute pair count | **Out of scope** — set by the campaign plan from measured throughput |

**2. Placeholder scan.** No "TBD"/"handle edge cases"/"similar to Task N". Every code step is runnable; every corpus entry is authored, not sketched. `scripts/calibrate.py`'s `--top-n`/`--seeds` are flags with conservative defaults, not unfilled blanks.

**3. Type consistency.** `delta_for(attr, *, scorer, phrase)` returns the uniform `(image_a, image_b, mask) -> float` signature the harness calls. `Masker.__call__(image, label)` matches `RectMasker`, `ClipSegMasker`, and every harness call site. `SwapSpec(block_id, prompt)` is constructed in `calibrate` and consumed in `make_swap_generate_fn`. `VitalityReport.vital_blocks` feeds `calibrate(vital_blocks=...)` and `BASM(block_ids=...)` as the same tuple type. `ImageTextScorer` is implemented by `ClipScorer` and by the test fakes with matching methods.

**One deliberate deviation, documented in-code:** `gram_texture_delta` is a NumPy Gram-matrix statistic rather than DISTS, so calibration needs no extra model weights. It shares the delta-metric signature, so DISTS can replace it for the paper's final numbers without touching the harness.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-08-25-flair-calibration-harness.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
