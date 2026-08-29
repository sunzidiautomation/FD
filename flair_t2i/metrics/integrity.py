"""Reject generations that fell apart, before they are measured.

Every delta metric answers "how much did property X change". A generation
that collapses -- into a saturated field, a tiled artefact, noise --
maximises that question's answer without controlling the attribute at all.
Left ungated, the most DESTRUCTIVE intervention outranks the most
SELECTIVE one, and min-max normalisation then anchors the whole scale to
that corrupt maximum, compressing every genuine result toward zero.

This was not hypothetical: the first head-level lighting sweep ranked a
head whose swap produced a yellow checkerboard above every head that
actually changed the light.

Three independent signals, because each alone is escapable:

- **colour spread** catches a frame that has collapsed into one hue. A
  scrambled frame keeps its histogram, so this misses that.
- **structural similarity** catches a frame whose layout is destroyed. A
  saturated field can stay structurally smooth, so this misses that.
- **speckle** catches mottled high-frequency noise laid over an intact
  scene. The first two signals are both global, and this failure is
  local: the specks are colourful, so spread survives, and the layout is
  untouched, so SSIM survives. A real action sweep ranked such a frame
  first before this signal existed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


def colour_spread(image: Image.Image) -> float:
    """Mean per-channel standard deviation. Zero for a solid colour."""
    array = np.asarray(image.convert("RGB"), dtype=np.float64)
    return float(array.std(axis=(0, 1)).mean())


def speckle(image: Image.Image) -> float:
    """Mean absolute deviation from a 3x3 median, on [0, 1].

    Isolates high-frequency noise from legitimate sharpness. A median
    filter erases isolated pixels and leaves edges standing, so what
    survives the subtraction is speckle rather than detail -- which
    matters because removing motion blur, a legitimate outcome of an
    action swap, raises high-frequency energy just as noise does.
    """
    from PIL import ImageFilter

    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.float64)
    smoothed = np.asarray(rgb.filter(ImageFilter.MedianFilter(3)), dtype=np.float64)
    return float(np.abs(array - smoothed).mean() / 255.0)


def structural_change(a: Image.Image, b: Image.Image) -> float:
    """``1 - SSIM`` over luminance. Zero when the two images match."""
    from skimage.metrics import structural_similarity

    grey_a = np.asarray(a.convert("L"), dtype=np.float64)
    grey_b = np.asarray(b.convert("L"), dtype=np.float64)
    return float(1.0 - structural_similarity(grey_a, grey_b, data_range=255))


@dataclass(frozen=True)
class IntegrityVerdict:
    ok: bool
    colour_ratio: float
    structural_change: float
    speckle_ratio: float = 1.0
    reason: str | None = None


@dataclass(frozen=True)
class IntegrityGate:
    """Decide whether a swapped frame is worth measuring at all.

    The first two defaults were read off a real 576-unit lighting sweep,
    where they rejected 22 frames (3.8%) -- every one of them visibly
    broken -- and kept every frame that showed a genuine lighting change.

    The speckle default was read off three sweeps (lighting, colour,
    action) after inspecting the images: every cell confirmed clean by eye
    scored at most 1.36, while the confetti frames that both other signals
    passed scored 1.92 and 2.09. 1.50 sits in that gap. The asymmetry
    argues for the lower end -- one contaminated cell becomes the
    normalisation ceiling for its whole plane, while one honest cell lost
    out of 576 costs almost nothing.
    """

    #: colour spread as a fraction of the baseline's
    min_colour_ratio: float = 0.75
    #: 1 - SSIM against the baseline
    max_structural_change: float = 0.60
    #: speckle as a multiple of the baseline's
    max_speckle_ratio: float = 1.50

    def check(
        self, baseline: Image.Image, candidate: Image.Image
    ) -> IntegrityVerdict:
        base_spread = colour_spread(baseline)
        ratio = (
            colour_spread(candidate) / base_spread if base_spread > 1e-9 else 1.0
        )
        change = structural_change(baseline, candidate)

        base_speckle = speckle(baseline)
        speckle_ratio = (
            speckle(candidate) / base_speckle if base_speckle > 1e-6 else 1.0
        )

        reason = None
        if ratio < self.min_colour_ratio:
            reason = (
                f"colour collapse: spread is {ratio:.2f} of the baseline's, "
                f"below {self.min_colour_ratio:.2f}"
            )
        elif change > self.max_structural_change:
            reason = (
                f"structure collapse: 1-SSIM is {change:.2f}, "
                f"above {self.max_structural_change:.2f}"
            )
        elif speckle_ratio > self.max_speckle_ratio:
            reason = (
                f"speckle: high-frequency noise is {speckle_ratio:.2f}x the "
                f"baseline's, above {self.max_speckle_ratio:.2f}"
            )

        return IntegrityVerdict(
            ok=reason is None,
            colour_ratio=ratio,
            structural_change=change,
            speckle_ratio=speckle_ratio,
            reason=reason,
        )
