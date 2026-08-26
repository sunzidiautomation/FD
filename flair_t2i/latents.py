"""Capture intermediate latents during denoising.

Routing's effect is easiest to read as a sequence: the attribute appears
early, while ``timestep_scale`` still has weight, and the rest of the run
refines what is already there. Decoding is injected rather than imported
so nothing here needs a VAE -- the tests pass a stub.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from PIL import Image


@dataclass
class LatentRecorder:
    decode_fn: Callable[[Any], Image.Image]
    at: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)
    frames: list[tuple[float, Image.Image]] = field(default_factory=list)

    def target_steps(self, total_steps: int) -> set[int]:
        """The step indices closest to each requested fraction."""
        if total_steps <= 0:
            return set()
        return {
            min(total_steps - 1, int(round(frac * total_steps))) for frac in self.at
        }

    def reset(self) -> None:
        self.frames = []

    def __call__(self, step_index: int, total_steps: int, latents: Any) -> None:
        if step_index not in self.target_steps(total_steps):
            return
        frac = step_index / total_steps if total_steps else 0.0
        self.frames.append((frac, self.decode_fn(latents)))
