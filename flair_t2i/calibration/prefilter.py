"""Vital-layer prefilter (spec section 3.4).

Each block is bypassed in turn and the output compared against the
unmodified run; blocks whose removal changes the image most are the vital
ones. This narrows the BASM sweep from every block to a handful, which is
what makes calibration affordable -- the campaign cost is linear in the
number of blocks kept.

Baselines are generated once per (prompt, seed) and reused across every
block, so the cost is ``prompts x seeds x (1 + n_blocks)`` rather than
``prompts x seeds x 2 x n_blocks``.

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
    def __call__(self, prompt: str, seed: int, bypass: int | None) -> Image.Image: ...


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

    def elbow(self, low: int = 6, high: int = 12) -> tuple[int, ...]:
        """Blocks scoring at least half the top block's, clamped to [low, high].

        The campaign plan's rule for choosing how many blocks to keep: the
        cost is linear in this number, and every block kept that is not
        actually attribute-selective just adds a noise row to the BASM.
        """
        ranked = sorted(self.scores.items(), key=lambda pair: (-pair[1], pair[0]))
        if not ranked:
            return ()
        cutoff = ranked[0][1] * 0.5
        keep = [block for block, score in ranked if score >= cutoff]
        keep = keep[:high] if len(keep) > high else keep
        if len(keep) < low:
            keep = [block for block, _ in ranked[: min(low, len(ranked))]]
        return tuple(sorted(keep))


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
