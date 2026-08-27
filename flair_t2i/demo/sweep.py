"""The demonstration sweep.

Every attribute over every head unit at ONE contrastive pair each -- one
fifth the full campaign's cost, and enough to show which block and which
head each attribute actually responds to. Unlike the campaign, this keeps
every generated image, because the images are the deliverable.

Run against a fresh output directory. A checkpointed cell is never
regenerated, so its images would be missing from the bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..attributes import AttributeClass
from ..calibration.harness import calibrate
from ..hasm import HASM
from ..heads import HeadUnit


@dataclass
class DemoPaths:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        for directory in (self.heads, self.blocks, self.latents, self.baselines):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def heads(self) -> Path:
        return self.root / "heads"

    @property
    def blocks(self) -> Path:
        return self.root / "blocks"

    @property
    def latents(self) -> Path:
        return self.root / "latents"

    @property
    def baselines(self) -> Path:
        return self.root / "baselines"

    def head_image(self, attr: AttributeClass, unit: HeadUnit) -> Path:
        return self.heads / f"{attr.value}_b{unit.block}_h{unit.head}.png"

    def block_image(self, attr: AttributeClass, block: int) -> Path:
        return self.blocks / f"{attr.value}_b{block}.png"

    def baseline_image(self, attr: AttributeClass) -> Path:
        return self.baselines / f"{attr.value}.png"


def run_demo_sweep(
    generate_fn,
    corpus,
    block_ids: tuple[int, ...],
    head_ids: tuple[int, ...],
    masker,
    paths: DemoPaths,
    seeds: list[int],
    scorer=None,
    progress=None,
) -> HASM:
    """Sweep every head unit, keeping every image, and return the HASM."""

    def on_pair(attr, unit, pair, seed, baseline: Image.Image, swapped: Image.Image):
        baseline_path = paths.baseline_image(attr)
        if not baseline_path.exists():
            baseline.save(baseline_path)
        swapped.save(paths.head_image(attr, unit))

    hasm = calibrate(
        generate_fn,
        corpus=corpus,
        block_ids=block_ids,
        head_ids=head_ids,
        masker=masker,
        seeds=seeds,
        scorer=scorer,
        progress=progress,
        checkpoint_dir=None,
        on_pair=on_pair,
    )

    # Block-level counterpart: the same swap with every head of a block
    # selected at once, which is what block-level routing does.
    from ..calibration.harness import SwapSpec

    for attr, pairs in corpus.items():
        pair = pairs[0]
        for block in block_ids:
            image = generate_fn(
                prompt=pair.base,
                seed=seeds[0],
                swap=SwapSpec(
                    units=tuple(
                        HeadUnit(block=block, head=head) for head in head_ids
                    ),
                    prompt=pair.changed,
                ),
            )
            image.save(paths.block_image(attr, block))

    hasm.save(paths.root / "hasm.npz")
    return hasm
