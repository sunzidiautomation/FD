"""HASM calibration harness (spec section 3.4).

For each attribute a, each routable head unit (l, h), and each contrastive
pair p: run the base prompt everywhere, then run it again with p.changed
swapped in at that head alone, and measure the attribute-specific change
inside the object mask. Averaging over pairs gives a raw sensitivity, and
min-max normalising each attribute's whole plane gives S[l, h, a] in
[0, 1].

Checkpointing is not optional in practice. A full sweep is
``attributes x pairs x seeds x (1 + blocks x heads)`` generations -- far
more than a thousand -- which can outlast a Kaggle session's 12-hour cap.
Each (attribute, block, head) cell is written to disk as it completes and
skipped on resume, so an interrupted campaign continues instead of
starting over. Cells store the RAW mean, not the normalised score, because
normalisation depends on the whole plane.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from PIL import Image

from ..attributes import AttributeClass
from ..hasm import HASM
from ..heads import HeadUnit
from ..metrics.embedding import ImageTextScorer
from ..metrics.masking import Masker
from ..metrics.photometric import size_delta
from ..metrics.registry import delta_for
from .corpus import ContrastivePair


@dataclass(frozen=True)
class SwapSpec:
    unit: HeadUnit
    prompt: str


class SwapGenerateFn(Protocol):
    def __call__(
        self, prompt: str, seed: int, swap: SwapSpec | None
    ) -> Image.Image: ...


ProgressFn = Callable[[AttributeClass, HeadUnit, float], None]
PairFn = Callable[
    [AttributeClass, HeadUnit, ContrastivePair, int, Image.Image, Image.Image], None
]


def _cell_path(
    checkpoint_dir: str | Path, attr: AttributeClass, unit: HeadUnit
) -> Path:
    return (
        Path(checkpoint_dir)
        / "cells"
        / f"{attr.value}_{unit.block}_{unit.head}.json"
    )


def _load_cell(path: Path) -> float | None:
    """The cell's raw value, or None if absent or unreadable."""
    try:
        return float(json.loads(path.read_text(encoding="utf-8"))["raw"])
    except (OSError, KeyError, TypeError, ValueError):
        return None  # missing, truncated by an interrupted session, or corrupt


def _save_cell(
    path: Path, attr: AttributeClass, unit: HeadUnit, raw: float, samples: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "attribute": attr.value,
                "block": unit.block,
                "head": unit.head,
                "raw": raw,
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )


def _normalise(column: np.ndarray) -> np.ndarray:
    """Min-max a raw sensitivity column onto [0, 1]."""
    low, high = float(column.min()), float(column.max())
    if high - low <= 1e-12:
        return np.zeros_like(column)  # flat response: no signal, not a peak
    return (column - low) / (high - low)


def _measure_cell(
    generate_fn: SwapGenerateFn,
    attr: AttributeClass,
    unit: HeadUnit,
    pairs: list[ContrastivePair],
    seeds: list[int],
    masker: Masker,
    scorer: ImageTextScorer | None,
    on_pair: PairFn | None = None,
) -> tuple[float, int]:
    """Mean attribute change for one (attribute, head unit), over every pair."""
    deltas: list[float] = []

    for pair in pairs:
        metric = delta_for(attr, scorer=scorer, phrase=pair.phrase)
        for seed in seeds:
            baseline = generate_fn(prompt=pair.base, seed=seed, swap=None)
            swapped = generate_fn(
                prompt=pair.base,
                seed=seed,
                swap=SwapSpec(unit=unit, prompt=pair.changed),
            )
            if on_pair is not None:
                on_pair(attr, unit, pair, seed, baseline, swapped)
            mask = masker(baseline, pair.object_label)

            if attr is AttributeClass.SIZE:
                # Area change is only visible by re-segmenting the changed
                # image -- one shared mask always reads 0. See registry.py.
                deltas.append(
                    size_delta(
                        baseline, swapped, mask, masker(swapped, pair.object_label)
                    )
                )
            else:
                deltas.append(float(metric(baseline, swapped, mask)))

    return (float(np.mean(deltas)) if deltas else 0.0), len(deltas)


def calibrate(
    generate_fn: SwapGenerateFn,
    corpus: dict[AttributeClass, list[ContrastivePair]],
    block_ids: tuple[int, ...],
    head_ids: tuple[int, ...],
    masker: Masker,
    seeds: list[int],
    scorer: ImageTextScorer | None = None,
    progress: ProgressFn | None = None,
    checkpoint_dir: str | Path | None = None,
    on_pair: PairFn | None = None,
) -> HASM:
    """Measure per-head sensitivity for every attribute in ``corpus``.

    Pass ``checkpoint_dir`` to make the sweep resumable. At
    ``blocks x heads x attributes`` cells this is not optional in practice.
    """
    attributes = tuple(a for a in AttributeClass if a in corpus)
    raw = np.zeros((len(block_ids), len(head_ids), len(attributes)))

    for plane, attr in enumerate(attributes):
        for i, block_id in enumerate(block_ids):
            for j, head_id in enumerate(head_ids):
                unit = HeadUnit(block=block_id, head=head_id)
                cached = (
                    _load_cell(_cell_path(checkpoint_dir, attr, unit))
                    if checkpoint_dir is not None
                    else None
                )

                if cached is not None:
                    raw[i, j, plane] = cached
                else:
                    value, samples = _measure_cell(
                        generate_fn,
                        attr,
                        unit,
                        corpus[attr],
                        seeds,
                        masker,
                        scorer,
                        on_pair=on_pair,
                    )
                    raw[i, j, plane] = value
                    if checkpoint_dir is not None:
                        _save_cell(
                            _cell_path(checkpoint_dir, attr, unit),
                            attr,
                            unit,
                            value,
                            samples,
                        )

                if progress is not None:
                    progress(attr, unit, raw[i, j, plane])

        # Normalise across every unit for this attribute, not per block.
        raw[:, :, plane] = _normalise(raw[:, :, plane])

    return HASM(
        tensor=raw,
        block_ids=block_ids,
        head_ids=head_ids,
        attributes=attributes,
    )


def make_swap_generate_fn(flair_pipeline, steps: int) -> SwapGenerateFn:
    """Bind a FlairPipeline into the harness's generate signature.

    A swap is the routing residual at full strength: with alpha = 1.0 on one
    head, that head's Q/K/V become exactly what they would be under
    H_changed, and every other head is untouched. Calibration therefore
    reuses the routing machinery instead of adding a second injection path
    that could drift from it.
    """
    import torch

    from ..components import Component
    from ..config import FlairConfig
    from ..patching import install_head_routing, uninstall_head_routing
    from ..processor import PlanRef
    from ..routing import RoutedComponent, RoutingPlan

    swap_cfg = FlairConfig(
        device=flair_pipeline.cfg.device,
        alpha_0=1.0,
        t_window=(0.0, 1.0),
        model_id=flair_pipeline.cfg.model_id,
        max_sequence_length=flair_pipeline.cfg.max_sequence_length,
    )

    @torch.inference_mode()
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
                    units=((swap.unit, 1.0),),
                ),
            ),
            cfg=swap_cfg,
        )

        ref = PlanRef(plan=plan, total_steps=steps, do_cfg=True)
        handles = install_head_routing(flair_pipeline.pipe.transformer, ref)
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
            uninstall_head_routing(handles)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result.images[0]

    return generate
