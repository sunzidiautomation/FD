"""Routing plans and the head-level residual (spec 2026-08-26, section 3).

    H_l = H_base + sum_i alpha_i(t) * (H_i - H_base)
    alpha_i(l, h, t) = alpha_0 * S[l, h, a] * intensity_i * sched(t) * alpha_scale

The residual is applied to the OUTPUT of a text-stream projection rather
than its input. Projection is linear, so for weight ``A``:

    proj(x + d) = proj(x) + d @ A.T

meaning a residual added after projection is identical to one added
before -- provided it is projected weight-only, with no bias, since the
bias cancels in a difference. Because each head owns a contiguous slice of
the projection output, scaling that projected residual by a per-head alpha
vector selects heads and applies their individual strengths in one step.

Selecting every head of a block at one score therefore reproduces the
block-level blend exactly. ``tests/test_routing.py`` pins that against the
frozen oracle in ``tests/reference_blend.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .components import Component
from .config import FlairConfig
from .hasm import HASM
from .heads import HeadUnit, alpha_vector
from .schedule import timestep_scale


@dataclass
class RoutedComponent:
    component: Component
    embedding: torch.Tensor  # [seq, dim]
    units: tuple[tuple[HeadUnit, float], ...]  # (head unit, sensitivity score)
    intensity: float = 1.0


@dataclass
class RoutingPlan:
    routed: tuple[RoutedComponent, ...]
    cfg: FlairConfig
    active: bool = True
    alpha_scale: float = 1.0
    _blocks: frozenset[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._blocks = frozenset(
            unit.block for rc in self.routed for unit, _ in rc.units
        )

    def blocks_touched(self) -> frozenset[int]:
        return self._blocks

    def alpha(self, rc: RoutedComponent, unit: HeadUnit, step_frac: float) -> float:
        score = next((s for u, s in rc.units if u == unit), None)
        if score is None:
            return 0.0
        sched = timestep_scale(step_frac, self.cfg.t_window)
        return self.cfg.alpha_0 * score * rc.intensity * sched * self.alpha_scale

    def head_residual(
        self,
        block_id: int,
        x: torch.Tensor,
        weight: torch.Tensor,
        step_frac: float,
        cond_slice: slice,
        n_heads: int,
        head_dim: int,
    ) -> torch.Tensor | None:
        """The masked residual to add to this projection's output.

        ``x`` is the projection's input (the text stream), ``weight`` its
        ``nn.Linear`` weight. Returns ``None`` when this block is not
        routed -- the fast path, which is most calls.
        """
        if not self.active or block_id not in self._blocks:
            return None

        base = x[cond_slice]
        seq = x.shape[-2]
        total: torch.Tensor | None = None

        for rc in self.routed:
            alphas = {
                unit.head: self.alpha(rc, unit, step_frac)
                for unit, _ in rc.units
                if unit.block == block_id
            }
            alphas = {head: a for head, a in alphas.items() if a != 0.0}
            if not alphas:
                continue

            if rc.embedding.shape[-2] != seq:
                raise ValueError(
                    f"component {rc.component.id} sequence length "
                    f"{rc.embedding.shape[-2]} does not match states {seq}"
                )

            target = rc.embedding.to(device=base.device, dtype=base.dtype)
            delta = target.unsqueeze(0) - base
            # Weight only. A bias here would break the equivalence invariant.
            projected = torch.nn.functional.linear(delta, weight)
            scaled = projected * alpha_vector(
                alphas, n_heads, head_dim, device=base.device, dtype=base.dtype
            )
            total = scaled if total is None else total + scaled

        return total


def build_routing_plan(
    components: list[Component],
    embeddings: dict[str, torch.Tensor],
    hasm: HASM,
    cfg: FlairConfig,
    intensities: dict[str, float] | None = None,
    k_overrides: dict[str, int] | None = None,
    granularity: str = "head",
    reduce: str = "max",
) -> RoutingPlan:
    """Select target head units per component from the calibrated HASM.

    ``granularity="block"`` selects every head of the top-k blocks at that
    block's reduced score -- one mechanism, two selection functions.
    """
    if granularity not in ("head", "block"):
        raise ValueError(f"unknown granularity {granularity!r}; use 'head' or 'block'")

    intensities = intensities or {}
    k_overrides = k_overrides or {}

    routed: list[RoutedComponent] = []
    for component in components:
        if component.attr not in hasm.attributes:
            continue  # not calibrated; nothing to route
        k = k_overrides.get(component.id, cfg.top_k_default)

        if granularity == "head":
            units = tuple(hasm.top_k(component.attr, k))
        else:
            blocks = hasm.to_basm(reduce=reduce).top_k(component.attr, k)
            units = tuple(
                (HeadUnit(block=block, head=head), score)
                for block, score in blocks
                for head in hasm.head_ids
            )

        if not units:
            continue
        routed.append(
            RoutedComponent(
                component=component,
                embedding=embeddings[component.id],
                units=units,
                intensity=intensities.get(component.id, 1.0),
            )
        )

    return RoutingPlan(routed=tuple(routed), cfg=cfg)
