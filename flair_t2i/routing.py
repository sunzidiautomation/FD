"""Routing plans and the blend operation (spec section 3.5).

    H_l = H_base + sum_i alpha_i(t) * (H_i - H_base)
    alpha_i(t) = alpha_0 * S[l, a] * intensity_i * sched(t)

Component embeddings are encoded once and held on the plan. They are never
stacked into the denoising batch, so there is no base-row arithmetic to get
wrong -- ``blend`` writes only into the conditional rows the caller names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .basm import BASM
from .components import Component
from .config import FlairConfig
from .schedule import timestep_scale


@dataclass
class RoutedComponent:
    component: Component
    embedding: torch.Tensor  # [seq, dim]
    blocks: tuple[tuple[int, float], ...]  # (block_id, basm_score)
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
            block_id for rc in self.routed for block_id, _ in rc.blocks
        )

    def blocks_touched(self) -> frozenset[int]:
        return self._blocks

    def alpha(self, rc: RoutedComponent, block_id: int, step_frac: float) -> float:
        score = next((s for b, s in rc.blocks if b == block_id), None)
        if score is None:
            return 0.0
        sched = timestep_scale(step_frac, self.cfg.t_window)
        return self.cfg.alpha_0 * score * rc.intensity * sched * self.alpha_scale

    def blend(
        self,
        encoder_hidden_states: torch.Tensor,
        block_id: int,
        step_frac: float,
        cond_slice: slice,
    ) -> torch.Tensor:
        """Return states with component residuals added at ``block_id``."""
        if not self.active or block_id not in self._blocks:
            return encoder_hidden_states

        contributions = [
            (rc, self.alpha(rc, block_id, step_frac))
            for rc in self.routed
            if block_id in {b for b, _ in rc.blocks}
        ]
        contributions = [(rc, a) for rc, a in contributions if a != 0.0]
        if not contributions:
            return encoder_hidden_states

        seq = encoder_hidden_states.shape[-2]
        for rc, _ in contributions:
            if rc.embedding.shape[-2] != seq:
                raise ValueError(
                    f"component {rc.component.id} sequence length "
                    f"{rc.embedding.shape[-2]} does not match states {seq}"
                )

        out = encoder_hidden_states.clone()
        base = encoder_hidden_states[cond_slice]

        for rc, alpha in contributions:
            target = rc.embedding.to(device=base.device, dtype=base.dtype)
            out[cond_slice] = out[cond_slice] + alpha * (target.unsqueeze(0) - base)

        return out


def build_routing_plan(
    components: list[Component],
    embeddings: dict[str, torch.Tensor],
    basm: BASM,
    cfg: FlairConfig,
    intensities: dict[str, float] | None = None,
    k_overrides: dict[str, int] | None = None,
) -> RoutingPlan:
    """Select target blocks per component from the calibrated BASM."""
    intensities = intensities or {}
    k_overrides = k_overrides or {}

    routed: list[RoutedComponent] = []
    for component in components:
        if component.attr not in basm.attributes:
            continue  # not calibrated; nothing to route
        k = k_overrides.get(component.id, cfg.top_k_default)
        blocks = tuple(basm.top_k(component.attr, k))
        if not blocks:
            continue
        routed.append(
            RoutedComponent(
                component=component,
                embedding=embeddings[component.id],
                blocks=blocks,
                intensity=intensities.get(component.id, 1.0),
            )
        )

    return RoutingPlan(routed=tuple(routed), cfg=cfg)
