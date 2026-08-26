"""FROZEN reference implementation of block-level blending. DO NOT EDIT.

A verbatim copy of ``RoutingPlan.blend`` and the dataclass shape it read,
taken from the commit that shipped block-level routing. Task 5's
equivalence test proves the head-level implementation reproduces it
exactly when every head of a block is selected.

Freezing it here rather than leaving it on the production path is
deliberate: an oracle that lives beside the implementation it validates
gets refactored alongside it and stops being independent evidence. It
imports nothing from ``flair_t2i.routing``, so it cannot drift with it.
"""

from dataclasses import dataclass

import torch

from flair_t2i.components import Component
from flair_t2i.schedule import timestep_scale


@dataclass
class ReferenceRouted:
    component: Component
    embedding: torch.Tensor  # [seq, dim]
    blocks: tuple[tuple[int, float], ...]  # (block_id, basm_score)
    intensity: float = 1.0


def reference_blend(
    routed,
    cfg,
    encoder_hidden_states: torch.Tensor,
    block_id: int,
    step_frac: float,
    cond_slice: slice,
    alpha_scale: float = 1.0,
) -> torch.Tensor:
    """The shipped block-level blend, frozen."""
    touched = {b for rc in routed for b, _ in rc.blocks}
    if block_id not in touched:
        return encoder_hidden_states

    def alpha(rc) -> float:
        score = next((s for b, s in rc.blocks if b == block_id), None)
        if score is None:
            return 0.0
        sched = timestep_scale(step_frac, cfg.t_window)
        return cfg.alpha_0 * score * rc.intensity * sched * alpha_scale

    contributions = [
        (rc, alpha(rc)) for rc in routed if block_id in {b for b, _ in rc.blocks}
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

    for rc, a in contributions:
        target = rc.embedding.to(device=base.device, dtype=base.dtype)
        out[cond_slice] = out[cond_slice] + a * (target.unsqueeze(0) - base)

    return out
