"""Attention processor wrapper that injects routed streams (spec section 3.5).

The wrapper modifies ``encoder_hidden_states`` and then delegates to the
backbone's own processor, so no attention maths is reimplemented here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .routing import RoutingPlan


@dataclass
class PlanRef:
    """Mutable handle the denoise loop updates each step."""

    plan: RoutingPlan | None = None
    step: int = 0
    total_steps: int = 1
    do_cfg: bool = True

    def step_frac(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        return self.step / self.total_steps

    def cond_slice(self, batch_size: int) -> slice:
        """Rows carrying the positive prompt.

        diffusers concatenates ``[negative, positive]`` when guidance is on,
        so the conditional half is the tail.
        """
        if not self.do_cfg:
            return slice(0, batch_size)
        return slice(batch_size // 2, batch_size)


class FlairJointProcessor:
    def __init__(self, inner, block_id: int, ref: PlanRef) -> None:
        self.inner = inner
        self.block_id = block_id
        self.ref = ref

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        plan = self.ref.plan
        if plan is not None and encoder_hidden_states is not None:
            encoder_hidden_states = plan.blend(
                encoder_hidden_states,
                block_id=self.block_id,
                step_frac=self.ref.step_frac(),
                cond_slice=self.ref.cond_slice(encoder_hidden_states.shape[0]),
            )
        return self.inner(attn, hidden_states, encoder_hidden_states, *args, **kwargs)
