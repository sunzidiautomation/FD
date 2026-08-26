"""The mutable handle the denoise loop shares with the routing wrappers.

One value -- the current step -- has to reach every wrapped projection on
every block. ``PlanRef`` is the box they all hold a reference to, which is
how that value crosses diffusers without being threaded through its call
signatures.
"""

from __future__ import annotations

from dataclasses import dataclass

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
