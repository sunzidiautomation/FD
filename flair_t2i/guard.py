"""Runtime coherence guard (spec section 3.6).

Two checks are specified. The cross-stream cosine check lives here and is
crisp. The attribute-distortion check becomes a fuzzy-membership
evaluation once the fuzzy module lands -- see Task 10.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch

from .config import FlairConfig
from .routing import RoutingPlan


@dataclass
class GuardEvent:
    step: int
    reason: str
    value: float


class CoherenceGuard:
    def __init__(self, cfg: FlairConfig) -> None:
        self.cfg = cfg
        self.events: list[GuardEvent] = []

    def check_streams(self, plan: RoutingPlan, step: int) -> GuardEvent | None:
        """Flag routed streams that have drifted apart into incoherence."""
        if len(plan.routed) < 2:
            return None

        worst = 1.0
        for a, b in combinations(plan.routed, 2):
            sim = torch.nn.functional.cosine_similarity(
                a.embedding.flatten().float(),
                b.embedding.flatten().float(),
                dim=0,
            ).item()
            worst = min(worst, sim)

        if worst >= self.cfg.guard_cos_threshold:
            return None
        return GuardEvent(step=step, reason="cross_stream_similarity", value=worst)

    def apply(self, plan: RoutingPlan, event: GuardEvent | None) -> None:
        if event is None:
            return
        plan.alpha_scale *= self.cfg.guard_backoff
        self.events.append(event)
