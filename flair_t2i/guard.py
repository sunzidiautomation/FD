"""Runtime coherence guard (spec section 3.6).

Two checks are specified: a crisp cross-stream cosine check, and an
attribute-distortion check that evaluates membership in the intended fuzzy
region rather than a percentile band.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch

from .attributes import AttributeClass
from .config import FlairConfig
from .fuzzy.membership import membership_at
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

    def check_membership(
        self,
        attr: AttributeClass,
        label: str,
        measured: float,
        step: int,
    ) -> GuardEvent | None:
        """Flag a measured metric that has left its intended fuzzy region.

        This replaces the crisp percentile band described in the reference
        design (spec section 3.6) with a graded membership evaluation.
        """
        mu = membership_at(attr, label, measured)
        if mu >= self.cfg.guard_membership_threshold:
            return None
        return GuardEvent(step=step, reason="attribute_membership", value=mu)

    def apply(self, plan: RoutingPlan, event: GuardEvent | None) -> None:
        if event is None:
            return
        plan.alpha_scale *= self.cfg.guard_backoff
        self.events.append(event)
