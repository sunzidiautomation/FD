"""Attention-head routing units and per-head masking.

diffusers reshapes a projection's output as
``view(batch, -1, heads, head_dim).transpose(1, 2)``, so head ``h`` owns the
contiguous range ``[h * head_dim, (h + 1) * head_dim)`` of the projection's
output dimension. Masking those ranges is therefore exactly masking per
head -- which is what lets FLAIR route at head granularity without
reimplementing any attention maths. ``scripts/verify_api.py`` checks that
reshape convention; if it ever changes, this premise is invalid.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, order=True)
class HeadUnit:
    """One routable attention head. Field order fixes the sort order."""

    block: int
    head: int


def head_slice(head: int, head_dim: int) -> slice:
    """The projection-output dimensions owned by ``head``."""
    return slice(head * head_dim, (head + 1) * head_dim)


def alpha_vector(
    alphas: dict[int, float],
    n_heads: int,
    head_dim: int,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Per-head injection strengths, laid out over the projection output.

    Heads absent from ``alphas`` get 0.0, so multiplying a projected
    residual by this vector both scales the selected heads and erases the
    unselected ones in a single operation.
    """
    vector = torch.zeros(n_heads * head_dim, device=device, dtype=dtype)
    for head, alpha in alphas.items():
        if not 0 <= head < n_heads:
            raise ValueError(f"head {head} out of range for {n_heads} heads")
        vector[head_slice(head, head_dim)] = alpha
    return vector
