"""Text-stream projection wrapper that injects a head-masked residual.

Wrapping ``add_q_proj`` / ``add_k_proj`` / ``add_v_proj`` -- rather than
reimplementing joint attention -- is what keeps FLAIR's guarantee that no
attention maths is reimplemented anywhere. The residual lands immediately
after the projection and therefore BEFORE the QK norm, which is what makes
the all-heads case exactly equal to the block-level blend (RMSNorm is not
linear, so a residual added after it would not be).
"""

from __future__ import annotations

import torch

from .processor import PlanRef


class HeadResidualProj(torch.nn.Module):
    def __init__(
        self,
        inner: torch.nn.Linear,
        block_id: int,
        ref: PlanRef,
        n_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.block_id = block_id
        self.ref = ref
        self.n_heads = n_heads
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)

        plan = self.ref.plan
        if plan is None:
            return out

        cond = self.ref.cond_slice(x.shape[0])
        residual = plan.head_residual(
            block_id=self.block_id,
            x=x,
            weight=self.inner.weight,
            step_frac=self.ref.step_frac(),
            cond_slice=cond,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
        )
        if residual is None:
            return out

        out = out.clone()
        out[cond] = out[cond] + residual
        return out
