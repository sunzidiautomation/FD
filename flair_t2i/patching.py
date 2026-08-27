"""Install FLAIR's head-routing wrappers on an SD3.5 transformer, and
bypass blocks.

Routing is applied by wrapping the text stream's Q/K/V projections rather
than by replacing an attention processor -- see ``head_proj.py`` for why.
``bypass_blocks`` is unrelated to routing: it implements the residual
bypass the vital-layer prefilter needs (spec section 3.4).
"""

from __future__ import annotations

from contextlib import contextmanager

from .head_proj import HeadResidualProj
from .processor import PlanRef

#: The text stream's projections. Modifying their inputs is what
#: block-level routing used to do; wrapping their outputs is how head-level
#: routing does it.
TEXT_PROJECTIONS = ("add_q_proj", "add_k_proj", "add_v_proj")

#: Every attention module a transformer block may carry. SD3.5-M gives part
#: of its stack a second attention (``attn2``), and a block whose attention
#: is not listed here routes nothing -- silently, since the wrapper simply
#: never gets installed. ``scripts/verify_api.py`` asserts this covers what
#: diffusers actually defines, so a future ``attn3`` fails loudly instead.
ATTENTION_MODULES = ("attn", "attn2")


def install_head_routing(transformer, ref: PlanRef) -> list[tuple]:
    """Wrap every text-stream projection. Returns handles for removal."""
    handles: list[tuple] = []

    for block_id, block in enumerate(transformer.transformer_blocks):
        for attn_name in ATTENTION_MODULES:
            attn = getattr(block, attn_name, None)
            if attn is None:
                continue
            n_heads = getattr(attn, "heads", None)
            if n_heads is None:
                continue

            for name in TEXT_PROJECTIONS:
                inner = getattr(attn, name, None)
                if inner is None:
                    continue  # final block: context_pre_only=True
                setattr(
                    attn,
                    name,
                    HeadResidualProj(
                        inner,
                        block_id=block_id,
                        ref=ref,
                        n_heads=n_heads,
                        head_dim=inner.out_features // n_heads,
                    ),
                )
                handles.append((attn, name, inner))

    return handles


def uninstall_head_routing(handles: list[tuple]) -> None:
    for attn, name, inner in handles:
        setattr(attn, name, inner)


@contextmanager
def bypass_blocks(transformer, block_ids: set[int]):
    """Temporarily skip the given blocks, passing their inputs straight through."""
    originals = {}

    for block_id in block_ids:
        block = transformer.transformer_blocks[block_id]
        originals[block_id] = block.forward

        def passthrough(hidden_states, encoder_hidden_states=None, *args, **kwargs):
            if encoder_hidden_states is None:
                return hidden_states
            return encoder_hidden_states, hidden_states

        block.forward = passthrough

    try:
        yield
    finally:
        for block_id, original in originals.items():
            transformer.transformer_blocks[block_id].forward = original
