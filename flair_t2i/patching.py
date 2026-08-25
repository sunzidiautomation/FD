"""Install FLAIR processors onto an SD3.5 transformer, and bypass blocks.

``bypass_blocks`` implements the residual bypass the vital-layer prefilter
needs (spec section 3.4): the block is skipped and its input passes through
unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager

from .processor import FlairJointProcessor, PlanRef


def install_flair(transformer, ref: PlanRef) -> list[tuple]:
    """Wrap every joint attention processor. Returns handles for removal."""
    handles = []
    for block_id, block in enumerate(transformer.transformer_blocks):
        attn = block.attn
        original = attn.get_processor()
        attn.set_processor(FlairJointProcessor(original, block_id=block_id, ref=ref))
        handles.append((attn, original))
    return handles


def uninstall_flair(handles: list[tuple]) -> None:
    for attn, original in handles:
        attn.set_processor(original)


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
