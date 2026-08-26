import pytest

torch = pytest.importorskip("torch")

from flair_t2i.head_proj import HeadResidualProj
from flair_t2i.patching import (
    bypass_blocks,
    install_head_routing,
    uninstall_head_routing,
)
from flair_t2i.processor import PlanRef

DIM, N_HEADS, HEAD_DIM = 8, 3, 4
INNER = N_HEADS * HEAD_DIM

PROJECTIONS = ("add_q_proj", "add_k_proj", "add_v_proj")


class StubAttn:
    def __init__(self, with_q=True):
        self.heads = N_HEADS
        self.add_k_proj = torch.nn.Linear(DIM, INNER)
        self.add_v_proj = torch.nn.Linear(DIM, INNER)
        if with_q:
            self.add_q_proj = torch.nn.Linear(DIM, INNER)


class StubBlock:
    def __init__(self, with_q=True):
        self.attn = StubAttn(with_q=with_q)


class StubTransformer:
    def __init__(self, n_blocks=3, last_lacks_q=False):
        self.transformer_blocks = [
            StubBlock(with_q=not (last_lacks_q and i == n_blocks - 1))
            for i in range(n_blocks)
        ]


def test_install_wraps_every_text_projection():
    transformer = StubTransformer()
    install_head_routing(transformer, PlanRef())

    for block in transformer.transformer_blocks:
        for name in PROJECTIONS:
            assert isinstance(getattr(block.attn, name), HeadResidualProj)


def test_install_records_the_block_id():
    transformer = StubTransformer()
    install_head_routing(transformer, PlanRef())
    assert transformer.transformer_blocks[2].attn.add_q_proj.block_id == 2


def test_install_derives_head_geometry():
    transformer = StubTransformer()
    install_head_routing(transformer, PlanRef())

    wrapper = transformer.transformer_blocks[0].attn.add_v_proj
    assert wrapper.n_heads == N_HEADS
    assert wrapper.head_dim == HEAD_DIM


def test_uninstall_restores_the_originals():
    transformer = StubTransformer()
    before = [
        [getattr(b.attn, name) for name in PROJECTIONS]
        for b in transformer.transformer_blocks
    ]

    handles = install_head_routing(transformer, PlanRef())
    uninstall_head_routing(handles)

    after = [
        [getattr(b.attn, name) for name in PROJECTIONS]
        for b in transformer.transformer_blocks
    ]
    assert after == before


def test_absent_projection_is_skipped_not_crashed():
    transformer = StubTransformer(last_lacks_q=True)
    install_head_routing(transformer, PlanRef())

    last = transformer.transformer_blocks[-1].attn
    assert not hasattr(last, "add_q_proj")
    assert isinstance(last.add_v_proj, HeadResidualProj)


def test_bypass_blocks_still_restores_forward():
    transformer = StubTransformer()
    original = transformer.transformer_blocks[1].forward = lambda *a, **k: "real"

    with bypass_blocks(transformer, {1}):
        assert transformer.transformer_blocks[1].forward is not original
    assert transformer.transformer_blocks[1].forward is original
