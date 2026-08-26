import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flair_t2i.attributes import AttributeClass
from flair_t2i.components import Component
from flair_t2i.config import FlairConfig
from flair_t2i.hasm import HASM
from flair_t2i.heads import HeadUnit
from flair_t2i.pipeline import FlairPipeline

SEQ, DIM = 4, 8

PROJECTIONS = ("add_q_proj", "add_k_proj", "add_v_proj")


class StubAttn:
    def __init__(self):
        self.heads = 2
        self.add_q_proj = torch.nn.Linear(DIM, DIM)
        self.add_k_proj = torch.nn.Linear(DIM, DIM)
        self.add_v_proj = torch.nn.Linear(DIM, DIM)


class StubBlock:
    def __init__(self):
        self.attn = StubAttn()


class StubTransformer:
    def __init__(self, n_blocks=3):
        self.transformer_blocks = [StubBlock() for _ in range(n_blocks)]


class StubPipe:
    """Mimics the slice of the diffusers SD3 pipeline FlairPipeline touches."""

    def __init__(self):
        self.transformer = StubTransformer()
        self.calls = []

    def encode_prompt(self, prompt, prompt_2=None, prompt_3=None, **kwargs):
        n = len(prompt) if isinstance(prompt, list) else 1
        return (
            torch.ones((n, SEQ, DIM)),
            torch.zeros((n, SEQ, DIM)),
            torch.ones((n, DIM)),
            torch.zeros((n, DIM)),
        )

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return type("Out", (), {"images": ["IMAGE"]})()


def _hasm():
    # 3 blocks x 2 heads x COLOR; block 0 head 1 is the single peak
    tensor = np.array([[[0.4], [0.9]], [[0.2], [0.1]], [[0.3], [0.3]]])
    return HASM(tensor, (0, 1, 2), (0, 1), (AttributeClass.COLOR,))


def test_encode_components_returns_one_embedding_per_component():
    fp = FlairPipeline(StubPipe(), FlairConfig(device="cpu"), _hasm())
    components = [Component(id="c_color", text="a red car", attr=AttributeClass.COLOR)]

    embeddings = fp.encode_components(components)

    assert set(embeddings) == {"c_color"}
    assert embeddings["c_color"].shape == (SEQ, DIM)


def test_generate_installs_routing_and_returns_image(monkeypatch):
    pipe = StubPipe()
    fp = FlairPipeline(pipe, FlairConfig(device="cpu"), _hasm())
    monkeypatch.setattr(
        "flair_t2i.pipeline.parse_prompt",
        lambda prompt, nlp=None: [
            Component(id="c_color", text="a red car", attr=AttributeClass.COLOR)
        ],
    )

    image = fp.generate("a red car", seed=1, steps=4)

    assert image == "IMAGE"
    assert fp.last_plan is not None
    assert fp.last_plan.routed[0].units == ((HeadUnit(0, 1), 0.9),)
    assert fp.last_plan.blocks_touched() == frozenset({0})


def test_block_granularity_selects_every_head(monkeypatch):
    pipe = StubPipe()
    fp = FlairPipeline(
        pipe, FlairConfig(device="cpu"), _hasm(), granularity="block"
    )
    monkeypatch.setattr(
        "flair_t2i.pipeline.parse_prompt",
        lambda prompt, nlp=None: [
            Component(id="c_color", text="a red car", attr=AttributeClass.COLOR)
        ],
    )

    fp.generate("a red car", steps=4)

    assert fp.last_plan.routed[0].units == (
        (HeadUnit(0, 0), 0.9),
        (HeadUnit(0, 1), 0.9),
    )


def test_generate_with_routing_disabled_builds_no_plan(monkeypatch):
    pipe = StubPipe()
    fp = FlairPipeline(pipe, FlairConfig(device="cpu"), _hasm())
    monkeypatch.setattr(
        "flair_t2i.pipeline.parse_prompt",
        lambda prompt, nlp=None: [
            Component(id="c_color", text="a red car", attr=AttributeClass.COLOR)
        ],
    )

    fp.generate("a red car", routing=False)

    assert fp.last_plan is None


def test_generate_restores_original_projections(monkeypatch):
    pipe = StubPipe()
    before = [
        [getattr(b.attn, name) for name in PROJECTIONS]
        for b in pipe.transformer.transformer_blocks
    ]
    fp = FlairPipeline(pipe, FlairConfig(device="cpu"), _hasm())
    monkeypatch.setattr(
        "flair_t2i.pipeline.parse_prompt",
        lambda prompt, nlp=None: [
            Component(id="c_color", text="a red car", attr=AttributeClass.COLOR)
        ],
    )

    fp.generate("a red car")

    after = [
        [getattr(b.attn, name) for name in PROJECTIONS]
        for b in pipe.transformer.transformer_blocks
    ]
    assert after == before


def test_encode_components_projects_through_context_embedder_when_present():
    pipe = StubPipe()
    # Mock context_embedder mapping DIM (8) to OUT_DIM (12)
    OUT_DIM = 12
    pipe.transformer.context_embedder = torch.nn.Linear(DIM, OUT_DIM)
    fp = FlairPipeline(pipe, FlairConfig(device="cpu"), _hasm())
    components = [Component(id="c_color", text="a red car", attr=AttributeClass.COLOR)]

    embeddings = fp.encode_components(components)

    assert set(embeddings) == {"c_color"}
    assert embeddings["c_color"].shape == (SEQ, OUT_DIM)

