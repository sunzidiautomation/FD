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
        self.encoded: list[list[str]] = []

    def encode_prompt(self, prompt, prompt_2=None, prompt_3=None, **kwargs):
        self.encoded.append(list(prompt) if isinstance(prompt, list) else [prompt])
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


def test_encode_components_caches_by_text():
    """The text encoders are frozen, so the same phrase always encodes the same.

    make_swap_generate_fn calls this once per generation, and a sweep has
    thousands of generations over a handful of distinct phrases. Without a
    cache, T5-XXL runs thousands of times -- and under cpu_offload each run
    drags 9.5GB onto the GPU and back.
    """
    pipe = StubPipe()
    fp = FlairPipeline(pipe, FlairConfig(device="cpu"), _hasm())
    component = Component(id="swap", text="a blue car", attr=AttributeClass.COLOR)

    for _ in range(5):
        fp.encode_components([component])

    assert pipe.encoded == [["a blue car"]]


def test_encode_components_encodes_each_distinct_text_once():
    pipe = StubPipe()
    fp = FlairPipeline(pipe, FlairConfig(device="cpu"), _hasm())

    fp.encode_components([Component(id="a", text="a red car", attr=AttributeClass.COLOR)])
    fp.encode_components([Component(id="b", text="a blue car", attr=AttributeClass.COLOR)])
    fp.encode_components([Component(id="c", text="a red car", attr=AttributeClass.COLOR)])

    assert pipe.encoded == [["a red car"], ["a blue car"]]


def test_encode_components_dedupes_within_one_call():
    pipe = StubPipe()
    fp = FlairPipeline(pipe, FlairConfig(device="cpu"), _hasm())

    fp.encode_components(
        [
            Component(id="a", text="a red car", attr=AttributeClass.COLOR),
            Component(id="b", text="a red car", attr=AttributeClass.SIZE),
        ]
    )

    assert pipe.encoded == [["a red car"]]


def test_cached_embeddings_are_returned_per_component_id():
    pipe = StubPipe()
    fp = FlairPipeline(pipe, FlairConfig(device="cpu"), _hasm())
    shared = "a red car"

    first = fp.encode_components([Component(id="a", text=shared, attr=AttributeClass.COLOR)])
    second = fp.encode_components([Component(id="b", text=shared, attr=AttributeClass.SIZE)])

    assert set(first) == {"a"} and set(second) == {"b"}
    torch.testing.assert_close(first["a"], second["b"])


def test_encode_components_returns_one_embedding_per_component():
    fp = FlairPipeline(StubPipe(), FlairConfig(device="cpu"), _hasm())
    components = [Component(id="c_color", text="a red car", attr=AttributeClass.COLOR)]

    embeddings = fp.encode_components(components)

    assert set(embeddings) == {"c_color"}
    assert embeddings["c_color"].shape == (SEQ, DIM)


def test_generate_pins_resolution_from_config():
    """SD3.5 defaults to 1024x1024 when height/width are omitted.

    That is 4-6x the cost of 512 and silently invalidates every budget in
    the campaign plan, which assumes 512. Pin it here so it cannot drift
    back by omission.
    """
    pipe = StubPipe()
    fp = FlairPipeline(pipe, FlairConfig(device="cpu"), _hasm())

    fp.generate("a red car", steps=2, routing=False)

    assert pipe.calls[0]["height"] == 512
    assert pipe.calls[0]["width"] == 512


def test_generate_honours_a_resolution_override():
    pipe = StubPipe()
    cfg = FlairConfig(device="cpu", height=768, width=768)
    fp = FlairPipeline(pipe, cfg, _hasm())

    fp.generate("a red car", steps=2, routing=False)

    assert pipe.calls[0]["height"] == 768
    assert pipe.calls[0]["width"] == 768


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

