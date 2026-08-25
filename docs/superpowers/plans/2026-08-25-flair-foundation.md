# FLAIR Foundation (Weeks 1-2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and unit-test the FLAIR code foundation — semantic parsing, BASM container, block-routed residual injection into SD3.5's MM-DiT, coherence guard, and the fuzzy hedge/value module — so that Week 3's BASM calibration runs have a verified, GPU-cheap codebase to sit on.

**Architecture:** A plain Python package (`flair_t2i`) that separates CPU-testable logic (parsing, fuzzy math, BASM, routing decisions, blend arithmetic) from the thin GPU-dependent layer (attention-processor patching). Every routing decision is unit-tested with fake tensors on CPU so Kaggle GPU quota is spent on calibration, not on debugging logic. Component text streams are encoded once upfront and stored in the `RoutingPlan`; they never enter the denoising batch, which eliminates the batch-layout bug class entirely rather than patching it.

**Tech Stack:** Python 3.10+, PyTorch, diffusers (SD3.5-Medium `SD3Transformer2DModel`), transformers (T5/CLIP encoders), spaCy `en_core_web_sm`, scikit-fuzzy, NumPy, pytest.

**Spec:** [`docs/superpowers/specs/2026-08-25-flair-cvpr-publication-plan-design.md`](../specs/2026-08-25-flair-cvpr-publication-plan-design.md)
**Architecture overview:** [`docs/superpowers/specs/2026-08-25-flair-architecture-overview.md`](../specs/2026-08-25-flair-architecture-overview.md)

## Scope

This plan covers **spec §3.1-§3.6 for Weeks 1-2 only** — the code foundation. It deliberately excludes:

- **Week 3+ BASM calibration runs** (§3.4 execution) — the container and query API are built here (Task 3), but the actual GPU calibration campaign gets its own plan once this foundation is verified.
- **Evaluation pipeline** (§4) — separate plan.
- **§3.7 FLUX port and §3.8 fuzzy conflict resolution** — both gated behind the Week 7 checkpoint; not built here.

Tasks 1-7 are Week 1 (crisp pipeline). Tasks 8-10 are Week 2 (fuzzy layer). Per spec §3.3, the crisp pipeline must be validated end-to-end (Task 7) before the fuzzy layer is started (Task 8) — do not interleave them.

## Global Constraints

- **Package name:** `flair_t2i` (NOT `flair` — that name collides with the flairNLP package on PyPI and would shadow it in a Kaggle environment).
- **Python:** 3.10+ (uses `X | None` union syntax).
- **Pinned dependencies** (exact strings, from the spec's reference notebook):
  `diffusers==0.39.0`, `transformers>=4.44.0`, `accelerate`, `safetensors`, `sentencepiece`, `protobuf`, `scikit-fuzzy`, `spacy`, `lpips`, `scipy`, `scikit-image`
- **spaCy model:** `en_core_web_sm`
- **The 7 attribute classes** are exactly: `identity`, `color`, `size`, `lighting`, `texture`, `style`, `action`. The 4 **core** attributes (used for the controllability curve, §4, and the FLUX mini-BASM, §3.7) are exactly: `identity`, `color`, `size`, `lighting`.
- **Backbone:** `stabilityai/stable-diffusion-3.5-medium`. All GPU work targets Kaggle free tier (T4/P100, 16GB) — code must never assume >16GB VRAM.
- **No test may require a GPU or download SD3.5.** Tests that would need the real transformer use stubs. Only the Task 7 smoke test runs on Kaggle GPU, and it is a script, not a pytest test.
- **Injection formula** (spec §3.3, §3.5), authoritative:
  `H_ℓ = H_base + Σ_i α_i(t) · (H_i − H_base)` where `α_i(t) = α_0 · S[ℓ,a] · intensity_i · sched(t)`
- **Batch-layout rule:** component streams are NEVER stacked into the denoising batch. They are text-encoded once and stored in the `RoutingPlan`. Any code computing a base-row index by arithmetic on batch size is a bug.

---

### Task 1: Scaffolding, core types, and the batch-layout fix

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `flair_t2i/__init__.py`
- Create: `flair_t2i/attributes.py`
- Create: `flair_t2i/config.py`
- Create: `flair_t2i/components.py`
- Test: `tests/test_components.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `AttributeClass` (str Enum): `IDENTITY, COLOR, SIZE, LIGHTING, TEXTURE, STYLE, ACTION`; module constant `CORE_ATTRIBUTES: tuple[AttributeClass, ...]`
  - `FlairConfig` dataclass: fields `device: str = "cuda"`, `alpha_0: float = 0.75`, `t_window: tuple[float, float] = (0.0, 0.6)`, `top_k_default: int = 1`, `guard_cos_threshold: float = 0.55`, `guard_membership_threshold: float = 0.5`, `guard_backoff: float = 0.5`, `model_id: str = "stabilityai/stable-diffusion-3.5-medium"`, `max_sequence_length: int = 256`; module constant `DEFAULT_CONFIG: FlairConfig`
  - `Component` frozen dataclass: `id: str`, `text: str`, `attr: AttributeClass`, `hedge: str | None = None`
  - `TextBatchLayout` frozen dataclass: `component_ids: tuple[str, ...]`; class attr `BASE_ROW: int = 0`; methods `n_rows() -> int`, `row_for(component_id: str) -> int`, `validate(batch_size: int) -> None`

- [ ] **Step 1: Initialize the repository**

`C:\FD` is not yet a git repo. Run from `C:\FD`:

```bash
git init
git config user.name "Sunzidul Islam"
git config user.email "sunzidulislam12@gmail.com"
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
*.egg-info/
.ipynb_checkpoints/
outputs/
calibration_runs/
*.safetensors
*.ckpt
*.png
!docs/**/*.png
```

- [ ] **Step 3: Create `requirements.txt`**

```
diffusers==0.39.0
transformers>=4.44.0
accelerate
safetensors
sentencepiece
protobuf
scikit-fuzzy
spacy
lpips
scipy
scikit-image
numpy
pytest
```

- [ ] **Step 4: Create `pyproject.toml`**

```toml
[project]
name = "flair-t2i"
version = "0.1.0"
description = "FLAIR: block-routed attribute control for MM-DiT text-to-image generation"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
include = ["flair_t2i*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 5: Write the failing test for core types**

Create `tests/test_components.py`:

```python
import pytest

from flair_t2i.attributes import AttributeClass, CORE_ATTRIBUTES
from flair_t2i.components import Component, TextBatchLayout


def test_seven_attribute_classes_exist():
    assert len(list(AttributeClass)) == 7
    assert AttributeClass.COLOR.value == "color"
    assert AttributeClass.ACTION.value == "action"


def test_core_attributes_are_the_documented_four():
    assert CORE_ATTRIBUTES == (
        AttributeClass.IDENTITY,
        AttributeClass.COLOR,
        AttributeClass.SIZE,
        AttributeClass.LIGHTING,
    )


def test_layout_row_zero_is_always_base():
    layout = TextBatchLayout(component_ids=("c_color", "c_size"))
    assert layout.BASE_ROW == 0
    assert layout.n_rows() == 3
    assert layout.row_for("c_color") == 1
    assert layout.row_for("c_size") == 2


def test_layout_rejects_wrong_batch_size():
    layout = TextBatchLayout(component_ids=("c_color",))
    layout.validate(2)  # base + 1 component: OK
    with pytest.raises(ValueError, match="expected 2 rows"):
        layout.validate(3)


def test_layout_rejects_unknown_component():
    layout = TextBatchLayout(component_ids=("c_color",))
    with pytest.raises(KeyError, match="c_bogus"):
        layout.row_for("c_bogus")


def test_layout_rejects_duplicate_component_ids():
    with pytest.raises(ValueError, match="duplicate"):
        TextBatchLayout(component_ids=("c_color", "c_color"))


def test_component_is_frozen():
    c = Component(id="c_color", text="a red sports car", attr=AttributeClass.COLOR)
    assert c.hedge is None
    with pytest.raises(Exception):
        c.text = "changed"
```

- [ ] **Step 6: Run the test to verify it fails**

Run: `python -m pytest tests/test_components.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i'`

- [ ] **Step 7: Create `flair_t2i/__init__.py`**

```python
"""FLAIR: block-routed attribute control for MM-DiT text-to-image generation."""

__version__ = "0.1.0"
```

- [ ] **Step 8: Create `flair_t2i/attributes.py`**

```python
"""The seven attribute classes FLAIR routes, per spec section 3.1."""

from enum import Enum


class AttributeClass(str, Enum):
    IDENTITY = "identity"
    COLOR = "color"
    SIZE = "size"
    LIGHTING = "lighting"
    TEXTURE = "texture"
    STYLE = "style"
    ACTION = "action"


#: The four attributes used for the controllability curve (spec section 4)
#: and the FLUX mini-BASM (spec section 3.7).
CORE_ATTRIBUTES: tuple[AttributeClass, ...] = (
    AttributeClass.IDENTITY,
    AttributeClass.COLOR,
    AttributeClass.SIZE,
    AttributeClass.LIGHTING,
)
```

- [ ] **Step 9: Create `flair_t2i/config.py`**

```python
"""Runtime configuration. Defaults follow spec sections 3.3-3.6."""

from dataclasses import dataclass


@dataclass
class FlairConfig:
    device: str = "cuda"

    # Injection (spec section 3.5)
    alpha_0: float = 0.75
    t_window: tuple[float, float] = (0.0, 0.6)
    top_k_default: int = 1

    # Coherence guard (spec section 3.6)
    guard_cos_threshold: float = 0.55
    guard_membership_threshold: float = 0.5
    guard_backoff: float = 0.5

    # Backbone
    model_id: str = "stabilityai/stable-diffusion-3.5-medium"
    max_sequence_length: int = 256


DEFAULT_CONFIG = FlairConfig()
```

- [ ] **Step 10: Create `flair_t2i/components.py`**

Note the deliberate absence of any `batch_size - n_components` arithmetic — the layout is explicit and validated. This is the fix for the bug flagged in spec §3.5.

```python
"""Parsed prompt components and the explicit text-encoding batch layout.

Spec section 3.5 flags an implicit batch-layout assumption
(``base_i = B - n_rows``) as the single most dangerous implementation
detail in FLAIR. This module removes that arithmetic entirely: row 0 is
always the base prompt, component rows are named, and every access is
validated.

Note that this layout describes the TEXT-ENCODING batch only. Component
streams are never stacked into the denoising batch -- see
``flair_t2i.routing``.
"""

from dataclasses import dataclass

from .attributes import AttributeClass


@dataclass(frozen=True)
class Component:
    """One attribute component extracted from a prompt."""

    id: str
    text: str
    attr: AttributeClass
    hedge: str | None = None


@dataclass(frozen=True)
class TextBatchLayout:
    """Maps text-encoder batch rows to streams. Row 0 is always the base."""

    component_ids: tuple[str, ...]

    BASE_ROW: int = 0

    def __post_init__(self) -> None:
        if len(set(self.component_ids)) != len(self.component_ids):
            raise ValueError(f"duplicate component ids: {self.component_ids}")

    def n_rows(self) -> int:
        return 1 + len(self.component_ids)

    def row_for(self, component_id: str) -> int:
        try:
            return 1 + self.component_ids.index(component_id)
        except ValueError:
            raise KeyError(
                f"{component_id} is not in this layout: {self.component_ids}"
            ) from None

    def validate(self, batch_size: int) -> None:
        if batch_size != self.n_rows():
            raise ValueError(
                f"expected {self.n_rows()} rows "
                f"(base + {len(self.component_ids)} components), got {batch_size}"
            )
```

- [ ] **Step 11: Run the test to verify it passes**

Run: `python -m pytest tests/test_components.py -v`
Expected: PASS — 7 passed

- [ ] **Step 12: Commit**

```bash
git add .gitignore requirements.txt pyproject.toml flair_t2i/ tests/ docs/
git commit -m "feat: scaffold flair_t2i package with explicit batch layout

Row 0 is always the base prompt and every component row is named and
validated, replacing the implicit base_i = B - n_rows arithmetic flagged
in spec section 3.5."
```

---

### Task 2: Semantic parser

**Files:**
- Create: `flair_t2i/parsing.py`
- Test: `tests/test_parsing.py`

**Interfaces:**
- Consumes: `Component`, `AttributeClass` from Task 1.
- Produces:
  - `ATTRIBUTE_LEXICON: dict[AttributeClass, frozenset[str]]`
  - `HEDGE_WORDS: frozenset[str]`
  - `parse_prompt(prompt: str, nlp=None) -> list[Component]` — component ids are `f"c_{attr.value}"`; at most one component per attribute class (first wins); the identity component's text is the head noun chunk; every other component's text is `f"{modifier} {head_noun}"` for object-bound attributes and the bare phrase for scene-level ones (lighting, style).

- [ ] **Step 1: Write the failing test**

Create `tests/test_parsing.py`:

```python
import pytest

spacy = pytest.importorskip("spacy")

from flair_t2i.attributes import AttributeClass
from flair_t2i.parsing import parse_prompt


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


def _by_attr(components):
    return {c.attr: c for c in components}


def test_parses_the_running_example(nlp):
    got = _by_attr(parse_prompt("A small red sports car under warm evening light", nlp))

    assert set(got) == {
        AttributeClass.IDENTITY,
        AttributeClass.SIZE,
        AttributeClass.COLOR,
        AttributeClass.LIGHTING,
    }
    assert got[AttributeClass.IDENTITY].text == "sports car"
    assert got[AttributeClass.SIZE].text == "small sports car"
    assert got[AttributeClass.COLOR].text == "red sports car"
    assert "warm evening light" in got[AttributeClass.LIGHTING].text


def test_component_ids_are_stable_and_attribute_derived(nlp):
    components = parse_prompt("a red car", nlp)
    ids = {c.id for c in components}
    assert "c_color" in ids
    assert "c_identity" in ids


def test_detects_hedge_words(nlp):
    got = _by_attr(parse_prompt("a very red car", nlp))
    assert got[AttributeClass.COLOR].hedge == "very"


def test_no_hedge_is_none(nlp):
    got = _by_attr(parse_prompt("a red car", nlp))
    assert got[AttributeClass.COLOR].hedge is None


def test_prompt_with_no_attributes_yields_identity_only(nlp):
    components = parse_prompt("a car", nlp)
    assert [c.attr for c in components] == [AttributeClass.IDENTITY]


def test_texture_and_action_are_recognised(nlp):
    got = _by_attr(parse_prompt("a rusty car driving", nlp))
    assert got[AttributeClass.TEXTURE].text == "rusty car"
    assert AttributeClass.ACTION in got
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_parsing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.parsing'`

(If it instead fails/skips with a spaCy model error, run `python -m spacy download en_core_web_sm` first.)

- [ ] **Step 3: Write the parser**

Create `flair_t2i/parsing.py`:

```python
"""Deterministic prompt parsing into attribute components (spec section 3.2).

A dependency parse binds modifiers to the head noun. Each modifier is
classified into one of the seven attribute classes by lexicon lookup.
Unknown modifiers are ignored rather than guessed at -- the calibration
prompts (spec section 3.4) are authored against this lexicon.
"""

from __future__ import annotations

from .attributes import AttributeClass
from .components import Component

ATTRIBUTE_LEXICON: dict[AttributeClass, frozenset[str]] = {
    AttributeClass.COLOR: frozenset(
        {
            "red", "blue", "green", "yellow", "orange", "purple", "pink",
            "black", "white", "grey", "gray", "brown", "silver", "golden",
            "crimson", "teal", "navy", "beige",
        }
    ),
    AttributeClass.SIZE: frozenset(
        {
            "small", "large", "big", "tiny", "huge", "miniature", "massive",
            "compact", "enormous", "petite", "oversized",
        }
    ),
    AttributeClass.TEXTURE: frozenset(
        {
            "rusty", "smooth", "rough", "glossy", "matte", "shiny", "worn",
            "polished", "weathered", "cracked", "woven", "furry", "metallic",
        }
    ),
    AttributeClass.STYLE: frozenset(
        {
            "cyberpunk", "vintage", "retro", "futuristic", "minimalist",
            "baroque", "impressionist", "cartoon", "photorealistic",
            "watercolor", "sketch",
        }
    ),
    AttributeClass.LIGHTING: frozenset(
        {
            "warm", "cool", "bright", "dim", "golden", "moody", "harsh",
            "soft", "backlit", "sunlit", "neon", "candlelit",
        }
    ),
}

#: Scene-level attributes keep their own phrase rather than binding to the
#: object's head noun.
_SCENE_LEVEL = (AttributeClass.LIGHTING, AttributeClass.STYLE)

HEDGE_WORDS: frozenset[str] = frozenset(
    {
        "very", "extremely", "super", "highly", "intensely",
        "quite", "fairly", "moderately", "rather",
        "slightly", "somewhat", "mildly", "faintly", "barely",
        "not",
    }
)

_LIGHTING_HEADS = frozenset({"light", "lighting", "sunlight", "glow", "illumination"})


def _classify(token) -> AttributeClass | None:
    word = token.lemma_.lower()
    for attr, vocab in ATTRIBUTE_LEXICON.items():
        if word in vocab:
            return attr
    return None


def _hedge_for(token) -> str | None:
    for child in token.children:
        if child.lemma_.lower() in HEDGE_WORDS:
            return child.lemma_.lower()
    if token.head is not token and token.head.lemma_.lower() in HEDGE_WORDS:
        return token.head.lemma_.lower()
    return None


def _head_noun_chunk(doc):
    """The first noun chunk that is not part of a lighting/scene phrase."""
    for chunk in doc.noun_chunks:
        if chunk.root.lemma_.lower() in _LIGHTING_HEADS:
            continue
        words = [t.text for t in chunk if not t.is_stop or t.pos_ == "NOUN"]
        stripped = [t.text for t in chunk if _classify(t) is None and not t.is_stop]
        return chunk, " ".join(stripped) if stripped else " ".join(words)
    return None, ""


def parse_prompt(prompt: str, nlp=None) -> list[Component]:
    """Parse ``prompt`` into at most one Component per attribute class."""
    if nlp is None:  # pragma: no cover - convenience path
        import spacy

        nlp = spacy.load("en_core_web_sm")

    doc = nlp(prompt)
    chunk, identity_text = _head_noun_chunk(doc)

    found: dict[AttributeClass, Component] = {}

    if identity_text:
        found[AttributeClass.IDENTITY] = Component(
            id=f"c_{AttributeClass.IDENTITY.value}",
            text=identity_text,
            attr=AttributeClass.IDENTITY,
        )

    for token in doc:
        attr = _classify(token)
        if attr is None or attr in found:
            continue

        if attr in _SCENE_LEVEL:
            phrase = " ".join(
                t.text for t in token.head.subtree if t.lemma_.lower() not in HEDGE_WORDS
            ) if token.head is not token else token.text
            text = phrase.strip() or token.text
        else:
            text = f"{token.text} {identity_text}".strip()

        found[attr] = Component(
            id=f"c_{attr.value}",
            text=text,
            attr=attr,
            hedge=_hedge_for(token),
        )

    for token in doc:
        if AttributeClass.ACTION in found:
            break
        if token.pos_ == "VERB" and token.lemma_.lower() not in {"be"}:
            found[AttributeClass.ACTION] = Component(
                id=f"c_{AttributeClass.ACTION.value}",
                text=f"{identity_text} {token.text}".strip(),
                attr=AttributeClass.ACTION,
                hedge=_hedge_for(token),
            )

    order = list(AttributeClass)
    return [found[a] for a in order if a in found]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_parsing.py -v`
Expected: PASS — 6 passed

If `test_parses_the_running_example` fails on the lighting phrase, print the parse (`[(t.text, t.dep_, t.head.text) for t in doc]`) and adjust `_SCENE_LEVEL` phrase extraction until the assertion holds. Do not weaken the assertion.

- [ ] **Step 5: Commit**

```bash
git add flair_t2i/parsing.py tests/test_parsing.py
git commit -m "feat: deterministic prompt parsing into 7 attribute components"
```

---

### Task 3: BASM container

**Files:**
- Create: `flair_t2i/basm.py`
- Test: `tests/test_basm.py`

**Interfaces:**
- Consumes: `AttributeClass` from Task 1.
- Produces:
  - `class BASM` with:
    - `BASM(matrix: np.ndarray, block_ids: tuple[int, ...], attributes: tuple[AttributeClass, ...])` — `matrix` shape `(len(block_ids), len(attributes))`, values in `[0, 1]`
    - `.score(block_id: int, attr: AttributeClass) -> float`
    - `.top_k(attr: AttributeClass, k: int) -> list[tuple[int, float]]` — descending by score, ties broken by ascending block id
    - `.save(path: str | Path) -> None` / `BASM.load(path) -> BASM` (npz round-trip)
    - `BASM.uniform(block_ids, attributes) -> BASM` — all 0.5, for tests and pre-calibration smoke runs

- [ ] **Step 1: Write the failing test**

Create `tests/test_basm.py`:

```python
import numpy as np
import pytest

from flair_t2i.attributes import AttributeClass
from flair_t2i.basm import BASM

BLOCKS = (3, 7, 11)
ATTRS = (AttributeClass.COLOR, AttributeClass.SIZE)


def _basm():
    matrix = np.array(
        [
            [0.22, 0.81],  # block 3
            [0.93, 0.14],  # block 7
            [0.19, 0.16],  # block 11
        ]
    )
    return BASM(matrix=matrix, block_ids=BLOCKS, attributes=ATTRS)


def test_score_lookup_by_block_and_attribute():
    assert _basm().score(7, AttributeClass.COLOR) == pytest.approx(0.93)
    assert _basm().score(3, AttributeClass.SIZE) == pytest.approx(0.81)


def test_top_k_is_descending_by_score():
    assert _basm().top_k(AttributeClass.COLOR, 2) == [(7, 0.93), (3, 0.22)]
    assert _basm().top_k(AttributeClass.SIZE, 1) == [(3, 0.81)]


def test_top_k_clamps_to_available_blocks():
    assert len(_basm().top_k(AttributeClass.COLOR, 99)) == 3


def test_ties_break_by_ascending_block_id():
    matrix = np.array([[0.5], [0.5]])
    basm = BASM(matrix=matrix, block_ids=(9, 4), attributes=(AttributeClass.COLOR,))
    assert basm.top_k(AttributeClass.COLOR, 2) == [(4, 0.5), (9, 0.5)]


def test_rejects_out_of_range_scores():
    with pytest.raises(ValueError, match="within"):
        BASM(matrix=np.array([[1.4]]), block_ids=(3,), attributes=(AttributeClass.COLOR,))


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        BASM(matrix=np.zeros((2, 2)), block_ids=(3,), attributes=ATTRS)


def test_unknown_attribute_raises():
    with pytest.raises(KeyError, match="lighting"):
        _basm().score(3, AttributeClass.LIGHTING)


def test_save_load_round_trip(tmp_path):
    original = _basm()
    path = tmp_path / "basm.npz"
    original.save(path)
    restored = BASM.load(path)

    assert restored.block_ids == original.block_ids
    assert restored.attributes == original.attributes
    np.testing.assert_allclose(restored.matrix, original.matrix)


def test_uniform_factory_is_all_half():
    basm = BASM.uniform((1, 2), ATTRS)
    assert basm.score(1, AttributeClass.COLOR) == 0.5
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_basm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.basm'`

- [ ] **Step 3: Write the BASM container**

Create `flair_t2i/basm.py`:

```python
"""The Block-Attribute Sensitivity Matrix (spec section 3.4).

Calibration is offline and runs once per backbone; this module is only the
container and query API. The calibration campaign that fills it in has its
own plan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .attributes import AttributeClass


class BASM:
    def __init__(
        self,
        matrix: np.ndarray,
        block_ids: tuple[int, ...],
        attributes: tuple[AttributeClass, ...],
    ) -> None:
        matrix = np.asarray(matrix, dtype=np.float64)
        expected = (len(block_ids), len(attributes))
        if matrix.shape != expected:
            raise ValueError(f"matrix shape {matrix.shape} does not match {expected}")
        if matrix.size and (matrix.min() < 0.0 or matrix.max() > 1.0):
            raise ValueError("sensitivity scores must be within [0, 1]")

        self.matrix = matrix
        self.block_ids = tuple(block_ids)
        self.attributes = tuple(attributes)
        self._block_index = {b: i for i, b in enumerate(self.block_ids)}
        self._attr_index = {a: i for i, a in enumerate(self.attributes)}

    @classmethod
    def uniform(
        cls, block_ids: tuple[int, ...], attributes: tuple[AttributeClass, ...]
    ) -> "BASM":
        """An uncalibrated matrix, for tests and pre-calibration smoke runs."""
        return cls(np.full((len(block_ids), len(attributes)), 0.5), block_ids, attributes)

    def _col(self, attr: AttributeClass) -> int:
        if attr not in self._attr_index:
            raise KeyError(f"{attr.value} is not calibrated in this BASM")
        return self._attr_index[attr]

    def score(self, block_id: int, attr: AttributeClass) -> float:
        if block_id not in self._block_index:
            raise KeyError(f"block {block_id} is not in this BASM")
        return float(self.matrix[self._block_index[block_id], self._col(attr)])

    def top_k(self, attr: AttributeClass, k: int) -> list[tuple[int, float]]:
        col = self.matrix[:, self._col(attr)]
        ranked = sorted(
            zip(self.block_ids, col), key=lambda pair: (-pair[1], pair[0])
        )
        return [(int(b), float(s)) for b, s in ranked[: max(0, k)]]

    def save(self, path: str | Path) -> None:
        np.savez(
            Path(path),
            matrix=self.matrix,
            block_ids=np.array(self.block_ids),
            attributes=np.array([a.value for a in self.attributes]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BASM":
        data = np.load(Path(path), allow_pickle=False)
        return cls(
            matrix=data["matrix"],
            block_ids=tuple(int(b) for b in data["block_ids"]),
            attributes=tuple(AttributeClass(a) for a in data["attributes"]),
        )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_basm.py -v`
Expected: PASS — 9 passed

- [ ] **Step 5: Commit**

```bash
git add flair_t2i/basm.py tests/test_basm.py
git commit -m "feat: BASM container with top-k query and npz persistence"
```

---

### Task 4: RoutingPlan and the blend operation

This is the heart of FLAIR (spec §3.5). Note what it does **not** do: it never indexes the denoising batch by arithmetic. Component embeddings are precomputed and held here; `blend` adds residuals to the conditional rows of whatever batch the pipeline hands it.

**Files:**
- Create: `flair_t2i/schedule.py`
- Create: `flair_t2i/routing.py`
- Test: `tests/test_schedule.py`
- Test: `tests/test_routing.py`

**Interfaces:**
- Consumes: `Component`, `FlairConfig` (Task 1), `BASM` (Task 3).
- Produces:
  - `flair_t2i/schedule.py`: `timestep_scale(step_frac: float, t_window: tuple[float, float]) -> float` — 1.0 at window start, linearly decaying to 0.0 at window end, 0.0 outside.
  - `flair_t2i/routing.py`:
    - `@dataclass RoutedComponent`: `component: Component`, `embedding: torch.Tensor` (shape `[seq, dim]`), `blocks: tuple[tuple[int, float], ...]`, `intensity: float = 1.0`
    - `@dataclass RoutingPlan`: `routed: tuple[RoutedComponent, ...]`, `cfg: FlairConfig`, `active: bool = True`, `alpha_scale: float = 1.0`
      - `.alpha(rc: RoutedComponent, block_id: int, step_frac: float) -> float`
      - `.blend(encoder_hidden_states: torch.Tensor, block_id: int, step_frac: float, cond_slice: slice) -> torch.Tensor`
      - `.blocks_touched() -> frozenset[int]`
    - `build_routing_plan(components, embeddings: dict[str, torch.Tensor], basm: BASM, cfg: FlairConfig, intensities: dict[str, float] | None = None, k_overrides: dict[str, int] | None = None) -> RoutingPlan`

- [ ] **Step 1: Write the failing schedule test**

Create `tests/test_schedule.py`:

```python
import pytest

from flair_t2i.schedule import timestep_scale

WINDOW = (0.0, 0.6)


def test_full_strength_at_window_start():
    assert timestep_scale(0.0, WINDOW) == pytest.approx(1.0)


def test_decays_linearly_to_zero_at_window_end():
    assert timestep_scale(0.3, WINDOW) == pytest.approx(0.5)
    assert timestep_scale(0.6, WINDOW) == pytest.approx(0.0)


def test_zero_outside_window():
    assert timestep_scale(0.75, WINDOW) == 0.0
    assert timestep_scale(0.2, (0.4, 0.8)) == 0.0


def test_offset_window_starts_at_full_strength():
    assert timestep_scale(0.4, (0.4, 0.8)) == pytest.approx(1.0)


def test_degenerate_window_is_a_point():
    assert timestep_scale(0.5, (0.5, 0.5)) == pytest.approx(1.0)
    assert timestep_scale(0.6, (0.5, 0.5)) == 0.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_schedule.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.schedule'`

- [ ] **Step 3: Write the schedule**

Create `flair_t2i/schedule.py`:

```python
"""Timestep weighting for injection (spec section 3.5).

Text influence is strongest early in denoising and fades to nothing by the
end of the window, following the schedule shape HeadRouter reports for
MM-DiT editing.
"""


def timestep_scale(step_frac: float, t_window: tuple[float, float]) -> float:
    """Return the injection scale in [0, 1] at ``step_frac`` of denoising."""
    start, end = t_window
    if step_frac < start or step_frac > end:
        return 0.0
    if end <= start:
        return 1.0
    return 1.0 - (step_frac - start) / (end - start)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest tests/test_schedule.py -v`
Expected: PASS — 5 passed

- [ ] **Step 5: Write the failing routing test**

Create `tests/test_routing.py`:

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flair_t2i.attributes import AttributeClass
from flair_t2i.basm import BASM
from flair_t2i.components import Component
from flair_t2i.config import FlairConfig
from flair_t2i.routing import RoutedComponent, RoutingPlan, build_routing_plan

SEQ, DIM = 4, 8
CFG = FlairConfig(alpha_0=1.0, t_window=(0.0, 1.0), top_k_default=1)


def _plan(blocks=((7, 1.0),), intensity=1.0, embedding_fill=1.0):
    component = Component(id="c_color", text="a red car", attr=AttributeClass.COLOR)
    routed = RoutedComponent(
        component=component,
        embedding=torch.full((SEQ, DIM), embedding_fill),
        blocks=blocks,
        intensity=intensity,
    )
    return RoutingPlan(routed=(routed,), cfg=CFG)


def _states(batch=2, fill=0.0):
    return torch.full((batch, SEQ, DIM), fill)


def test_blend_is_identity_on_untouched_blocks():
    plan = _plan()
    states = _states()
    out = plan.blend(states, block_id=99, step_frac=0.0, cond_slice=slice(1, 2))
    assert out is states


def test_blend_is_identity_when_inactive():
    plan = _plan()
    plan.active = False
    states = _states()
    out = plan.blend(states, block_id=7, step_frac=0.0, cond_slice=slice(1, 2))
    assert out is states


def test_blend_moves_conditional_rows_toward_the_component():
    plan = _plan()
    states = _states(fill=0.0)
    out = plan.blend(states, block_id=7, step_frac=0.0, cond_slice=slice(1, 2))

    # alpha = alpha_0(1.0) * S(1.0) * intensity(1.0) * sched(1.0) = 1.0
    # H = 0 + 1.0 * (1 - 0) = 1.0
    assert out[1].mean().item() == pytest.approx(1.0)


def test_blend_leaves_unconditional_rows_untouched():
    plan = _plan()
    states = _states(fill=0.0)
    out = plan.blend(states, block_id=7, step_frac=0.0, cond_slice=slice(1, 2))
    assert out[0].abs().max().item() == pytest.approx(0.0)


def test_blend_does_not_mutate_its_input():
    plan = _plan()
    states = _states(fill=0.0)
    plan.blend(states, block_id=7, step_frac=0.0, cond_slice=slice(1, 2))
    assert states.abs().max().item() == pytest.approx(0.0)


def test_alpha_scales_with_basm_score_and_intensity():
    plan = _plan(blocks=((7, 0.5),), intensity=0.4)
    rc = plan.routed[0]
    # 1.0 * 0.5 * 0.4 * 1.0
    assert plan.alpha(rc, 7, 0.0) == pytest.approx(0.2)


def test_alpha_respects_guard_backoff():
    plan = _plan()
    plan.alpha_scale = 0.5
    assert plan.alpha(plan.routed[0], 7, 0.0) == pytest.approx(0.5)


def test_alpha_is_zero_outside_timestep_window():
    plan = _plan()
    plan.cfg = FlairConfig(alpha_0=1.0, t_window=(0.0, 0.5))
    assert plan.alpha(plan.routed[0], 7, 0.9) == 0.0


def test_blend_rejects_sequence_length_mismatch():
    plan = _plan()
    bad = torch.zeros((2, SEQ + 1, DIM))
    with pytest.raises(ValueError, match="sequence length"):
        plan.blend(bad, block_id=7, step_frac=0.0, cond_slice=slice(1, 2))


def test_build_routing_plan_selects_top_block_per_attribute():
    components = [
        Component(id="c_color", text="a red car", attr=AttributeClass.COLOR),
        Component(id="c_size", text="a small car", attr=AttributeClass.SIZE),
    ]
    basm = BASM(
        matrix=np.array([[0.22, 0.81], [0.93, 0.14]]),
        block_ids=(3, 7),
        attributes=(AttributeClass.COLOR, AttributeClass.SIZE),
    )
    embeddings = {c.id: torch.zeros((SEQ, DIM)) for c in components}

    plan = build_routing_plan(components, embeddings, basm, CFG)
    by_id = {rc.component.id: rc for rc in plan.routed}

    assert by_id["c_color"].blocks == ((7, 0.93),)
    assert by_id["c_size"].blocks == ((3, 0.81),)
    assert plan.blocks_touched() == frozenset({3, 7})


def test_build_routing_plan_skips_uncalibrated_attributes():
    components = [Component(id="c_action", text="a car driving", attr=AttributeClass.ACTION)]
    basm = BASM.uniform((3,), (AttributeClass.COLOR,))
    plan = build_routing_plan(components, {"c_action": torch.zeros((SEQ, DIM))}, basm, CFG)
    assert plan.routed == ()


def test_build_routing_plan_applies_k_override():
    components = [Component(id="c_color", text="a red car", attr=AttributeClass.COLOR)]
    basm = BASM(
        matrix=np.array([[0.4], [0.9], [0.6]]),
        block_ids=(3, 7, 11),
        attributes=(AttributeClass.COLOR,),
    )
    plan = build_routing_plan(
        components,
        {"c_color": torch.zeros((SEQ, DIM))},
        basm,
        CFG,
        k_overrides={"c_color": 2},
    )
    assert plan.routed[0].blocks == ((7, 0.9), (11, 0.6))
```

- [ ] **Step 6: Run it to verify it fails**

Run: `python -m pytest tests/test_routing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.routing'`

- [ ] **Step 7: Write the routing module**

Create `flair_t2i/routing.py`:

```python
"""Routing plans and the blend operation (spec section 3.5).

    H_l = H_base + sum_i alpha_i(t) * (H_i - H_base)
    alpha_i(t) = alpha_0 * S[l, a] * intensity_i * sched(t)

Component embeddings are encoded once and held on the plan. They are never
stacked into the denoising batch, so there is no base-row arithmetic to get
wrong -- ``blend`` writes only into the conditional rows the caller names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .basm import BASM
from .components import Component
from .config import FlairConfig
from .schedule import timestep_scale


@dataclass
class RoutedComponent:
    component: Component
    embedding: torch.Tensor  # [seq, dim]
    blocks: tuple[tuple[int, float], ...]  # (block_id, basm_score)
    intensity: float = 1.0


@dataclass
class RoutingPlan:
    routed: tuple[RoutedComponent, ...]
    cfg: FlairConfig
    active: bool = True
    alpha_scale: float = 1.0
    _blocks: frozenset[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._blocks = frozenset(
            block_id for rc in self.routed for block_id, _ in rc.blocks
        )

    def blocks_touched(self) -> frozenset[int]:
        return self._blocks

    def alpha(self, rc: RoutedComponent, block_id: int, step_frac: float) -> float:
        score = next((s for b, s in rc.blocks if b == block_id), None)
        if score is None:
            return 0.0
        sched = timestep_scale(step_frac, self.cfg.t_window)
        return self.cfg.alpha_0 * score * rc.intensity * sched * self.alpha_scale

    def blend(
        self,
        encoder_hidden_states: torch.Tensor,
        block_id: int,
        step_frac: float,
        cond_slice: slice,
    ) -> torch.Tensor:
        """Return states with component residuals added at ``block_id``."""
        if not self.active or block_id not in self._blocks:
            return encoder_hidden_states

        contributions = [
            (rc, self.alpha(rc, block_id, step_frac))
            for rc in self.routed
            if block_id in {b for b, _ in rc.blocks}
        ]
        contributions = [(rc, a) for rc, a in contributions if a != 0.0]
        if not contributions:
            return encoder_hidden_states

        seq = encoder_hidden_states.shape[-2]
        for rc, _ in contributions:
            if rc.embedding.shape[-2] != seq:
                raise ValueError(
                    f"component {rc.component.id} sequence length "
                    f"{rc.embedding.shape[-2]} does not match states {seq}"
                )

        out = encoder_hidden_states.clone()
        base = encoder_hidden_states[cond_slice]

        for rc, alpha in contributions:
            target = rc.embedding.to(device=base.device, dtype=base.dtype)
            out[cond_slice] = out[cond_slice] + alpha * (target.unsqueeze(0) - base)

        return out


def build_routing_plan(
    components: list[Component],
    embeddings: dict[str, torch.Tensor],
    basm: BASM,
    cfg: FlairConfig,
    intensities: dict[str, float] | None = None,
    k_overrides: dict[str, int] | None = None,
) -> RoutingPlan:
    """Select target blocks per component from the calibrated BASM."""
    intensities = intensities or {}
    k_overrides = k_overrides or {}

    routed: list[RoutedComponent] = []
    for component in components:
        if component.attr not in basm.attributes:
            continue  # not calibrated; nothing to route
        k = k_overrides.get(component.id, cfg.top_k_default)
        blocks = tuple(basm.top_k(component.attr, k))
        if not blocks:
            continue
        routed.append(
            RoutedComponent(
                component=component,
                embedding=embeddings[component.id],
                blocks=blocks,
                intensity=intensities.get(component.id, 1.0),
            )
        )

    return RoutingPlan(routed=tuple(routed), cfg=cfg)
```

- [ ] **Step 8: Run it to verify it passes**

Run: `python -m pytest tests/test_routing.py -v`
Expected: PASS — 12 passed

- [ ] **Step 9: Commit**

```bash
git add flair_t2i/schedule.py flair_t2i/routing.py tests/test_schedule.py tests/test_routing.py
git commit -m "feat: routing plan and blend with precomputed component streams

Component embeddings live on the plan rather than in the denoising batch,
removing the base-row arithmetic flagged in spec section 3.5."
```

---

### Task 5: Coherence guard (crisp)

**Files:**
- Create: `flair_t2i/guard.py`
- Test: `tests/test_guard.py`

**Interfaces:**
- Consumes: `FlairConfig` (Task 1), `RoutingPlan` (Task 4).
- Produces:
  - `@dataclass GuardEvent`: `step: int`, `reason: str`, `value: float`
  - `class CoherenceGuard`:
    - `CoherenceGuard(cfg: FlairConfig)`
    - `.check_streams(plan: RoutingPlan, step: int) -> GuardEvent | None` — flags when any two routed component embeddings have cosine similarity **below** `cfg.guard_cos_threshold` (they have drifted apart into incoherence)
    - `.apply(plan: RoutingPlan, event: GuardEvent | None) -> None` — multiplies `plan.alpha_scale` by `cfg.guard_backoff` when an event fires
    - `.events: list[GuardEvent]`
  - Task 10 extends this class with `check_membership`; do not add it here.

- [ ] **Step 1: Write the failing test**

Create `tests/test_guard.py`:

```python
import pytest

torch = pytest.importorskip("torch")

from flair_t2i.attributes import AttributeClass
from flair_t2i.components import Component
from flair_t2i.config import FlairConfig
from flair_t2i.guard import CoherenceGuard
from flair_t2i.routing import RoutedComponent, RoutingPlan

SEQ, DIM = 4, 8
CFG = FlairConfig(guard_cos_threshold=0.55, guard_backoff=0.5)


def _plan(*embeddings):
    routed = tuple(
        RoutedComponent(
            component=Component(id=f"c{i}", text="x", attr=AttributeClass.COLOR),
            embedding=e,
            blocks=((7, 1.0),),
        )
        for i, e in enumerate(embeddings)
    )
    return RoutingPlan(routed=routed, cfg=CFG)


def test_no_event_for_aligned_streams():
    a = torch.ones((SEQ, DIM))
    plan = _plan(a, a.clone())
    assert CoherenceGuard(CFG).check_streams(plan, step=3) is None


def test_event_for_opposed_streams():
    a = torch.ones((SEQ, DIM))
    plan = _plan(a, -a)
    event = CoherenceGuard(CFG).check_streams(plan, step=3)
    assert event is not None
    assert event.step == 3
    assert event.reason == "cross_stream_similarity"
    assert event.value < 0.55


def test_single_stream_never_fires():
    plan = _plan(torch.ones((SEQ, DIM)))
    assert CoherenceGuard(CFG).check_streams(plan, step=0) is None


def test_apply_backs_off_alpha_scale():
    a = torch.ones((SEQ, DIM))
    plan = _plan(a, -a)
    guard = CoherenceGuard(CFG)
    event = guard.check_streams(plan, step=1)
    guard.apply(plan, event)
    assert plan.alpha_scale == pytest.approx(0.5)


def test_apply_is_a_noop_without_an_event():
    plan = _plan(torch.ones((SEQ, DIM)))
    guard = CoherenceGuard(CFG)
    guard.apply(plan, None)
    assert plan.alpha_scale == pytest.approx(1.0)


def test_backoff_compounds_across_steps():
    a = torch.ones((SEQ, DIM))
    plan = _plan(a, -a)
    guard = CoherenceGuard(CFG)
    for step in range(2):
        guard.apply(plan, guard.check_streams(plan, step))
    assert plan.alpha_scale == pytest.approx(0.25)
    assert len(guard.events) == 2
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_guard.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.guard'`

- [ ] **Step 3: Write the guard**

Create `flair_t2i/guard.py`:

```python
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
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest tests/test_guard.py -v`
Expected: PASS — 6 passed

- [ ] **Step 5: Commit**

```bash
git add flair_t2i/guard.py tests/test_guard.py
git commit -m "feat: crisp coherence guard with compounding alpha backoff"
```

---

### Task 6: Attention processor and block patching

**Files:**
- Create: `flair_t2i/processor.py`
- Create: `flair_t2i/patching.py`
- Test: `tests/test_processor.py`

**Interfaces:**
- Consumes: `RoutingPlan` (Task 4).
- Produces:
  - `flair_t2i/processor.py`:
    - `@dataclass PlanRef`: `plan: RoutingPlan | None = None`, `step: int = 0`, `total_steps: int = 1`, `do_cfg: bool = True`; `.step_frac() -> float`; `.cond_slice(batch_size: int) -> slice`
    - `class FlairJointProcessor`: `FlairJointProcessor(inner, block_id: int, ref: PlanRef)`; `__call__(attn, hidden_states, encoder_hidden_states=None, *args, **kwargs)` blends `encoder_hidden_states` then delegates to `inner`.
  - `flair_t2i/patching.py`:
    - `install_flair(transformer, ref: PlanRef) -> list[tuple]` (handles)
    - `uninstall_flair(handles) -> None`
    - `bypass_blocks(transformer, block_ids: set[int])` — context manager used by the vital-layer prefilter (spec §3.4)

- [ ] **Step 1: Write the failing test**

Create `tests/test_processor.py`:

```python
import pytest

torch = pytest.importorskip("torch")

from flair_t2i.attributes import AttributeClass
from flair_t2i.components import Component
from flair_t2i.config import FlairConfig
from flair_t2i.processor import FlairJointProcessor, PlanRef
from flair_t2i.routing import RoutedComponent, RoutingPlan

SEQ, DIM = 4, 8
CFG = FlairConfig(alpha_0=1.0, t_window=(0.0, 1.0))


class RecordingProcessor:
    """Stands in for diffusers' JointAttnProcessor2_0."""

    def __init__(self):
        self.seen = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, **kwargs):
        self.seen = encoder_hidden_states
        return hidden_states, encoder_hidden_states


def _ref(block_blocks=((7, 1.0),), total_steps=10, step=0):
    routed = RoutedComponent(
        component=Component(id="c_color", text="a red car", attr=AttributeClass.COLOR),
        embedding=torch.ones((SEQ, DIM)),
        blocks=block_blocks,
    )
    return PlanRef(
        plan=RoutingPlan(routed=(routed,), cfg=CFG),
        step=step,
        total_steps=total_steps,
        do_cfg=True,
    )


def test_step_frac_is_step_over_total():
    assert PlanRef(step=5, total_steps=10).step_frac() == pytest.approx(0.5)


def test_step_frac_handles_zero_total():
    assert PlanRef(step=0, total_steps=0).step_frac() == 0.0


def test_cond_slice_is_second_half_under_cfg():
    assert PlanRef(do_cfg=True).cond_slice(4) == slice(2, 4)


def test_cond_slice_is_everything_without_cfg():
    assert PlanRef(do_cfg=False).cond_slice(2) == slice(0, 2)


def test_processor_blends_before_delegating():
    inner = RecordingProcessor()
    proc = FlairJointProcessor(inner, block_id=7, ref=_ref())
    states = torch.zeros((2, SEQ, DIM))

    proc(attn=None, hidden_states=torch.zeros((2, SEQ, DIM)), encoder_hidden_states=states)

    assert inner.seen[1].mean().item() == pytest.approx(1.0)  # conditional row moved
    assert inner.seen[0].abs().max().item() == pytest.approx(0.0)  # uncond untouched


def test_processor_passes_through_on_unrouted_block():
    inner = RecordingProcessor()
    proc = FlairJointProcessor(inner, block_id=99, ref=_ref())
    states = torch.zeros((2, SEQ, DIM))

    proc(attn=None, hidden_states=torch.zeros((2, SEQ, DIM)), encoder_hidden_states=states)

    assert inner.seen is states


def test_processor_passes_through_when_plan_is_none():
    inner = RecordingProcessor()
    proc = FlairJointProcessor(inner, block_id=7, ref=PlanRef(plan=None))
    states = torch.zeros((2, SEQ, DIM))

    proc(attn=None, hidden_states=torch.zeros((2, SEQ, DIM)), encoder_hidden_states=states)

    assert inner.seen is states


def test_processor_passes_through_when_encoder_states_absent():
    inner = RecordingProcessor()
    proc = FlairJointProcessor(inner, block_id=7, ref=_ref())
    proc(attn=None, hidden_states=torch.zeros((2, SEQ, DIM)), encoder_hidden_states=None)
    assert inner.seen is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_processor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.processor'`

- [ ] **Step 3: Write the processor**

Create `flair_t2i/processor.py`:

```python
"""Attention processor wrapper that injects routed streams (spec section 3.5).

The wrapper modifies ``encoder_hidden_states`` and then delegates to the
backbone's own processor, so no attention maths is reimplemented here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

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


class FlairJointProcessor:
    def __init__(self, inner, block_id: int, ref: PlanRef) -> None:
        self.inner = inner
        self.block_id = block_id
        self.ref = ref

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        plan = self.ref.plan
        if plan is not None and encoder_hidden_states is not None:
            encoder_hidden_states = plan.blend(
                encoder_hidden_states,
                block_id=self.block_id,
                step_frac=self.ref.step_frac(),
                cond_slice=self.ref.cond_slice(encoder_hidden_states.shape[0]),
            )
        return self.inner(
            attn, hidden_states, encoder_hidden_states, *args, **kwargs
        )
```

- [ ] **Step 4: Write the patching helpers**

Create `flair_t2i/patching.py`:

```python
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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_processor.py -v`
Expected: PASS — 8 passed

- [ ] **Step 6: Commit**

```bash
git add flair_t2i/processor.py flair_t2i/patching.py tests/test_processor.py
git commit -m "feat: FlairJointProcessor, processor patching, and block bypass"
```

---

### Task 7: Pipeline wiring and Kaggle smoke test

End of Week 1. After this task the crisp pipeline is validated end-to-end and the fuzzy layer may begin.

**Files:**
- Create: `flair_t2i/pipeline.py`
- Create: `scripts/smoke_test.py`
- Create: `scripts/verify_env.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces:
  - `class FlairPipeline`:
    - `FlairPipeline(pipe, cfg: FlairConfig, basm: BASM, nlp=None)`
    - `.encode_components(components: list[Component]) -> dict[str, torch.Tensor]`
    - `.generate(prompt: str, seed: int = 0, steps: int = 20, guidance_scale: float = 4.5, routing: bool = True) -> PIL.Image.Image`
    - `.last_plan: RoutingPlan | None`, `.last_guard: CoherenceGuard | None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline.py`. It uses a stub pipe — no GPU, no model download.

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flair_t2i.attributes import AttributeClass
from flair_t2i.basm import BASM
from flair_t2i.components import Component
from flair_t2i.config import FlairConfig
from flair_t2i.pipeline import FlairPipeline

SEQ, DIM = 4, 8


class StubBlock:
    def __init__(self):
        self.attn = StubAttn()


class StubAttn:
    def __init__(self):
        self._processor = object()

    def get_processor(self):
        return self._processor

    def set_processor(self, processor):
        self._processor = processor


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


def _basm():
    return BASM(
        matrix=np.array([[0.9], [0.2], [0.4]]),
        block_ids=(0, 1, 2),
        attributes=(AttributeClass.COLOR,),
    )


def test_encode_components_returns_one_embedding_per_component():
    fp = FlairPipeline(StubPipe(), FlairConfig(device="cpu"), _basm())
    components = [Component(id="c_color", text="a red car", attr=AttributeClass.COLOR)]

    embeddings = fp.encode_components(components)

    assert set(embeddings) == {"c_color"}
    assert embeddings["c_color"].shape == (SEQ, DIM)


def test_generate_installs_routing_and_returns_image(monkeypatch):
    pipe = StubPipe()
    fp = FlairPipeline(pipe, FlairConfig(device="cpu"), _basm())
    monkeypatch.setattr(
        "flair_t2i.pipeline.parse_prompt",
        lambda prompt, nlp=None: [
            Component(id="c_color", text="a red car", attr=AttributeClass.COLOR)
        ],
    )

    image = fp.generate("a red car", seed=1, steps=4)

    assert image == "IMAGE"
    assert fp.last_plan is not None
    assert fp.last_plan.blocks_touched() == frozenset({0})


def test_generate_with_routing_disabled_builds_no_plan(monkeypatch):
    pipe = StubPipe()
    fp = FlairPipeline(pipe, FlairConfig(device="cpu"), _basm())
    monkeypatch.setattr(
        "flair_t2i.pipeline.parse_prompt",
        lambda prompt, nlp=None: [
            Component(id="c_color", text="a red car", attr=AttributeClass.COLOR)
        ],
    )

    fp.generate("a red car", routing=False)

    assert fp.last_plan is None


def test_generate_restores_original_processors(monkeypatch):
    pipe = StubPipe()
    before = [b.attn.get_processor() for b in pipe.transformer.transformer_blocks]
    fp = FlairPipeline(pipe, FlairConfig(device="cpu"), _basm())
    monkeypatch.setattr(
        "flair_t2i.pipeline.parse_prompt",
        lambda prompt, nlp=None: [
            Component(id="c_color", text="a red car", attr=AttributeClass.COLOR)
        ],
    )

    fp.generate("a red car")

    after = [b.attn.get_processor() for b in pipe.transformer.transformer_blocks]
    assert after == before
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.pipeline'`

- [ ] **Step 3: Write the pipeline**

Create `flair_t2i/pipeline.py`:

```python
"""Top-level FLAIR pipeline (spec sections 3.2-3.6).

Component streams are text-encoded once here, before denoising, and handed
to the RoutingPlan. The denoising batch keeps its usual
``[negative, positive]`` shape.
"""

from __future__ import annotations

import torch

from .basm import BASM
from .components import Component
from .config import FlairConfig
from .guard import CoherenceGuard
from .parsing import parse_prompt
from .patching import install_flair, uninstall_flair
from .processor import PlanRef
from .routing import RoutingPlan, build_routing_plan


class FlairPipeline:
    def __init__(self, pipe, cfg: FlairConfig, basm: BASM, nlp=None) -> None:
        self.pipe = pipe
        self.cfg = cfg
        self.basm = basm
        self.nlp = nlp
        self.last_plan: RoutingPlan | None = None
        self.last_guard: CoherenceGuard | None = None

    def encode_components(
        self, components: list[Component]
    ) -> dict[str, torch.Tensor]:
        """Encode every component's text once, batched."""
        if not components:
            return {}

        texts = [c.text for c in components]
        prompt_embeds, _, _, _ = self.pipe.encode_prompt(
            prompt=texts,
            prompt_2=texts,
            prompt_3=texts,
            do_classifier_free_guidance=False,
            max_sequence_length=self.cfg.max_sequence_length,
        )
        return {c.id: prompt_embeds[i] for i, c in enumerate(components)}

    def generate(
        self,
        prompt: str,
        seed: int = 0,
        steps: int = 20,
        guidance_scale: float = 4.5,
        routing: bool = True,
    ):
        self.last_plan = None
        self.last_guard = None

        ref = PlanRef(total_steps=steps, do_cfg=guidance_scale > 1.0)

        if routing:
            components = parse_prompt(prompt, self.nlp)
            routable = [c for c in components if c.attr in self.basm.attributes]
            embeddings = self.encode_components(routable)
            plan = build_routing_plan(routable, embeddings, self.basm, self.cfg)
            if plan.routed:
                ref.plan = plan
                self.last_plan = plan
                self.last_guard = CoherenceGuard(self.cfg)
                self.last_guard.apply(
                    plan, self.last_guard.check_streams(plan, step=0)
                )

        handles = install_flair(self.pipe.transformer, ref)
        try:
            def on_step(pipe, step_index, timestep, callback_kwargs):
                ref.step = step_index
                return callback_kwargs

            result = self.pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                callback_on_step_end=on_step,
            )
        finally:
            uninstall_flair(handles)

        return result.images[0]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS — 4 passed

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -v`
Expected: PASS — all tests from Tasks 1-7

- [ ] **Step 6: Write the environment verification script**

Spec §7 flags the `lpips`/`skfuzzy` import failures. This script fails loudly at the top of a Kaggle session instead of 40 cells in.

Create `scripts/verify_env.py`:

```python
"""Verify every FLAIR dependency imports. Run this first in any session."""

import importlib
import sys

REQUIRED = [
    "torch",
    "diffusers",
    "transformers",
    "spacy",
    "skfuzzy",
    "lpips",
    "numpy",
    "scipy",
    "skimage",
]

failed = []
for name in REQUIRED:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "(no __version__)")
        print(f"  OK    {name:<14} {version}")
    except ImportError as exc:
        failed.append(name)
        print(f"  FAIL  {name:<14} {exc}")

try:
    import spacy

    spacy.load("en_core_web_sm")
    print("  OK    en_core_web_sm")
except Exception as exc:  # noqa: BLE001
    failed.append("en_core_web_sm")
    print(f"  FAIL  en_core_web_sm  {exc}")

if failed:
    print(f"\nMissing: {', '.join(failed)}")
    print("Install, then RESTART the session before running anything else.")
    sys.exit(1)

print("\nEnvironment OK.")
```

- [ ] **Step 7: Write the Kaggle smoke test**

Create `scripts/smoke_test.py`:

```python
"""Week 1 exit criterion: routed generation runs end-to-end on SD3.5-M.

Run on Kaggle GPU:
    python scripts/smoke_test.py --steps 20 --out outputs/

This is a manual script, not a pytest test -- it needs a GPU and the
SD3.5-M weights. It uses an UNCALIBRATED uniform BASM, so the images prove
the plumbing works, not that routing helps. Real BASM values arrive in
Week 3.
"""

import argparse
from pathlib import Path

import spacy
import torch
from diffusers import StableDiffusion3Pipeline

from flair_t2i.attributes import CORE_ATTRIBUTES
from flair_t2i.basm import BASM
from flair_t2i.config import FlairConfig
from flair_t2i.pipeline import FlairPipeline

PROMPT = "A small red sports car under warm evening light"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = FlairConfig(device="cuda")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        cfg.model_id, torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()

    n_blocks = len(pipe.transformer.transformer_blocks)
    print(f"SD3.5-M has {n_blocks} transformer blocks")

    basm = BASM.uniform(tuple(range(n_blocks)), CORE_ATTRIBUTES)
    fp = FlairPipeline(pipe, cfg, basm, nlp=spacy.load("en_core_web_sm"))

    baseline = fp.generate(PROMPT, seed=args.seed, steps=args.steps, routing=False)
    baseline.save(args.out / "smoke_baseline.png")
    print("wrote smoke_baseline.png")

    routed = fp.generate(PROMPT, seed=args.seed, steps=args.steps, routing=True)
    routed.save(args.out / "smoke_routed.png")
    print("wrote smoke_routed.png")

    assert fp.last_plan is not None, "routing produced no plan"
    print(f"routed components: {[rc.component.id for rc in fp.last_plan.routed]}")
    print(f"blocks touched:    {sorted(fp.last_plan.blocks_touched())}")
    print(f"guard events:      {len(fp.last_guard.events)}")
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run the smoke test on Kaggle**

Upload the repo to a Kaggle notebook with GPU enabled, then run:

```bash
!pip install -q -r requirements.txt
!python -m spacy download en_core_web_sm -q
!python scripts/verify_env.py
!python scripts/smoke_test.py --steps 20 --out outputs/
```

Expected: `verify_env.py` prints all OK; `smoke_test.py` writes both PNGs and prints a non-empty routed-components list and the blocks touched.

**Week 1 exit criterion:** both images exist, are not identical, and neither is noise. With a uniform BASM the routed image should differ only mildly from baseline — a wildly corrupted routed image means the injection sign or `cond_slice` is wrong; investigate before Task 8.

- [ ] **Step 9: Commit**

```bash
git add flair_t2i/pipeline.py scripts/ tests/test_pipeline.py
git commit -m "feat: FlairPipeline end-to-end wiring, env check, and smoke test"
```

---

### Task 8: Fuzzy attribute-value membership (spec §3.3-A)

Week 2 starts here. Do not start until Task 7's exit criterion is met.

**Files:**
- Create: `flair_t2i/fuzzy/__init__.py`
- Create: `flair_t2i/fuzzy/membership.py`
- Test: `tests/test_membership.py`

**Interfaces:**
- Consumes: `AttributeClass` (Task 1).
- Produces:
  - `UNIVERSE: np.ndarray` — `np.linspace(0.0, 1.0, 201)`, shared grid for every attribute
  - `@dataclass AttributeUniverse`: `attr: AttributeClass`, `metric: str`, `labels: dict[str, np.ndarray]`
  - `UNIVERSES: dict[AttributeClass, AttributeUniverse]` — all 7 attributes
  - `membership_curve(attr: AttributeClass, label: str) -> np.ndarray`
  - `membership_at(attr: AttributeClass, label: str, x: float) -> float`
  - `default_label(attr: AttributeClass) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_membership.py`:

```python
import numpy as np
import pytest

pytest.importorskip("skfuzzy")

from flair_t2i.attributes import AttributeClass
from flair_t2i.fuzzy.membership import (
    UNIVERSE,
    UNIVERSES,
    default_label,
    membership_at,
    membership_curve,
)


def test_every_attribute_has_a_universe():
    assert set(UNIVERSES) == set(AttributeClass)


def test_universe_is_the_unit_interval():
    assert UNIVERSE[0] == pytest.approx(0.0)
    assert UNIVERSE[-1] == pytest.approx(1.0)
    assert len(UNIVERSE) == 201


def test_curves_are_valid_memberships():
    for attr, universe in UNIVERSES.items():
        for label, curve in universe.labels.items():
            assert curve.shape == UNIVERSE.shape, (attr, label)
            assert curve.min() >= 0.0 and curve.max() <= 1.0, (attr, label)
            assert curve.max() == pytest.approx(1.0), (attr, label)


def test_size_labels_peak_where_expected():
    # 'small' peaks at low area ratio, 'large' at high
    assert membership_at(AttributeClass.SIZE, "small", 0.05) == pytest.approx(1.0)
    assert membership_at(AttributeClass.SIZE, "small", 0.9) == pytest.approx(0.0)
    assert membership_at(AttributeClass.SIZE, "large", 0.9) == pytest.approx(1.0)


def test_color_membership_rises_toward_target():
    low = membership_at(AttributeClass.COLOR, "match", 0.2)
    high = membership_at(AttributeClass.COLOR, "match", 0.95)
    assert high > low


def test_default_label_is_defined_for_every_attribute():
    for attr in AttributeClass:
        label = default_label(attr)
        assert label in UNIVERSES[attr].labels


def test_unknown_label_raises():
    with pytest.raises(KeyError, match="enormous"):
        membership_curve(AttributeClass.SIZE, "enormous")


def test_membership_at_clamps_out_of_range_inputs():
    assert membership_at(AttributeClass.SIZE, "small", -5.0) == pytest.approx(1.0)
    assert membership_at(AttributeClass.SIZE, "small", 5.0) == pytest.approx(0.0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_membership.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.fuzzy'`

- [ ] **Step 3: Write the membership module**

Create `flair_t2i/fuzzy/__init__.py`:

```python
"""Fuzzy attribute-value membership and linguistic hedges (spec section 3.3)."""
```

Create `flair_t2i/fuzzy/membership.py`:

```python
"""Fuzzy membership over attribute metric universes (spec section 3.3-A).

Every attribute's metric is normalised onto [0, 1] so one shared grid
serves all seven. The four core attributes carry real linguistic labels;
the remaining three carry a single 'match' label -- their metrics
(Gram/DISTS distance, CLIP action score) are similarity scores without a
natural linguistic scale.

These curves do two jobs: they give BASM calibration a graded target
(section 3.4) and they replace the coherence guard's crisp percentile band
(section 3.6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import skfuzzy as fuzz

from ..attributes import AttributeClass

#: Shared normalised grid. Each attribute maps its own metric onto [0, 1].
UNIVERSE: np.ndarray = np.linspace(0.0, 1.0, 201)


@dataclass(frozen=True)
class AttributeUniverse:
    attr: AttributeClass
    metric: str
    labels: dict[str, np.ndarray]


def _rising() -> np.ndarray:
    """A shoulder that saturates as the metric approaches its target."""
    return fuzz.trapmf(UNIVERSE, [0.35, 0.85, 1.0, 1.0])


UNIVERSES: dict[AttributeClass, AttributeUniverse] = {
    AttributeClass.SIZE: AttributeUniverse(
        attr=AttributeClass.SIZE,
        metric="object mask area ratio",
        labels={
            "small": fuzz.trapmf(UNIVERSE, [0.0, 0.0, 0.12, 0.28]),
            "medium": fuzz.trimf(UNIVERSE, [0.18, 0.38, 0.58]),
            "large": fuzz.trapmf(UNIVERSE, [0.48, 0.70, 1.0, 1.0]),
        },
    ),
    AttributeClass.LIGHTING: AttributeUniverse(
        attr=AttributeClass.LIGHTING,
        metric="normalised colour temperature (0 cool, 1 warm)",
        labels={
            "cool": fuzz.trapmf(UNIVERSE, [0.0, 0.0, 0.20, 0.42]),
            "neutral": fuzz.trimf(UNIVERSE, [0.32, 0.50, 0.68]),
            "warm": fuzz.trapmf(UNIVERSE, [0.58, 0.80, 1.0, 1.0]),
        },
    ),
    AttributeClass.COLOR: AttributeUniverse(
        attr=AttributeClass.COLOR,
        metric="1 - normalised CIELAB dE to target hue",
        labels={"match": _rising()},
    ),
    AttributeClass.IDENTITY: AttributeUniverse(
        attr=AttributeClass.IDENTITY,
        metric="CLIP similarity to identity anchors",
        labels={"match": _rising()},
    ),
    AttributeClass.TEXTURE: AttributeUniverse(
        attr=AttributeClass.TEXTURE,
        metric="1 - normalised Gram/DISTS distance",
        labels={"match": _rising()},
    ),
    AttributeClass.STYLE: AttributeUniverse(
        attr=AttributeClass.STYLE,
        metric="CLIP style similarity",
        labels={"match": _rising()},
    ),
    AttributeClass.ACTION: AttributeUniverse(
        attr=AttributeClass.ACTION,
        metric="CLIP score for the action phrase",
        labels={"match": _rising()},
    ),
}

_DEFAULT_LABELS: dict[AttributeClass, str] = {
    AttributeClass.SIZE: "small",
    AttributeClass.LIGHTING: "warm",
}


def default_label(attr: AttributeClass) -> str:
    """The label used when a prompt names no specific linguistic value."""
    return _DEFAULT_LABELS.get(attr, "match")


def membership_curve(attr: AttributeClass, label: str) -> np.ndarray:
    labels = UNIVERSES[attr].labels
    if label not in labels:
        raise KeyError(f"{label!r} is not a label of {attr.value}: {sorted(labels)}")
    return labels[label]


def membership_at(attr: AttributeClass, label: str, x: float) -> float:
    """Membership of a measured metric value, clamped to the universe."""
    curve = membership_curve(attr, label)
    clamped = float(np.clip(x, UNIVERSE[0], UNIVERSE[-1]))
    return float(np.interp(clamped, UNIVERSE, curve))
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest tests/test_membership.py -v`
Expected: PASS — 8 passed

- [ ] **Step 5: Commit**

```bash
git add flair_t2i/fuzzy/ tests/test_membership.py
git commit -m "feat: fuzzy membership functions over attribute metric universes"
```

---

### Task 9: Linguistic hedge operators (spec §3.3-B)

**Resolves a spec defect.** §3.3-B specifies scalar operators on `μ_base = 1.0`, but `1.0² = 1.0` and `1.0^0.5 = 1.0` — every hedge collapses to the same value. Zadeh's operators act on a membership *curve*, not a scalar. This task applies them pointwise to Task 8's curves and derives the scalar from the resulting set's **specificity**, normalised against the unhedged set. That yields exactly `1.0` for no hedge (backward compatible with the crisp pipeline), `> 1` for "very", `< 1` for "somewhat", and sensible behaviour for "not".

**Files:**
- Create: `flair_t2i/fuzzy/hedges.py`
- Test: `tests/test_hedges.py`

**Interfaces:**
- Consumes: `membership_curve`, `UNIVERSE` (Task 8); `HEDGE_WORDS` (Task 2).
- Produces:
  - `class HedgeKind(str, Enum)`: `NONE, CONCENTRATE, DILATE, COMPLEMENT`
  - `HEDGE_KINDS: dict[str, HedgeKind]` — maps every word in `HEDGE_WORDS` to a kind
  - `apply_hedge(curve: np.ndarray, kind: HedgeKind) -> np.ndarray`
  - `specificity(curve: np.ndarray) -> float` — `1 - mean(curve)`
  - `@dataclass HedgeResult`: `kind: HedgeKind`, `intensity: float`, `k: int`, `curve: np.ndarray`
  - `resolve_hedge(attr, label, hedge_word: str | None) -> HedgeResult` — `intensity = clip(s_hedged / s_base, 0.3, 1.6)`; `k = 1 if intensity >= 1.0 else 2 if intensity >= 0.6 else 3`

- [ ] **Step 1: Write the failing test**

Create `tests/test_hedges.py`:

```python
import numpy as np
import pytest

pytest.importorskip("skfuzzy")

from flair_t2i.attributes import AttributeClass
from flair_t2i.fuzzy.hedges import (
    HEDGE_KINDS,
    HedgeKind,
    apply_hedge,
    resolve_hedge,
    specificity,
)
from flair_t2i.fuzzy.membership import membership_curve
from flair_t2i.parsing import HEDGE_WORDS

SIZE_SMALL = membership_curve(AttributeClass.SIZE, "small")


def test_every_parser_hedge_word_has_a_kind():
    assert set(HEDGE_WORDS) <= set(HEDGE_KINDS)


def test_concentration_narrows_and_dilation_widens():
    con = specificity(apply_hedge(SIZE_SMALL, HedgeKind.CONCENTRATE))
    base = specificity(apply_hedge(SIZE_SMALL, HedgeKind.NONE))
    dil = specificity(apply_hedge(SIZE_SMALL, HedgeKind.DILATE))
    assert con > base > dil


def test_complement_inverts_the_curve():
    comp = apply_hedge(SIZE_SMALL, HedgeKind.COMPLEMENT)
    np.testing.assert_allclose(comp, 1.0 - SIZE_SMALL)


def test_no_hedge_leaves_the_curve_untouched():
    np.testing.assert_allclose(apply_hedge(SIZE_SMALL, HedgeKind.NONE), SIZE_SMALL)


def test_specificity_of_full_set_is_zero():
    assert specificity(np.ones(201)) == pytest.approx(0.0)


def test_specificity_of_empty_set_is_one():
    assert specificity(np.zeros(201)) == pytest.approx(1.0)


def test_unhedged_intensity_is_exactly_one():
    result = resolve_hedge(AttributeClass.SIZE, "small", None)
    assert result.intensity == pytest.approx(1.0)
    assert result.k == 1
    assert result.kind is HedgeKind.NONE


def test_very_strengthens_and_narrows():
    result = resolve_hedge(AttributeClass.SIZE, "small", "very")
    assert result.intensity > 1.0
    assert result.k == 1


def test_slightly_weakens_and_widens():
    result = resolve_hedge(AttributeClass.SIZE, "small", "slightly")
    assert result.intensity < 1.0
    assert result.k >= 2


def test_negation_is_weak_and_diffuse():
    result = resolve_hedge(AttributeClass.SIZE, "small", "not")
    assert result.kind is HedgeKind.COMPLEMENT
    assert result.intensity < 0.6
    assert result.k == 3


def test_intensity_is_clipped_to_the_documented_range():
    for word in HEDGE_KINDS:
        result = resolve_hedge(AttributeClass.SIZE, "small", word)
        assert 0.3 <= result.intensity <= 1.6


def test_unknown_hedge_word_is_treated_as_no_hedge():
    result = resolve_hedge(AttributeClass.SIZE, "small", "purple")
    assert result.kind is HedgeKind.NONE
    assert result.intensity == pytest.approx(1.0)


def test_monotonic_across_the_intensity_ladder():
    ladder = ["slightly", "quite", "very"]
    values = [resolve_hedge(AttributeClass.SIZE, "small", w).intensity for w in ladder]
    assert values == sorted(values)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_hedges.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.fuzzy.hedges'`

- [ ] **Step 3: Write the hedges module**

Create `flair_t2i/fuzzy/hedges.py`:

```python
"""Zadeh linguistic hedge operators (spec section 3.3-B).

Spec section 3.3-B as written applies the operators to a scalar mu_base of
1.0, where mu**2 == mu**0.5 == 1.0 and every hedge collapses to the same
value. Zadeh's operators act on a membership CURVE, so they are applied
pointwise here and the scalar is derived from the resulting set:

    specificity(mu) = 1 - mean(mu)
    intensity       = clip(spec(hedged) / spec(unhedged), 0.3, 1.6)

Concentration ("very") narrows the set, raising specificity and pushing
harder through fewer blocks. Dilation ("slightly") widens it, giving a
weaker, more diffuse push. An unhedged label yields exactly 1.0, so the
crisp pipeline's behaviour is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..attributes import AttributeClass
from .membership import membership_curve


class HedgeKind(str, Enum):
    NONE = "none"
    CONCENTRATE = "concentrate"
    DILATE = "dilate"
    COMPLEMENT = "complement"


HEDGE_KINDS: dict[str, HedgeKind] = {
    # concentration -- stronger, narrower
    "very": HedgeKind.CONCENTRATE,
    "extremely": HedgeKind.CONCENTRATE,
    "super": HedgeKind.CONCENTRATE,
    "highly": HedgeKind.CONCENTRATE,
    "intensely": HedgeKind.CONCENTRATE,
    # mild concentration -- treated as no-op so the ladder stays ordered
    "quite": HedgeKind.NONE,
    "fairly": HedgeKind.NONE,
    "moderately": HedgeKind.NONE,
    "rather": HedgeKind.NONE,
    # dilation -- weaker, wider
    "slightly": HedgeKind.DILATE,
    "somewhat": HedgeKind.DILATE,
    "mildly": HedgeKind.DILATE,
    "faintly": HedgeKind.DILATE,
    "barely": HedgeKind.DILATE,
    # negation
    "not": HedgeKind.COMPLEMENT,
}

_INTENSITY_MIN, _INTENSITY_MAX = 0.3, 1.6


@dataclass(frozen=True)
class HedgeResult:
    kind: HedgeKind
    intensity: float
    k: int
    curve: np.ndarray


def apply_hedge(curve: np.ndarray, kind: HedgeKind) -> np.ndarray:
    if kind is HedgeKind.CONCENTRATE:
        return curve**2
    if kind is HedgeKind.DILATE:
        return curve**0.5
    if kind is HedgeKind.COMPLEMENT:
        return 1.0 - curve
    return curve


def specificity(curve: np.ndarray) -> float:
    """How selective a fuzzy set is: 1 for empty, 0 for the whole universe."""
    return float(1.0 - np.mean(curve))


def _breadth(intensity: float) -> int:
    if intensity >= 1.0:
        return 1
    if intensity >= 0.6:
        return 2
    return 3


def resolve_hedge(
    attr: AttributeClass, label: str, hedge_word: str | None
) -> HedgeResult:
    """Turn a hedge word into an injection intensity and a routing breadth."""
    base = membership_curve(attr, label)
    kind = HEDGE_KINDS.get((hedge_word or "").lower(), HedgeKind.NONE)
    hedged = apply_hedge(base, kind)

    base_spec = specificity(base)
    ratio = 1.0 if base_spec <= 0.0 else specificity(hedged) / base_spec
    intensity = float(np.clip(ratio, _INTENSITY_MIN, _INTENSITY_MAX))

    return HedgeResult(
        kind=kind, intensity=intensity, k=_breadth(intensity), curve=hedged
    )
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest tests/test_hedges.py -v`
Expected: PASS — 13 passed

- [ ] **Step 5: Commit**

```bash
git add flair_t2i/fuzzy/hedges.py tests/test_hedges.py
git commit -m "feat: Zadeh hedge operators via curve specificity

Applies concentration/dilation/complement to the membership curve rather
than to a scalar, which spec section 3.3-B's literal formulation leaves
degenerate at mu_base = 1.0."
```

---

### Task 10: Wire fuzzy into routing and upgrade the guard

Week 2 exit. After this, `α_i(t) = α_0 · S[ℓ,a] · intensity · sched(t)` is fully realised and the guard's distortion check is fuzzy.

**Files:**
- Modify: `flair_t2i/guard.py`
- Modify: `flair_t2i/pipeline.py`
- Create: `flair_t2i/fuzzy/resolve.py`
- Test: `tests/test_fuzzy_integration.py`
- Test: `tests/test_guard_membership.py`

**Interfaces:**
- Consumes: `resolve_hedge`, `HedgeResult` (Task 9); `default_label`, `membership_at` (Task 8); `build_routing_plan` (Task 4); `CoherenceGuard` (Task 5).
- Produces:
  - `flair_t2i/fuzzy/resolve.py`: `resolve_components(components: list[Component]) -> tuple[dict[str, float], dict[str, int], dict[str, HedgeResult]]` returning `(intensities, k_overrides, results)` keyed by component id
  - `CoherenceGuard.check_membership(attr, label, measured: float, step: int) -> GuardEvent | None` — fires when `membership_at(attr, label, measured) < cfg.guard_membership_threshold`
  - `FlairPipeline.generate` gains `fuzzy: bool = True`

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_fuzzy_integration.py`:

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("skfuzzy")

from flair_t2i.attributes import AttributeClass
from flair_t2i.basm import BASM
from flair_t2i.components import Component
from flair_t2i.config import FlairConfig
from flair_t2i.fuzzy.resolve import resolve_components
from flair_t2i.routing import build_routing_plan

SEQ, DIM = 4, 8
CFG = FlairConfig(alpha_0=1.0, t_window=(0.0, 1.0), top_k_default=1)


def _basm():
    return BASM(
        matrix=np.array([[0.4], [0.9], [0.6]]),
        block_ids=(3, 7, 11),
        attributes=(AttributeClass.SIZE,),
    )


def _component(hedge):
    return Component(
        id="c_size", text="a small car", attr=AttributeClass.SIZE, hedge=hedge
    )


def test_unhedged_component_keeps_intensity_one():
    intensities, k_overrides, _ = resolve_components([_component(None)])
    assert intensities["c_size"] == pytest.approx(1.0)
    assert k_overrides["c_size"] == 1


def test_very_raises_intensity_and_keeps_k_at_one():
    intensities, k_overrides, _ = resolve_components([_component("very")])
    assert intensities["c_size"] > 1.0
    assert k_overrides["c_size"] == 1


def test_slightly_lowers_intensity_and_widens_k():
    intensities, k_overrides, _ = resolve_components([_component("slightly")])
    assert intensities["c_size"] < 1.0
    assert k_overrides["c_size"] >= 2


def test_hedge_flows_through_to_routed_blocks():
    components = [_component("slightly")]
    intensities, k_overrides, _ = resolve_components(components)
    embeddings = {"c_size": torch.zeros((SEQ, DIM))}

    plan = build_routing_plan(
        components, embeddings, _basm(), CFG, intensities, k_overrides
    )

    assert len(plan.routed[0].blocks) >= 2  # widened by dilation
    assert plan.routed[0].intensity < 1.0


def test_hedge_changes_the_resulting_alpha():
    embeddings = {"c_size": torch.zeros((SEQ, DIM))}

    plain = build_routing_plan([_component(None)], embeddings, _basm(), CFG)
    hedged_components = [_component("very")]
    intensities, k_overrides, _ = resolve_components(hedged_components)
    hedged = build_routing_plan(
        hedged_components, embeddings, _basm(), CFG, intensities, k_overrides
    )

    a_plain = plain.alpha(plain.routed[0], 7, 0.0)
    a_hedged = hedged.alpha(hedged.routed[0], 7, 0.0)
    assert a_hedged > a_plain


def test_uncalibrated_attribute_is_still_skipped():
    components = [
        Component(id="c_style", text="cyberpunk", attr=AttributeClass.STYLE, hedge="very")
    ]
    intensities, k_overrides, _ = resolve_components(components)
    plan = build_routing_plan(
        components, {"c_style": torch.zeros((SEQ, DIM))}, _basm(), CFG,
        intensities, k_overrides,
    )
    assert plan.routed == ()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_fuzzy_integration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.fuzzy.resolve'`

- [ ] **Step 3: Write the resolver**

Create `flair_t2i/fuzzy/resolve.py`:

```python
"""Turn parsed components into per-component injection intensity and breadth."""

from __future__ import annotations

from ..components import Component
from .hedges import HedgeResult, resolve_hedge
from .membership import default_label


def resolve_components(
    components: list[Component],
) -> tuple[dict[str, float], dict[str, int], dict[str, HedgeResult]]:
    """Return (intensities, k_overrides, results) keyed by component id."""
    intensities: dict[str, float] = {}
    k_overrides: dict[str, int] = {}
    results: dict[str, HedgeResult] = {}

    for component in components:
        label = default_label(component.attr)
        result = resolve_hedge(component.attr, label, component.hedge)
        intensities[component.id] = result.intensity
        k_overrides[component.id] = result.k
        results[component.id] = result

    return intensities, k_overrides, results
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest tests/test_fuzzy_integration.py -v`
Expected: PASS — 6 passed

- [ ] **Step 5: Write the failing guard-membership test**

Create `tests/test_guard_membership.py`:

```python
import pytest

pytest.importorskip("skfuzzy")

from flair_t2i.attributes import AttributeClass
from flair_t2i.config import FlairConfig
from flair_t2i.guard import CoherenceGuard

CFG = FlairConfig(guard_membership_threshold=0.5, guard_backoff=0.5)


def test_no_event_when_measurement_sits_inside_the_fuzzy_region():
    guard = CoherenceGuard(CFG)
    event = guard.check_membership(AttributeClass.SIZE, "small", measured=0.05, step=2)
    assert event is None


def test_event_when_measurement_falls_outside_the_region():
    guard = CoherenceGuard(CFG)
    event = guard.check_membership(AttributeClass.SIZE, "small", measured=0.9, step=2)
    assert event is not None
    assert event.reason == "attribute_membership"
    assert event.step == 2
    assert event.value < 0.5


def test_membership_event_backs_off_alpha_like_the_cosine_check():
    from flair_t2i.components import Component
    from flair_t2i.routing import RoutedComponent, RoutingPlan
    import torch

    plan = RoutingPlan(
        routed=(
            RoutedComponent(
                component=Component(id="c", text="x", attr=AttributeClass.SIZE),
                embedding=torch.ones((2, 2)),
                blocks=((7, 1.0),),
            ),
        ),
        cfg=CFG,
    )
    guard = CoherenceGuard(CFG)
    guard.apply(
        plan, guard.check_membership(AttributeClass.SIZE, "small", 0.9, step=1)
    )
    assert plan.alpha_scale == pytest.approx(0.5)
```

- [ ] **Step 6: Run it to verify it fails**

Run: `python -m pytest tests/test_guard_membership.py -v`
Expected: FAIL — `AttributeError: 'CoherenceGuard' object has no attribute 'check_membership'`

- [ ] **Step 7: Add the fuzzy check to the guard**

In `flair_t2i/guard.py`, add this import at the top:

```python
from .attributes import AttributeClass
from .fuzzy.membership import membership_at
```

Then add this method to `CoherenceGuard`, immediately after `check_streams`:

```python
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
```

Also update the module docstring's last line to read:

```python
"""Runtime coherence guard (spec section 3.6).

Two checks are specified: a crisp cross-stream cosine check, and an
attribute-distortion check that evaluates membership in the intended fuzzy
region rather than a percentile band.
"""
```

- [ ] **Step 8: Run it to verify it passes**

Run: `python -m pytest tests/test_guard_membership.py -v`
Expected: PASS — 3 passed

- [ ] **Step 9: Wire fuzzy resolution into the pipeline**

In `flair_t2i/pipeline.py`, add the import:

```python
from .fuzzy.resolve import resolve_components
```

Change the `generate` signature to add the flag:

```python
    def generate(
        self,
        prompt: str,
        seed: int = 0,
        steps: int = 20,
        guidance_scale: float = 4.5,
        routing: bool = True,
        fuzzy: bool = True,
    ):
```

Replace the plan-building block inside `if routing:` with:

```python
            components = parse_prompt(prompt, self.nlp)
            routable = [c for c in components if c.attr in self.basm.attributes]
            embeddings = self.encode_components(routable)

            if fuzzy:
                intensities, k_overrides, _ = resolve_components(routable)
            else:
                intensities, k_overrides = {}, {}

            plan = build_routing_plan(
                routable, embeddings, self.basm, self.cfg, intensities, k_overrides
            )
```

- [ ] **Step 10: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS — every test from Tasks 1-10, no failures

- [ ] **Step 11: Extend the smoke test with a hedge ladder**

In `scripts/smoke_test.py`, add before `print("\nSmoke test passed.")`:

```python
    for hedge in ["slightly", "", "very"]:
        text = f"A {hedge} small red sports car under warm evening light".replace(
            "  ", " "
        )
        image = fp.generate(text, seed=args.seed, steps=args.steps)
        name = hedge or "plain"
        image.save(args.out / f"smoke_hedge_{name}.png")
        intensity = fp.last_plan.routed[0].intensity if fp.last_plan.routed else None
        print(f"  {name:<9} intensity={intensity}")
```

- [ ] **Step 12: Run the extended smoke test on Kaggle**

Run: `python scripts/smoke_test.py --steps 20 --out outputs/`

Expected: three hedge images written; printed intensities strictly increase across `slightly` → `plain` → `very`.

**Week 2 exit criterion:** the printed intensity ladder is monotonic, and the three hedge images differ visibly from one another. This is the qualitative precursor to the controllability curve in spec §4 — if the images are indistinguishable, `alpha_0` is too low; raise it and re-run before starting BASM calibration.

- [ ] **Step 13: Commit**

```bash
git add flair_t2i/ scripts/smoke_test.py tests/
git commit -m "feat: wire fuzzy hedges into routing and add fuzzy guard check

Completes the spec section 3.3 alpha formula and replaces the guard's
crisp percentile band with a membership evaluation."
```

---

## Self-Review

**1. Spec coverage.**

| Spec section | Covered by |
|---|---|
| §3.1 scope (7 attrs, Channel A, no head-level) | Task 1 (`AttributeClass`, `CORE_ATTRIBUTES`) |
| §3.2 semantic parsing | Task 2 |
| §3.3-A value membership | Task 8 |
| §3.3-B hedge operators | Task 9 (spec defect resolved) |
| §3.3 α formula + routing breadth | Tasks 4, 10 |
| §3.4 BASM container + vital-layer bypass | Tasks 3, 6 (`bypass_blocks`); **calibration campaign is a separate plan** |
| §3.5 routing, `FlairJointProcessor`, batch-layout fix | Tasks 4, 6, 7 |
| §3.6 coherence guard (both checks) | Tasks 5, 10 |
| §3.7 FLUX, §3.8 fuzzy conflict | Out of scope — gated behind Week 7 |
| §7 env-failure risk | Task 7 (`verify_env.py`) |

**2. Placeholder scan.** No "TBD"/"TODO"/"add error handling" steps; every code step carries runnable code; every test step carries an exact command and expected result.

**3. Type consistency.** `RoutingPlan.blend(encoder_hidden_states, block_id, step_frac, cond_slice)` is called with exactly those names in `FlairJointProcessor` (Task 6). `build_routing_plan(components, embeddings, basm, cfg, intensities, k_overrides)` matches its call sites in Tasks 4, 7, and 10. `GuardEvent(step, reason, value)` is constructed identically in `check_streams` and `check_membership`. `HedgeResult.intensity`/`.k` feed `resolve_components`, which feeds `build_routing_plan`'s `intensities`/`k_overrides` parameters.

**Two known deviations from the spec, both deliberate and documented in-code:**
1. Task 9 replaces §3.3-B's scalar operators with curve-based specificity (the scalar form is degenerate at μ=1.0).
2. Task 4 keeps component streams out of the denoising batch entirely, rather than fixing the `base_i = B - n_rows` arithmetic in place — the stronger fix for the §3.5 risk.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-08-25-flair-foundation.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
