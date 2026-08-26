# FLAIR Head-Level Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move FLAIR's routing unit from the transformer block to the individual attention head, with block-level routing retained as a derived special case of head-level routing.

**Architecture:** Projection is linear and diffusers gives each attention head a contiguous slice of the projection output, so head-level injection is achieved by wrapping three `nn.Linear` modules per block (`add_q_proj`, `add_k_proj`, `add_v_proj`) rather than reimplementing any attention maths. Selecting every head of a block reduces algebraically to the shipped block-level result, which is pinned by a frozen oracle in the test tree. Sensitivity is calibrated as a `[blocks × heads × attributes]` tensor (HASM); the `[blocks × attributes]` BASM is derived from it by reduction over heads at no measurement cost.

**Tech Stack:** Python 3.10+, PyTorch (CPU for all tests), NumPy, pytest, diffusers 0.39.0 (introspection only — no test downloads weights or needs a GPU).

**Spec:** [`docs/superpowers/specs/2026-08-26-flair-head-level-routing-design.md`](../specs/2026-08-26-flair-head-level-routing-design.md)

## Global Constraints

- **Package:** `flair_t2i`. Python 3.10+ (`X | None` union syntax).
- **Tests run in Docker only.** `docker build -t flair-test . && ./run-local.sh test`. There is no local Python environment for this repo. Do not strip `MSYS_NO_PATHCONV=1` from `run-local.sh`.
- **Baseline: 175 tests passing.** Every task must leave the suite green. Report the new count at each commit.
- **No test may require a GPU or download SD3.5.** Tests that would need the real transformer use stubs.
- **Ruff IS configured** in `pyproject.toml` — line-length 100, target py310, `E402` ignored under `tests/`. A `PostToolUse` hook runs `ruff check` on every `.py` write, so lint errors surface inline. Keep imports used: an unused `import pytest` in a test file is an `F401` failure. No formatter is configured — do not reformat existing code. (CLAUDE.md claimed no linter existed; that was stale and has been corrected.)
- **`tests/` is a package** — `tests/__init__.py` exists, so test modules may use relative imports such as `from .reference_blend import reference_blend`. Without it, pytest's default `prepend` import mode fails with `attempted relative import with no known parent package`.
- **`./run-local.sh test` always rebuilds the image.** The Dockerfile `COPY`s the source, so a cached image would test stale code. Do not add a build-skip back.
- **`docs/`, `notebooks/`, `calibration_runs/` are gitignored.** New files under them need `git add -f`.
- **The two load-bearing correctness conditions** (spec §3.3), which Task 5 and Task 6 must both honour:
  1. The residual is projected **weight-only, with no bias term**.
  2. The residual is applied **immediately after projection, before the QK norm**.
- **Injection formula, authoritative:** `α_i(ℓ,h,t) = α_0 · S[ℓ,h,a] · intensity_i · sched(t) · alpha_scale`
- **Attribute classes** remain exactly the 7 in `attributes.py`. This change does not touch them.

---

### Task 1: Verify the head-level assumptions against diffusers

**This task is a gate.** Spec §6: none of the implementation is safe to build on until these pass, because each assumption, if false, silently produces a wrong result rather than an error. Check 2 in particular invalidates the entire design.

`verify_api.py` is introspection-only — it downloads nothing and needs no GPU.

**Files:**
- Modify: `scripts/verify_api.py:89-106` (append new checks before the summary block)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: no importable symbols. Produces a pass/fail gate for Tasks 2-10.

- [ ] **Step 1: Add the head-level checks**

Insert this block into `scripts/verify_api.py` immediately after the existing `bypass_blocks` checks (after line 106, before `failed = [...]`):

```python
    # --- head-level routing depends on all of the following --------------
    proc_src = "".join(inspect.getsource(JointAttnProcessor2_0.__call__).split())

    # The entire head-masking premise: head h must own the contiguous
    # output range [h*head_dim, (h+1)*head_dim). What matters is the SHAPE
    # of the reshape -- heads in axis 2, then transposed to axis 1 -- not
    # what diffusers happens to name its locals, so this matches structure
    # and tolerates a rename of batch_size or head_dim.
    check(
        "head reshape gives each head a contiguous dim slice",
        re.search(r"\.view\([^)]+,-1,attn\.heads,[^)]+\)\.transpose\(1,2\)", proc_src)
        is not None,
        "reshape SHAPE changed -- read the diffusers source before concluding "
        "the head-masking premise is dead; a rename alone does not break it",
    )

    # patching.install_head_routing wraps exactly these three modules.
    for name in ("add_q_proj", "add_k_proj", "add_v_proj"):
        check(f"processor projects text through attn.{name}", f"attn.{name}" in proc_src)

    # head_proj.HeadResidualProj derives head_dim from attn.heads.
    attn_src = "".join(inspect.getsource(Attention.__init__).split())
    check("Attention exposes .heads", "self.heads=" in attn_src)

    # install_head_routing wraps block.attn only. A second attention module
    # would route unrouted -- a gap that predates head-level routing.
    block_src = "".join(inspect.getsource(JointTransformerBlock.__init__).split())
    check(
        "no unhandled second attention (attn2) on JointTransformerBlock",
        "self.attn2=" not in block_src,
        "attn2 exists -- install_head_routing must wrap its projections too",
    )

    # The final block sets context_pre_only=True and may lack add_*_proj;
    # install_head_routing skips absent modules via getattr(..., None).
    check(
        "context_pre_only is a JointTransformerBlock parameter",
        "context_pre_only"
        in inspect.signature(JointTransformerBlock.__init__).parameters,
    )
```

- [ ] **Step 2: Run it**

This needs `diffusers` installed, which the Docker test image deliberately omits. Run it on Kaggle, or in any environment with `diffusers==0.39.0`:

```bash
python scripts/verify_api.py
```

Expected: `All 19 checks passed -- FLAIR's hooks match this diffusers.` (12 existing + 7 new)

Two of the existing checks — `Attention.get_processor` / `set_processor` — become vestigial once Task 7 retires `install_flair`. Leave them: they still describe a diffusers contract worth noticing if it breaks.

- [ ] **Step 3: Act on the result before writing any other code**

| Result | Action |
|---|---|
| All pass | Proceed to Task 2. |
| "head reshape gives each head a contiguous dim slice" FAILS | **Stop and read the diffusers source first.** The check matches structure, not identifier names, so a failure should mean a real convention change — but confirm by eye before concluding. If heads genuinely no longer occupy a contiguous output range, the masking premise is invalid and spec §3.1 needs rework. |
| "no unhandled second attention (attn2)" FAILS | Proceed, but add `attn2` to the module list in Task 7 Step 3 and record the deviation in the commit message. |
| Any other FAIL | Fix the named assumption in the code it points at before continuing. |

- [ ] **Step 4: Commit**

```bash
git add scripts/verify_api.py
git commit -m "test: verify diffusers head-reshape and projection assumptions

The reshape check is the gate for head-level routing: head masking is
only equivalent to head selection if head h owns a contiguous output
range."
```

---

### Task 2: HeadUnit and the per-head alpha vector

**Files:**
- Create: `flair_t2i/heads.py`
- Test: `tests/test_heads.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `HeadUnit` frozen dataclass, `order=True`: fields `block: int`, `head: int` (in that order, so sorting is `(block, head)` ascending)
  - `head_slice(head: int, head_dim: int) -> slice`
  - `alpha_vector(alphas: dict[int, float], n_heads: int, head_dim: int, device=None, dtype=None) -> torch.Tensor` — shape `[n_heads * head_dim]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_heads.py`:

```python
import pytest

torch = pytest.importorskip("torch")

from flair_t2i.heads import HeadUnit, alpha_vector, head_slice

N_HEADS, HEAD_DIM = 3, 4
INNER = N_HEADS * HEAD_DIM


def test_head_slice_is_contiguous_and_non_overlapping():
    assert head_slice(0, HEAD_DIM) == slice(0, 4)
    assert head_slice(2, HEAD_DIM) == slice(8, 12)


def test_head_units_sort_by_block_then_head():
    units = [HeadUnit(7, 2), HeadUnit(3, 5), HeadUnit(3, 1)]
    assert sorted(units) == [HeadUnit(3, 1), HeadUnit(3, 5), HeadUnit(7, 2)]


def test_head_unit_is_frozen_and_hashable():
    unit = HeadUnit(block=3, head=1)
    assert {unit: "ok"}[HeadUnit(3, 1)] == "ok"
    with pytest.raises(Exception):
        unit.block = 9


def test_alpha_vector_fills_only_the_selected_head_slice():
    vector = alpha_vector({1: 0.75}, N_HEADS, HEAD_DIM)

    assert vector.shape == (INNER,)
    assert vector[head_slice(1, HEAD_DIM)].tolist() == [0.75] * HEAD_DIM
    assert vector[head_slice(0, HEAD_DIM)].abs().max().item() == 0.0
    assert vector[head_slice(2, HEAD_DIM)].abs().max().item() == 0.0


def test_alpha_vector_supports_different_alpha_per_head():
    vector = alpha_vector({0: 0.2, 2: 0.9}, N_HEADS, HEAD_DIM)

    # approx, not equality: alpha_vector builds a float32 tensor and
    # .tolist() widens each element to float64, so 0.2 comes back as
    # 0.20000000298023224. The zero assertion stays exact -- 0.0 is.
    assert vector[head_slice(0, HEAD_DIM)].tolist() == pytest.approx([0.2] * HEAD_DIM)
    assert vector[head_slice(1, HEAD_DIM)].abs().max().item() == 0.0
    assert vector[head_slice(2, HEAD_DIM)].tolist() == pytest.approx([0.9] * HEAD_DIM)


def test_alpha_vector_of_every_head_is_uniform():
    vector = alpha_vector({h: 0.5 for h in range(N_HEADS)}, N_HEADS, HEAD_DIM)
    assert vector.tolist() == [0.5] * INNER


def test_alpha_vector_rejects_out_of_range_head():
    with pytest.raises(ValueError, match="out of range"):
        alpha_vector({9: 1.0}, N_HEADS, HEAD_DIM)


def test_alpha_vector_honours_dtype():
    vector = alpha_vector({0: 1.0}, N_HEADS, HEAD_DIM, dtype=torch.float16)
    assert vector.dtype == torch.float16
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./run-local.sh test`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.heads'`

- [ ] **Step 3: Write the module**

Create `flair_t2i/heads.py`:

```python
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
```

- [ ] **Step 4: Run it to verify it passes**

Run: `./run-local.sh test`
Expected: PASS — 183 passed (175 + 8)

- [ ] **Step 5: Commit**

```bash
git add flair_t2i/heads.py tests/test_heads.py
git commit -m "feat: HeadUnit and per-head alpha vector"
```

---

### Task 3: The HASM container

**Files:**
- Create: `flair_t2i/hasm.py`
- Test: `tests/test_hasm.py`

**Interfaces:**
- Consumes: `HeadUnit` (Task 2), `BASM` and `AttributeClass` (existing, unmodified).
- Produces:
  - `HASM(tensor: np.ndarray, block_ids: tuple[int, ...], head_ids: tuple[int, ...], attributes: tuple[AttributeClass, ...])`, tensor shape `(len(block_ids), len(head_ids), len(attributes))`, values in `[0, 1]`
  - `.score(unit: HeadUnit, attr) -> float`
  - `.top_k(attr, k) -> list[tuple[HeadUnit, float]]` — descending by score, ties by ascending `(block, head)`
  - `.to_basm(reduce: str = "max") -> BASM` — `"max"` or `"mean"` over the head axis
  - `.save(path)` / `HASM.load(path)` — npz round-trip
  - `HASM.uniform(block_ids, head_ids, attributes) -> HASM` — all 0.5

- [ ] **Step 1: Write the failing test**

Create `tests/test_hasm.py`:

```python
import numpy as np
import pytest

from flair_t2i.attributes import AttributeClass
from flair_t2i.hasm import HASM
from flair_t2i.heads import HeadUnit

BLOCKS = (3, 7)
HEADS = (0, 1)
ATTRS = (AttributeClass.COLOR, AttributeClass.SIZE)


def _hasm():
    # [block, head, attribute]
    tensor = np.array(
        [
            [[0.20, 0.80], [0.10, 0.40]],  # block 3: head 0, head 1
            [[0.90, 0.30], [0.50, 0.60]],  # block 7: head 0, head 1
        ]
    )
    return HASM(tensor=tensor, block_ids=BLOCKS, head_ids=HEADS, attributes=ATTRS)


def test_score_lookup_by_unit_and_attribute():
    assert _hasm().score(HeadUnit(7, 0), AttributeClass.COLOR) == pytest.approx(0.90)
    assert _hasm().score(HeadUnit(3, 1), AttributeClass.SIZE) == pytest.approx(0.40)


def test_top_k_is_descending_by_score():
    assert _hasm().top_k(AttributeClass.COLOR, 2) == [
        (HeadUnit(7, 0), 0.90),
        (HeadUnit(7, 1), 0.50),
    ]


def test_top_k_clamps_to_available_units():
    assert len(_hasm().top_k(AttributeClass.COLOR, 99)) == 4


def test_ties_break_by_ascending_block_then_head():
    tensor = np.full((2, 2, 1), 0.5)
    hasm = HASM(tensor, (9, 4), (1, 0), (AttributeClass.COLOR,))
    units = [unit for unit, _ in hasm.top_k(AttributeClass.COLOR, 4)]
    assert units == [HeadUnit(4, 0), HeadUnit(4, 1), HeadUnit(9, 0), HeadUnit(9, 1)]


def test_to_basm_max_reduces_over_heads():
    basm = _hasm().to_basm(reduce="max")
    assert basm.block_ids == BLOCKS
    assert basm.attributes == ATTRS
    assert basm.score(3, AttributeClass.COLOR) == pytest.approx(0.20)
    assert basm.score(7, AttributeClass.COLOR) == pytest.approx(0.90)
    assert basm.score(3, AttributeClass.SIZE) == pytest.approx(0.80)


def test_to_basm_mean_reduces_over_heads():
    basm = _hasm().to_basm(reduce="mean")
    assert basm.score(3, AttributeClass.COLOR) == pytest.approx(0.15)
    assert basm.score(7, AttributeClass.SIZE) == pytest.approx(0.45)


def test_to_basm_rejects_unknown_reduction():
    with pytest.raises(ValueError, match="unknown reduction"):
        _hasm().to_basm(reduce="median")


def test_rejects_out_of_range_scores():
    with pytest.raises(ValueError, match="within"):
        HASM(np.full((1, 1, 1), 1.4), (3,), (0,), (AttributeClass.COLOR,))


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        HASM(np.zeros((2, 2, 2)), (3,), (0,), ATTRS)


def test_unknown_attribute_raises():
    with pytest.raises(KeyError, match="lighting"):
        _hasm().score(HeadUnit(3, 0), AttributeClass.LIGHTING)


def test_unknown_head_raises():
    with pytest.raises(KeyError, match="head 9"):
        _hasm().score(HeadUnit(3, 9), AttributeClass.COLOR)


def test_save_load_round_trip(tmp_path):
    original = _hasm()
    path = tmp_path / "hasm.npz"
    original.save(path)
    restored = HASM.load(path)

    assert restored.block_ids == original.block_ids
    assert restored.head_ids == original.head_ids
    assert restored.attributes == original.attributes
    np.testing.assert_allclose(restored.tensor, original.tensor)


def test_uniform_factory_is_all_half():
    hasm = HASM.uniform((1, 2), (0, 1), ATTRS)
    assert hasm.score(HeadUnit(1, 0), AttributeClass.COLOR) == 0.5
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./run-local.sh test`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.hasm'`

- [ ] **Step 3: Write the container**

Create `flair_t2i/hasm.py`:

```python
"""The Head-Attribute Sensitivity Matrix.

A ``[blocks x heads x attributes]`` tensor, calibrated by the same causal
contrastive-swap procedure that produced the BASM. Reducing over the head
axis yields an ordinary BASM at no additional measurement cost, which is
what keeps block-level routing available as a derived special case rather
than a second calibration campaign.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .attributes import AttributeClass
from .basm import BASM
from .heads import HeadUnit


class HASM:
    def __init__(
        self,
        tensor: np.ndarray,
        block_ids: tuple[int, ...],
        head_ids: tuple[int, ...],
        attributes: tuple[AttributeClass, ...],
    ) -> None:
        tensor = np.asarray(tensor, dtype=np.float64)
        expected = (len(block_ids), len(head_ids), len(attributes))
        if tensor.shape != expected:
            raise ValueError(f"tensor shape {tensor.shape} does not match {expected}")
        if tensor.size and (tensor.min() < 0.0 or tensor.max() > 1.0):
            raise ValueError("sensitivity scores must be within [0, 1]")

        self.tensor = tensor
        self.block_ids = tuple(block_ids)
        self.head_ids = tuple(head_ids)
        self.attributes = tuple(attributes)
        self._block_index = {b: i for i, b in enumerate(self.block_ids)}
        self._head_index = {h: i for i, h in enumerate(self.head_ids)}
        self._attr_index = {a: i for i, a in enumerate(self.attributes)}

    @classmethod
    def uniform(
        cls,
        block_ids: tuple[int, ...],
        head_ids: tuple[int, ...],
        attributes: tuple[AttributeClass, ...],
    ) -> "HASM":
        """An uncalibrated tensor, for tests and pre-calibration smoke runs."""
        shape = (len(block_ids), len(head_ids), len(attributes))
        return cls(np.full(shape, 0.5), block_ids, head_ids, attributes)

    def _plane(self, attr: AttributeClass) -> int:
        if attr not in self._attr_index:
            raise KeyError(f"{attr.value} is not calibrated in this HASM")
        return self._attr_index[attr]

    def score(self, unit: HeadUnit, attr: AttributeClass) -> float:
        if unit.block not in self._block_index:
            raise KeyError(f"block {unit.block} is not in this HASM")
        if unit.head not in self._head_index:
            raise KeyError(f"head {unit.head} is not in this HASM")
        return float(
            self.tensor[
                self._block_index[unit.block],
                self._head_index[unit.head],
                self._plane(attr),
            ]
        )

    def top_k(self, attr: AttributeClass, k: int) -> list[tuple[HeadUnit, float]]:
        plane = self.tensor[:, :, self._plane(attr)]
        ranked = sorted(
            (
                (HeadUnit(block=b, head=h), float(plane[i, j]))
                for i, b in enumerate(self.block_ids)
                for j, h in enumerate(self.head_ids)
            ),
            key=lambda pair: (-pair[1], pair[0].block, pair[0].head),
        )
        return ranked[: max(0, k)]

    def to_basm(self, reduce: str = "max") -> BASM:
        """Collapse the head axis into an ordinary block-level BASM."""
        if reduce == "max":
            matrix = self.tensor.max(axis=1)
        elif reduce == "mean":
            matrix = self.tensor.mean(axis=1)
        else:
            raise ValueError(f"unknown reduction {reduce!r}; use 'max' or 'mean'")
        return BASM(
            matrix=matrix, block_ids=self.block_ids, attributes=self.attributes
        )

    def save(self, path: str | Path) -> None:
        np.savez(
            Path(path),
            tensor=self.tensor,
            block_ids=np.array(self.block_ids),
            head_ids=np.array(self.head_ids),
            attributes=np.array([a.value for a in self.attributes]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "HASM":
        data = np.load(Path(path), allow_pickle=False)
        return cls(
            tensor=data["tensor"],
            block_ids=tuple(int(b) for b in data["block_ids"]),
            head_ids=tuple(int(h) for h in data["head_ids"]),
            attributes=tuple(AttributeClass(a) for a in data["attributes"]),
        )
```

- [ ] **Step 4: Run it to verify it passes**

Run: `./run-local.sh test`
Expected: PASS — 196 passed (183 + 13)

- [ ] **Step 5: Commit**

```bash
git add flair_t2i/hasm.py tests/test_hasm.py
git commit -m "feat: HASM container with head top-k and BASM reduction"
```

---

### Task 4: Freeze the block-level oracle

Do this **before** modifying `routing.py`. The oracle must be captured from the code that currently works; copying it after the rewrite would copy the rewrite.

**Files:**
- Create: `tests/reference_blend.py`
- Test: `tests/test_reference_blend.py`

**Interfaces:**
- Consumes: `Component` and `timestep_scale` (existing, unmodified).
- Produces:
  - `ReferenceRouted` dataclass: `component`, `embedding`, `blocks: tuple[tuple[int, float], ...]`, `intensity: float = 1.0`
  - `reference_blend(routed, cfg, encoder_hidden_states, block_id, step_frac, cond_slice, alpha_scale=1.0) -> torch.Tensor`

- [ ] **Step 1: Write the frozen oracle**

Create `tests/reference_blend.py`. This is a verbatim transcription of `RoutingPlan.blend` and the `RoutedComponent` shape it reads, as they exist at `flair_t2i/routing.py:24-89` today:

```python
"""FROZEN reference implementation of block-level blending. DO NOT EDIT.

A verbatim copy of ``RoutingPlan.blend`` and the dataclass shape it read,
taken from the commit that shipped block-level routing. Task 5's
equivalence test proves the head-level implementation reproduces it
exactly when every head of a block is selected.

Freezing it here rather than leaving it on the production path is
deliberate: an oracle that lives beside the implementation it validates
gets refactored alongside it and stops being independent evidence. It
imports nothing from ``flair_t2i.routing``, so it cannot drift with it.
"""

from dataclasses import dataclass

import torch

from flair_t2i.components import Component
from flair_t2i.schedule import timestep_scale


@dataclass
class ReferenceRouted:
    component: Component
    embedding: torch.Tensor  # [seq, dim]
    blocks: tuple[tuple[int, float], ...]  # (block_id, basm_score)
    intensity: float = 1.0


def reference_blend(
    routed,
    cfg,
    encoder_hidden_states: torch.Tensor,
    block_id: int,
    step_frac: float,
    cond_slice: slice,
    alpha_scale: float = 1.0,
) -> torch.Tensor:
    """The shipped block-level blend, frozen."""
    touched = {b for rc in routed for b, _ in rc.blocks}
    if block_id not in touched:
        return encoder_hidden_states

    def alpha(rc) -> float:
        score = next((s for b, s in rc.blocks if b == block_id), None)
        if score is None:
            return 0.0
        sched = timestep_scale(step_frac, cfg.t_window)
        return cfg.alpha_0 * score * rc.intensity * sched * alpha_scale

    contributions = [
        (rc, alpha(rc)) for rc in routed if block_id in {b for b, _ in rc.blocks}
    ]
    contributions = [(rc, a) for rc, a in contributions if a != 0.0]
    if not contributions:
        return encoder_hidden_states

    out = encoder_hidden_states.clone()
    base = encoder_hidden_states[cond_slice]

    for rc, a in contributions:
        target = rc.embedding.to(device=base.device, dtype=base.dtype)
        out[cond_slice] = out[cond_slice] + a * (target.unsqueeze(0) - base)

    return out
```

- [ ] **Step 2: Write the test that pins the oracle to today's behaviour**

Create `tests/test_reference_blend.py`. These assertions are ported from `tests/test_routing.py` and must produce identical numbers:

```python
import pytest

torch = pytest.importorskip("torch")

from flair_t2i.attributes import AttributeClass
from flair_t2i.components import Component
from flair_t2i.config import FlairConfig

from .reference_blend import ReferenceRouted, reference_blend

SEQ, DIM = 4, 8
CFG = FlairConfig(alpha_0=1.0, t_window=(0.0, 1.0), top_k_default=1)


def _routed(blocks=((7, 1.0),), intensity=1.0, fill=1.0):
    return [
        ReferenceRouted(
            component=Component(id="c_color", text="a red car", attr=AttributeClass.COLOR),
            embedding=torch.full((SEQ, DIM), fill),
            blocks=blocks,
            intensity=intensity,
        )
    ]


def _states(batch=2, fill=0.0):
    return torch.full((batch, SEQ, DIM), fill)


def test_identity_on_untouched_block():
    states = _states()
    assert reference_blend(_routed(), CFG, states, 99, 0.0, slice(1, 2)) is states


def test_moves_conditional_rows_toward_the_component():
    out = reference_blend(_routed(), CFG, _states(), 7, 0.0, slice(1, 2))
    assert out[1].mean().item() == pytest.approx(1.0)


def test_leaves_unconditional_rows_untouched():
    out = reference_blend(_routed(), CFG, _states(), 7, 0.0, slice(1, 2))
    assert out[0].abs().max().item() == pytest.approx(0.0)


def test_does_not_mutate_its_input():
    states = _states()
    reference_blend(_routed(), CFG, states, 7, 0.0, slice(1, 2))
    assert states.abs().max().item() == pytest.approx(0.0)


def test_scales_with_score_and_intensity():
    out = reference_blend(
        _routed(blocks=((7, 0.5),), intensity=0.4), CFG, _states(), 7, 0.0, slice(1, 2)
    )
    assert out[1].mean().item() == pytest.approx(0.2)


def test_honours_alpha_scale():
    out = reference_blend(
        _routed(), CFG, _states(), 7, 0.0, slice(1, 2), alpha_scale=0.5
    )
    assert out[1].mean().item() == pytest.approx(0.5)
```

- [ ] **Step 3: Run it to verify it passes**

The oracle is a copy of working code, so this passes immediately — that is the point. It pins the numbers before anything changes.

Run: `./run-local.sh test`
Expected: PASS — 202 passed (196 + 6)

- [ ] **Step 4: Commit**

```bash
git add tests/reference_blend.py tests/test_reference_blend.py
git commit -m "test: freeze block-level blend as the head-routing oracle

Captured before routing.py changes, so it records the behaviour that
shipped rather than the behaviour that replaces it."
```

---

### Task 5: RoutingPlan on head units

**Files:**
- Modify: `flair_t2i/routing.py` (whole file)
- Modify: `tests/test_routing.py` (whole file)
- Modify: `tests/test_guard.py:17-22`, `tests/test_guard_membership.py:35-40`, `tests/test_artifacts.py:44-50`, `tests/test_processor.py:27-32`, `tests/test_fuzzy_integration.py:59`

**Interfaces:**
- Consumes: `HeadUnit`, `alpha_vector` (Task 2); `HASM` (Task 3); `reference_blend` (Task 4).
- Produces:
  - `RoutedComponent`: `component`, `embedding`, `units: tuple[tuple[HeadUnit, float], ...]`, `intensity: float = 1.0`
  - `RoutingPlan.alpha(rc, unit: HeadUnit, step_frac: float) -> float`
  - `RoutingPlan.head_residual(block_id, x, weight, step_frac, cond_slice, n_heads, head_dim) -> torch.Tensor | None`
  - `RoutingPlan.blocks_touched() -> frozenset[int]` (unchanged name — `smoke_test.py:100` and `explain.py:145` depend on it)
  - `build_routing_plan(components, embeddings, hasm, cfg, intensities=None, k_overrides=None, granularity="head", reduce="max") -> RoutingPlan`
  - `RoutingPlan.blend` is **removed**.

- [ ] **Step 1: Write the failing test**

Replace `tests/test_routing.py` entirely:

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flair_t2i.attributes import AttributeClass
from flair_t2i.components import Component
from flair_t2i.config import FlairConfig
from flair_t2i.hasm import HASM
from flair_t2i.heads import HeadUnit
from flair_t2i.routing import RoutedComponent, RoutingPlan, build_routing_plan

from .reference_blend import ReferenceRouted, reference_blend

SEQ, DIM = 4, 8
N_HEADS, HEAD_DIM = 3, 4
INNER = N_HEADS * HEAD_DIM
CFG = FlairConfig(alpha_0=1.0, t_window=(0.0, 1.0), top_k_default=1)

COMPONENT = Component(id="c_color", text="a red car", attr=AttributeClass.COLOR)


def _plan(units, intensity=1.0, fill=1.0):
    return RoutingPlan(
        routed=(
            RoutedComponent(
                component=COMPONENT,
                embedding=torch.full((SEQ, DIM), fill),
                units=units,
                intensity=intensity,
            ),
        ),
        cfg=CFG,
    )


def _residual(plan, block_id, x, weight, step_frac=0.0):
    return plan.head_residual(
        block_id=block_id,
        x=x,
        weight=weight,
        step_frac=step_frac,
        cond_slice=slice(1, 2),
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
    )


def test_untouched_block_returns_none():
    plan = _plan(((HeadUnit(7, 0), 1.0),))
    assert _residual(plan, 99, torch.zeros((2, SEQ, DIM)), torch.zeros((INNER, DIM))) is None


def test_inactive_plan_returns_none():
    plan = _plan(((HeadUnit(7, 0), 1.0),))
    plan.active = False
    assert _residual(plan, 7, torch.zeros((2, SEQ, DIM)), torch.zeros((INNER, DIM))) is None


def test_residual_touches_only_the_selected_head_slice():
    plan = _plan(((HeadUnit(7, 1), 1.0),))
    residual = _residual(plan, 7, torch.zeros((2, SEQ, DIM)), torch.ones((INNER, DIM)))

    assert residual.shape == (1, SEQ, INNER)
    assert residual[..., 0:4].abs().max().item() == pytest.approx(0.0)
    assert residual[..., 4:8].abs().max().item() > 0.0
    assert residual[..., 8:12].abs().max().item() == pytest.approx(0.0)


def test_two_heads_in_one_block_scale_independently():
    plan = _plan(((HeadUnit(7, 0), 0.25), (HeadUnit(7, 2), 1.0)))
    residual = _residual(plan, 7, torch.zeros((2, SEQ, DIM)), torch.ones((INNER, DIM)))

    head0 = residual[..., 0:4].abs().max().item()
    head2 = residual[..., 8:12].abs().max().item()
    assert head2 == pytest.approx(head0 * 4.0)


def test_alpha_scales_with_score_and_intensity():
    plan = _plan(((HeadUnit(7, 0), 0.5),), intensity=0.4)
    assert plan.alpha(plan.routed[0], HeadUnit(7, 0), 0.0) == pytest.approx(0.2)


def test_alpha_respects_guard_backoff():
    plan = _plan(((HeadUnit(7, 0), 1.0),))
    plan.alpha_scale = 0.5
    assert plan.alpha(plan.routed[0], HeadUnit(7, 0), 0.0) == pytest.approx(0.5)


def test_alpha_is_zero_outside_timestep_window():
    plan = _plan(((HeadUnit(7, 0), 1.0),))
    plan.cfg = FlairConfig(alpha_0=1.0, t_window=(0.0, 0.5))
    assert plan.alpha(plan.routed[0], HeadUnit(7, 0), 0.9) == 0.0


def test_alpha_is_zero_for_an_unrouted_unit():
    plan = _plan(((HeadUnit(7, 0), 1.0),))
    assert plan.alpha(plan.routed[0], HeadUnit(7, 2), 0.0) == 0.0


def test_rejects_sequence_length_mismatch():
    plan = _plan(((HeadUnit(7, 0), 1.0),))
    with pytest.raises(ValueError, match="sequence length"):
        _residual(plan, 7, torch.zeros((2, SEQ + 1, DIM)), torch.ones((INNER, DIM)))


def test_blocks_touched_reports_blocks_not_units():
    plan = _plan(((HeadUnit(7, 0), 1.0), (HeadUnit(3, 2), 1.0)))
    assert plan.blocks_touched() == frozenset({3, 7})


# --- the equivalence invariant (spec section 3.3) ------------------------


def test_all_heads_reproduces_the_frozen_block_level_oracle():
    """Selecting every head of a block must equal the shipped block blend.

    The bias on the Linear is deliberately non-zero: the residual must be
    projected weight-only, or this test fails.
    """
    torch.manual_seed(0)
    linear = torch.nn.Linear(DIM, INNER)
    torch.nn.init.normal_(linear.bias)

    x = torch.randn((2, SEQ, DIM))
    embedding = torch.randn((SEQ, DIM))
    score, cond = 0.9, slice(1, 2)

    reference = reference_blend(
        [ReferenceRouted(COMPONENT, embedding, ((7, score),))],
        CFG,
        x,
        block_id=7,
        step_frac=0.0,
        cond_slice=cond,
    )
    expected = linear(reference)[cond]

    plan = RoutingPlan(
        routed=(
            RoutedComponent(
                component=COMPONENT,
                embedding=embedding,
                units=tuple((HeadUnit(7, h), score) for h in range(N_HEADS)),
            ),
        ),
        cfg=CFG,
    )
    got = linear(x)[cond] + _residual(plan, 7, x, linear.weight)

    torch.testing.assert_close(got, expected)


def test_two_components_into_one_block_sum_against_the_original_base():
    torch.manual_seed(1)
    linear = torch.nn.Linear(DIM, INNER)
    torch.nn.init.normal_(linear.bias)

    x = torch.randn((2, SEQ, DIM))
    e1, e2 = torch.randn((SEQ, DIM)), torch.randn((SEQ, DIM))
    other = Component(id="c_size", text="a small car", attr=AttributeClass.SIZE)
    cond = slice(1, 2)

    reference = reference_blend(
        [
            ReferenceRouted(COMPONENT, e1, ((7, 0.6),)),
            ReferenceRouted(other, e2, ((7, 0.3),)),
        ],
        CFG,
        x,
        block_id=7,
        step_frac=0.0,
        cond_slice=cond,
    )
    expected = linear(reference)[cond]

    plan = RoutingPlan(
        routed=(
            RoutedComponent(COMPONENT, e1, tuple((HeadUnit(7, h), 0.6) for h in range(N_HEADS))),
            RoutedComponent(other, e2, tuple((HeadUnit(7, h), 0.3) for h in range(N_HEADS))),
        ),
        cfg=CFG,
    )
    got = linear(x)[cond] + _residual(plan, 7, x, linear.weight)

    torch.testing.assert_close(got, expected)


# --- build_routing_plan --------------------------------------------------


def _hasm():
    # blocks (3, 7) x heads (0, 1) x attrs (COLOR, SIZE)
    tensor = np.array(
        [
            [[0.20, 0.80], [0.10, 0.40]],
            [[0.90, 0.30], [0.50, 0.60]],
        ]
    )
    return HASM(tensor, (3, 7), (0, 1), (AttributeClass.COLOR, AttributeClass.SIZE))


def test_head_granularity_selects_the_single_best_head():
    plan = build_routing_plan(
        [COMPONENT], {"c_color": torch.zeros((SEQ, DIM))}, _hasm(), CFG
    )
    assert plan.routed[0].units == ((HeadUnit(7, 0), 0.90),)


def test_block_granularity_expands_to_every_head_at_the_block_score():
    plan = build_routing_plan(
        [COMPONENT],
        {"c_color": torch.zeros((SEQ, DIM))},
        _hasm(),
        CFG,
        granularity="block",
    )
    # to_basm(max) gives block 7 a COLOR score of 0.90
    assert plan.routed[0].units == ((HeadUnit(7, 0), 0.90), (HeadUnit(7, 1), 0.90))


def test_rejects_unknown_granularity():
    with pytest.raises(ValueError, match="unknown granularity"):
        build_routing_plan(
            [COMPONENT],
            {"c_color": torch.zeros((SEQ, DIM))},
            _hasm(),
            CFG,
            granularity="layer",
        )


def test_skips_uncalibrated_attributes():
    action = Component(id="c_action", text="a car driving", attr=AttributeClass.ACTION)
    plan = build_routing_plan(
        [action], {"c_action": torch.zeros((SEQ, DIM))}, _hasm(), CFG
    )
    assert plan.routed == ()


def test_applies_k_override():
    plan = build_routing_plan(
        [COMPONENT],
        {"c_color": torch.zeros((SEQ, DIM))},
        _hasm(),
        CFG,
        k_overrides={"c_color": 2},
    )
    assert plan.routed[0].units == ((HeadUnit(7, 0), 0.90), (HeadUnit(7, 1), 0.50))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./run-local.sh test`
Expected: FAIL — `ImportError: cannot import name 'head_residual'` / `RoutedComponent` has no field `units`

- [ ] **Step 3: Rewrite the routing module**

Replace `flair_t2i/routing.py` entirely:

```python
"""Routing plans and the head-level residual (spec 2026-08-26, section 3).

    H_l = H_base + sum_i alpha_i(t) * (H_i - H_base)
    alpha_i(l, h, t) = alpha_0 * S[l, h, a] * intensity_i * sched(t) * alpha_scale

The residual is applied to the OUTPUT of a text-stream projection rather
than its input. Projection is linear, so for weight ``A``:

    proj(x + d) = proj(x) + d @ A.T

meaning a residual added after projection is identical to one added
before -- provided it is projected weight-only, with no bias, since the
bias cancels in a difference. Because each head owns a contiguous slice of
the projection output, scaling that projected residual by a per-head alpha
vector selects heads and applies their individual strengths in one step.

Selecting every head of a block at one score therefore reproduces the
block-level blend exactly. ``tests/test_routing.py`` pins that against the
frozen oracle in ``tests/reference_blend.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .components import Component
from .config import FlairConfig
from .hasm import HASM
from .heads import HeadUnit, alpha_vector
from .schedule import timestep_scale


@dataclass
class RoutedComponent:
    component: Component
    embedding: torch.Tensor  # [seq, dim]
    units: tuple[tuple[HeadUnit, float], ...]  # (head unit, sensitivity score)
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
            unit.block for rc in self.routed for unit, _ in rc.units
        )

    def blocks_touched(self) -> frozenset[int]:
        return self._blocks

    def alpha(self, rc: RoutedComponent, unit: HeadUnit, step_frac: float) -> float:
        score = next((s for u, s in rc.units if u == unit), None)
        if score is None:
            return 0.0
        sched = timestep_scale(step_frac, self.cfg.t_window)
        return self.cfg.alpha_0 * score * rc.intensity * sched * self.alpha_scale

    def head_residual(
        self,
        block_id: int,
        x: torch.Tensor,
        weight: torch.Tensor,
        step_frac: float,
        cond_slice: slice,
        n_heads: int,
        head_dim: int,
    ) -> torch.Tensor | None:
        """The masked residual to add to this projection's output.

        ``x`` is the projection's input (the text stream), ``weight`` its
        ``nn.Linear`` weight. Returns ``None`` when this block is not
        routed -- the fast path, which is most calls.
        """
        if not self.active or block_id not in self._blocks:
            return None

        base = x[cond_slice]
        seq = x.shape[-2]
        total: torch.Tensor | None = None

        for rc in self.routed:
            alphas = {
                unit.head: self.alpha(rc, unit, step_frac)
                for unit, _ in rc.units
                if unit.block == block_id
            }
            alphas = {head: a for head, a in alphas.items() if a != 0.0}
            if not alphas:
                continue

            if rc.embedding.shape[-2] != seq:
                raise ValueError(
                    f"component {rc.component.id} sequence length "
                    f"{rc.embedding.shape[-2]} does not match states {seq}"
                )

            target = rc.embedding.to(device=base.device, dtype=base.dtype)
            delta = target.unsqueeze(0) - base
            # Weight only. A bias here would break the equivalence invariant.
            projected = torch.nn.functional.linear(delta, weight)
            scaled = projected * alpha_vector(
                alphas, n_heads, head_dim, device=base.device, dtype=base.dtype
            )
            total = scaled if total is None else total + scaled

        return total


def build_routing_plan(
    components: list[Component],
    embeddings: dict[str, torch.Tensor],
    hasm: HASM,
    cfg: FlairConfig,
    intensities: dict[str, float] | None = None,
    k_overrides: dict[str, int] | None = None,
    granularity: str = "head",
    reduce: str = "max",
) -> RoutingPlan:
    """Select target head units per component from the calibrated HASM.

    ``granularity="block"`` selects every head of the top-k blocks at that
    block's reduced score -- one mechanism, two selection functions.
    """
    if granularity not in ("head", "block"):
        raise ValueError(f"unknown granularity {granularity!r}; use 'head' or 'block'")

    intensities = intensities or {}
    k_overrides = k_overrides or {}

    routed: list[RoutedComponent] = []
    for component in components:
        if component.attr not in hasm.attributes:
            continue  # not calibrated; nothing to route
        k = k_overrides.get(component.id, cfg.top_k_default)

        if granularity == "head":
            units = tuple(hasm.top_k(component.attr, k))
        else:
            blocks = hasm.to_basm(reduce=reduce).top_k(component.attr, k)
            units = tuple(
                (HeadUnit(block=block, head=head), score)
                for block, score in blocks
                for head in hasm.head_ids
            )

        if not units:
            continue
        routed.append(
            RoutedComponent(
                component=component,
                embedding=embeddings[component.id],
                units=units,
                intensity=intensities.get(component.id, 1.0),
            )
        )

    return RoutingPlan(routed=tuple(routed), cfg=cfg)
```

- [ ] **Step 4: Update the five test files that construct RoutedComponent**

These construct `RoutedComponent(...)` with `blocks=`; each needs `units=` with `HeadUnit`s. Apply exactly these edits:

In `tests/test_guard.py`, add `from flair_t2i.heads import HeadUnit` to the imports and change line 21:
```python
            blocks=((7, 1.0),),
```
to:
```python
            units=((HeadUnit(7, 0), 1.0),),
```

In `tests/test_guard_membership.py`, add `from flair_t2i.heads import HeadUnit` inside the same local import block at line 31 and change:
```python
                blocks=((7, 1.0),),
```
to:
```python
                units=((HeadUnit(7, 0), 1.0),),
```

In `tests/test_artifacts.py`, add `from flair_t2i.heads import HeadUnit` to the imports and change the `blocks=` keyword in the `RoutedComponent(` call at line 44 to `units=((HeadUnit(7, 0), 0.9),)`, preserving whatever score the existing literal used.

In `tests/test_processor.py`, add `from flair_t2i.heads import HeadUnit` and change `blocks=block_blocks` in `_ref()` to `units=block_units`, renaming the parameter `block_blocks=((7, 1.0),)` to `block_units=((HeadUnit(7, 0), 1.0),)`.

In `tests/test_fuzzy_integration.py`, change line 59:
```python
    assert len(plan.routed[0].blocks) >= 2  # widened by dilation
```
to:
```python
    assert len(plan.routed[0].units) >= 2  # widened by dilation
```

- [ ] **Step 5: Run the suite**

Removing `RoutedComponent.blocks` and `RoutingPlan.blend` breaks three downstream files at once. All three are fixed by later tasks — **do not fix them here.**

Run: `./run-local.sh test`
Expected: **206 passed, 7 failed**, with `tests/test_routing.py` fully green (17 tests) and the failures exactly:

| File | Failures | Cause | Fixed by |
|---|---|---|---|
| `tests/test_processor.py` | 2 | `FlairJointProcessor` calls the removed `plan.blend` | Task 7 |
| `tests/test_pipeline.py` | 2 | `pipeline.py` uses `install_flair` and a BASM | Task 8 |
| `tests/test_artifacts.py` | 3 | `artifacts.py:110` reads the removed `rc.blocks` | Task 8 |

Note `tests/test_fuzzy_integration.py` needs more than the one-line edit named in Step 4: its `_basm()` helper builds a `BASM` and hands it to `build_routing_plan`, which now takes a `HASM`. `BASM.top_k` duck-types far enough to return, then dies in `RoutingPlan.__post_init__` on `unit.block` where `unit` is an `int`. Replace the helper with a `_hasm()` building `blocks (3, 7, 11) × head (0,) × SIZE` at the same 0.4/0.9/0.6 scores, and change its two `plan.alpha(..., 7, 0.0)` calls to pass `HeadUnit(7, 0)`.

- [ ] **Step 6: Commit**

```bash
git add flair_t2i/routing.py tests/test_routing.py tests/test_guard.py \
        tests/test_guard_membership.py tests/test_artifacts.py \
        tests/test_processor.py tests/test_fuzzy_integration.py
git commit -m "feat: route to head units; residual applied post-projection

Selecting every head of a block reproduces the frozen block-level oracle
exactly, so head routing is a strict generalization of what shipped.
test_processor.py is left red here; Task 7 retires FlairJointProcessor."
```

---

### Task 6: The HeadResidualProj wrapper

**Files:**
- Create: `flair_t2i/head_proj.py`
- Test: `tests/test_head_proj.py`

**Interfaces:**
- Consumes: `PlanRef` (existing `processor.py`), `RoutingPlan` (Task 5).
- Produces:
  - `HeadResidualProj(inner: torch.nn.Linear, block_id: int, ref: PlanRef)` — an `nn.Module` with `.forward(x)`; derives `n_heads` and `head_dim` from the wrapped module and the owning attention at construction time via explicit arguments `n_heads`, `head_dim`.
  - Full signature: `HeadResidualProj(inner, block_id, ref, n_heads, head_dim)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_head_proj.py`:

```python
import pytest

torch = pytest.importorskip("torch")

from flair_t2i.attributes import AttributeClass
from flair_t2i.components import Component
from flair_t2i.config import FlairConfig
from flair_t2i.head_proj import HeadResidualProj
from flair_t2i.heads import HeadUnit
from flair_t2i.processor import PlanRef
from flair_t2i.routing import RoutedComponent, RoutingPlan

SEQ, DIM = 4, 8
N_HEADS, HEAD_DIM = 3, 4
INNER = N_HEADS * HEAD_DIM
CFG = FlairConfig(alpha_0=1.0, t_window=(0.0, 1.0))

COMPONENT = Component(id="c_color", text="a red car", attr=AttributeClass.COLOR)


def _linear():
    torch.manual_seed(0)
    linear = torch.nn.Linear(DIM, INNER)
    torch.nn.init.normal_(linear.bias)
    return linear


def _ref(units=((HeadUnit(7, 1), 1.0),), plan=True):
    routed = RoutedComponent(
        component=COMPONENT, embedding=torch.ones((SEQ, DIM)), units=units
    )
    return PlanRef(
        plan=RoutingPlan(routed=(routed,), cfg=CFG) if plan else None,
        step=0,
        total_steps=10,
        do_cfg=True,
    )


def test_passthrough_when_no_plan():
    linear = _linear()
    proj = HeadResidualProj(linear, 7, _ref(plan=False), N_HEADS, HEAD_DIM)
    x = torch.randn((2, SEQ, DIM))
    torch.testing.assert_close(proj(x), linear(x))


def test_passthrough_on_unrouted_block():
    linear = _linear()
    proj = HeadResidualProj(linear, 99, _ref(), N_HEADS, HEAD_DIM)
    x = torch.randn((2, SEQ, DIM))
    torch.testing.assert_close(proj(x), linear(x))


def test_unconditional_rows_are_never_written():
    linear = _linear()
    proj = HeadResidualProj(linear, 7, _ref(), N_HEADS, HEAD_DIM)
    x = torch.randn((2, SEQ, DIM))
    torch.testing.assert_close(proj(x)[0], linear(x)[0])


def test_only_the_selected_head_slice_changes():
    linear = _linear()
    proj = HeadResidualProj(linear, 7, _ref(), N_HEADS, HEAD_DIM)
    x = torch.randn((2, SEQ, DIM))

    delta = (proj(x) - linear(x))[1]
    assert delta[..., 0:4].abs().max().item() == pytest.approx(0.0)
    assert delta[..., 4:8].abs().max().item() > 0.0
    assert delta[..., 8:12].abs().max().item() == pytest.approx(0.0)


def test_does_not_mutate_the_input():
    linear = _linear()
    proj = HeadResidualProj(linear, 7, _ref(), N_HEADS, HEAD_DIM)
    x = torch.randn((2, SEQ, DIM))
    before = x.clone()
    proj(x)
    torch.testing.assert_close(x, before)


def test_matches_a_pre_projection_blend_for_all_heads():
    """The wrapper must honour condition 1: weight-only, no bias."""
    linear = _linear()
    units = tuple((HeadUnit(7, h), 1.0) for h in range(N_HEADS))
    proj = HeadResidualProj(linear, 7, _ref(units=units), N_HEADS, HEAD_DIM)

    x = torch.randn((2, SEQ, DIM))
    blended = x.clone()
    blended[1:2] = blended[1:2] + 1.0 * (torch.ones((SEQ, DIM)) - blended[1:2])

    torch.testing.assert_close(proj(x)[1], linear(blended)[1])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./run-local.sh test`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.head_proj'`

- [ ] **Step 3: Write the wrapper**

Create `flair_t2i/head_proj.py`:

```python
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
```

- [ ] **Step 4: Run it to verify it passes**

Run: `./run-local.sh test`
Expected: `tests/test_head_proj.py` — 6 passed. `tests/test_processor.py` still red (Task 7).

- [ ] **Step 5: Commit**

```bash
git add flair_t2i/head_proj.py tests/test_head_proj.py
git commit -m "feat: HeadResidualProj wraps a text projection, no attention rewrite"
```

---

### Task 7: Install head routing; retire FlairJointProcessor

**Deviation from the spec, recorded here.** Spec §8 lists `processor.py` as "unchanged contract; still delegates." Planning found that unreachable: with `blend()` frozen into the test tree, `FlairJointProcessor` has nothing to call, and block-level routing now travels the same `head_residual` path as head-level routing. Keeping it installed would double-inject. `PlanRef` stays — it is still the step/CFG handle. `FlairJointProcessor` and `install_flair` are removed.

**Files:**
- Modify: `flair_t2i/processor.py` (delete `FlairJointProcessor`, keep `PlanRef`)
- Modify: `flair_t2i/patching.py` (replace `install_flair`/`uninstall_flair`)
- Modify: `tests/test_processor.py` (drop the `FlairJointProcessor` tests, keep the `PlanRef` tests)
- Create: `tests/test_patching.py`

**Interfaces:**
- Consumes: `HeadResidualProj` (Task 6), `PlanRef`.
- Produces:
  - `install_head_routing(transformer, ref: PlanRef) -> list[tuple]` — wraps `add_q_proj`, `add_k_proj`, `add_v_proj` on every `block.attn`; derives `n_heads` from `attn.heads` and `head_dim` from `inner.out_features // n_heads`; skips modules that are absent (the final block sets `context_pre_only=True`).
  - `uninstall_head_routing(handles) -> None`
  - `bypass_blocks` — unchanged, still exported.

- [ ] **Step 1: Write the failing test**

Create `tests/test_patching.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./run-local.sh test`
Expected: FAIL — `ImportError: cannot import name 'install_head_routing'`

- [ ] **Step 3: Rewrite patching**

Replace the `install_flair` / `uninstall_flair` functions in `flair_t2i/patching.py` with the following, keeping `bypass_blocks` exactly as it is and updating the module docstring:

```python
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


def install_head_routing(transformer, ref: PlanRef) -> list[tuple]:
    """Wrap every text-stream projection. Returns handles for removal."""
    handles: list[tuple] = []

    for block_id, block in enumerate(transformer.transformer_blocks):
        attn = block.attn
        n_heads = attn.heads

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
```

- [ ] **Step 4: Retire FlairJointProcessor**

In `flair_t2i/processor.py`, delete the entire `FlairJointProcessor` class (lines 41-63) and the now-unused `from .routing import RoutingPlan` import is still needed for `PlanRef.plan`'s type annotation — keep it. Update the module docstring to:

```python
"""The mutable handle the denoise loop shares with the routing wrappers.

One value -- the current step -- has to reach every wrapped projection on
every block. ``PlanRef`` is the box they all hold a reference to, which is
how that value crosses diffusers without being threaded through its call
signatures.
"""
```

In `tests/test_processor.py`, delete `RecordingProcessor`, the `_ref` helper, and every test whose name starts with `test_processor_`. Keep the four `PlanRef` tests (`test_step_frac_is_step_over_total`, `test_step_frac_handles_zero_total`, `test_cond_slice_is_second_half_under_cfg`, `test_cond_slice_is_everything_without_cfg`) and delete the now-unused imports, leaving only:

```python
import pytest

from flair_t2i.processor import PlanRef
```

- [ ] **Step 5: Run the suite**

`tests/test_pipeline.py` and `tests/test_harness.py` will now fail on the missing `install_flair` — Tasks 8 and 9 fix them. Confirm failures are confined to those two files:

Run: `./run-local.sh test`
Expected: `tests/test_patching.py` and `tests/test_processor.py` green; failures confined to `test_pipeline.py` and `test_harness.py`

- [ ] **Step 6: Commit**

```bash
git add flair_t2i/patching.py flair_t2i/processor.py \
        tests/test_patching.py tests/test_processor.py
git commit -m "feat: install head routing by wrapping text projections

Retires FlairJointProcessor: with blend() frozen into the test tree it has
nothing to call, and block-level routing now travels the same
head_residual path. PlanRef stays as the step handle.

Deviates from spec section 8, which listed processor.py as unchanged."
```

---

### Task 8: Pipeline, artifacts, and explain on HASM

**Files:**
- Modify: `flair_t2i/pipeline.py`
- Modify: `flair_t2i/artifacts.py:110`
- Modify: `scripts/explain.py:98,143,171,179`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `HASM` (Task 3), `build_routing_plan` (Task 5), `install_head_routing` (Task 7).
- Produces:
  - `FlairPipeline(pipe, cfg, hasm: HASM, nlp=None, granularity: str = "head")`
  - `.hasm` attribute replaces `.basm`
  - `.generate(...)` signature unchanged.

- [ ] **Step 1: Update the pipeline test**

In `tests/test_pipeline.py`, replace the `_basm()` helper and every `FlairPipeline(...)` construction:

```python
from flair_t2i.hasm import HASM
from flair_t2i.heads import HeadUnit


def _hasm():
    # 3 blocks x 2 heads x COLOR; block 0 head 1 is the single peak
    tensor = np.array([[[0.4], [0.9]], [[0.2], [0.1]], [[0.3], [0.3]]])
    return HASM(tensor, (0, 1, 2), (0, 1), (AttributeClass.COLOR,))
```

Change every `FlairPipeline(pipe, FlairConfig(device="cpu"), _basm())` to
`FlairPipeline(pipe, FlairConfig(device="cpu"), _hasm())`, and replace the
routing assertion at line 89:

```python
    assert fp.last_plan.blocks_touched() == frozenset({0})
```

with:

```python
    assert fp.last_plan.routed[0].units == ((HeadUnit(0, 1), 0.9),)
    assert fp.last_plan.blocks_touched() == frozenset({0})
```

Add one test for the granularity switch:

```python
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
```

Also update `StubTransformer`/`StubAttn` in that file so `install_head_routing` can wrap them — give `StubAttn` a `heads` attribute and three `torch.nn.Linear(DIM, DIM)` projections, matching `tests/test_patching.py`'s stubs:

```python
class StubAttn:
    def __init__(self):
        self.heads = 2
        self.add_q_proj = torch.nn.Linear(DIM, DIM)
        self.add_k_proj = torch.nn.Linear(DIM, DIM)
        self.add_v_proj = torch.nn.Linear(DIM, DIM)
```

and delete `get_processor` / `set_processor`, which nothing calls any more.

- [ ] **Step 2: Run it to verify it fails**

Run: `./run-local.sh test`
Expected: FAIL — `TypeError: FlairPipeline.__init__() got an unexpected keyword argument 'granularity'`

- [ ] **Step 3: Update the pipeline**

In `flair_t2i/pipeline.py`, change the imports:

```python
from .hasm import HASM
from .patching import install_head_routing, uninstall_head_routing
```

(remove `from .basm import BASM` and `from .patching import install_flair, uninstall_flair`)

Replace `__init__`:

```python
    def __init__(
        self,
        pipe,
        cfg: FlairConfig,
        hasm: HASM,
        nlp=None,
        granularity: str = "head",
    ) -> None:
        self.pipe = pipe
        self.cfg = cfg
        self.hasm = hasm
        self.nlp = nlp
        self.granularity = granularity
        self.last_plan: RoutingPlan | None = None
        self.last_guard: CoherenceGuard | None = None
```

In `generate`, change line 63 and the `build_routing_plan` call:

```python
            routable = [c for c in components if c.attr in self.hasm.attributes]
```

```python
            plan = build_routing_plan(
                routable,
                embeddings,
                self.hasm,
                self.cfg,
                intensities,
                k_overrides,
                granularity=self.granularity,
            )
```

and swap the install/uninstall calls:

```python
        handles = install_head_routing(self.pipe.transformer, ref)
```
```python
            uninstall_head_routing(handles)
```

- [ ] **Step 4: Update artifacts and explain**

In `flair_t2i/artifacts.py`, replace line 110:

```python
            "blocks": [[int(b), float(s)] for b, s in rc.blocks],
```

with:

```python
            "units": [
                [int(u.block), int(u.head), float(s)] for u, s in rc.units
            ],
```

**This renames the emitted JSON key, so `tests/test_artifacts.py` must move with it.** Task 5 already converted that file's `RoutedComponent` fixture to `units=((HeadUnit(7, 0), 0.93), (HeadUnit(3, 0), 0.22))`, but its assertion still reads the old key. Change:

```python
    assert entry["blocks"] == [[7, 0.93], [3, 0.22]]
```

to:

```python
    assert entry["units"] == [[7, 0, 0.93], [3, 0, 0.22]]
```

Three `tests/test_artifacts.py` tests are red coming into this task (they broke in Task 5 when `rc.blocks` was removed); this edit plus the `artifacts.py` change is what makes them green again.

In `scripts/explain.py`, import `from flair_t2i.hasm import HASM` (replacing the `BASM` import) and replace the `synthetic_basm` helper (lines 50-63) with:

```python
def synthetic_hasm(n_blocks: int, n_heads: int, attributes) -> HASM:
    """A stand-in HASM: each attribute peaks on a different (block, head)."""
    rng = np.random.default_rng(0)
    tensor = rng.uniform(0.05, 0.35, size=(n_blocks, n_heads, len(attributes)))
    stride = max(1, n_blocks // max(1, len(attributes)))
    for col in range(len(attributes)):
        block = (3 + col * stride) % n_blocks
        head = col % n_heads
        tensor[block, head, col] = rng.uniform(0.80, 0.95)
    return HASM(
        tensor=tensor,
        block_ids=tuple(range(n_blocks)),
        head_ids=tuple(range(n_heads)),
        attributes=attributes,
    )
```

Then apply these four call-site edits:

- after line 75, add `parser.add_argument("--heads", type=int, default=24, help="N_HEADS to assume")`
- lines 94-99, replace the `--basm` branch:
  ```python
      if args.hasm:
          hasm = HASM.load(args.hasm)
          source = f"calibrated -- {args.hasm}"
      else:
          hasm = synthetic_hasm(args.blocks, args.heads, CORE_ATTRIBUTES)
          source = "SYNTHETIC placeholder (real one arrives in Week 3-4)"
  ```
  and rename the argument on line 77 to `parser.add_argument("--hasm", type=Path, help="a real calibrated hasm.npz")`. Change the `BASM` label on line 102 to `HASM`.
- line 143: `blocks = "  ".join(f"B{b}({s:.2f})" for b, s in rc.blocks)` becomes
  `units = "  ".join(f"B{u.block}H{u.head}({s:.2f})" for u, s in rc.units)` — rename the local and its use on the following print line.
- line 171: `top_block = rc.blocks[0][0]` becomes `top_unit = rc.units[0][0]`, renaming its use below.
- line 179: `plan.alpha(rc, rc.blocks[0][0], step / args.steps) > 0` becomes
  `plan.alpha(rc, rc.units[0][0], step / args.steps) > 0`.

Wherever `build_routing_plan(...)` is called in this script, pass `hasm` in place of `basm`.

- [ ] **Step 5: Run the suite**

Run: `./run-local.sh test`
Expected: `tests/test_pipeline.py` green; failures confined to `tests/test_harness.py`

- [ ] **Step 6: Verify explain still runs end to end**

Run: `./run-local.sh explain "A small red sports car under warm evening light"`
Expected: four parsed components, each showing `B<block>H<head>(score)` units, α decaying to 0

- [ ] **Step 7: Commit**

```bash
git add flair_t2i/pipeline.py flair_t2i/artifacts.py scripts/explain.py \
        tests/test_pipeline.py
git commit -m "feat: pipeline routes from a HASM with a granularity switch"
```

---

### Task 9: Calibration harness on head units

**Files:**
- Modify: `flair_t2i/calibration/harness.py`
- Modify: `tests/test_harness.py`

**Interfaces:**
- Consumes: `HeadUnit` (Task 2), `HASM` (Task 3), `install_head_routing` (Task 7).
- Produces:
  - `SwapSpec(unit: HeadUnit, prompt: str)` — frozen dataclass
  - `calibrate(generate_fn, corpus, block_ids, head_ids, masker, seeds, scorer=None, progress=None, checkpoint_dir=None) -> HASM`
  - `ProgressFn = Callable[[AttributeClass, HeadUnit, float], None]`
  - `make_swap_generate_fn(flair_pipeline, steps) -> SwapGenerateFn` — unchanged name and signature
  - Checkpoint cells at `cells/{attr}_{block}_{head}.json`

- [ ] **Step 1: Update the harness test**

In `tests/test_harness.py`, add `from flair_t2i.heads import HeadUnit` to the imports and replace lines 12-15:

```python
BLOCK_IDS = (0, 1, 2)
HEAD_IDS = (0, 1)
SEEDS = [0]
#: The fake model routes the attribute through exactly one head.
LIVE_UNIT = HeadUnit(block=1, head=1)
```

Replace `fake_generate` (lines 30-34):

```python
def fake_generate(prompt: str, seed: int, swap: SwapSpec | None) -> Image.Image:
    """Only a swap at LIVE_UNIT actually changes the image."""
    if swap is not None and swap.unit == LIVE_UNIT and "blue" in swap.prompt:
        return Image.new("RGB", (64, 64), (30, 30, 220))
    return Image.new("RGB", (64, 64), (220, 30, 30))
```

Replace the `_calibrate` helper (lines 48-54):

```python
def _calibrate(**kw):
    kw.setdefault("generate_fn", fake_generate)
    kw.setdefault("corpus", _corpus())
    kw.setdefault("block_ids", BLOCK_IDS)
    kw.setdefault("head_ids", HEAD_IDS)
    kw.setdefault("masker", FULL_MASKER)
    kw.setdefault("seeds", SEEDS)
    return calibrate(**kw)
```

Then update the three assertions that named blocks. Line 62 becomes `assert hasm.block_ids == BLOCK_IDS` (and add `assert hasm.head_ids == HEAD_IDS`); line 68 and line 163 become:

```python
    assert hasm.top_k(AttributeClass.COLOR, 1) == [(LIVE_UNIT, pytest.approx(1.0))]
```

and lines 73-74's `basm.score(0, ...)` / `basm.score(2, ...)` become
`hasm.score(HeadUnit(0, 0), ...)` / `hasm.score(HeadUnit(2, 0), ...)`. Rename every local `basm` in this file to `hasm`.

Add a test that the checkpoint key carries the head:

```python
def test_checkpoint_cell_is_keyed_by_block_and_head(tmp_path):
    calibrate(
        _generate_fn(),
        corpus=_corpus(),
        block_ids=BLOCK_IDS,
        head_ids=HEAD_IDS,
        masker=_masker,
        seeds=[0],
        checkpoint_dir=tmp_path,
    )
    written = {p.name for p in (tmp_path / "cells").glob("*.json")}
    assert "color_7_1.json" in written
    assert len(written) == len(BLOCK_IDS) * len(HEAD_IDS)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./run-local.sh test`
Expected: FAIL — `TypeError: calibrate() got an unexpected keyword argument 'block_ids'`

- [ ] **Step 3: Update the harness**

In `flair_t2i/calibration/harness.py`, replace the imports of `BASM` with `HASM` and add `HeadUnit`:

```python
from ..hasm import HASM
from ..heads import HeadUnit
```

Replace `SwapSpec`, `_cell_path`, `_save_cell`, and the `ProgressFn` alias:

```python
@dataclass(frozen=True)
class SwapSpec:
    unit: HeadUnit
    prompt: str


ProgressFn = Callable[[AttributeClass, HeadUnit, float], None]


def _cell_path(
    checkpoint_dir: str | Path, attr: AttributeClass, unit: HeadUnit
) -> Path:
    return (
        Path(checkpoint_dir)
        / "cells"
        / f"{attr.value}_{unit.block}_{unit.head}.json"
    )


def _save_cell(
    path: Path, attr: AttributeClass, unit: HeadUnit, raw: float, samples: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "attribute": attr.value,
                "block": unit.block,
                "head": unit.head,
                "raw": raw,
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
```

Change `_measure_cell`'s `block_id: int` parameter to `unit: HeadUnit` and its one use:

```python
                swap=SwapSpec(unit=unit, prompt=pair.changed),
```

Replace `calibrate` entirely:

```python
def calibrate(
    generate_fn: SwapGenerateFn,
    corpus: dict[AttributeClass, list[ContrastivePair]],
    block_ids: tuple[int, ...],
    head_ids: tuple[int, ...],
    masker: Masker,
    seeds: list[int],
    scorer: ImageTextScorer | None = None,
    progress: ProgressFn | None = None,
    checkpoint_dir: str | Path | None = None,
) -> HASM:
    """Measure per-head sensitivity for every attribute in ``corpus``.

    Pass ``checkpoint_dir`` to make the sweep resumable. At
    ``blocks x heads x attributes`` cells this is not optional in practice.
    """
    attributes = tuple(a for a in AttributeClass if a in corpus)
    raw = np.zeros((len(block_ids), len(head_ids), len(attributes)))

    for plane, attr in enumerate(attributes):
        for i, block_id in enumerate(block_ids):
            for j, head_id in enumerate(head_ids):
                unit = HeadUnit(block=block_id, head=head_id)
                cached = (
                    _load_cell(_cell_path(checkpoint_dir, attr, unit))
                    if checkpoint_dir is not None
                    else None
                )

                if cached is not None:
                    raw[i, j, plane] = cached
                else:
                    value, samples = _measure_cell(
                        generate_fn, attr, unit, corpus[attr], seeds, masker, scorer
                    )
                    raw[i, j, plane] = value
                    if checkpoint_dir is not None:
                        _save_cell(
                            _cell_path(checkpoint_dir, attr, unit),
                            attr,
                            unit,
                            value,
                            samples,
                        )

                if progress is not None:
                    progress(attr, unit, raw[i, j, plane])

        # Normalise across every unit for this attribute, not per block.
        raw[:, :, plane] = _normalise(raw[:, :, plane])

    return HASM(
        tensor=raw,
        block_ids=block_ids,
        head_ids=head_ids,
        attributes=attributes,
    )
```

Finally, in `make_swap_generate_fn`, update the local imports and the plan construction:

```python
    from ..patching import install_head_routing, uninstall_head_routing
```
```python
                RoutedComponent(
                    component=component,
                    embedding=embeddings["swap"],
                    units=((swap.unit, 1.0),),
                ),
```
```python
        handles = install_head_routing(flair_pipeline.pipe.transformer, ref)
```
```python
            uninstall_head_routing(handles)
```

Update the function's docstring closing sentence to read:

```
    A swap is the routing residual at full strength: with alpha = 1.0 on one
    head, that head's Q/K/V become exactly what they would be under
    H_changed, and every other head is untouched. Calibration therefore
    reuses the routing machinery instead of adding a second injection path
    that could drift from it.
```

- [ ] **Step 4: Run the suite**

Run: `./run-local.sh test`
Expected: PASS — the full suite green

- [ ] **Step 5: Commit**

```bash
git add flair_t2i/calibration/harness.py tests/test_harness.py
git commit -m "feat: calibrate per head unit into a HASM

Cells are keyed by (attribute, block, head) and normalisation runs across
every unit for an attribute rather than down a single block column."
```

---

### Task 10: Campaign script and documentation

**Files:**
- Modify: `scripts/calibrate.py`
- Modify: `scripts/smoke_test.py:100` (verify only — `blocks_touched()` survives)
- Modify: `docs/CODE_WALKTHROUGH.md`, `docs/EXECUTION_TREE.md`, `docs/RUNBOOK.md`

**Interfaces:**
- Consumes: everything from Tasks 2-9.
- Produces: a runnable `python scripts/calibrate.py hasm --out calibration_runs/`.

- [ ] **Step 1: Update the campaign script**

In `scripts/calibrate.py`:

Change the imports:

```python
from flair_t2i.hasm import HASM
```
(replacing `from flair_t2i.basm import BASM`)

In `_pipeline`, replace the placeholder line:

```python
    # Calibration never reads this matrix; it swaps prompts directly.
    placeholder = HASM.uniform((0,), (0,), CORE_ATTRIBUTES)
    return FlairPipeline(pipe, cfg, placeholder, nlp=spacy.load("en_core_web_sm"))
```

Change the phase choices and add the head arguments:

```python
    parser.add_argument("phase", choices=["prefilter", "hasm"])
```

Replace the whole `basm` phase body (from `if args.vitality is None:` to the end of `main`) with:

```python
    corpus = load_corpus(DEFAULT_CORPUS_PATH)
    n_blocks = len(fp.pipe.transformer.transformer_blocks)
    n_heads = fp.pipe.transformer.transformer_blocks[0].attn.heads
    block_ids = tuple(range(n_blocks))
    head_ids = tuple(range(n_heads))

    pairs = sum(len(v) for v in corpus.values())
    units = n_blocks * n_heads
    total = pairs * len(args.seeds) * (1 + units)
    print(f"calibrating {len(corpus)} attributes over {units} head units")
    print(f"  {n_blocks} blocks x {n_heads} heads")
    print(f"  {pairs} pairs x {len(args.seeds)} seed(s) -- up to {total} generations")
    print(f"checkpointing to {args.out / 'cells'} -- safe to re-run after a timeout\n")

    hasm = calibrate(
        make_swap_generate_fn(fp, steps=args.steps),
        corpus=corpus,
        block_ids=block_ids,
        head_ids=head_ids,
        masker=ClipSegMasker(device=cfg.device),
        seeds=args.seeds,
        scorer=ClipScorer(device=cfg.device),
        checkpoint_dir=args.out,
        progress=lambda attr, unit, value: print(
            f"  {attr.value:<9} B{unit.block:<3}H{unit.head:<3} raw={value:.4f}"
        ),
    )
    hasm.save(args.out / "hasm.npz")
    hasm.to_basm().save(args.out / "basm.npz")

    print("\ncalibrated HASM:")
    for attr in hasm.attributes:
        print(f"  {attr.value:<9} top units: {hasm.top_k(attr, 3)}")

    basm = hasm.to_basm()
    peaks = {attr: basm.top_k(attr, 1)[0][0] for attr in basm.attributes}
    if len(set(peaks.values())) == 1:
        print("\n  WARNING: every attribute peaks on the same block.")
        print("  There is no disentanglement to exploit -- see roadmap 2.5.")

    print(f"\nwrote {args.out / 'hasm.npz'} and {args.out / 'basm.npz'}")
```

Update the module docstring's usage block to:

```
    python scripts/calibrate.py prefilter --top-n 10 --out calibration_runs/
    python scripts/calibrate.py hasm --seeds 0 --out calibration_runs/

The HASM phase sweeps every (attribute, block, head) cell and checkpoints
each one as it completes. Re-run the same command after a session times
out and it resumes. The prefilter phase is retained for the FLUX port; the
HASM phase does not read its output.
```

Keep the `--vitality` argument, the `VitalityReport` import, and the entire `prefilter` phase exactly as they are: spec §5.1 retains the block prefilter for the FLUX port. What goes away is only the *former* `basm` phase's dependence on it — the `hasm` phase reads no vitality file, so the `if args.vitality is None: parser.error(...)` guard and the `VitalityReport.load(...)` call in that phase are deleted.

- [ ] **Step 2: Update the smoke test call site**

`scripts/smoke_test.py:100` calls `fp.last_plan.blocks_touched()`, which Task 5 preserved. Confirm it still reads correctly and change its label:

```python
    print(f"units touched:     {sorted(fp.last_plan.routed[0].units)}")
    print(f"blocks touched:    {sorted(fp.last_plan.blocks_touched())}")
```

Also update the `BASM.uniform(...)` construction in that script to
`HASM.uniform(tuple(range(n_blocks)), tuple(range(n_heads)), CORE_ATTRIBUTES)`,
reading `n_heads` the same way `calibrate.py` does, and import `HASM`.

- [ ] **Step 3: Run the suite and both local scripts**

```bash
./run-local.sh test
./run-local.sh explain "A small red sports car under warm evening light"
```

Expected: full suite green; `explain` prints per-head units.

- [ ] **Step 4: Update the three prose docs**

Each describes block-level routing throughout. Make these specific changes:

- `docs/CODE_WALKTHROUGH.md` — in the "1a" table, replace row 13's `processor.py` description with `PlanRef` only, and add `heads.py`, `hasm.py`, `head_proj.py` rows. In the "core equation" block, change `S[ℓ,a]` to `S[ℓ,h,a]`. In Part 2 Phase C, replace `plan.blend(ehs, ℓ, ...)` with `HeadResidualProj` wrapping the three text projections.
- `docs/EXECUTION_TREE.md` — in TREE 1 section B6/C, replace the `blend` call chain with the projection-wrapper path; in TREE 3, replace "per vital block" with "per head unit" and update the cell path to `{attr}_{block}_{head}.json`.
- `docs/RUNBOOK.md` — replace Phase 6's command with `python scripts/calibrate.py hasm --seeds 0 --out calibration_runs/`, update the "Total cells" line to `attributes × blocks × heads`, and in Phase 7 change `BASM.load` to `HASM.load` with `top_k` printing head units.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate.py scripts/smoke_test.py
git add -f docs/CODE_WALKTHROUGH.md docs/EXECUTION_TREE.md docs/RUNBOOK.md
git commit -m "feat: head-level calibration campaign and doc updates

calibrate.py hasm sweeps every (attribute, block, head) cell and writes
both hasm.npz and the derived basm.npz."
```

---

### Task 11: Intermediate latent capture

The demo's most legible evidence is watching a routed attribute emerge across denoising. FLAIR has no way to see intermediate latents today.

**Files:**
- Create: `flair_t2i/latents.py`
- Modify: `flair_t2i/pipeline.py` (`generate` gains a `recorder` parameter)
- Test: `tests/test_latents.py`

**Interfaces:**
- Consumes: nothing from Tasks 2-10.
- Produces:
  - `LatentRecorder(decode_fn, at=(0.0, 0.25, 0.5, 0.75))` — `decode_fn: Callable[[Any], Image.Image]` is injected so tests never need a VAE
  - `.target_steps(total_steps) -> set[int]`
  - `.__call__(step_index, total_steps, latents) -> None`
  - `.frames: list[tuple[float, Image.Image]]`
  - `FlairPipeline.generate(..., recorder: LatentRecorder | None = None)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_latents.py`:

```python
import pytest
from PIL import Image

from flair_t2i.latents import LatentRecorder


def _decode(latents):
    return Image.new("RGB", (8, 8), (int(latents), 0, 0))


def test_target_steps_map_fractions_onto_step_indices():
    recorder = LatentRecorder(_decode, at=(0.0, 0.5))
    assert recorder.target_steps(20) == {0, 10}


def test_target_steps_never_exceed_the_last_index():
    recorder = LatentRecorder(_decode, at=(1.0,))
    assert recorder.target_steps(10) == {9}


def test_records_only_at_target_steps():
    recorder = LatentRecorder(_decode, at=(0.0, 0.5))
    for step in range(20):
        recorder(step, 20, step)

    assert [frac for frac, _ in recorder.frames] == [0.0, 0.5]


def test_frames_carry_the_decoded_image():
    recorder = LatentRecorder(_decode, at=(0.0,))
    recorder(0, 4, 42)

    assert recorder.frames[0][1].getpixel((0, 0)) == (42, 0, 0)


def test_zero_total_steps_records_nothing():
    recorder = LatentRecorder(_decode, at=(0.0,))
    recorder(0, 0, 1)
    assert recorder.frames == []


def test_reset_clears_frames_between_generations():
    recorder = LatentRecorder(_decode, at=(0.0,))
    recorder(0, 4, 1)
    recorder.reset()
    assert recorder.frames == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./run-local.sh test`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.latents'`

- [ ] **Step 3: Write the recorder**

Create `flair_t2i/latents.py`:

```python
"""Capture intermediate latents during denoising.

Routing's effect is easiest to read as a sequence: the attribute appears
early, while ``timestep_scale`` still has weight, and the rest of the run
refines what is already there. Decoding is injected rather than imported
so nothing here needs a VAE -- the tests pass a stub.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from PIL import Image


@dataclass
class LatentRecorder:
    decode_fn: Callable[[Any], Image.Image]
    at: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)
    frames: list[tuple[float, Image.Image]] = field(default_factory=list)

    def target_steps(self, total_steps: int) -> set[int]:
        """The step indices closest to each requested fraction."""
        if total_steps <= 0:
            return set()
        return {
            min(total_steps - 1, int(round(frac * total_steps))) for frac in self.at
        }

    def reset(self) -> None:
        self.frames = []

    def __call__(self, step_index: int, total_steps: int, latents: Any) -> None:
        if step_index not in self.target_steps(total_steps):
            return
        frac = step_index / total_steps if total_steps else 0.0
        self.frames.append((frac, self.decode_fn(latents)))
```

- [ ] **Step 4: Wire it into the pipeline**

In `flair_t2i/pipeline.py`, add the import:

```python
from .latents import LatentRecorder
```

Add the parameter to `generate`'s signature, after `fuzzy: bool = True`:

```python
        recorder: LatentRecorder | None = None,
```

and replace the `on_step` callback body:

```python
            def on_step(pipe, step_index, timestep, callback_kwargs):
                ref.step = step_index
                if recorder is not None and "latents" in callback_kwargs:
                    recorder(step_index, steps, callback_kwargs["latents"])
                return callback_kwargs
```

- [ ] **Step 5: Run it to verify it passes**

Run: `./run-local.sh test`
Expected: PASS — 6 new tests, full suite green

- [ ] **Step 6: Commit**

```bash
git add flair_t2i/latents.py flair_t2i/pipeline.py tests/test_latents.py
git commit -m "feat: capture intermediate latents at chosen step fractions"
```

---

### Task 12: The demo sweep

Runs all 7 attributes over every head unit at **one contrastive pair each**, keeping every image. This is the deliverable the supervisor sees, and it is one fifth the cost of the full campaign (`7 × 1 × (1 + 576) = 4,039` generations, versus 20,195 at five pairs).

**Files:**
- Modify: `flair_t2i/calibration/harness.py` (add the `on_pair` hook)
- Create: `flair_t2i/demo/__init__.py`
- Create: `flair_t2i/demo/sweep.py`
- Test: `tests/test_demo_sweep.py`

**Interfaces:**
- Consumes: `calibrate` (Task 9), `HeadUnit` (Task 2), `HASM` (Task 3), `LatentRecorder` (Task 11).
- Produces:
  - `calibrate(..., on_pair: PairFn | None = None)` where `PairFn = Callable[[AttributeClass, HeadUnit, ContrastivePair, int, Image.Image, Image.Image], None]`
  - `DemoPaths(root)` with `.heads`, `.blocks`, `.latents`, `.baselines` directories and `.head_image(attr, unit) -> Path`, `.block_image(attr, block) -> Path`
  - `run_demo_sweep(generate_fn, corpus, block_ids, head_ids, masker, paths, seeds, scorer=None, progress=None) -> HASM`

- [ ] **Step 1: Add the image hook to the harness**

In `flair_t2i/calibration/harness.py`, add the type alias next to `ProgressFn`:

```python
PairFn = Callable[
    [AttributeClass, HeadUnit, ContrastivePair, int, Image.Image, Image.Image], None
]
```

Add an `on_pair: PairFn | None = None` parameter to **both** `_measure_cell` and `calibrate` (last parameter in each), thread it through the `calibrate` → `_measure_cell` call, and inside `_measure_cell`'s seed loop, immediately after `swapped = generate_fn(...)`:

```python
            if on_pair is not None:
                on_pair(attr, unit, pair, seed, baseline, swapped)
```

**Note the interaction with checkpointing:** a cell restored from `cells/` never regenerates, so `on_pair` never fires for it. The demo therefore runs against a **fresh** `checkpoint_dir` (or none), which Step 3's docstring states.

- [ ] **Step 2: Write the failing test**

Create `tests/test_demo_sweep.py`:

```python
import numpy as np
import pytest
from PIL import Image

from flair_t2i.attributes import AttributeClass
from flair_t2i.calibration.corpus import ContrastivePair
from flair_t2i.calibration.harness import SwapSpec
from flair_t2i.demo.sweep import DemoPaths, run_demo_sweep
from flair_t2i.heads import HeadUnit
from flair_t2i.metrics.masking import RectMasker

BLOCK_IDS = (0, 1)
HEAD_IDS = (0, 1)
LIVE_UNIT = HeadUnit(block=1, head=1)
FULL_MASKER = RectMasker((0.0, 0.0, 1.0, 1.0))


def _corpus():
    return {
        AttributeClass.COLOR: [
            ContrastivePair("a red car on a road", "a blue car on a road", "car", None)
        ]
    }


def _generate(prompt: str, seed: int, swap: SwapSpec | None) -> Image.Image:
    if swap is not None and swap.unit == LIVE_UNIT and "blue" in swap.prompt:
        return Image.new("RGB", (16, 16), (30, 30, 220))
    return Image.new("RGB", (16, 16), (220, 30, 30))


def _sweep(tmp_path):
    return run_demo_sweep(
        generate_fn=_generate,
        corpus=_corpus(),
        block_ids=BLOCK_IDS,
        head_ids=HEAD_IDS,
        masker=FULL_MASKER,
        paths=DemoPaths(tmp_path),
        seeds=[0],
    )


def test_writes_one_image_per_head_unit(tmp_path):
    _sweep(tmp_path)
    written = sorted(p.name for p in (tmp_path / "heads").glob("*.png"))
    assert written == [
        "color_b0_h0.png",
        "color_b0_h1.png",
        "color_b1_h0.png",
        "color_b1_h1.png",
    ]


def test_writes_one_image_per_block(tmp_path):
    _sweep(tmp_path)
    written = sorted(p.name for p in (tmp_path / "blocks").glob("*.png"))
    assert written == ["color_b0.png", "color_b1.png"]


def test_writes_a_baseline_per_attribute(tmp_path):
    _sweep(tmp_path)
    assert (tmp_path / "baselines" / "color.png").exists()


def test_returns_a_hasm_scoring_the_live_unit_highest(tmp_path):
    hasm = _sweep(tmp_path)
    assert hasm.top_k(AttributeClass.COLOR, 1) == [(LIVE_UNIT, pytest.approx(1.0))]


def test_scores_are_saved_alongside_the_images(tmp_path):
    _sweep(tmp_path)
    assert (tmp_path / "hasm.npz").exists()


def test_paths_are_created_on_construction(tmp_path):
    paths = DemoPaths(tmp_path / "bundle")
    for directory in (paths.heads, paths.blocks, paths.latents, paths.baselines):
        assert directory.is_dir()


def test_head_image_path_is_stable(tmp_path):
    paths = DemoPaths(tmp_path)
    got = paths.head_image(AttributeClass.COLOR, HeadUnit(3, 7))
    assert got == tmp_path / "heads" / "color_b3_h7.png"
```

- [ ] **Step 3: Run it to verify it fails**

Run: `./run-local.sh test`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.demo'`

- [ ] **Step 4: Write the sweep**

Create `flair_t2i/demo/__init__.py`:

```python
"""Demonstration bundle: a runnable, self-contained view of what routing does."""
```

Create `flair_t2i/demo/sweep.py`:

```python
"""The demonstration sweep.

Every attribute over every head unit at ONE contrastive pair each -- one
fifth the full campaign's cost, and enough to show which block and which
head each attribute actually responds to. Unlike the campaign, this keeps
every generated image, because the images are the deliverable.

Run against a fresh output directory. A checkpointed cell is never
regenerated, so its images would be missing from the bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..attributes import AttributeClass
from ..calibration.harness import calibrate
from ..hasm import HASM
from ..heads import HeadUnit


@dataclass
class DemoPaths:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        for directory in (self.heads, self.blocks, self.latents, self.baselines):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def heads(self) -> Path:
        return self.root / "heads"

    @property
    def blocks(self) -> Path:
        return self.root / "blocks"

    @property
    def latents(self) -> Path:
        return self.root / "latents"

    @property
    def baselines(self) -> Path:
        return self.root / "baselines"

    def head_image(self, attr: AttributeClass, unit: HeadUnit) -> Path:
        return self.heads / f"{attr.value}_b{unit.block}_h{unit.head}.png"

    def block_image(self, attr: AttributeClass, block: int) -> Path:
        return self.blocks / f"{attr.value}_b{block}.png"

    def baseline_image(self, attr: AttributeClass) -> Path:
        return self.baselines / f"{attr.value}.png"


def run_demo_sweep(
    generate_fn,
    corpus,
    block_ids: tuple[int, ...],
    head_ids: tuple[int, ...],
    masker,
    paths: DemoPaths,
    seeds: list[int],
    scorer=None,
    progress=None,
) -> HASM:
    """Sweep every head unit, keeping every image, and return the HASM."""

    def on_pair(attr, unit, pair, seed, baseline: Image.Image, swapped: Image.Image):
        baseline_path = paths.baseline_image(attr)
        if not baseline_path.exists():
            baseline.save(baseline_path)
        swapped.save(paths.head_image(attr, unit))

    hasm = calibrate(
        generate_fn,
        corpus=corpus,
        block_ids=block_ids,
        head_ids=head_ids,
        masker=masker,
        seeds=seeds,
        scorer=scorer,
        progress=progress,
        checkpoint_dir=None,
        on_pair=on_pair,
    )

    # Block-level counterpart: the same swap with every head of a block
    # selected at once, which is what block-level routing does.
    from ..calibration.harness import SwapSpec

    for attr, pairs in corpus.items():
        pair = pairs[0]
        for block in block_ids:
            image = generate_fn(
                prompt=pair.base,
                seed=seeds[0],
                swap=SwapSpec(
                    unit=HeadUnit(block=block, head=head_ids[0]), prompt=pair.changed
                ),
            )
            image.save(paths.block_image(attr, block))

    hasm.save(paths.root / "hasm.npz")
    return hasm
```

- [ ] **Step 5: Run it to verify it passes**

Run: `./run-local.sh test`
Expected: PASS — 7 new tests, full suite green

- [ ] **Step 6: Commit**

```bash
git add flair_t2i/demo/ flair_t2i/calibration/harness.py tests/test_demo_sweep.py
git commit -m "feat: demo sweep keeps every per-head and per-block image

calibrate() gains an on_pair hook so the demo can retain images the
campaign discards after measuring."
```

---

### Task 13: The handoff report

4,039 loose PNGs communicate nothing. The heatmap carries the argument; the images are what you click into from it. Output is one self-contained directory with an `index.html` that needs no Python, no install, and no repo.

**Files:**
- Create: `flair_t2i/demo/report.py`
- Create: `scripts/demo.py`
- Test: `tests/test_demo_report.py`

**Interfaces:**
- Consumes: `HASM` (Task 3), `DemoPaths` (Task 12).
- Produces:
  - `heat_color(score: float) -> str` — CSS colour for a score in `[0, 1]`
  - `render_report(hasm: HASM, paths: DemoPaths, title: str) -> str` — returns HTML
  - `write_report(hasm, paths, title) -> Path` — writes `index.html`, returns its path

- [ ] **Step 1: Write the failing test**

Create `tests/test_demo_report.py`:

```python
import numpy as np
import pytest

from flair_t2i.attributes import AttributeClass
from flair_t2i.demo.report import heat_color, render_report, write_report
from flair_t2i.demo.sweep import DemoPaths
from flair_t2i.hasm import HASM

ATTRS = (AttributeClass.COLOR, AttributeClass.SIZE)


def _hasm():
    tensor = np.array(
        [
            [[0.10, 0.90], [0.20, 0.30]],
            [[1.00, 0.05], [0.40, 0.60]],
        ]
    )
    return HASM(tensor, (0, 1), (0, 1), ATTRS)


def test_heat_color_is_a_css_colour():
    assert heat_color(0.0).startswith("rgb(")
    assert heat_color(1.0).startswith("rgb(")


def test_heat_color_is_monotonic_in_score():
    assert heat_color(0.0) != heat_color(1.0)


def test_report_names_every_attribute(tmp_path):
    html = render_report(_hasm(), DemoPaths(tmp_path), title="FLAIR")
    assert "color" in html
    assert "size" in html


def test_report_marks_the_peak_unit_per_attribute(tmp_path):
    html = render_report(_hasm(), DemoPaths(tmp_path), title="FLAIR")
    # COLOR peaks at block 1, head 0
    assert "block 1, head 0" in html


def test_report_links_every_head_image(tmp_path):
    html = render_report(_hasm(), DemoPaths(tmp_path), title="FLAIR")
    for block in (0, 1):
        for head in (0, 1):
            assert f"heads/color_b{block}_h{head}.png" in html


def test_report_is_self_contained_html(tmp_path):
    html = render_report(_hasm(), DemoPaths(tmp_path), title="FLAIR")
    assert html.lstrip().startswith("<!doctype html>")
    assert "<style>" in html
    assert "http://" not in html and "https://" not in html


def test_write_report_creates_index_html(tmp_path):
    path = write_report(_hasm(), DemoPaths(tmp_path), title="FLAIR")
    assert path == tmp_path / "index.html"
    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./run-local.sh test`
Expected: FAIL — `ModuleNotFoundError: No module named 'flair_t2i.demo.report'`

- [ ] **Step 3: Write the report generator**

Create `flair_t2i/demo/report.py`:

```python
"""Render the demo sweep as one self-contained HTML page.

The heatmap carries the argument -- which block and which head each
attribute responds to -- and every cell links to the image that produced
its score. No external assets, so the output directory can be zipped and
opened anywhere.
"""

from __future__ import annotations

from pathlib import Path

from ..hasm import HASM
from .sweep import DemoPaths

_CSS = """
body { font: 15px/1.6 system-ui, sans-serif; margin: 0; padding: 2rem;
       background: #14151e; color: #eaeaf2; }
h1 { font-size: 1.6rem; margin: 0 0 .3rem; }
h2 { font-size: 1.15rem; margin: 2.4rem 0 .2rem; text-transform: capitalize; }
.sub { color: #9a9db3; margin: 0 0 2rem; }
.peak { color: #9089d8; font-weight: 600; }
table { border-collapse: collapse; margin-top: .6rem; }
caption { text-align: left; color: #9a9db3; font-size: .85rem;
          padding-bottom: .4rem; }
th { font: 500 11px monospace; color: #787c93; padding: 2px 4px; }
td { padding: 0; }
a.cell { display: block; width: 22px; height: 22px; border: 1px solid #14151e; }
a.cell:hover { outline: 2px solid #eaeaf2; outline-offset: -2px; }
.scroll { overflow-x: auto; }
"""


def heat_color(score: float) -> str:
    """Dark indigo through to hot amber, for a score in [0, 1]."""
    score = max(0.0, min(1.0, float(score)))
    red = int(38 + score * (217 - 38))
    green = int(36 + score * (167 - 36))
    blue = int(61 + score * (91 - 61))
    return f"rgb({red}, {green}, {blue})"


def render_report(hasm: HASM, paths: DemoPaths, title: str) -> str:
    parts = [
        "<!doctype html>",
        f"<title>{title}</title>",
        f"<style>{_CSS}</style>",
        f"<h1>{title}</h1>",
        '<p class="sub">Each cell is one attention head. Colour is that '
        "head's measured sensitivity to the attribute; click a cell to see "
        "the image produced by swapping the attribute at that head alone.</p>",
    ]

    for attr in hasm.attributes:
        peak_unit, peak_score = hasm.top_k(attr, 1)[0]
        parts.append(f"<h2>{attr.value}</h2>")
        parts.append(
            f'<p class="sub">peak: <span class="peak">block {peak_unit.block}, '
            f"head {peak_unit.head}</span> at {peak_score:.3f}</p>"
        )
        parts.append('<div class="scroll"><table>')
        parts.append(
            "<caption>rows: blocks &nbsp;&middot;&nbsp; columns: heads</caption>"
        )

        parts.append(
            "<tr><th></th>"
            + "".join(f"<th>{head}</th>" for head in hasm.head_ids)
            + "</tr>"
        )
        for block in hasm.block_ids:
            cells = []
            for head in hasm.head_ids:
                from ..heads import HeadUnit

                unit = HeadUnit(block=block, head=head)
                score = hasm.score(unit, attr)
                href = paths.head_image(attr, unit).relative_to(paths.root).as_posix()
                cells.append(
                    f'<td><a class="cell" href="{href}" '
                    f'style="background:{heat_color(score)}" '
                    f'title="block {block}, head {head} -- {score:.3f}"></a></td>'
                )
            parts.append(f"<tr><th>B{block}</th>" + "".join(cells) + "</tr>")
        parts.append("</table></div>")

    return "\n".join(parts)


def write_report(hasm: HASM, paths: DemoPaths, title: str) -> Path:
    path = paths.root / "index.html"
    path.write_text(render_report(hasm, paths, title), encoding="utf-8")
    return path
```

- [ ] **Step 4: Write the driver script**

Create `scripts/demo.py`:

```python
"""Produce the supervisor demonstration bundle.

    python scripts/demo.py --out flair_head_demo --steps 12

Sweeps all 7 attributes over every (block, head) unit at one contrastive
pair each -- 7 x (1 + blocks x heads) generations, roughly one fifth the
full campaign. Writes every image plus an index.html that needs no Python
to read. Zip the output directory and hand it over.
"""

import argparse
from pathlib import Path

import spacy
import torch
from diffusers import StableDiffusion3Pipeline

from flair_t2i.attributes import CORE_ATTRIBUTES
from flair_t2i.calibration.corpus import DEFAULT_CORPUS_PATH, load_corpus
from flair_t2i.calibration.harness import make_swap_generate_fn
from flair_t2i.config import FlairConfig
from flair_t2i.demo.report import write_report
from flair_t2i.demo.sweep import DemoPaths, run_demo_sweep
from flair_t2i.hasm import HASM
from flair_t2i.metrics.embedding import ClipScorer
from flair_t2i.metrics.masking import ClipSegMasker
from flair_t2i.pipeline import FlairPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("flair_head_demo"))
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pairs", type=int, default=1, help="contrastive pairs per attribute"
    )
    args = parser.parse_args()

    cfg = FlairConfig(device="cuda")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        cfg.model_id, torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()

    n_blocks = len(pipe.transformer.transformer_blocks)
    n_heads = pipe.transformer.transformer_blocks[0].attn.heads
    block_ids, head_ids = tuple(range(n_blocks)), tuple(range(n_heads))

    fp = FlairPipeline(
        pipe,
        cfg,
        HASM.uniform((0,), (0,), CORE_ATTRIBUTES),
        nlp=spacy.load("en_core_web_sm"),
    )

    corpus = {
        attr: pairs[: args.pairs]
        for attr, pairs in load_corpus(DEFAULT_CORPUS_PATH).items()
    }
    total = len(corpus) * args.pairs * (1 + n_blocks * n_heads)
    print(f"{n_blocks} blocks x {n_heads} heads = {n_blocks * n_heads} units")
    print(f"{len(corpus)} attributes x {args.pairs} pair(s) -- {total} generations\n")

    hasm = run_demo_sweep(
        make_swap_generate_fn(fp, steps=args.steps),
        corpus=corpus,
        block_ids=block_ids,
        head_ids=head_ids,
        masker=ClipSegMasker(device=cfg.device),
        paths=DemoPaths(args.out),
        seeds=[args.seed],
        scorer=ClipScorer(device=cfg.device),
        progress=lambda attr, unit, value: print(
            f"  {attr.value:<9} B{unit.block:<3}H{unit.head:<3} raw={value:.4f}"
        ),
    )

    report = write_report(hasm, DemoPaths(args.out), title="FLAIR — head-level routing")
    print(f"\nwrote {report}")
    for attr in hasm.attributes:
        unit, score = hasm.top_k(attr, 1)[0]
        print(f"  {attr.value:<9} peaks at B{unit.block} H{unit.head}  ({score:.3f})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run it to verify it passes**

Run: `./run-local.sh test`
Expected: PASS — 7 new tests, full suite green

- [ ] **Step 6: Commit**

```bash
git add flair_t2i/demo/report.py scripts/demo.py tests/test_demo_report.py
git commit -m "feat: self-contained HTML demo report

One index.html with a per-attribute block-by-head heatmap; every cell
links to the image that produced its score. No external assets, so the
bundle opens anywhere."
```

---

## Where each piece runs

**Every one of the 13 tasks is written, implemented, and tested on the laptop.** There is no GPU here and `diffusers` is deliberately absent from the Docker image, so nothing in the task list may import it at test time. Only the *sweeps* need Kaggle.

| | Laptop (Docker, CPU) | Kaggle (GPU) |
|---|---|---|
| Tasks 1-9, 11-13 — all code and all tests | ✅ `./run-local.sh test` | — |
| `scripts/explain.py` — routing decisions, no image | ✅ | ✅ |
| `scripts/verify_api.py` — 19 introspection checks | ❌ needs `diffusers` installed | ✅ (no GPU used, just the package) |
| `scripts/smoke_test.py` — images | ❌ | ✅ |
| `scripts/demo.py` — the 4,039-generation demo sweep | ❌ | ✅ |
| `scripts/calibrate.py` — the full campaign | ❌ | ✅ |

This is the same discipline the existing suite already follows: every routing *decision* is CPU-only and unit-tested with stub tensors; the GPU only turns those decisions into pixels. Do not add `diffusers` to the Dockerfile to work around a failing test — if a test needs it, the test is wrong.

## Execution order

Tasks are numbered in dependency order, but they do **not** all run before the supervisor sees results:

```
Tasks 1-9      build the machinery                    CPU only
Tasks 11-13    demo harness, latents, report          CPU only
     ↓
Demo sweep     scripts/demo.py                        ~4,039 generations
     ↓
👤 SUPERVISOR REVIEW — index.html
     ↓
Task 10        full campaign script + docs            ~20,195 generations
```

Task 10 is written last deliberately. There is no reason to build the full-campaign driver, or to spend five-pair compute, before the supervisor has agreed the direction is right. The demo de-risks roughly 15 A100-hours behind a 3-hour check.

## Two risks the CPU suite structurally cannot catch

Both come from `install_head_routing` doing `setattr(attn, name, wrapper)` where the wrapper is itself an `nn.Module`. On CPU stubs this is invisible; on a real offloaded SD3.5-M it may not be. Check both during the Kaggle verification below, before the demo sweep.

**1. The module tree changes shape while patched.** Assigning an `nn.Module` to an `nn.Module` attribute *registers* it, so the original `Linear` becomes a child of the wrapper and `state_dict()` keys shift from `add_q_proj.weight` to `add_q_proj.inner.weight`. Harmless as long as nothing saves or reloads weights while FLAIR is installed — and `pipeline.generate` uninstalls in a `finally`, so the window is narrow. Do not save a checkpoint from inside a routed generation.

**2. `enable_model_cpu_offload()` may not see the wrapper.** accelerate installs offload hooks by walking the module tree at offload time. FLAIR wraps *after* that, so the hook stays on `inner` while calls now arrive at the wrapper. The expected behaviour is fine — `wrapper.forward` → `inner.forward` → hook fires as before — but this is an interaction between two libraries' patching, not something the stub tests exercise. Confirm empirically with step 3 below: if routed generation produces noise, produces a device-mismatch error, or is dramatically slower than baseline, suspect this first.

If either bites, the fallback is to wrap the *function* rather than the module — keep `attn.add_q_proj` as the original `Linear` and instead patch `attn.add_q_proj.forward` with a closure, which leaves the module tree untouched. That change is confined to `patching.py` and needs no change to `head_proj.py`'s residual maths.

## Post-plan verification

Before the demo sweep, on Kaggle:

1. `python scripts/verify_api.py` — all 19 checks pass
2. `python -m pytest -q` — full suite green
3. `python scripts/smoke_test.py --steps 20 --out outputs/` — images generate, `units touched` lists head units, **and the routed image is neither noise nor visibly slower than the baseline** (see the two risks above)
4. Record `N_BLOCKS`, `N_HEADS`, and `T_GEN` in `calibration_runs/measurements.txt`

Then `python scripts/demo.py --out flair_head_demo --steps 12`, zip the output, and hand it over. Recompute the full-campaign budget as `A × P × S × (1 + N_BLOCKS × N_HEADS)` only after that review.
