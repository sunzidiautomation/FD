# FLAIR — Head-Level Routing Design

**Status:** Approved design, ready for implementation planning
**Date:** 2026-08-26
**Supersedes:** §2 and §3.1 of [`2026-08-25-flair-cvpr-publication-plan-design.md`](./2026-08-25-flair-cvpr-publication-plan-design.md) on routing granularity only. Every other section of that spec stands unchanged.
**Precondition:** Tasks 1-16b complete, 175 tests passing, no calibration campaign yet run.

---

## 1. What changes, in one paragraph

FLAIR currently routes each attribute's text stream into whole transformer
*blocks*, selected from a `[blocks × attributes]` sensitivity matrix (BASM).
This design moves the routing unit from the block to the individual attention
*head*: a `[blocks × heads × attributes]` tensor (HASM) calibrated by the same
causal contrastive-swap procedure, with block-level routing retained as a
derived special case rather than a separate mechanism. Compute is no longer a
binding constraint on this project, which removes the cost argument that
originally motivated both the block granularity and the vital-layer prefilter.

## 2. Why

**Mechanical reason.** A block contains many heads that attend to different
things. Block-level routing pushes an attribute's stream into all of them at
once, so a head that is highly selective for colour is driven with the same
strength as a head in the same block that is not. Head-level routing addresses
the attribute at the granularity at which specialization actually occurs.

**Why now.** The original spec (§2) excluded head-level routing deliberately,
to preserve a clean differentiation from HeadRouter. That exclusion was sound
under the original constraint set. Two things have changed: the GPU budget is
no longer free-tier-limited, and §5 below establishes that head-level
calibration yields the block-level matrix at zero additional cost — which
converts the differentiation problem into a contribution.

### 2.1 Novelty positioning after this change

The claim no longer rests on granularity. It rests on two axes that remain
intact, plus one that this change creates:

| | HeadRouter (ACM TOG 2026) | FLAIR after this change |
|---|---|---|
| When routing is computed | per-instance, at inference | **offline, once, reused across all prompts** |
| Requires a source image | yes (editing) | **no (generation from scratch)** |
| Granularity | heads only | **block and head, compared directly** |

The paper's contribution claim gains a clause:

> Because the same offline calibration produces sensitivity scores at both
> block and head granularity, the choice of routing granularity becomes an
> empirically settled variable rather than an assumption. A per-instance
> adaptive router cannot make this comparison, because its scores are not
> reusable across prompts and therefore cannot be aggregated.

**Reviewer risk this accepts:** "this is HeadRouter, offline." The mitigation is
the granularity ablation (§7.3) and the offline/generation axis, both of which
must be argued explicitly in §1 of the paper, not left implicit.

## 3. The mechanism

### 3.1 Why no attention code is reimplemented

The obvious implementation — reimplement the joint attention forward pass so
the code can reach between head-splitting and head-concatenation — is not
necessary and is rejected. Two facts make a much smaller change sufficient.

**Projection is linear.** For `nn.Linear`, `proj(x) = xAᵀ + b`. Therefore:

```
proj(x + δ)  =  proj(x) + δAᵀ
```

The bias cancels in the difference. A residual added to the projection *output*
is identical to the same residual added to the projection *input* — provided it
is projected weight-only, with no bias term.

**Head slices are contiguous.** diffusers reshapes projection output as
`[B, seq, inner_dim] → view(B, seq, H, head_dim) → transpose(1, 2)`. In
row-major order, head `h` therefore owns exactly the contiguous range
`dims[h·head_dim : (h+1)·head_dim]` of `inner_dim`. Masking those dimension
ranges *is* masking per head.

Together these mean head-level injection is achieved by wrapping three
`nn.Linear` modules per block — `add_q_proj`, `add_k_proj`, `add_v_proj`, the
text stream's projections — and leaving the attention processor delegating
exactly as it does today. `FlairJointProcessor`'s guarantee that *"no attention
maths is reimplemented here"* survives this change intact.

### 3.2 The wrapper

```python
class HeadResidualProj(nn.Module):
    """Wraps one text-stream projection; adds a head-masked residual."""

    def forward(self, x):                       # x: [B, seq_txt, dim]
        out = self.inner(x)                     # untouched projection
        residual = self.ref.head_residual(
            self.block_id, x, weight=self.inner.weight
        )                                       # [B_cond, seq_txt, inner_dim]
        if residual is not None:
            out[cond] = out[cond] + residual
        return out
```

and the residual itself, for one component `i` routed to a set of heads:

```
δ_i      = H_i − x[cond]                        # text-encoder space, unscaled
r_i      = linear(δ_i, weight)                  # weight only, NO bias
w_i      = per-head α vector, length inner_dim  # α_i(ℓ,h,t) in head h's slice,
                                                # 0.0 in every unselected slice
residual = Σ_i  r_i * w_i
```

Scaling commutes with the projection, so applying the per-head α *after*
projection is exact, not an approximation. Multiple components routed into the
same block sum their residuals, each against the **original** `x[cond]` — the
same semantics as today's `routing.py:83`, where `base` is captured once before
the contribution loop.

### 3.3 The invariant

Selecting every head in block ℓ, all at the same α, yields:

```
proj(x)[cond] + α·(H_i − x[cond])·Aᵀ  ≡  proj(x[cond] + α·(H_i − x[cond]))
```

which is exactly the current block-level result. This is not merely a test that
should pass — it is an algebraic identity, and it holds through the
`norm_added_q` / `norm_added_k` RMSNorm as well, because both paths produce a
bitwise-identical tensor *before* the norm is applied.

**Two conditions are load-bearing and must be asserted in code:**

1. The residual is projected **weight-only** (no bias), or the identity breaks.
2. The residual is applied **immediately after projection, before the QK norm**.
   Applying it after the norm breaks the identity, because RMSNorm is not
   linear.

Head-level routing is therefore a strict generalization of block-level routing,
not a replacement for it.

## 4. Data model

```python
@dataclass(frozen=True)
class HeadUnit:
    block: int
    head: int


class HASM:
    """[blocks × heads × attributes], values in [0, 1]."""
    tensor: np.ndarray
    block_ids: tuple[int, ...]
    head_ids: tuple[int, ...]
    attributes: tuple[AttributeClass, ...]

    def score(self, unit: HeadUnit, attr) -> float: ...
    def top_k(self, attr, k) -> list[tuple[HeadUnit, float]]: ...
    def to_basm(self, reduce: str = "max") -> BASM: ...
    def save(path) / load(path)                      # npz, as BASM
    @classmethod
    def uniform(cls, block_ids, head_ids, attributes) -> "HASM": ...
```

`to_basm` reduces over the head axis (`max` by default, `mean` available) and
returns an ordinary `BASM`. **`basm.py` is not modified.** Its class, its tests,
and the block-level routing path all continue to work unchanged, now fed by a
matrix derived from real head-level measurements instead of its own campaign.

**Routing plan.** `RoutedComponent.blocks: tuple[tuple[int, float], ...]`
becomes `RoutedComponent.units: tuple[tuple[HeadUnit, float], ...]`.
Block-level routing is expressed as *"every head in block ℓ, each carrying
S[ℓ,a]"* — so there is one blending mechanism and two selection functions:

| Granularity | Selector | Produces |
|---|---|---|
| head | `hasm.top_k(attr, k)` | k individual `HeadUnit`s |
| block | `hasm.to_basm().top_k(attr, k)`, expanded over heads | k × H `HeadUnit`s |

Ties in `HASM.top_k` break by ascending `(block, head)`, matching `BASM`'s
existing ascending-block-id rule.

`RoutingPlan.blend()` is replaced on the generation path by
`RoutingPlan.head_residual(block_id, x, weight)`, which returns the summed
masked residual for that block or `None` when the block is untouched. The fast
path — most blocks are not routed on most steps — is preserved.

**`blend()` is not deleted; it is frozen.** It moves verbatim, together with the
`RoutedComponent.blocks` shape it reads, into `tests/reference_blend.py`, where
it becomes the equivalence oracle for §7.1. Freezing the reference in the test
tree rather than leaving it on the production path is deliberate: an oracle that
lives beside the implementation it validates will eventually be refactored
alongside it and stop being independent evidence. Its existing tests move with
it and continue to pin the reference behaviour.

## 5. Calibration

The procedure is unchanged in kind. Only the swept unit changes.

- `SwapSpec(block_id, text)` → `SwapSpec(unit: HeadUnit, text)`.
- The exact-replacement trick still holds: `alpha_0 = 1.0` with
  `t_window = (0.0, 1.0)` gives `H = H_base + 1.0·(H_changed − H_base) =
  H_changed`, now confined to one head's slice. Calibration continues to reuse
  the generation-path routing machinery rather than a second injection path
  that could drift from it.
- Checkpoint cells move from `cells/{attr}_{block}.json` to
  `cells/{attr}_{block}_{head}.json`. Resume-on-restart behaviour is otherwise
  unchanged and remains mandatory at this volume.
- Min-max normalisation moves from per-attribute-over-blocks to
  per-attribute-over-all-units. A flat column still normalises to all zeros
  rather than a spurious peak.

### 5.1 No prefilter

The vital-layer prefilter exists because *"the campaign cost is linear in the
number of blocks kept"* (`prefilter.py:6`). With the cost constraint removed,
its only remaining effect is selection bias on the artefact being measured, and
it would additionally require a head-bypass mechanism that does not exist.

**Every `[block × head × attribute]` cell is calibrated.** Consequences:

- `bypass_blocks` (`patching.py:32`) is untouched and retained — it still
  serves the block prefilter for the FLUX port (§3.7 of the original spec).
- **Head bypass is not built.** It was only ever needed to serve a head
  prefilter. This removes a component from scope rather than adding one.
- The resulting tensor is dense, which makes the BASM/HASM heatmap a
  substantially stronger figure and forecloses a "why these blocks?" question.

### 5.2 Budget

```
units  = n_blocks × n_heads                    (read from config, never assumed)
total  = A × P × S × (1 + units)
```

At A=7, P=5, S=1 and a measured 24 blocks × 24 heads:

| | Count |
|---|---|
| Baselines (`A × P × S`) | 35 |
| Swaps (`A × P × S × units`) | 20,160 |
| **Total generations** | **20,195** |
| At T_GEN = 12.4 s | ≈ 70 GPU-hours (T4) |
| On a rented A100 | ≈ 20 GPU-hours |

`n_blocks` and `n_heads` are read from the loaded model, never hard-coded; the
table above is illustrative of the expected configuration, and the campaign
budget is recomputed from the measured values.

## 6. GPU verification prerequisites

`scripts/verify_api.py` gains the checks below. **None of the implementation is
safe to build on until these pass on a real SD3.5-M**, because each one, if
false, silently produces a wrong result rather than an error.

| # | Assumption | Failure mode if false |
|---|---|---|
| 1 | `num_attention_heads` and `attention_head_dim` are readable from config, and `heads × head_dim == inner_dim` | head slicing is misaligned |
| 2 | Reshape convention is `view(B, seq, H, head_dim).transpose(1, 2)`, so head `h` owns contiguous dims `[h·d, (h+1)·d)` | **the entire masking premise is invalid** |
| 3 | `add_q_proj`, `add_k_proj`, `add_v_proj` exist on `block.attn` and are `nn.Linear` | wrapper cannot install |
| 4 | Which blocks carry dual attention (`attn2`), and whether its text projections need the same wrapping | a second attention path routes unrouted |
| 5 | The final block's `context_pre_only=True` and which `add_*_proj` modules it therefore lacks | crash or silent skip on the last block |

Checks 2 and 4 are the ones most likely to be false. Check 4 in particular is a
pre-existing gap: `install_flair` (`patching.py:18`) wraps `block.attn` only,
and SD3.5-M's dual-attention blocks may carry a second attention module that
block-level routing has been quietly leaving untouched.

## 7. Testing

Everything below runs on CPU with stub `nn.Linear` modules. No GPU, no SD3.5
download — matching the existing suite's discipline.

### 7.1 The equivalence test

The headline test, and the reason §3.3 matters:

> Build a routing plan selecting **every** head of block ℓ at a uniform score,
> run it through `HeadResidualProj`, and assert the result equals
> `tests/reference_blend.py`'s frozen block-level output to floating-point
> tolerance.

If this test passes, head-level routing is provably a superset of the shipped,
already-validated block-level behaviour.

### 7.2 Unit coverage

- `HeadUnit` ordering and `HASM.top_k` tie-breaking by `(block, head)`.
- `HASM.to_basm` under both `max` and `mean` reduction, including a
  degenerate all-equal tensor.
- Head masking: a residual routed to head `h` leaves every other head's
  dimension slice bitwise unchanged.
- Per-head α: two heads in one block with different HASM scores receive
  correspondingly different residual magnitudes.
- Weight-only projection: a wrapped `Linear` **with a non-zero bias** still
  satisfies the §7.1 equivalence test. This is the test that catches the
  single most likely implementation error.
- Multi-component summation into one block, each against the original base.
- Unconditional (negative-prompt) rows are never written.
- Uncalibrated attributes are dropped, as today.

### 7.3 Ablation enabled by this design

Block-routing vs. head-routing vs. random-head control, all three running the
same mechanism with a different selector — the comparison that §2.1 promotes
from a defensive answer into a contribution.

## 8. Impact on existing work

| Artefact | Change |
|---|---|
| `flair_t2i/basm.py` | none |
| `flair_t2i/routing.py` | `blocks` → `units`; `blend` → `head_residual` |
| `tests/reference_blend.py` | new — frozen copy of `blend()` as the §7.1 oracle |
| `flair_t2i/processor.py` | unchanged contract; still delegates |
| `flair_t2i/patching.py` | installs projection wrappers alongside processors |
| `flair_t2i/hasm.py` | new |
| `flair_t2i/calibration/harness.py` | `SwapSpec` takes a `HeadUnit`; cell keys gain a head field |
| `flair_t2i/calibration/prefilter.py` | unused by the campaign; retained for FLUX |
| `scripts/verify_api.py` | +5 checks (§6) |
| `docs/CODE_WALKTHROUGH.md`, `docs/EXECUTION_TREE.md`, `docs/RUNBOOK.md` | describe block-level throughout; must be revised after implementation |
| Master roadmap Weeks 3-4 | campaign scope and budget replaced by §5 |

## 9. Risks

| Risk | Trigger | Response |
|---|---|---|
| Reshape convention differs from §6 check 2 | `verify_api.py` | **Stop.** The masking premise fails and the design needs rework before any code is written. |
| Dual attention (`attn2`) carries routable text projections | `verify_api.py` check 4 | Wrap them and treat their heads as additional units; this also fixes a pre-existing block-level gap. |
| 576 units per attribute produce a noisy, low-contrast tensor | Week 4 gate | More seeds (`S`) directly reduces per-cell variance and is now affordable. Report head-level and block-level (`to_basm`) selectivity side by side. |
| "This is HeadRouter, offline" | Review | §2.1's granularity ablation plus the offline/generation axis, argued explicitly in paper §1. |
| Schedule: this is a fork mid-project, before any calibration data exists | Now | Implementation is bounded by §8's file list; the §7.1 invariant means the existing block-level behaviour cannot silently regress. |

## 10. Out of scope

- **Head bypass and any head prefilter** — see §5.1.
- **Changes to the fuzzy module.** `intensity` and `k` still multiply into
  `α`; `k` now selects k heads rather than k blocks. `hedges.py` is untouched.
- **Changes to the coherence guard.** It operates on component embeddings in
  text-encoder space, upstream of any routing granularity.
- **FLUX port (§3.7) and fuzzy conflict resolution (§3.8).** Both remain gated
  behind the Week 7 checkpoint, unchanged.

---

*Companion to `2026-08-25-flair-cvpr-publication-plan-design.md` and
`2026-08-25-flair-architecture-overview.md`. Implementation plan to follow via
the writing-plans workflow.*
