# FLAIR — AdaIN Latent Appearance/Structure Split

**Status:** Design, awaiting approval
**Date:** 2026-08-31
**Sibling:** [`2026-08-31-lcs-colour-basis-design.md`](./2026-08-31-lcs-colour-basis-design.md) — the perceptually-grounded metric this one serves as a baseline for.
**Context:** [`2026-08-31-colour-localization-research.md`](./2026-08-31-colour-localization-research.md) — this spec stands unchanged as the baseline, but see §2 there for the corrected cost model (the VAE is not the bottleneck; do not argue this metric on speed).
**Precondition:** Tasks 1-16b complete, 175 tests passing. `scripts/test_latent_color_blocks.py` exists and produces block rankings that are not trustworthy (§2).

---

## 1. What this builds, in one paragraph

A latent-space metric that splits the change between two SD3.5 latents into an
**appearance** term and a **structure** term, so a block-routing swap that
recolours an object can be told apart from one that reshapes or moves it. It
requires no calibration, no VAE in either direction, and no generations to fit —
it is pure arithmetic on the two latent tensors. It is the cheap, defensible
baseline against which the LCS basis metric (sibling spec) must prove its worth.

## 2. Why the current metrics cannot work

`compute_latent_metrics` in `scripts/test_latent_color_blocks.py:43` returns
`relative_l2`, `cosine_distance`, and `gram_matrix_shift`, and combines them into
`combined_color_score`. All three are **total-change** metrics computed over the
whole frame. None of them contains a notion of *which part* of `Δz` is colour.

The consequence is that a swap which turns the car blue and a swap which turns
the car into a truck both produce a large `relative_l2`, and the block
leaderboard ranks them identically. Reweighting the three terms cannot repair
this, because the information needed to separate the cases is not present in
any of them.

One of the three is closer than it looks. `gram_matrix_shift` computes the
channel second-moment matrix, and the Gram matrix *is* the style term in Gatys
et al. The defect is not that Gram is the wrong object; it is that the design
never computes the corresponding **content** term to divide it out, and never
gates the computation to the object region. This spec supplies both.

## 3. The mechanism

The split rests on the AdaIN assumption (Huang & Belongie, 2017): for a
convolutional feature map, per-channel first and second moments carry *style*,
and the spatially-normalised map carries *content*.

For a latent `z ∈ R^{16×H×W}` and a region mask `M`:

```
μ_c      = mean of z[c] over M
σ_c      = std  of z[c] over M
s        = (μ, σ) ∈ R³²                      # the appearance vector
ẑ[c,p]   = (z[c,p] − μ_c) / (σ_c + ε)        # the content map, p ∈ M
```

The two deltas, given a base latent and a swapped latent:

```
appearance_delta = ‖s_swap − s_base‖ / ‖s_base‖
structure_delta  = mean_{p∈M} ‖ẑ_swap[:,p] − ẑ_base[:,p]‖ / √16
purity           = appearance_delta / (appearance_delta + structure_delta)
```

`purity ∈ [0,1]`. A pure recolour drives it toward 1; a pure reshape drives it
toward 0.

### 3.1 The normalisers are a real decision, not a detail

`‖s_base‖` and `√16` make the two terms roughly commensurate, but "roughly" is
not good enough for a ratio that will rank 24 blocks against each other. The
correct normalisers must be **settled empirically against the validation
triplet in §6**, not assumed. If the triplet does not produce a clean ordering
under the normalisers above, the normalisers are wrong before the metric is.

## 4. Why the spatial roll is the load-bearing test

A cyclic spatial roll of `z` permutes positions without altering the multiset of
values in any channel. Therefore `μ_c` and `σ_c` are **provably unchanged**,
while `ẑ` changes everywhere.

That gives an exact, analytic ground truth for "structure changed, appearance did
not" — constructible on random tensors, on CPU, in microseconds, with no model
loaded. Symmetrically, adding a per-channel constant is exact ground truth for
"appearance changed, structure did not."

These two facts turn what is usually a soft perceptual claim into a hard unit
test. That is the strongest reason to build this metric first even if the LCS
basis (sibling spec) ultimately supersedes it: the test harness it forces into
existence is reusable.

## 5. Files

| Path | Change |
|---|---|
| `flair_t2i/metrics/latent_color.py` | **new** — `channel_stats()`, `adain_split()`, `LatentSplit` dataclass |
| `tests/test_latent_color.py` | **new** — synthetic ground-truth tests (§6.1) |
| `scripts/test_latent_color_blocks.py` | rewire `compute_latent_metrics` to emit the split alongside existing columns |

The new module sits beside `flair_t2i/metrics/photometric.py` as its latent-space
sibling: `photometric.py` measures colour in CIELAB on decoded PIL images,
`latent_color.py` measures it on raw latents with no decode.

## 6. Build steps (TDD, per repo convention)

**Phase 1 — the metric core, no GPU.**

1. Write `tests/test_latent_color.py` first, with three synthetic cases:
   - `z_s = z_b + k` (constant per-channel offset) → assert `structure_delta ≈ 0`, `purity ≈ 1`
   - `z_s = torch.roll(z_b, shifts, dims=(-2,-1))` → assert `appearance_delta ≈ 0`, `purity ≈ 0`
   - `z_s = z_b` → assert both ≈ 0
2. Implement `channel_stats()` and `adain_split()` until green.
3. Add masked variants; test that a mask genuinely restricts the region read.

**Phase 2 — region gating.** Both this spec and the sibling need an object mask
on the 64×64 latent grid, and with no decoder available ClipSeg is unusable. The
mask must come from **cross-attention to the object token** (DAAM-style), pooled
over heads and steps. The hooking machinery already exists in
`flair_t2i/patching.py`.

Gating is deliberately deferred to phase 2. If the metric cannot separate colour
from structure on a clean pair *ungated*, masking will not rescue it, and
discovering that costs six generations rather than a mask implementation.

**Phase 3 — integration.** Wire into the sweep script, emit the new columns
beside the existing three, and run the §6.2 validation.

### 6.1 Synthetic acceptance (phase 1)

All three unit tests green. These are analytic, not statistical — they should
pass to tight tolerance.

### 6.2 Real-pair acceptance (phase 3)

Same seed, no routing installed, three prompt pairs:

| pair | expected |
|---|---|
| `"a red car on a road"` → itself | appearance ≈ 0, structure ≈ 0 |
| `"a red car on a road"` → `"a blue car on a road"` | **purity high** |
| `"a red car on a road"` → `"a red truck on a road"` | **purity low** |

Acceptance is the **ordering**, with a clear margin:
`purity(colour pair) > purity(structure pair)`.

Six generations. Zero VAE at scoring time.

## 7. Known limit — and the experiment that exposes it

This metric measures **appearance**, not colour. Brightness, contrast, and
texture-density changes all land in `(μ, σ)` and are indistinguishable from hue
changes.

The discriminating case:

```
"a red car"  →  "a dark red car"
```

Here hue is unchanged and only lightness moved. This metric reports a large
`appearance_delta` — the same signature it gives `"red car" → "blue car"`. It
cannot tell the two apart.

This is not a defect to be patched. It is the precise boundary of what
channel statistics can express, and it is the empirical justification for the
LCS basis in the sibling spec, which resolves the case by giving the axes
perceptual labels.

## 8. Cost

| | |
|---|---|
| Fitting | none — no calibration stage exists |
| Phase-1 tests | ~0 GPU, CPU-only, seconds |
| Validation | 6 generations |
| Per-block scoring | negligible — a mean and a std over 16 channels |

The 24-block sweep gets no slower.

## 9. References

- Huang & Belongie, *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*, ICCV 2017 — the μ/σ = style, normalised map = content decomposition.
- Gatys, Ecker & Bethge, *Image Style Transfer Using Convolutional Neural Networks*, CVPR 2016 — the Gram matrix as style representation; the reason `gram_matrix_shift` was half-right.
