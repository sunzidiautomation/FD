# FLAIR — Research Note: How the Two-Stage Screen-and-Confirm Pipeline Works

**Status:** Research findings, mechanism detail
**Date:** 2026-08-31
**Parent:** [`2026-08-31-colour-localization-research.md`](./2026-08-31-colour-localization-research.md) §7 proposed this pipeline; this note works out how it actually runs, and **corrects two cost claims made there** (§8 below).
**Siblings:**
- [`2026-08-31-lcs-colour-basis-design.md`](./2026-08-31-lcs-colour-basis-design.md)
- [`2026-08-31-adain-latent-split-design.md`](./2026-08-31-adain-latent-split-design.md)

---

## 1. Purpose

The parent note recommends screening all `(block, head)` pairs cheaply, then
confirming only the finalists causally. That is a shape, not a mechanism. This
note answers: **what exactly is measured during the screen, where is it captured
in SD3.5's joint attention, how is colour separated from structure at that point,
and what does the whole thing cost.**

## 2. The pipeline at a glance

| stage | what runs | what it produces | cost |
|---|---|---|---|
| **1 — screen** | contrastive prompt triplets, batched, all heads hooked | ranking of all 576 `(block, head)` pairs by colour selectivity | ~8 batched runs |
| **1b — screen validation** | exhaustive Stage 2 on 2 blocks | Spearman ρ between screen and ground truth | 48 generations, one time |
| **2 — confirm** | real FLAIR injection on top-K, decoded | CIELAB ΔE per finalist, the published ranking | ~30 generations |
| **3 — free bonus** | reuse Stage 2's outputs | paired `(latent, true Lab)` data → validated decoder-free probe | 0 |

## 3. Stage 1 — the per-head screen

### 3.1 Which signal: head output, not final latent

The current sweep diffs **final latents**, which means every measurement passes
through all downstream blocks before being read. That is both expensive (one
generation per unit) and blurry (the signal is mixed with everything that
happened after).

HeadRouter's insight is to read the head itself. For head `h`, the quantity of
interest is its output `A_h · V_h` on the image tokens — available in every
forward pass, for every head, simultaneously.

### 3.2 The product-rule split — colour term vs structure term

Parent note §4 established that **`A` carries structure and `V` carries
appearance**. That insight becomes operational here.

A head's output change between a base run and a swapped run is:

```
Δ(A·V)  =  A_base·ΔV  +  ΔA·V_base  +  ΔA·ΔV
```

To first order, the two leading terms are exactly the decomposition wanted:

| term | interpretation |
|---|---|
| `A_base · ΔV` | attention held fixed, **values changed** → **colour / appearance** |
| `ΔA · V_base` | values held fixed, **attention changed** → **structure / layout** |

This is a genuine disentanglement rather than a metric that hopes to approximate
one. It requires no colour basis in head space, and it falls out of the algebra
of attention itself.

**Note the space mismatch this avoids.** The LCS basis of spec ① lives in the
VAE's 16-channel latent space. Head values live in `head_dim` space (64, for
SD3.5-M at 1536/24). The two are *not* the same space and the basis cannot be
projected onto `V` directly. The product-rule split sidesteps the problem
entirely by never needing a colour basis at the head.

### 3.3 Computing both terms without materialising the attention matrix

The obvious objection: `A` is `seq × seq`, and with 4096 image tokens plus ~333
text tokens that is ~19.6M entries per head — roughly 39 MB in fp16, ~940 MB per
block. Materialising it is not viable, and `F.scaled_dot_product_attention` does
not return it.

**It does not need to be materialised.** SDPA is **linear in `V`**, so:

```
A_base · ΔV   =  SDPA( Q_base, K_base, ΔV )

ΔA · V_base   =  SDPA( Q_swap, K_swap, V_base ) − SDPA( Q_base, K_base, V_base )
```

Both terms are ordinary SDPA calls with substituted arguments. Memory stays at
normal attention cost, and the fused kernel is still used. This is the single
implementation detail that makes the whole screen practical.

### 3.4 Where to hook

`flair_t2i/patching.py` already establishes the pattern: wrap the projection
`nn.Linear` modules rather than replacing an attention processor, because
projection is linear and wrapping composes cleanly (`head_proj.py`).

The screen needs `Q`, `K`, `V` for both streams:

| stream | modules |
|---|---|
| image tokens | `to_q`, `to_k`, `to_v` |
| text tokens | `add_q_proj`, `add_k_proj`, `add_v_proj` — **already wrapped by `install_head_routing`** |

Iterate `ATTENTION_MODULES = ("attn", "attn2")` exactly as `install_head_routing`
does — SD3.5-M carries a second attention on part of its stack, and a screen that
ignores `attn2` would silently under-report those blocks. Reshape each projection
output `[B, seq, n_heads·head_dim] → [B, seq, n_heads, head_dim]` using the
`n_heads` / `head_dim` the existing wrapper already derives.

Two API facts to confirm against `scripts/verify_api.py` before building: the
concatenation **order** of text and image tokens in the joint sequence, and
whether the final block's missing `add_*_proj` (`context_pre_only=True`, noted at
`patching.py:46`) needs special handling in the screen. Both are cheap to assert
and expensive to get silently wrong.

### 3.5 The contrastive triplet — why two runs is not enough

The parent note said the screen costs "2 generations." **That was wrong**, and
here is why.

Two runs — base prompt and colour-swapped prompt — yield `‖ΔV‖` per head. But a
head with a large `‖ΔV‖` is merely *responsive to the prompt changing*, not
*selective for colour*. Every head that responds to text at all will score high.

Selectivity requires a **structure control**:

```
family = (
    "a red car on a road",      # base
    "a blue car on a road",     # colour swap    → ΔV_colour
    "a red truck on a road",    # structure swap → ΔV_struct
)
```

Three prompts, not two.

### 3.6 The score

Per `(block, head)`, using only the colour term from §3.2:

```
c_colour = ‖ SDPA(Q_base, K_base, ΔV_colour) ‖        # response to colour swap
c_struct = ‖ SDPA(Q_base, K_base, ΔV_struct) ‖        # response to structure swap

colour_selectivity = c_colour / (c_colour + c_struct + ε)
```

A head that moves for colour and not for shape scores near 1; a head that moves
for everything scores near 0.5; an inert head is excluded by a floor on
`c_colour` so that noise-dominated heads cannot reach the top of the ranking by
having a small denominator.

Average over `P` prompt families before ranking. A score from `"red car"` alone
would rank heads for *that scene*, not for colour.

### 3.7 Timestep aggregation

`ΔV` is a function of the denoising step. Colour commits early — low frequencies
first — so an unweighted mean over all steps dilutes the signal with late-step
refinement.

Capture at a small set of steps and report the score as a curve over `t`, then
aggregate over the early window. `flair_t2i/schedule.py` already carries the
timestep-weighting concept from the head-routing spec and should supply the
weights rather than a second mechanism being introduced.

The curve is worth keeping, not just the aggregate: "this head is colour-selective
only in the first third of denoising" is a finding, and it is invisible in a
scalar.

### 3.8 Memory: batch the triplet, reduce online

Storing `V` for later comparison does not fit. Per head per step, image tokens
alone are `4096 × 64` at fp16 ≈ 512 KB; across 24 blocks × 24 heads that is
~295 MB **per step**.

Two design choices remove the problem entirely:

1. **Batch the triplet.** Run all three prompts in one batch with identical
   initial noise. Both `V_base` and `V_swap` are then present in the same tensor
   inside the same hook call — no cross-run storage is needed at all.
2. **Reduce inside the hook.** Compute the §3.6 scalars immediately and
   accumulate into a `[blocks × heads]` array; discard the tensors.

Peak memory becomes one batch of normal attention, and the accumulator is
`24 × 24` floats. With CFG the batch is 6 rather than 3 — still one run.

**This is also why the triplet is not three times the cost:** batched, it is one
generation-equivalent, not three.

## 4. Stage 2 — causal confirm

### 4.1 What is different

Stage 1 measures *correlation*: how a head responds when the prompt changes.
Stage 2 measures *capacity*: what happens when FLAIR actually routes the swap
embedding into that head via `install_head_routing`. These are different claims,
and only the second supports the paper's thesis.

Run the existing injection sweep on the top K ≈ 30 heads, unchanged.

### 4.2 Masking becomes easy again

Both sibling specs plan a "phase 2" cross-attention mask on the 64×64 latent
grid, because with no decoder `ClipSegMasker` is unusable.

**Stage 2 decodes.** So `ClipSegMasker` and the whole existing pixel-space masking
path work exactly as they already do, and the DAAM-style latent mask is not needed
here at all. Only Stage 1 needs a latent-grid mask — and Stage 1 is already
hooking attention, so the cross-attention map is available at zero marginal cost.

This meaningfully reduces the work in both sibling specs.

### 4.3 The metric

`flair_t2i/metrics/photometric.py`, plus the one addition identified in parent
note §8.4: the `ΔE`-then-mean partner to `color_delta`, so that

```
purity = colour_delta / total_delta
```

can be formed. `target_colour_delta` is used unchanged — parent note §8.5 records
why plain ΔE-from-baseline is not safe on its own.

## 5. Stage 3 — the free validation dataset

Stage 2 decodes ~30 images, and for each it already holds the final latent. That
yields paired samples with no extra compute:

```
per run:  z ∈ R^{16×64×64}   and   true Lab ∈ R^{3×512×512}
```

Downsample Lab to the latent grid (or upsample `z`), and ~30 runs give ~123,000
paired positions — far more than needed to both fit and test a 16→3 linear probe.

Fit on half, evaluate on the held-out half, report per-channel R² and the
correlation between decoder-free ΔE and true ΔE.

**This partly supersedes spec ①'s Stage A.** Fitting on latents from real
generations is *in-distribution*, whereas synthetic flat swatches are not — the
out-of-distribution risk that spec ①'s risk table flags is avoided rather than
mitigated. The swatch synthesis becomes optional: useful for spanning colours the
generations happen not to cover, not required for the fit.

## 6. Validating the screen itself

A prefilter does not need to be causal, but it **does** need high recall, and that
must be demonstrated rather than assumed.

**Protocol.** Pick 2 blocks. Run Stage 2 exhaustively on all 24 heads of each (48
generations). Then:

- **Spearman ρ** between the Stage 1 score and the Stage 2 ground truth. Report
  it. ρ > 0.7 makes the screen defensible; below that, the screen is a filter of
  unknown quality and the top-K claim does not hold.
- **Recall@K** — what fraction of the true top-5 heads the screen placed in its
  top-K.
- **Random-K control** — the screened K must beat a randomly chosen K by a clear
  margin, or the screen contributed nothing.

This costs 48 generations once and is the difference between a defensible
pipeline and an unfalsifiable one. It belongs in the paper.

## 7. Honest cost accounting

| item | generations |
|---|---|
| Stage 1 screen — 8 families, batched | ~8 |
| Stage 1b screen validation — 2 blocks exhaustive | 48 *(one time)* |
| Stage 2 confirm — top K=30 | ~30 |
| **Total** | **~86** |
| Naïve full sweep `1 + 24×24` | **577** |
| **Reduction** | **~85%** |

At ~20 s/generation: roughly **29 minutes** versus **3.2 hours**, per attribute.

Note this is **85%, not the 94% claimed in the parent note** — that figure omitted
both the multi-family screen and the screen-validation cost. 85% is the number to
use.

## 8. Corrections to the parent research note

| parent claim | correction |
|---|---|
| §6 D2/D3: "2 generations" scores all heads | **3 prompts** are required (colour swap *and* structure control, §3.5). Batched, they remain **one generation-equivalent**, so the cost claim survives but the design does not. |
| §7: "~32 generations, ~94% reduction" | **~86 generations, ~85%** (§7 here). The omission was the screen-validation cost, which is not optional. |
| §8.3 / spec ①: fit the probe on synthetic swatches | Stage 2 supplies **in-distribution** paired data for free (§5). Swatches become a supplement for colour coverage, not the primary fit. |
| Both sibling specs: "phase 2 cross-attention masking" | Needed for **Stage 1 only**. Stage 2 decodes, so existing `ClipSegMasker` applies (§4.2). |

## 9. Risks and open questions

| Risk | Detail |
|---|---|
| **First-order split may be insufficient** | `Δ(A·V)` drops the `ΔA·ΔV` cross-term. If it is large the two-way split misattributes. **Measure it** — it costs one more SDPA call — and report its relative magnitude rather than assuming it is negligible. |
| **Screen measures response, not capacity** | Structural, not fixable within Stage 1. It is why §6 exists and why the published ranking rests on Stage 2. |
| **Colour may be distributed, not localized** | Carried over from parent note §10. The screen will show this honestly as a flat ranking; design the write-up so a flat result is still a finding. |
| **`attn2` asymmetry** | SD3.5-M's second attention exists only on part of the stack. Scores from `attn`-only and `attn`+`attn2` blocks may not be directly comparable; decide the normalisation before running, not after seeing results. |
| **Text/image concat order** | Getting the joint-sequence split wrong silently scores the wrong tokens. Assert it in `scripts/verify_api.py` (§3.4). |
| **Early-step capture window** | The claim that colour commits early is plausible and standard, but unverified *for SD3.5 specifically*. The §3.7 curve tests it as a by-product. |
