# FLAIR — Research Note: Localizing Colour to Blocks and Heads in MM-DiT

**Status:** Research findings, informs the two sibling specs
**Date:** 2026-08-31
**Siblings:**
- [`2026-08-31-lcs-colour-basis-design.md`](./2026-08-31-lcs-colour-basis-design.md)
- [`2026-08-31-adain-latent-split-design.md`](./2026-08-31-adain-latent-split-design.md)

**Extended and partly corrected by:**
[`2026-08-31-two-stage-screen-confirm-research.md`](./2026-08-31-two-stage-screen-confirm-research.md)
— works out the mechanism of §7 and revises two cost claims made here. Where the
two notes disagree, that one is current.

**Questions researched:**
1. MM-DiT produces a latent; that latent goes to the VAE and becomes an image. If
   the VAE is skipped and colour difference is measured on the raw latent, which
   block and which head is responsible for colour generation — and how is that
   measurement actually done?
2. Can CIELAB be used for this task, given that it failed when applied to raw
   latents? (§8)

---

## 1. Summary of findings

Four findings, in descending order of how much they change the plan.

1. **The VAE is not the bottleneck** (§2). Skipping the decoder saves ~2-5% of
   runtime. The cost is in the number of generations. The time argument for a
   latent-space metric does not survive arithmetic and must be dropped from the
   paper; the *scientific* arguments for it remain sound.
2. **Colour and structure are already disentangled inside every attention head**
   (§4). Attention maps carry structure, value tokens carry appearance. The
   separation this project has been trying to recover from the final latent
   exists cleanly at the source.
3. **A two-stage screen-and-confirm design cuts the sweep by ~85%** (§7) without
   weakening the causal claim.
4. **CIELAB is usable after all** (§8), because finding 1 removed the reason it
   was abandoned — and the difference between two ways of averaging ΔE *is* the
   colour/structure split, needing no new metric at all.

## 2. The cost model, corrected

| | estimate |
|---|---|
| SD3.5-medium, 12 steps, free-tier GPU | ~12-25 s / generation |
| VAE decode at 1024² | ~0.3-0.8 s |
| **Decoder share of runtime** | **~2-5%** |

A full `[blocks × heads]` sweep at 24×24:

```
1 base + (24 × 24) = 577 generations
577 × ~20 s ≈ 3.2 hours       per prompt pair, per attribute
```

Dropping the VAE saves roughly four minutes out of 3.2 hours.

**Consequence for the paper.** Do not claim latent-space measurement as a speed
optimization; a reviewer will do this arithmetic. Claim it on the grounds that
actually hold: no decoder artefacts, no dependence on a pretrained perceptual
model's biases, differentiability, and measurement in the space the intervention
actually occurs in.

## 3. How prior work measures "which layer or head matters"

| Work | Intervention | Signal measured | Space | Cost |
|---|---|---|---|---|
| **Stable Flow** | bypass layer via its residual connection | DINOv2 perceptual distance | pixel — decodes | 1 gen / layer |
| **HeadRouter** | reconstruction branch vs editing branch | cosine similarity of **per-head output features** | feature — no decode | 2 gens, **all heads at once** |
| **LocoGen** | cross-attention intervention | direct effect on generation | pixel | 1 gen / layer set |
| **ColorCtrl** | manipulate attention maps and value tokens separately | — (control method, not measurement) | attention internals | — |

### 3.1 Stable Flow's vitality formula

```
vitality(ℓ) = 1 − (1/k) Σ_{s,p} d( M_full(s,p), M_−ℓ(s,p) )
```

with k = 64 prompts, S random seeds, 15 Euler steps, guidance scale 3.5, and
`d` = DINOv2 perceptual distance. Layers above threshold `τ_vit` are vital.

Reported results: **SD3 → 4 vital layers `[0, 7, 8, 9]`**; FLUX.1-dev → 10 vital
layers `[0, 1, 2, 17, 18, 25, 28, 53, 54, 56]`.

Note the causal direction differs from FLAIR's. Stable Flow **ablates** (bypass a
layer, see what breaks). FLAIR **injects** (route an embedding into a unit, see
what moves). These answer different questions — necessity versus capacity — and
should not be conflated when comparing results.

### 3.2 HeadRouter's head scoring

Sensitivity is the cosine similarity between the **output features of
corresponding attention heads** in a reconstruction branch versus an editing
branch, converted to a normalised dissimilarity `d̃_h` and applied as a weight:

```
w_h = 1 + γ · σ( k · (d̃_h − δ) )
```

Computed **per instance, at inference**. Critically, it never decodes and never
requires one generation per head — every head is scored from the same pair of
forward passes. This is the efficiency mechanism FLAIR should adopt (§6, D2).

## 4. The load-bearing finding: attention maps vs value tokens

ColorCtrl achieves training-free colour editing in MM-DiT by *"disentangling
structure and colour through targeted manipulation of attention maps and value
tokens."* The underlying principle, corroborated across sources:

> In attention `out = A · V` — the **attention map `A` carries structure** (which
> token attends where), and the **value tokens `V` carry appearance and colour**
> (what content is moved).

**Why this matters more than anything else in this note.** Both sibling specs
attempt to un-mix colour from structure *after* 24 blocks of processing have
already blended them into the final latent. That is a deconvolution problem. But
the two signals are already separate *inside each head*:

- a head's contribution to **colour** ≈ how much the swap moves its **V**
- a head's contribution to **structure** ≈ how much the swap moves its **A**

Measuring at the head is better-conditioned than measuring at the output, and it
is also where the intervention happens, so attribution is direct rather than
inferred through downstream propagation.

## 5. External ground truth available for free

Stable Flow reports SD3 vital layers `[0, 7, 8, 9]`. Seg4Diff independently finds
SD3 **layer 9** exhibits strong semantic grounding. Two papers, different
methods, converging on the same region of the network.

**Use this as a validation target.** If a decoder-free latent metric reproduces
`[0, 7, 8, 9]` without ever decoding, that is external corroboration that the
measurement apparatus works — and is itself a publishable claim. If it does not
reproduce them, the metric is broken, and that is discovered cheaply.

**Stated precisely:** these are layers vital to **image formation in general**,
not to **colour specifically**. This is a sanity check on the apparatus, not a
colour ground truth. Do not overclaim it.

## 6. Three measurement designs

**D1 — final-latent difference.** *(what `scripts/test_latent_color_blocks.py`
does today)*
Inject at `(block, head)`, denoise fully, diff the final latents.
*Measures:* true end-to-end causal effect.
*Costs:* 577 generations; signal confounded by all downstream blocks.

**D2 — per-head output difference.** *(HeadRouter-style)*
Two generations — base prompt and colour-swapped prompt, same seed — hooking
**every head simultaneously**; compare each head's output features across runs.
*Measures:* each head's responsiveness to the colour word.
*Costs:* **one batched run for all 576 `(block, head)` pairs.**
> **Corrected:** two prompts measure *responsiveness*, not *selectivity*. A
> structure control is required, making it a three-prompt triplet — batched into
> a single run. See the extension note §3.5.

**D3 — V-token colour projection.** *(D2 refined by §4)*
Same two runs, but split each head's change into its **V** component (colour) and
its **A** component (structure), then project the V-change onto a colour basis.
*Measures:* colour response, disentangled by construction rather than by
post-hoc metric design.
*Costs:* same as D2.

## 7. Recommended design: two-stage screen-and-confirm

Do not choose one. LocoGen explicitly reports that causal tracing *"fails in
pinpointing localized knowledge"* in recent models, and a correlational head
score is not evidence of causal capacity. So separate the roles:

| stage | design | purpose | cost |
|---|---|---|---|
| **1 — screen** | D3 | rank all 576 `(block, head)` pairs by V-space colour response | ~8 batched runs |
| **1b — validate the screen** | D1 | 2 blocks exhaustively, to measure the screen's recall | 48 gens, one time |
| **2 — confirm** | D1 | run the real injection sweep on the top K ≈ 20-30 only | ~30 generations |

**Total ≈ 86 generations instead of 577 — a ~85% reduction**, with the causal
claim intact, because the published ranking still rests on injection experiments.

> **Corrected from an earlier ~32 / ~94%**, which omitted the multi-family screen
> and the screen-validation cost. The latter is not optional: without it the
> prefilter's recall is unmeasured. See the extension note §6 and §7.
A prefilter does not need to be causal; it needs **high recall**, which should be
verified by checking that a random-K control performs materially worse than the
screened K.

### 7.1 Effect on the novelty claim

This strengthens the positioning already argued in
[`2026-08-26-flair-head-level-routing-design.md`](./2026-08-26-flair-head-level-routing-design.md) §2.1.
HeadRouter's scores are per-instance and computed at inference. FLAIR's would be
**screened cheaply, confirmed causally, and frozen offline for reuse across every
prompt** — the offline/reusable axis, now with a defensible cost story that no
longer depends on the incorrect VAE argument.

## 8. Using CIELAB after all

CIELAB was abandoned early in this project because it "does not work for latent
space." That diagnosis was correct but the conclusion drawn from it was too
strong.

### 8.1 Why it genuinely fails on raw latents

CIELAB is a **colorimetric** transform. `rgb2lab` assumes its input *is light* —
tristimulus values with a defined white point and gamma. A 16-channel SD3 latent
is a learned compression with no white point, no gamma, and no channel that means
"red." Passing latents to `rgb2lab` is a type error that happens to execute
without raising. This is not a tuning problem and no amount of rescaling fixes it.

### 8.2 The premise that changed

CIELAB was dropped because it requires a decode, and a decode was assumed to cost
time. §2 removes that premise: the decoder is 2-5% of runtime, and under the
two-stage design (§7) Stage 2 is ~30 generations — about **20 seconds of total
decode**.

Correct, tested, masked CIELAB already exists in this repo at
`flair_t2i/metrics/photometric.py:118`. The cheapest correct route is to use it.

### 8.3 Route B — fit latent → Lab directly

If a decoder-free ΔE is still wanted, fit it **supervised** rather than by PCA.
The swatches of spec ① are synthesised, so their Lab is known exactly:

```
encode swatches      → z ∈ R^{600×16}
known Lab of swatch  → Y ∈ R^{600×3}
least squares        → W = argmin ‖zW − Y‖²        W ∈ R^{16×3}

Lab_hat = z @ W            ΔE = ‖Lab_hat_a − Lab_hat_b‖
```

Strictly preferable to the unsupervised basis in spec ① for this purpose:

- the axes **are** L\*, a\*, b\* by construction, so the hue-circularity gate is
  not needed to interpret them;
- ΔE lands in the **same units** as `photometric.py`, making latent-space and
  pixel-space numbers directly comparable;
- per-channel **R²** is a built-in honesty metric — it states how much of Lab the
  latent linearly predicts, rather than assuming the answer.

### 8.4 Route C — the averaging order *is* the disentanglement

There are two ways to compute ΔE over a region, and the difference between them
separates colour from structure with no new metric invented:

| computation | property |
|---|---|
| `ΔE( mean_p Lab_a[p], mean_p Lab_b[p] )` — **mean, then ΔE** | spatial averaging destroys layout → structure-blind → **pure colour** |
| `mean_p ΔE( Lab_a[p], Lab_b[p] )` — **ΔE, then mean** | per-position → structure-sensitive → **total change** |

```
colour_delta = mean-then-ΔE
total_delta  = ΔE-then-mean
purity       = colour_delta / total_delta
```

A uniform recolour moves every position in the same Lab direction, so the mean
preserves it and `purity → 1`. A rearrangement moves positions in directions that
cancel under averaging, so `purity → 0`.

**Half of this is already implemented.** `photometric.py:118 color_delta`
computes mean-then-ΔE over `masked_mean_rgb`. The ΔE-then-mean partner was never
written, so the ratio was never formed. This is structurally the identical defect
to `gram_matrix_shift` computing a style term with no content term to divide out.

Lab also supplies the axis split the sibling specs had to construct by hand:

| quantity | meaning |
|---|---|
| **ΔL\*** | lightness |
| **ΔC\*ab** = √(Δa² + Δb²) | chroma |
| **ΔH\*** | hue |

So `"red car" → "dark red car"` reads as large ΔL\*, ≈0 ΔH\* — the discriminating
experiment of both sibling specs, expressed in standard colorimetric units rather
than invented ones.

### 8.5 An existing asset not to lose

`photometric.py:89` records a real failure from an earlier sweep: *"the top three
colour cells were a sepia-washed frame, a reoriented car and a recomposed scene —
every one of them still red."* Plain ΔE-from-baseline is maximised by anything
that shifts mean colour, including global tone washes.

`target_colour_delta` fixed this by measuring movement **toward the named swap
colour**, which is exact for colour because the swapped-in word names a fixed
point in Lab. Whichever space the final metric uses, this correction must be
carried forward or the same failure recurs.

### 8.6 Recommended use of CIELAB

1. **Stage 1 screen** — V-token ranking in latent space. No Lab required; only an
   ordering is needed.
2. **Stage 2 confirm** — decode the ~30 finalists and use the existing
   `photometric.py` CIELAB, extended with the §8.4 ratio.
3. **Free validation data** — Stage 2 decodes regardless, so those runs yield
   paired `(latent, true Lab)` samples at **zero additional cost**. Use them to
   validate the §8.3 probe and report the agreement: *"decoder-free ΔE tracks
   true ΔE at R² = x."* That validation is itself a contribution, and the data
   for it is a by-product of work already being done.

## 9. Consequences for the two sibling specs

| spec | status after this research |
|---|---|
| **① LCS colour basis** | Still needed, with two amendments. The higher-value input is **V-token changes** (D3), not final-latent changes. And the basis should be fit by **supervised regression onto Lab** (§8.3) rather than PCA — same data, interpretable axes for free, and the hue-circularity gate becomes unnecessary. |
| **③ AdaIN split** | Unchanged as the zero-calibration baseline. Note §8.4 supplies a *perceptually grounded* purity ratio of the same shape, which is the stronger version of what ③ approximates. |
| **`photometric.py`** | Promoted from "too slow to use" to the Stage 2 metric of record. Needs one addition: the ΔE-then-mean partner to `color_delta`, so the §8.4 purity ratio can be formed. `target_colour_delta` carries forward unchanged (§8.5). |
| **new** | A `[blocks × heads]` per-head **A/V probe** — the direct answer to "which block's which head generates colour." Highest-value piece, not yet specced. |

## 10. Risks this research surfaced

| Risk | Detail |
|---|---|
| **Colour may not be localized in SD3 at all** | Sources indicate SD3 shows no clean layer-based separation of style from other image elements; colour control appears distributed. The honest outcome may be "distributed across many heads with a weak peak," not "block 12 head 7 is the colour head." **Design the experiment so that result is still publishable.** |
| **Correlational screen ≠ causal capacity** | Mitigated by the two-stage design (§7) and the random-K control. |
| **Ablation vs injection are different questions** | Stable Flow's `[0,7,8,9]` measures necessity; FLAIR measures capacity. Comparable as a sanity check only (§5). |
| **Per-instance variance** | HeadRouter recomputes per instance for a reason. An offline frozen score must be shown stable across prompts and seeds, or the reuse claim fails. |

## 11. Bibliography

- Ku, et al. — *Stable Flow: Vital Layers for Training-Free Image Editing* — [arXiv:2411.14430](https://arxiv.org/abs/2411.14430)
- *HeadRouter: A Training-free Image Editing Framework for MM-DiTs by Adaptively Routing Attention Heads*, ACM TOG 2026 — [arXiv:2411.15034](https://arxiv.org/abs/2411.15034) · [code](https://github.com/ICTMCG/HeadRouter)
- *Training-Free Text-Guided Color Editing with Multi-Modal Diffusion Transformer* (ColorCtrl) — [arXiv:2508.09131](https://arxiv.org/pdf/2508.09131)
- *On Mechanistic Knowledge Localization in Text-to-Image Generative Models* (LocoGen / LocoEdit) — [arXiv:2405.01008](https://arxiv.org/pdf/2405.01008)
- *Seg4Diff: Unveiling Open-Vocabulary Segmentation in Text-to-Image Diffusion Transformers* — [arXiv:2509.18096](https://arxiv.org/pdf/2509.18096)
- Pach, Bader, Bouniot, Belongie & Akata — *The Latent Color Subspace: Emergent Order in High-Dimensional Chaos* — [arXiv:2603.12261](https://arxiv.org/abs/2603.12261)
- Arias, Solà, Armengod & Vanrell — *Color encoding in Latent Space of Stable Diffusion Models* — [arXiv:2512.09477](https://arxiv.org/abs/2512.09477)
- *Ctrl-X: Controlling Structure and Appearance for Text-To-Image Generation Without Guidance* — [arXiv:2406.07540](https://arxiv.org/pdf/2406.07540)
- *Towards Best Practices of Activation Patching in Language Models* — [arXiv:2309.16042](https://arxiv.org/pdf/2309.16042)
