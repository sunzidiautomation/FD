# FLAIR — CVPR 2027 Publication Plan

**Status:** Approved design, ready for implementation planning
**Author context:** Solo researcher, Kaggle-only compute, no implementation exists yet
**Target venue:** CVPR 2027 (author-targeted deadline: last week of November 2026 — **not yet official**; recheck when the CFP posts, likely ~Sept 2026)

## 1. Overview

FLAIR is a training-free, inference-time method for disentangled multi-attribute
control in text-to-image generation on SD3.5's MM-DiT architecture. It decomposes
a prompt into independent attribute components (identity, color, size, lighting),
calibrates — offline, once per model — which transformer blocks are causally most
sensitive to each attribute (the Block–Attribute Sensitivity Matrix, BASM), and at
generation time routes each attribute's own text stream into its most-sensitive
blocks as a weighted residual injection, with a runtime coherence guard that backs
off injection strength if it destabilizes the latent.

This document scopes FLAIR down from the full reference design (a fuzzy
linguistic module, three realization channels) to a version buildable solo,
from scratch, on Kaggle's free GPU tier, within roughly 13 weeks — while
keeping the core novel mechanism (BASM + routing), one deliberately-added
second contribution (fuzzy hedge-operator and attribute-value-membership
graded control, see §3.3), the full
7-attribute set (§3.1), and a lightweight multi-backbone generalization demo
on FLUX (§3.7) intact. Head-level routing granularity is deliberately
excluded — see §2.

## 2. Novelty Positioning

**Prior work this builds on and must explicitly differentiate from:**

- **Stable Flow** (Avrahami et al., CVPR 2025) — training-free vital-layer
  discovery in DiT via residual-bypass ablation; FLAIR's vital-layer prefilter
  (§3.4) is directly modeled on this. Differs: Stable Flow computes one generic
  per-layer vitality score for image *editing* (requires a source image,
  preserves unedited structure). FLAIR computes a *per-attribute* sensitivity
  score via causal contrastive-pair intervention, used for from-scratch
  *generation*.
- **HeadRouter** (ACM TOG 2026) — training-free MM-DiT image editing that
  adaptively routes text guidance to attention heads based on semantic
  sensitivity, with a per-timestep schedule (more text influence early, less
  late) that FLAIR's α(t) schedule reuses. Differs: HeadRouter's router is
  computed *per-instance at inference* from a source/reconstruction branch —
  nothing is precomputed. FLAIR's BASM is computed *offline, once, from
  contrastive text-only prompt pairs*, then reused across arbitrary prompts.
  Routing granularity also differs: heads (HeadRouter) vs. blocks (FLAIR) —
  **this is deliberately preserved as a differentiator.** Head-level routing
  for FLAIR is explicitly out of scope for this submission (see §8): adding
  it would erase this distinction and invite reviewers to read FLAIR as a
  narrower version of HeadRouter rather than a distinct offline/generation
  approach.
- **Training-free Color-Style Disentanglement for Constrained T2I Synthesis**
  (CVPR 2025) — closest generation-side disentanglement work, but is
  reference-image-conditioned and limited to two attributes; not block-routing
  based.
- **Attend-and-Excite / RPG / MultiDiffusion class** — the baseline family for
  compositional generation; these act via attention-map reweighting or spatial
  region binding, not causally-calibrated layer/block routing.

**Contribution claim (must appear explicitly in §1 of the paper, not left
implicit):**

> Per-attribute sensitivity in MM-DiT blocks can be causally calibrated offline,
> from contrastive text-only prompt pairs (no source image, no per-instance
> adaptation), into a reusable Block–Attribute Sensitivity Matrix. Routing
> independent attribute text-streams through this matrix at generation time
> gives disentangled multi-attribute control that a single concatenated prompt
> cannot achieve — extending training-free MM-DiT layer/head specialization
> from single-instance image editing to general-purpose compositional
> generation, and additionally supporting graded (fuzzy-hedge-controlled)
> attribute intensity, which neither prior method offers. We further show the
> calibration procedure transfers to a second MM-DiT backbone (FLUX),
> supporting the "general-purpose" claim.

The paper needs a short, explicit paragraph (end of §1 or start of Related
Work) arguing why "offline-causal-calibration-for-generation" is a different
problem from "per-instance-adaptive-routing-for-editing," not just an easier
version of it — this is the single most likely reviewer pushback.

## 3. Method Scope (MVP)

### 3.1 Cut from the reference design
- Three realization channels → **Channel A only** (T5 text sub-phrase
  realization). Channels B/C were marked "not used" / "not needed" in the
  reference design and have no evaluation payoff.
- **All 7 attributes kept: Color, Size, Identity, Lighting, Texture, Style,
  Action.** Color/Size/Identity/Lighting use the cheap standard metrics
  from the original scoping pass (§3.4). Texture/Style use Gram-matrix
  distance / DISTS; Action uses CLIP-score for action phrases, per the
  reference design's metric table. This is a ~1.75x increase in calibration
  volume over the original 4-attribute MVP (§5) — the first thing to cut
  back down if the Week 3-4 calibration falls behind schedule.
- Head-level routing granularity: **not attempted** — see §2 and §8.

### 3.2 Semantic parsing
Deterministic: spaCy dependency parse + modifier binding extracts
`(object identity, attribute-class, value)` tuples, same as reference design
step 1, without the fuzzy membership step (that's now folded into §3.3 instead
of duplicated here).

### 3.3 Fuzzy hedge & attribute-value module (build from the start, sequenced
after crisp pipeline is validated)

Purpose: handle graded/vague intensity modifiers ("slightly red," "very
small," "somewhat warm") — the classic Zadeh fuzzy-set use case for vague
predicates — and give them real causal effect on generation, not just pass
the hedge word through to the text encoder. Two mechanisms, built together
(both feed the same integration point below), plus a third gated behind the
Week 7 checkpoint (§3.8):

**(B) Linguistic hedge operators** — replaces a fixed lookup table with
Zadeh's standard operators applied to a base membership μ_base (=1.0 for the
stated crisp value, e.g. "red"):
  - "very X" → concentration: μ = μ_base²
  - "somewhat X" / "a bit X" → dilation: μ = μ_base^0.5
  - "not X" → complement: μ = 1 − μ_base
  - compound hedges ("not very X") → compose in sequence (μ = 1 − μ_base²)
  - no hedge → μ = μ_base = 1.0 (crisp, unchanged default — backward
    compatible)

  This is a small lexicon of *operators* (~5), not a lookup table of
  buckets, and is the first thing in the spec to handle negation at all.

**(A) Fuzzy value membership over attribute universes** — each attribute's
metric has a defined numeric universe (size → CLIPSeg object-mask area
ratio ∈ [0,1]; lighting → color-temperature mapped to [0,1]; color → target
hue distance; etc.) with triangular/trapezoidal membership functions for
that attribute's linguistic labels (e.g. "small"/"medium"/"large" as three
overlapping trapezoids over the size universe). This does two things:
  - Gives BASM calibration (§3.4) a graded target — did the swapped
    attribute land *inside the fuzzy region* for the intended label, not
    just "did the metric move."
  - Upgrades the coherence guard's distortion check (§3.6) from crisp
    percentile thresholds to real fuzzy-membership evaluation: a violation
    becomes "measured metric's membership in the intended fuzzy region
    fell below a guard threshold" (continuous), not "outside a fixed
    percentile band" (binary).

- **Integration point**: `α_i(t) = α_0 · S[ℓ,a] · μ_hedge(t) ·
  timestep_schedule(t)`, where `μ_hedge` now comes from (B)'s operators
  instead of a lookup table. `RoutingPlan`'s top-k block selection
  (`routing_breadth`) becomes a formula of μ_hedge (e.g.
  `k = clip(round(1 + 2·(1 − μ_hedge)), 1, 3)`) — same spirit as before
  (higher confidence → narrower/stronger push, lower confidence →
  wider/diffuse push) but continuous instead of bucketed. This remains the
  only structural change to `RoutingPlan` and `blend()`; still no new model
  calls, still implemented via `skfuzzy` (defuzzify via centroid where
  needed, e.g. for the fuzzy-region membership check in §3.6).
- **Build order**: implement and validate the crisp (no-hedge, no-fuzzy-
  region) pipeline first (week 1); layer (A)+(B) on top together in week 2,
  once base routing is confirmed correct — they share the same integration
  point and the guard upgrade only makes sense with both in place. Don't
  debug the crisp pipeline and the fuzzy layer simultaneously.

### 3.4 BASM calibration
Per reference design step 4, scoped down:
- 7 attributes × ~10 contrastive prompt pairs each (down from ~20)
- Vital blocks pre-filtered via Stable-Flow-style residual-bypass prefilter
  to ~8-10 candidate blocks (out of SD3.5-M's full block count)
- Metrics: CIELAB ΔE (color), CLIPSeg object-mask area-ratio change (size),
  1-CLIP-similarity-to-anchors (identity), CLIP-score to lighting descriptors
  / luminance-color-temperature (lighting), Gram-matrix distance / DISTS
  (texture, style), CLIP-score for action phrases (action) — all restricted
  to the object mask from cross-attention thresholding, per reference
  design.
- Estimated volume: ~1000-1400 short generations (up from ~600-800 at 4
  attributes), feasible across Kaggle sessions in ~2 weeks.

### 3.5 Routing & generation
Per reference design steps 5-7: `FlairJointProcessor` installed on every
MM-DiT block, modifies `encoder_hidden_states` before the original joint
attention forward runs. **Must fix first**: the batch-layout assumption in
`blend()` (`base_i = B - self.n_rows`) flagged in prior debugging — this must
be verified against the actual batch construction before any result can be
trusted, since a silent mismatch here would invalidate every downstream
number.

### 3.6 Coherence guard
Per reference design step 7 (cross-stream cosine similarity check +
attribute-change-range check; reduce α on violation). The attribute-change-
range check is upgraded per §3.3(A): instead of a fixed percentile band, it
evaluates the measured metric's membership in the intended attribute's fuzzy
region and flags a violation below a guard threshold (e.g. membership <
0.5). The cross-stream cosine-similarity check is unchanged (crisp
threshold — that's §3.8's territory, not this one). Still cheap, still
gives a clean on/off ablation (crisp-percentile guard vs. fuzzy-membership
guard is just swapping which check function runs).

### 3.7 Multi-backbone generalization demo (FLUX) — lightweight, not a
second full experimental track

Goal: show the calibration procedure transfers to a second MM-DiT backbone,
as supporting evidence for the "general-purpose" claim — without doubling
the entire engineering and compute budget.

- **Port**: implement a FLUX-equivalent of `FlairJointProcessor` against
  FLUX's own attention-processor API (necessary, unavoidable engineering —
  budgeted at up to ~1 week, see §6). The Stable-Flow-style residual-bypass
  vital-layer prefilter (§3.4) is architecture-agnostic and reusable as-is.
- **Mini-BASM**: recalibrate on FLUX using only the original **4 core
  attributes** (Color, Size, Identity, Lighting — not the full 7) with a
  reduced ~5 contrastive pairs each, over the pre-filtered vital blocks.
  Sensitivity patterns are architecture-specific and do not transfer from
  the SD3.5-M matrix — this must be computed fresh, but at greatly reduced
  volume (~150-250 generations).
- **Evidence produced**: a small qualitative side-by-side (~10-20 prompts,
  FLUX baseline vs. FLUX+FLAIR) plus, if time allows, the same primary
  metrics on that small set as a lightweight secondary table. This is
  supplementary/generalization evidence, not the paper's primary
  experimental section — no controllability curve, no T2I-CompBench++ pass,
  no ablations on FLUX.
- **VRAM risk**: FLUX (~12B) is much larger than SD3.5-M (~2.5B) and may not
  fit Kaggle's free-tier GPUs (T4/P100, 16GB) without aggressive
  quantization. Plan to use FLUX.1-schnell (distilled) and/or NF4/fp8
  quantization; if that still doesn't fit or produces unusable quality, this
  section is dropped and noted as future work rather than allowed to
  jeopardize the core SD3.5-M results. See §7.
- **Sequencing**: attempted only *after* the Week 7 go/no-go checkpoint on
  the core SD3.5-M method passes — no FLUX effort is spent before the core
  method is known to work.

### 3.8 Fuzzy conflict resolution (C) — stretch, gated post-checkpoint

Goal: replace the coherence guard's crisp cross-stream cosine-similarity
cutoff (§3.6) with a graded conflict-severity score between routed attribute
streams, using fuzzy T-norm/T-conorm composition (e.g. min/max or product)
over each stream pair's membership in a "conflicting" fuzzy set.

This is the least well-defined of the three fuzzy mechanisms — it has no
anchor in the original reference design (which only specifies per-attribute
membership, not inter-stream conflict logic) and risks ending up as
unablated, unmotivated machinery in the paper if attribute streams turn out
to rarely conflict in practice. **Not committed for this submission.**

- **Gate**: attempted only after the Week 7 go/no-go checkpoint passes and
  only if there's schedule slack — same treatment as §3.7. No time is
  budgeted for it in Weeks 1-8.
- **If attempted**: must be justified with its own small ablation (fuzzy
  conflict score vs. the existing crisp cosine cutoff) before it goes in the
  paper — if the two don't behave meaningfully differently, drop it rather
  than include unmotivated complexity.
- **If not attempted or dropped**: note in Future Work (§8), not in the
  submission.

## 4. Evaluation Plan

All of the following are on **SD3.5-M** (the primary backbone) unless noted;
the FLUX generalization demo (§3.7) has its own smaller, separate evidence
set and is not folded into these primary numbers.

- **Primary benchmark**: curated ~100-150 prompt set spanning combinations of
  the 7 attributes.
- **Controllability curve** (new, enabled by §3.3): ~100-150 additional
  generations across 5 intensity levels ("slightly" → "very") × 4 core
  attributes (Color, Size, Identity, Lighting — the ones with clean,
  continuous intensity semantics) × a few base scenes, plotting hedge
  intensity against the corresponding attribute metric. Should be monotonic
  — this is a cheap, high-signal figure and a concrete demonstration of the
  graded-control differentiator.
- **Secondary benchmark**: attribute-binding subset only of T2I-CompBench++
  (not the full suite) for standard-benchmark comparability.
- **Metrics**: CIELAB ΔE, CLIPSeg mask-area ratio, CLIP similarity,
  luminance/color-temperature, Gram-matrix distance/DISTS (texture, style),
  CLIP-score for action phrases — all primary, per attribute. LPIPS as
  secondary perceptual sanity-check only, not a primary attribute metric.
- **Baselines** (kept small, given solo/compute constraints): vanilla SD3.5-M
  (single concatenated prompt), a cheap attention-reweighting baseline, and
  one adapted training-free compositional method (e.g. RPG) if it ports to
  MM-DiT without significant engineering. More can be added post-submission
  or during rebuttal.
- **Ablations**: BASM-routed vs. random-block routing (control); coherence
  guard on/off; routed vs. global-scale-no-routing; fuzzy module on/off
  (crisp-only vs. graded).

## 5. Compute Budget

Kaggle free tier: ~30 GPU-hours/week, 12-hour session cap (P100 / T4x2).
Rough generation counts: ~1000-1400 (BASM calibration, 7 attributes) + ~150
(main eval) + ~150 (controllability curve) + baseline runs + ablations +
~150-250 (FLUX mini-BASM) + ~10-20 (FLUX qualitative) ≈ low thousands of
short (≤20-step, 512px) generations total — feasible within the weekly quota
across the ~13-week timeline below, but this is the tightest constraint in
the plan and the first thing to re-cut if behind schedule. If behind, cut in
this order: (1) FLUX demo dropped to future work entirely, (2) attributes
reduced back toward the 4-attribute core, (3) fewer contrastive pairs /
vital blocks in SD3.5-M BASM.

## 6. Timeline

Assumes a submission deadline in the last week of November 2026 (~13-13.5
weeks from plan start). Recheck against the official CFP once published.

| Week | Milestone |
|---|---|
| 1 | Fix environment (crisp pipeline first — no `skfuzzy` install yet); fix the `blend()` batch-layout bug; get one working end-to-end crisp (no-hedge) generation on SD3.5-M |
| 2 | Build fuzzy hedge-operator + attribute-value-membership module (§3.3) on top of validated crisp pipeline |
| 3-4 | Vital-layer prefilter + BASM calibration, SD3.5-M, all 7 attributes (reduced per-attribute budget) |
| 5 | Routing + coherence guard integration, qualitative iteration on α₀/timestep window |
| 6 | Build eval prompt set (main + controllability curve) + full 7-attribute metric pipeline |
| **7** | **Go/no-go checkpoint**: run FLAIR + baselines on the full eval set. If not beating vanilla SD3.5 on attribute metrics without quality/identity collapse, decide here — debug further, cut scope again, or retarget a workshop/later cycle. Neither FLUX (§3.7) nor fuzzy conflict resolution (§3.8) starts before this checkpoint passes. |
| 8 | Ablations + T2I-CompBench++ subset comparison (SD3.5-M) |
| 9 | FLUX generalization demo (§3.7): port processor, mini-BASM, qualitative set. If VRAM/quality issues block it, cut here and move to Future Work. If there's schedule slack instead, attempt fuzzy conflict resolution (§3.8) here or in parallel — but only one of the two stretch items if time is tight, not both. Do not let either slip into the writing weeks. |
| 10 | Qualitative/preference pass, finalize all figures (incl. FLUX generalization figure if it landed) |
| 11 | Full paper draft (Intro w/ explicit novelty paragraph, Related Work, Method, Experiments, Ablations, Limitations, Conclusion) |
| 12 | Self-review pass, polish, prep supplementary/anonymized code |
| 13 | Buffer: reruns, proofreading, final formatting, submission |

## 7. Risk Register

- **CVPR 2027 dates not official** — planning against the author's targeted
  window (last week of November 2026); recheck official CFP (~Sept 2026)
  and adjust the timeline immediately once published.
- **Kaggle GPU quota is the primary execution bottleneck** — if calibration
  or eval runs are falling behind, cut in the order given in §5.
- **`blend()` batch-layout bug** — must be fixed and verified before any
  result is trusted; a silent mismatch would invalidate all downstream
  numbers.
- **FLUX VRAM / feasibility on free-tier Kaggle GPUs** — FLUX (~12B) is
  ~5x SD3.5-M's size; may not fit even quantized. Mitigation: use
  FLUX.1-schnell + NF4/fp8 quantization; if still infeasible or quality is
  unusable, drop §3.7 to Future Work rather than let it consume time
  budgeted for the core method or the writing weeks.
- **Head-level routing temptation mid-project** — if it seems easy to "just
  add" once block-level routing works, revisit §2's reasoning first: it
  weakens the differentiation from HeadRouter and was deliberately excluded,
  not merely deferred for lack of time.
- **Fuzzy conflict resolution (§3.8) becoming unmotivated complexity** — if
  attempted, it must survive its own ablation (fuzzy conflict score vs. the
  existing crisp cosine cutoff) before inclusion in the paper. Cut it rather
  than include a mechanism that isn't shown to matter.
- **No second reader** — solo author, no advisor/co-author to sanity-check
  the draft. Recommend lining up at least one outside read (labmate, online
  research community, or a paid review pass) before submission.

## 8. Future Work (explicitly out of scope for this submission)
- Head-level (not just block-level) routing granularity — deliberately
  excluded to preserve the differentiation from HeadRouter (§2), not just
  deferred for time. Revisit only with a clear reframing (e.g. as a
  granularity-comparison study), not as a parallel mechanism.
- Full quantitative multi-backbone evaluation (beyond the lightweight FLUX
  demo in §3.7) — additional backbones, full BASM + full eval suite per
  backbone.
- Fuzzy conflict resolution (§3.8), if not attempted post-checkpoint or if
  attempted but dropped for lacking ablation support.
