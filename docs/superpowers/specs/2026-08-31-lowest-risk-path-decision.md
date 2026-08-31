# FLAIR — Which Path Is Lowest-Risk and Most Accurate

**Status:** Decision note
**Date:** 2026-08-31
**Question:** Of the paths explored across the research notes, which one yields
trustworthy results with the least chance of the work being wasted?

**Related:**
- [`2026-08-31-colour-localization-research.md`](./2026-08-31-colour-localization-research.md)
- [`2026-08-31-two-stage-screen-confirm-research.md`](./2026-08-31-two-stage-screen-confirm-research.md)
- [`2026-08-31-lcs-colour-basis-design.md`](./2026-08-31-lcs-colour-basis-design.md)
- [`2026-08-31-adain-latent-split-design.md`](./2026-08-31-adain-latent-split-design.md)

---

## 1. Short answer

**Run the exhaustive sweep with the pixel-space CIELAB you already have, across
multiple seeds. Add nothing new until that is producing stable numbers.**

Two reasons, and the second matters more than the first:

1. Every clever alternative — latent metrics, V-token screens, LCS bases — is
   *unfalsifiable without it*. They are all approximations of the exhaustive
   causal sweep, and an approximation cannot be validated against nothing.
2. **The largest source of error in the current setup is not the metric. It is
   that everything is being measured at a single seed** (§5). Fixing that
   changes the answers; changing the metric mostly changes the units.

"Risk-free" does not exist here. What follows is risk *ranked*.

## 2. There are six risks, not one

Optimizing for compute alone has been hiding five of these.

| # | Risk | Question it asks |
|---|---|---|
| **R1** | Measurement validity | Does the number actually track colour? |
| **R2** | Attribution | Does a high score mean *this head* caused it? |
| **R3** | Statistical | Would a different seed reorder the leaderboard? |
| **R4** | Implementation | Can it be built without subtle silent bugs? |
| **R5** | Reviewer | Does it survive CVPR peer review? |
| **R6** | Null result | What if colour simply is not localized? |

A path that scores well on compute and badly on R1-R4 produces numbers that look
like results and are not.

## 3. The four paths scored

| | **A** Exhaustive + pixel CIELAB | **B** Two-stage screen | **C** Latent metric (specs ①/③) | **D** Per-head A/V probe |
|---|---|---|---|---|
| **R1 validity** | **Lowest** — CIELAB is standard; `photometric.py` is tested | Lowest (uses A in Stage 2) | **High** — novel, unvalidated metric | High |
| **R2 attribution** | **Lowest** — direct causal injection, no prefilter | Medium — prefilter recall must be proven | Lowest if full sweep | **Highest** — correlational, not causal |
| **R3 statistical** | Depends on seeds — same for all paths | Same | Same | Same |
| **R4 implementation** | **Lowest** — nearly all of it exists | **High** — SDPA substitution, concat order, `attn2` asymmetry | Medium-high | **Highest** |
| **R5 reviewer** | Medium — exhaustive is not itself a contribution | Medium | Good if validated, bad if not | Good if validated |
| **R6 null result** | unaffected | unaffected | unaffected | unaffected |
| **Compute** | 577 gen/attribute | ~86 | ~560 | ~8 |

Note the pattern: **the paths that reduce compute increase R1, R2 and R4.** That
is the trade actually on the table, and compute is the cheapest of the four
things being traded.

## 4. Why the boring path wins: it is the ground truth the others need

This is the structural argument, and it is stronger than any of the individual
scores above.

- Path B's screen is defensible **only** if its recall is measured — which
  requires exhaustive Path A results on some blocks to measure against.
- Path C's latent metric is defensible **only** if it agrees with a perceptual
  metric — which requires Path A's CIELAB numbers to compare with.
- Path D's per-head probe is correlational and needs causal confirmation — which
  is Path A.

So Path A is not the timid option. **It is the prerequisite for publishing any of
the others.** Doing it first is what makes B, C and D into contributions instead
of unvalidated claims.

## 5. The accuracy problem nobody has raised: seed variance

`scripts/test_latent_color_blocks.py:173` defaults to `--seed 0` and every run in
the sweep uses that one seed. The base latent is generated once, at one seed, and
all 24 blocks are compared against it.

**This is the dominant error term, and it is larger than the metric choice.**

Diffusion output varies substantially with initial noise. At a single seed, the
gap between "block 5 scores 0.42" and "block 7 scores 0.39" carries no
information — there is no variance estimate, so there is no way to know whether
that ordering would survive `--seed 1`. A leaderboard built this way is not a
measurement; it is one sample presented as one.

Everything else in these notes — CIELAB versus latent, mean-then-ΔE versus
ΔE-then-mean, V-tokens versus final latents — is a second-order concern beside
this.

**Minimum fix:** N ≥ 5 seeds per cell, report mean and standard error, and treat
two cells as distinguishable only when their intervals separate. Better: report
the rank correlation between seeds. If block rankings do not correlate across
seeds, that is the finding, and no metric refinement will rescue it.

## 6. The recommendation: spend the savings on seeds, not on speed

This reframes every efficiency result in the research notes.

```
Path A, 1 seed   =  577 generations   → unreliable
Path A, 5 seeds  = 2885 generations   → reliable, ~16 h/attribute
Path B, 5 seeds  =  430 generations   → reliable, ~2.4 h/attribute
```

The screen's value is **not** that it finishes sooner. It is that it buys the
statistical power that makes the numbers mean anything. Efficiency work whose
savings are banked rather than reinvested in seeds has improved nothing that
matters.

But note the ordering constraint: Path B cannot be trusted until validated
against Path A. So the sequence is A first at small scale, then B at full scale.

## 7. The de-risked sequence

Each stage has a gate. If a gate fails, stop and reconsider rather than proceeding
on a broken foundation.

| # | Do | Gate |
|---|---|---|
| **1** | Add multi-seed support and variance reporting to the existing sweep | Same block ranked top across ≥3 of 5 seeds. **If not, stop — nothing downstream is measurable.** |
| **2** | Run exhaustive Path A on **one head** across all 24 blocks, 5 seeds, pixel CIELAB with `target_colour_delta` | A stable block ranking with separated intervals |
| **3** | Compare against Stable Flow's SD3 vital layers `[0,7,8,9]` | Sanity check on the apparatus only — see parent §5 for why this is not colour ground truth |
| **4** | Extend to all 24 heads for the 2-3 best blocks | Head-level structure exists, or is honestly flat (R6) |
| **5** | *Only now* build Path B's screen; validate recall against stages 2-4 | Spearman ρ > 0.7 |
| **6** | *Only now* build Path C's latent metric; validate against stage 2's CIELAB | Agreement reported with R² |

Stages 1-4 use code that already exists plus a seed loop. Stages 5-6 are the
research contributions, and by then they have something to be measured against.

## 8. If a gate fails

| Gate | If it fails |
|---|---|
| **1 — seeds disagree** | The intervention may be too weak to exceed sampling noise. Raise routing strength, or increase steps, before touching metrics. This is a *finding*, and an important one. |
| **2 — no stable ranking** | Colour is distributed (R6). Reframe the paper around distribution rather than localization; the HASM contribution survives, the "colour head" claim does not. |
| **5 — low ρ** | Ship Path A results. The screen becomes future work, not a retracted claim. |
| **6 — poor agreement** | Report it honestly: "latent-space ΔE does not track perceptual ΔE in SD3.5." That is a publishable negative result and saves others the same detour. |

Every failure mode above still leaves a paper. That is the property being
optimized for.

## 9. Unresolved: is compute binding or not?

Two project documents disagree, and this decision depends on which is current.

- `CLAUDE.md:7` — *"Kaggle free-tier GPU only for actual generation."*
- [`2026-08-26-flair-head-level-routing-design.md`](./2026-08-26-flair-head-level-routing-design.md) §2 — *"Compute is no longer a binding constraint on this project, which removes the cost argument that originally motivated both the block granularity and the vital-layer prefilter."*

If the spec is current, **§6's Path A at 5 seeds is affordable and the screen is
optional** — build it as a contribution, not a necessity. If CLAUDE.md is current,
the screen is required and stage 5 moves earlier, accepting its recall risk.

**Resolve this before starting.** It is the single input that changes the
recommended order, and it costs one sentence to settle.

## 10. Assets already built that reduce risk

Worth naming, because several of these solve problems the newer notes were about
to re-solve:

| Asset | What it already handles |
|---|---|
| `metrics/photometric.py` | Masked CIELAB ΔE, tested |
| `photometric.py:83 target_colour_delta` | The sepia-wash / reoriented-car failure mode — movement *toward* the named colour, not merely away from baseline |
| `guard.py` | Rejecting collapsed generations before they are measured |
| `metrics/masking.py` | ClipSeg masking, usable whenever output is decoded |
| `patching.py:69 bypass_blocks` | Stable Flow-style residual ablation, already implemented |
| `calibration/` | Resumable checkpointing — essential for multi-seed runs that exceed a Kaggle session |

The last row matters for §6: a 2885-generation campaign will not fit one session,
and the resumable harness already exists to span them.
