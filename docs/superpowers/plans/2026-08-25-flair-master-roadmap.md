# FLAIR — Master Roadmap to CVPR 2027

**Target:** submission in the last week of November 2026 (~13 weeks from 2026-08-25). Dates unofficial — recheck the CFP when it posts (~Sept 2026).
**Context:** solo author, Kaggle free-tier GPU only, no implementation existed at plan start.

This is the index. It says what is planned in executable detail, what is deliberately not yet planned and why, and what has to be decided at each gate.

## Document map

| Document | Covers | State |
|---|---|---|
| [`specs/…-cvpr-publication-plan-design.md`](../specs/2026-08-25-flair-cvpr-publication-plan-design.md) | The design: method scope, novelty positioning, evaluation, risks | Approved |
| [`specs/…-flair-architecture-overview.md`](../specs/2026-08-25-flair-architecture-overview.md) | Both pipelines as diagrams + component reference | Approved |
| [`plans/…-flair-foundation.md`](./2026-08-25-flair-foundation.md) | **Tasks 1-10** — code foundation (Weeks 1-2) | Executable |
| [`plans/…-flair-calibration-harness.md`](./2026-08-25-flair-calibration-harness.md) | **Tasks 11-16** — calibration code (Week 3) | Executable |
| This document | Weeks 3-13: campaign run-book, evaluation, paper, gates | Run-book + outline |

**Task numbering is global and continuous.** Tasks 1-16 are written. Tasks 17+ are named here but not yet expanded into TDD steps — see §4 for why.

---

## 1 · The 13 weeks

| Week | Work | Planned to | Gate |
|---|---|---|---|
| 1 | Scaffolding, core types, parser, BASM container, routing/blend, guard, processor, pipeline, smoke test | Tasks 1-7 | **Crisp pipeline runs** |
| 2 | Fuzzy membership, hedge operators, integration | Tasks 8-10 | **Hedge ladder monotonic** |
| 3 | Metrics, corpus, prefilter, harness; run the prefilter | Tasks 11-16 | **Vital blocks identified** |
| 4 | BASM calibration campaign | §2 run-book | **BASM is non-degenerate** |
| 5 | Routing integration, α₀/window tuning, qualitative iteration | §3.1 | — |
| 6 | Eval prompt set + absolute metric pipeline + baselines | Tasks 17-19 | — |
| **7** | **Full eval run: FLAIR vs. baselines** | §3.2 | **GO / NO-GO** |
| 8 | Ablations + T2I-CompBench++ attribute-binding subset | Task 20 | — |
| 9 | Gated stretch: FLUX demo (§3.7) *or* fuzzy conflict (§3.8) — one, not both | Tasks 21-22 | **Cut by Friday** |
| 10 | Qualitative pass, all figures final | §3.3 | **Figures frozen** |
| 11 | Full draft | §3.4 | — |
| 12 | Self-review, polish, anonymized supplementary code | §3.4 | — |
| 13 | Buffer: reruns, proofreading, formatting, submit | — | **Submit** |

---

## 2 · Weeks 3-4: the calibration campaign run-book

The harness (Tasks 11-16) is code. This is how to *run* it. Parameters are formulas, not invented numbers — they resolve once Week 1's smoke test prints the two measurements everything depends on.

### 2.1 The two measurements that set everything

From `scripts/smoke_test.py` (foundation Task 7):

- **`N_BLOCKS`** — SD3.5-M's real transformer block count (printed on startup).
- **`T_GEN`** — wall-clock seconds for one 20-step 512px generation on the session's GPU. Time the baseline generation.

Record both in `calibration_runs/measurements.txt` before proceeding.

### 2.2 Generation budget

Baselines are shared across blocks — the base-prompt image for a given (pair, seed) is identical no matter which block gets swapped, so it is generated once and reused. That makes the cost:

```
baselines = A × P × S
swaps     = A × P × S × B
total     = A × P × S × (1 + B)
```

where `A` = attributes (7), `P` = pairs per attribute, `S` = seeds per pair, `B` = vital blocks kept.

Worked examples at `B = 10`:

| P | S | Total generations |
|---|---|---|
| 5 | 1 | 385 |
| 5 | 2 | 770 |
| 10 | 1 | 770 |
| 10 | 2 | 1540 |

Spec §3.4's ~1000-1400 target corresponds to roughly `P = 10, S = 2`.

**Choose P and S to fit the weekly quota:**

```
P × S  ≤  (WEEKLY_GPU_HOURS × 3600) / (A × (1 + B) × T_GEN)
```

Kaggle free tier is ~30 GPU-hours/week. Budget **60%** of it for calibration (leave headroom for reruns and the prefilter), so use `WEEKLY_GPU_HOURS = 18` across the two weeks available.

**Prefilter cost is separate and small:** `prompts × seeds × (1 + N_BLOCKS)` generations. With 3 prompts and 1 seed at `N_BLOCKS = 24`, that is 75 generations.

### 2.3 Choosing B (how many vital blocks to keep)

`--top-n` for the prefilter. Trade-off: B drives the entire campaign cost linearly, and every block kept that isn't actually attribute-selective just adds noise rows to the BASM.

Rule: sort the vitality scores descending and keep blocks down to the **elbow** — where the score drops below 50% of the top block's score — clamped to `[6, 12]`. Record the chosen B and the reason in `measurements.txt`. Spec §3.4 assumes ~8-10; if the elbow puts you far outside that, say so in the paper rather than forcing the number.

### 2.4 Session discipline (Kaggle's 12-hour cap)

**Known gap — address before running:** `calibrate()` (Task 16) runs the full sweep in one call with no checkpointing. At 1540 generations × `T_GEN`, a campaign can exceed a 12-hour session and lose everything.

**Task 16b (do this first, ~1 hour):** add per-(attribute, block) checkpointing to `calibrate()` — write each cell's raw mean to `calibration_runs/cells/{attr}_{block}.json` as it completes, and skip cells whose file already exists on startup. Test it by killing a fake run mid-sweep and re-running: completed cells must not regenerate. This turns a lost session into a resumed one.

Run order:

```bash
python scripts/calibrate.py prefilter --top-n <B> --out calibration_runs/
python scripts/calibrate.py basm --vitality calibration_runs/vitality.json \
    --seeds 0 1 --out calibration_runs/
```

### 2.5 Week 4 gate — is the BASM real?

Before building anything on it, check three things. A BASM that fails these is measuring noise, and every downstream number would be meaningless.

1. **Non-degenerate columns.** No attribute column may be all-zero after normalisation (that means min-max saw no spread — the swap had no measurable effect for that attribute).
2. **Selectivity.** For at least the 4 core attributes, the top block's raw score should be clearly above the column median — not a flat field. If every block looks equally sensitive, the prefilter kept the wrong blocks or the metric is insensitive.
3. **Face validity.** Different attributes should not all peak on the *same* block. If Color, Size, and Lighting all route to block 7, there is no disentanglement to exploit and the paper's core premise is in trouble — investigate before Week 5.

Record the outcome. **If (3) fails, that is a Week-4 warning of a Week-7 NO-GO** — raise it early rather than discovering it in the full eval.

---

## 3 · Weeks 5-13 outline

### 3.1 Week 5 — integration and tuning

Wire the real BASM into `FlairPipeline`, replacing `BASM.uniform`. Tune two things qualitatively on ~10 prompts:

- **`alpha_0`** — raise until attribute changes are clearly visible; stop before identity or structure degrades. The Week 2 hedge-ladder images are the reference for "visible."
- **`t_window`** — the default `(0.0, 0.6)` is a starting guess; try `(0.0, 0.4)` and `(0.0, 0.8)`.

Deliverable: chosen values recorded with side-by-side evidence. No new code expected.

### 3.2 Weeks 6-7 — evaluation (Tasks 17-19, to be planned)

- **Task 17 — absolute metrics.** The calibration plan delivered *delta* metrics for all 7 attributes and absolute forms for the 3 photometric ones. Evaluation needs absolute forms for identity, texture, style, action (anchor sets and target descriptors), each landing on its declared membership universe.
- **Task 18 — eval prompt set.** ~100-150 prompts spanning attribute combinations, plus the controllability-curve set (5 intensity levels × 4 core attributes × a few base scenes). Authored as data with the same one-thing-varies discipline as the calibration corpus.
- **Task 19 — baseline runners + results table.** Vanilla SD3.5-M (single concatenated prompt), an attention-reweighting baseline, and RPG if it ports without significant engineering. Plus the aggregation that turns per-image metrics into the paper's table.

**Week 7 GO / NO-GO.** FLAIR must beat vanilla SD3.5-M on attribute metrics *without* quality or identity collapse. If not: debug, cut scope further, or retarget a workshop / later cycle. Decide here, not in Week 10. Neither gated stretch item starts before this passes.

### 3.3 Weeks 8-10 — ablations, stretch, figures

- **Task 20 — ablation runner.** BASM-routed vs. random-block routing (the control that proves calibration matters), guard on/off, routed vs. global-scale-no-routing, fuzzy on/off. Plus the T2I-CompBench++ attribute-binding subset.
- **Task 21 (gated) — FLUX demo** per spec §3.7: ported processor, mini-BASM on 4 core attributes, ~10-20 qualitative prompts. Drop it if FLUX won't fit Kaggle VRAM even quantized.
- **Task 22 (gated) — fuzzy conflict resolution** per spec §3.8. Must survive its own ablation or be cut.
- **Week 9 rule:** attempt *one* stretch item, not both. Whatever isn't done by that Friday moves to Future Work.
- **Week 10:** figures frozen. The controllability curve and the BASM heatmap are the two that carry the paper's argument — make those first.

### 3.4 Weeks 11-13 — the paper

Draft order (hardest-first, so the risky parts get the most revision):

1. **Method** (§3) — the mechanism is already fully specified across the spec and both plans; this is mostly transcription.
2. **Experiments + Ablations** (§4-5) — tables and figures exist by now.
3. **Intro** (§1) — must contain the explicit "why offline-causal-calibration-for-generation ≠ per-instance-adaptive-routing-for-editing" paragraph. This is the single most likely reviewer pushback; do not leave it implicit.
4. **Related Work** (§2) — Stable Flow, HeadRouter, the color-style disentanglement line, and the Attend-and-Excite/RPG/MultiDiffusion baseline family.
5. **Limitations + Conclusion.**

Week 12: self-review, anonymize the supplementary code drop. Week 13 is buffer — treat it as buffer, not as working time.

---

## 4 · What is deliberately not planned yet

Tasks 17-22 are named, not expanded into TDD steps. Three reasons, in order of weight:

1. **They depend on measurements that don't exist yet.** Evaluation thresholds, prompt-set size, and ablation scope all key off what the Week 4 BASM and the Week 7 baseline gap actually look like. Writing step-by-step tasks now means inventing those numbers — the exact failure mode the plan format forbids.
2. **A gate sits in the middle.** Week 7 can cut or retarget the project. Detailed plans for Weeks 8-13 written today may plan work that never happens.
3. **Weeks 11-13 aren't code.** The TDD task format doesn't fit paper writing; §3.4's draft order is the appropriate level of planning for it.

Expand Tasks 17-19 into a full plan at the **start of Week 6**, once the BASM is real and tuned.

---

## 5 · Standing risks

| Risk | Trigger to watch | Response |
|---|---|---|
| Kaggle GPU quota | Calibration slipping past Week 4 | Cut in order: FLUX demo → attributes back toward the core 4 → fewer pairs/blocks |
| BASM has no selectivity | Week 4 gate check (3) fails | Early warning of Week-7 NO-GO; investigate metric sensitivity and prefilter choice first |
| No checkpointing in `calibrate()` | Before any long run | Task 16b (§2.4) — do it before the campaign, not after a lost session |
| FLUX won't fit VRAM | Week 9 | Drop to Future Work; do not spend core-method or writing time on it |
| Head-level routing temptation | Any time | It erases the HeadRouter differentiation. Deliberately excluded, not deferred — see spec §2 |
| Public repo before submission | Now | Repo is public at `github.com/sunzidiautomation/FD`; consider private until acceptance (scooping + double-blind) |
| No second reader | Week 11 | Line up one outside read — labmate, community, or paid review — before submission |

---

## 6 · Immediate next actions

1. Decide repo visibility.
2. Execute foundation **Task 1** (scaffolding, core types, `TextBatchLayout`).
3. Work Tasks 1-7, then run the smoke test and record `N_BLOCKS` and `T_GEN` — the campaign's parameters are blocked on those two numbers.
