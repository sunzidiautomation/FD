# FLAIR Code Walkthrough

How the code runs, in execution order, and how to get an image out of it.

**Status:** Tasks 1-10 implemented, 87 tests passing. Tasks 11-16 (calibration) planned, not built.

## What runs where

| | Local (Docker, no GPU) | Kaggle (GPU) |
|---|---|---|
| Test suite (87) | ✅ | ✅ |
| Parsing, fuzzy hedges, BASM lookup, routing plan, α schedule | ✅ `scripts/explain.py` | ✅ |
| Text encoding (T5/CLIP), diffusion, images | ❌ | ✅ `scripts/smoke_test.py` |
| BASM calibration | ❌ | ✅ (after Tasks 11-16) |

Every *decision* FLAIR makes is CPU-only. The GPU is needed just to turn those decisions into pixels — so tune routing locally, then spend quota on generation.

```bash
docker build -t flair-test .
docker run --rm flair-test                                   # 87 tests, ~3s
docker run --rm flair-test python scripts/explain.py "a very red car"
```

Kaggle: **`notebooks/flair_kaggle.ipynb`**.

---

## Part 1 — The files, in the order they run

| # | File | Responsibility | Runs when |
|---|---|---|---|
| 1 | `flair_t2i/attributes.py` | The 7 attribute classes | import |
| 2 | `flair_t2i/config.py` | `FlairConfig` — α₀, timestep window, guard thresholds, model id | import |
| 3 | `flair_t2i/components.py` | `Component`, `TextBatchLayout` | import |
| 4 | `flair_t2i/basm.py` | Sensitivity matrix: `score()`, `top_k()`, save/load | setup + per prompt |
| 5 | `flair_t2i/pipeline.py` | `FlairPipeline` — the orchestrator | entry point |
| 6 | `flair_t2i/parsing.py` | Prompt → `Component` list (spaCy) | per prompt |
| 7 | `flair_t2i/fuzzy/membership.py` | Membership curves per attribute universe | per prompt |
| 8 | `flair_t2i/fuzzy/hedges.py` | Zadeh operators → intensity + routing breadth | per prompt |
| 9 | `flair_t2i/fuzzy/resolve.py` | Runs 7+8 over every component | per prompt |
| 10 | `flair_t2i/routing.py` | `build_routing_plan()`, and `blend()` — the injection | per prompt, then **per block per step** |
| 11 | `flair_t2i/guard.py` | Coherence checks, α back-off | per prompt |
| 12 | `flair_t2i/patching.py` | Installs/removes processors; `bypass_blocks` | per generation |
| 13 | `flair_t2i/processor.py` | `PlanRef`, `FlairJointProcessor` — the hook | **per block per step** |
| 14 | `flair_t2i/schedule.py` | `timestep_scale()` — early strong, late weak | **per block per step** |

Rows 10, 13, 14 run in the hot loop: `steps × blocks` times per image (20 steps × 24 blocks ≈ 480 calls).

---

## Part 2 — Serial execution trace

### Phase A · Setup (once per session)

```
scripts/smoke_test.py :: main()
├─ FlairConfig(device="cuda")                        config.py
├─ StableDiffusion3Pipeline.from_pretrained(...)     diffusers — downloads ~5GB
├─ pipe.enable_model_cpu_offload()                   fits 16GB VRAM
├─ n_blocks = len(pipe.transformer.transformer_blocks)   ← record as N_BLOCKS
├─ BASM.uniform(range(n_blocks), CORE_ATTRIBUTES)    basm.py — PLACEHOLDER
├─ spacy.load("en_core_web_sm")
└─ FlairPipeline(pipe, cfg, basm, nlp)               pipeline.py
```

### Phase B · Per prompt — building the routing plan

`FlairPipeline.generate(prompt, seed, steps, guidance_scale, routing=True, fuzzy=True)`

```
B1. PlanRef(total_steps=steps, do_cfg=guidance_scale > 1.0)      processor.py
    └─ mutable handle the denoise loop updates each step

B2. parse_prompt(prompt, nlp)                                     parsing.py
    ├─ _head_noun_chunk(doc)      → identity text ("sports car")
    │  └─ _fallback_identity(doc) → only if spaCy finds no noun chunk
    ├─ per token: _classify(t)    → lexicon lookup → AttributeClass
    ├─ per token: _hedge_for(t)   → "very" / "slightly" / "not" / None
    └─ → [Component(id="c_color", text="red sports car", attr=COLOR, hedge=None), ...]

B3. routable = [c for c in components if c.attr in basm.attributes]
    └─ attributes with no calibrated column are dropped here

B4. encode_components(routable)                                   pipeline.py
    └─ pipe.encode_prompt(texts, ...)  ← ONE batched call, before denoising
       → {"c_color": Tensor[seq, dim], "c_size": ..., ...}
       These embeddings live on the plan. They NEVER enter the denoising batch.

B5. resolve_components(routable)              (skipped if fuzzy=False)
    └─ per component:                                    fuzzy/resolve.py
       ├─ default_label(attr)                            fuzzy/membership.py
       ├─ membership_curve(attr, label)                  → μ over [0,1] grid
       ├─ HEDGE_KINDS[hedge] → NONE|CONCENTRATE|DILATE|COMPLEMENT
       ├─ apply_hedge(curve, kind)   → μ² | μ^0.5 | 1−μ | μ     fuzzy/hedges.py
       ├─ intensity = specificity(hedged) / specificity(base)
       │              clipped to [0.3, 1.6]   → no hedge = exactly 1.0
       └─ k = 1 if intensity ≥ 1.0 else 2 if ≥ 0.6 else 3
    → intensities {"c_color": 1.0}, k_overrides {"c_color": 1}

B6. build_routing_plan(routable, embeddings, basm, cfg, intensities, k_overrides)
    └─ per component:                                    routing.py
       └─ basm.top_k(attr, k)  → [(block_id, score), ...]
    → RoutingPlan(routed=(RoutedComponent, ...), cfg)

B7. CoherenceGuard(cfg).check_streams(plan, step=0)               guard.py
    └─ pairwise cosine between component embeddings
       below threshold → GuardEvent → plan.alpha_scale *= 0.5

B8. install_flair(pipe.transformer, ref)                          patching.py
    └─ every block's attn.processor wrapped in FlairJointProcessor
```

### Phase C · The denoise loop — where injection happens

`self.pipe(prompt=..., callback_on_step_end=on_step)` hands control to diffusers.

```
FOR each denoising step t (0 … steps-1):
  FOR each transformer block ℓ (0 … N_BLOCKS-1):

    FlairJointProcessor.__call__(attn, hidden_states, encoder_hidden_states)
    │                                                          processor.py
    ├─ ref.step_frac()          = step / total_steps
    ├─ ref.cond_slice(B)        = slice(B//2, B) under CFG
    │                             (diffusers packs [negative, positive])
    └─ plan.blend(ehs, ℓ, step_frac, cond_slice)               routing.py
       ├─ FAST PATH: block ℓ not routed, or plan inactive → return unchanged
       ├─ per contributing component:
       │  └─ alpha(rc, ℓ, step_frac)
       │     ├─ S[ℓ,a]                      ← BASM score for this block
       │     ├─ timestep_scale(...)         ← schedule.py, 1.0 → 0.0 over window
       │     └─ α = α₀ · S[ℓ,a] · intensity · sched(t) · alpha_scale
       └─ out[cond_slice] += α · (component_embedding − base_embedding)
    │
    └─ delegate to inner JointAttnProcessor2_0  ← real attention, untouched

  on_step callback → ref.step = step_index + 1
```

**The core equation, and where each term comes from:**

```
H_ℓ = H_base + Σᵢ αᵢ(t) · (Hᵢ − H_base)

αᵢ(t) = α₀        · S[ℓ,a]   · intensityᵢ  · sched(t)    · alpha_scale
        config.py   basm.py    hedges.py     schedule.py   guard.py
```

### Phase D · Teardown

```
finally: uninstall_flair(handles)     patching.py — always restores originals
return result.images[0]               PIL.Image
```

---

## Part 3 — How and when you get an image

### What you need

| Requirement | Why | Where |
|---|---|---|
| GPU, ≥16GB VRAM | SD3.5-M inference | Kaggle T4 ×2 or P100 (free tier) |
| Hugging Face account | SD3.5 is a **gated** model | accept the license on the model page |
| HF token | authenticate the download | Kaggle → Add-ons → Secrets |
| ~5GB disk | model weights | Kaggle provides |

**This will not run on your local machine** — no GPU, and the current Docker image deliberately omits `diffusers`/`transformers` because no test needs them.

### Steps on Kaggle

1. New notebook → **Settings → Accelerator → GPU T4 ×2**
2. **Add-ons → Secrets** → add `HF_TOKEN`
3. Accept the SD3.5-Medium license on its Hugging Face model page (one time, or the download 403s)
4. Run:

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

!git clone https://github.com/sunzidiautomation/FD.git
%cd FD
!pip install -q -r requirements.txt
!python -m spacy download en_core_web_sm -q
```

**Restart the session here.** The reference notebook's `lpips`/`skfuzzy` `ModuleNotFoundError`s came from skipping this.

```python
!python scripts/verify_env.py        # must print "Environment OK."
!python scripts/smoke_test.py --steps 20 --out outputs/
```

### What you get

```
outputs/smoke_baseline.png        routing off — plain SD3.5
outputs/smoke_routed.png          routing on
outputs/smoke_hedge_slightly.png  ┐
outputs/smoke_hedge_plain.png     ├ the hedge ladder
outputs/smoke_hedge_very.png      ┘
```

And on stdout — **write these two numbers into `calibration_runs/measurements.txt`**, the whole campaign budget derives from them:

```
N_BLOCKS = 24
T_GEN = 12.4s
```

### ⚠ Read this before judging the images

**The BASM is uncalibrated right now.** `BASM.uniform()` fills every cell with 0.5, and `top_k` breaks ties by ascending block id — so **every attribute routes to block 0**. All four streams pile into one block.

That is expected at this stage. The smoke test proves the *plumbing* works, not that routing helps. Judge it only on:

- ✅ Both images exist and neither is noise
- ✅ They differ from each other
- ✅ `routed components:` lists 4 ids, `blocks touched: [0]`
- ❌ Do **not** expect better colour/size fidelity yet — there is nothing calibrated to route by

### When results actually become meaningful

| Stage | What you get | Blocked on |
|---|---|---|
| **Now** | Plumbing images, all routed to block 0 | — |
| **After Week 3-4** | Real BASM → each attribute to *its own* block; routing becomes meaningful | calibration campaign (Tasks 11-16) |
| **After Week 5** | Good images — α₀ and timestep window tuned | qualitative tuning |
| **After Week 7** | Numbers vs. baselines; the go/no-go answer | eval pipeline (Tasks 17-19) |

**Answer to "when do I get the result?":** a *picture* today, on Kaggle, in about 15 minutes. A picture that *demonstrates the method* after calibration, ~2 weeks in. A picture worth putting in the paper after Week 5.

---

## Part 4 — Running the tests

```bash
docker build -t flair-test .
docker run --rm flair-test          # 87 passed
```

The image installs only what the suite needs: CPU torch, numpy, scipy, scikit-fuzzy, spaCy. No GPU, no model download, ~3 seconds to run.

**Not covered by any test** — these need real SD3.5 weights and surface on the first Kaggle run:

- `patching.bypass_blocks` return signature vs. diffusers' real `JointTransformerBlock.forward`
- `pipeline.encode_prompt(...)` 4-tuple shape and `prompt_2`/`prompt_3` kwargs

---

## Part 5 — Not built yet

| Tasks | What | Plan |
|---|---|---|
| 11-16 | Metrics, corpus, prefilter, BASM harness | `plans/…-calibration-harness.md` |
| 16b | Checkpointing in `calibrate()` — **do before any long run** | roadmap §2.4 |
| 17-19 | Absolute metrics, eval prompt set, baselines | named in roadmap, expand at Week 6 |
| 20-22 | Ablations; FLUX and fuzzy-conflict (both gated on Week 7) | roadmap §3.3 |
