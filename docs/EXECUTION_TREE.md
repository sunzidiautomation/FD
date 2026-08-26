# FLAIR Execution Tree

Every file, in the exact order it runs, and what each one does.

There are **three entry points**. Each is a separate tree.

| Entry point | What it produces | GPU |
|---|---|---|
| `scripts/explain.py` | routing decisions as text, no image | no |
| `scripts/smoke_test.py` | **images** | yes |
| `scripts/calibrate.py` | `vitality.json`, then `basm.npz` | yes |

The dependency between them runs one way:

```
calibrate.py  --produces-->  basm.npz  --read by-->  smoke_test.py / explain.py
```

Until `basm.npz` exists, the other two run on a **uniform placeholder** where
every cell is 0.5 — so every attribute routes to block 0 and routing does
nothing useful. That is the repo's current state.

---

# TREE 1 — Generating an image

```
scripts/smoke_test.py :: main()                                smoke_test.py:38
│
├── [A] SETUP — once per session ────────────────────────────────────────────
│   │
│   ├── FlairConfig(device="cuda")                                config.py
│   │      alpha_0=0.75, t_window=(0.0,0.6), top_k_default=1,
│   │      guard thresholds, model_id, max_sequence_length=256
│   │
│   ├── StableDiffusion3Pipeline.from_pretrained(cfg.model_id)    diffusers
│   │      downloads ~5GB on first run; gated -> 403 without licence
│   │
│   ├── pipe.enable_model_cpu_offload()
│   │      keeps SD3.5-M inside a 16GB T4
│   │
│   ├── n_blocks = len(pipe.transformer.transformer_blocks)   <-- N_BLOCKS
│   │
│   ├── BASM.uniform(range(n_blocks), CORE_ATTRIBUTES)            basm.py
│   │   or BASM.load(--basm)                                  <-- the real one
│   │      the block x attribute sensitivity matrix
│   │
│   ├── spacy.load("en_core_web_sm")                              spaCy
│   │
│   └── FlairPipeline(pipe, cfg, basm, nlp)                    pipeline.py:24
│          holds those four; owns last_plan / last_guard
│
├── [B] fp.generate(prompt, seed, steps, routing, fuzzy)      pipeline.py:47
│   │      ALL OF B RUNS ON CPU, BEFORE ANY DENOISING
│   │
│   ├── B1  ref = PlanRef(total_steps, do_cfg=guidance>1.0)  processor.py:17
│   │        A mutable box. The denoise loop writes ref.step into it each
│   │        step; the processors read it. This is how one value reaches
│   │        480 attention calls without threading it through diffusers.
│   │
│   ├── B2  parse_prompt(prompt, nlp)                          parsing.py:108
│   │    │     "A small red sports car under warm evening light"
│   │    │
│   │    ├── _head_noun_chunk(doc)                             parsing.py:97
│   │    │      first noun chunk that is not a lighting phrase
│   │    │      -> identity_text = "sports car"
│   │    │      └── _fallback_identity(doc)                    parsing.py:82
│   │    │             ONLY if spaCy found no noun chunk at all.
│   │    │             "a rusty car driving" parses with `driving` as ROOT
│   │    │             and `car` as its compound child -> noun_chunks empty.
│   │    │             Every action pair in the corpus has that shape.
│   │    │
│   │    ├── per token: _classify(t)                           parsing.py:65
│   │    │      lemma -> ATTRIBUTE_LEXICON lookup -> AttributeClass
│   │    │      unknown words are IGNORED, never guessed
│   │    │
│   │    ├── per token: _hedge_for(t)                          parsing.py:73
│   │    │      checks both children AND head for very/slightly/not
│   │    │
│   │    ├── scene-level attrs (LIGHTING, STYLE) keep their own phrase
│   │    │   object-level attrs bind to the head noun: "red sports car"
│   │    │
│   │    └── verb pass -> ACTION component                    parsing.py:153
│   │
│   │       -> [Component(c_identity), Component(c_color, hedge=None),
│   │           Component(c_size), Component(c_lighting)]
│   │
│   ├── B3  routable = [c for c in components
│   │                   if c.attr in basm.attributes]         pipeline.py:63
│   │        an attribute with no calibrated column is DROPPED here
│   │
│   ├── B4  encode_components(routable)                       pipeline.py:32
│   │        pipe.encode_prompt(texts, prompt_2, prompt_3,
│   │                           do_classifier_free_guidance=False)
│   │        ONE batched call. -> {"c_color": Tensor[seq, dim], ...}
│   │        These live on the plan. They NEVER enter the denoising batch,
│   │        so there is no base-row index arithmetic to get wrong.
│   │
│   ├── B5  resolve_components(routable)   [skipped if fuzzy=False]
│   │    │                                                fuzzy/resolve.py:10
│   │    └── per component: resolve_hedge(attr, label, hedge)  hedges.py:91
│   │         ├── membership_curve(attr, label)             membership.py
│   │         │      mu over a [0,1] grid, e.g. "red" over the hue universe
│   │         ├── HEDGE_KINDS[hedge]                          hedges.py:35
│   │         │      very->CONCENTRATE  slightly->DILATE  not->COMPLEMENT
│   │         │      quite/fairly->NONE, so the ladder stays ordered
│   │         ├── apply_hedge(curve, kind)                    hedges.py:68
│   │         │      mu**2   |  mu**0.5  |  1-mu  |  mu
│   │         ├── specificity(c) = 1 - mean(c)                hedges.py:78
│   │         ├── intensity = clip(spec(hedged)/spec(base), 0.3, 1.6)
│   │         │      no hedge -> EXACTLY 1.0 -> crisp behaviour unchanged
│   │         └── k = _breadth(intensity)                     hedges.py:83
│   │                >=1.0 -> 1 block   >=0.6 -> 2   else 3
│   │
│   │       -> intensities {"c_color": 1.138}, k_overrides {"c_color": 1}
│   │          Fuzzy is now FINISHED. Two numbers. Nothing else survives.
│   │
│   ├── B6  build_routing_plan(routable, embeddings, hasm,
│   │                          cfg, intensities, k_overrides)  routing.py:92
│   │    └── per component: hasm.top_k(attr, k)                   hasm.py
│   │           -> [(HeadUnit(block, head), score), ...]
│   │       -> RoutingPlan(routed=(RoutedComponent, ...))       routing.py:31
│   │
│   ├── B7  CoherenceGuard(cfg)                                  guard.py:28
│   │    └── check_streams(plan, step=0)                         guard.py:33
│   │           pairwise cosine between component embeddings.
│   │           worst < 0.55 -> GuardEvent -> apply() halves alpha_scale
│   │           (needs >=2 components, else returns None)
│   │
│   └── B8  install_head_routing(pipe.transformer, ref)         patching.py:15
│            for every block: wraps add_q_proj, add_k_proj, add_v_proj
│            in HeadResidualProj(inner, block_id, plan, ref).
│            Returns handles so teardown can restore the originals.
│
├── [C] DENOISING — self.pipe(...) hands control to diffusers  pipeline.py:87
│   │
│   │   FOR each step t in 0..steps-1:
│   │     FOR each block l in 0..N_BLOCKS-1:      <-- 20 x 24 = 480 calls
│   │
│   ├──── HeadResidualProj.__call__(hidden_states)           head_proj.py
│   │     │
│   │     ├── base_proj = inner(hidden_states)               Linear proj
│   │     ├── ref.step_frac()   = step / total_steps         processor.py:25
│   │     ├── ref.cond_slice(B) = slice(B//2, B) under CFG    processor.py:30
│   │     │      diffusers packs [negative, positive]; the conditional
│   │     │      half is the tail.
│   │     │
│   │     ├── plan.alpha_vector(block_id, step_frac)         routing.py
│   │     │    │
│   │     │    └── per head h: alpha = alpha_0 * S[l, h, a] * intensity
│   │     │                            * sched(t) * alpha_scale
│   │     │
│   │     └── out[cond] += Σ_i (Δ_emb_i @ W.T) * alpha_vector_i[head_slice]
│   │            applied post-projection, weight-only, before QK norm
│   │
│   └──── on_step callback -> ref.step = step_index          pipeline.py:83
│
├── [D] TEARDOWN — finally: uninstall_head_routing(handles)   pipeline.py:95
│        restores every original linear module
│        -> return result.images[0]                          pipeline.py:97
│
└── [E] SAVING — back in smoke_test.py
    │
    ├── describe_plan(fp.last_plan, fp.last_guard)             artifacts.py
    ├── save_run(out_dir, RunRecord(...), image)            artifacts.py:130
    │      outputs/<run_id>.png       the image
    │      outputs/<run_id>.json      prompt, seed, config, plan, guard
    │                                 events, git commit, package versions
    │      outputs/manifest.jsonl     ONE APPENDED LINE per run
    │
    │   Append-only: a Kaggle session killed mid-campaign still leaves
    │   every completed run readable.
    │
    ├── writes calibration_runs/measurements.txt            smoke_test.py:118
    │      N_BLOCKS, N_HEADS, T_GEN, STEPS  <-- the campaign budget derives from these
    │
    └── summarise(out_dir)                                     artifacts.py
```

**The equation, and which file owns each term:**

```
H_{l,h} = H_{base,h} + SUM_i  alpha_i(l,h,t) * (H_{i,h} - H_{base,h})

alpha_i(l,h,t) = alpha_0  *  S[l,h,a]  *  intensity_i  *  sched(t)  *  alpha_scale
                 config.py   hasm.py       hedges.py      schedule.py   guard.py
```

---

# TREE 2 — Calibration, phase 1: which blocks matter

Answers *which blocks are worth measuring at all*, so the expensive sweep
covers ~10 blocks instead of 24.

```
scripts/calibrate.py prefilter --top-n 10                    calibrate.py:56
│
├── _pipeline(cfg)                                           calibrate.py:46
│      loads SD3.5, fp16, cpu-offload
│      HASM.uniform((0,), (0,), CORE_ATTRIBUTES)  <-- placeholder, never read;
│                                                     this phase does not route
│
├── n_blocks = len(transformer.transformer_blocks)
│   total = prompts x seeds x (1 + n_blocks)  =  3 x 1 x 25  =  75 gens
│
└── run_prefilter(...)                                       prefilter.py:81
    │
    ├── baselines = { (prompt,seed): generate(bypass=None) }  prefilter.py:90
    │      ONE baseline per (prompt, seed), reused for every block.
    │      This is why the cost is (1 + n_blocks), not (2 * n_blocks).
    │
    ├── FOR block_id in range(n_blocks):                     prefilter.py:97
    │   │
    │   ├── generate(prompt, seed, bypass=block_id)
    │   │   └── make_bypass_generate_fn                     prefilter.py:109
    │   │       └── with bypass_blocks(transformer, {id}):    patching.py:32
    │   │              block.forward = passthrough
    │   │              the block is SKIPPED; its input passes through
    │   │           └── fp.generate(..., routing=False)
    │   │
    │   └── distance_fn(baseline, bypassed)                 prefilter.py:123
    │          lpips_distance — injected, so DINOv2 can replace it
    │
    │      scores[block] = mean distance
    │      big change when removed  ==  vital block
    │
    ├── vital = top_n by score                              prefilter.py:105
    └── report.save("calibration_runs/vitality.json")        prefilter.py:43

    then printed:  report.elbow(low=6, high=12)              prefilter.py:63
        blocks scoring >= half the top block's, clamped.
        PREFER THE ELBOW over --top-n: every extra block costs a full
        column of generations and adds a noise row to the BASM.
```

---

# TREE 3 — Calibration, phase 2: build the HASM

The expensive phase. Hours. This produces the first real result.

```
scripts/calibrate.py hasm --seeds 0                          calibrate.py:93
│
├── load_corpus(data/contrastive_pairs.json)                     corpus.py
│      35 pairs. validate_corpus enforces ONE WORD of difference
│      between base and changed — two diffs would attribute the image
│      change to the wrong cause.
│
└── calibrate(...)                                             harness.py:126
    │
    │   FOR each attribute a  (plane)
    │     FOR each block l    (row)
    │       FOR each head h   (col)
    │
    ├──── _load_cell(cells/<attr>_<block>_<head>.json)          harness.py:56
    │        already done -> SKIP, reuse the raw value.
    │        missing / truncated / corrupt -> None -> recompute.
    │        THIS is what makes a 12-hour Kaggle timeout survivable.
    │
    ├──── _measure_cell(...)   [only if not cached]            harness.py:89
    │     │
    │     │  FOR each contrastive pair, FOR each seed:
    │     │
    │     ├── delta_for(attr, scorer, phrase)                 registry.py:40
    │     │      color    -> color_delta         (Delta-E in Lab)
    │     │      lighting -> lighting_delta
    │     │      size     -> size_delta          (mask area ratio)
    │     │      texture  -> gram_texture_delta  (Gram matrix)
    │     │      identity -> identity_delta     ] CLIP
    │     │      style    -> style_delta        ] embedding
    │     │      action   -> action_delta       ] based
    │     │
    │     ├── baseline = generate(pair.base, seed, swap=None)
    │     │
    │     ├── swapped  = generate(pair.base, seed,
    │     │                       swap=SwapSpec(HeadUnit(block, head), pair.changed))
    │     │   └── make_swap_generate_fn                       harness.py:175
    │     │          alpha_0=1.0 AND t_window=(0.0,1.0), so
    │     │            H = H_base + 1.0*(H_changed - H_base) = H_changed
    │     │          i.e. an exact replacement at ONE head, reusing the
    │     │          routing machinery instead of a second injection path
    │     │          that could drift from it.
    │     │
    │     ├── mask = masker(baseline, pair.object_label)        masking.py
    │     │      ClipSeg. Confines the measurement to the object, so a
    │     │      background change cannot be scored as a colour change.
    │     │
    │     └── SIZE IS SPECIAL                                 harness.py:112
    │            re-segments the SWAPPED image too. Against one shared
    │            mask, area change always reads exactly 0.
    │
    │        -> mean over all pairs = the cell's RAW value
    │
    ├──── _save_cell(...)                                      harness.py:64
    │        writes the RAW value, not the normalised one — normalisation
    │        depends on the whole plane, which is not finished yet.
    │
    ├──── _normalise(plane)  after each attribute plane completes harness.py:81
    │        min-max onto [0,1]. A flat plane -> all zeros, not a
    │        spurious peak.
    │
    └── hasm.save("calibration_runs/hasm.npz")                    hasm.py
        + hasm.to_basm().save("calibration_runs/basm.npz")
        + WARNING if every attribute peaks on the same block
                                                              calibrate.py:123
```

---

# TREE 4 — Local, no GPU

```
scripts/explain.py "a very red car" --save outputs/
│
├── HASM.uniform(...)              synthetic; there is no real one yet
├── parse_prompt(...)              parsing.py     <-- real
├── resolve_components(...)        fuzzy/         <-- real
├── build_routing_plan(...)        routing.py     <-- real
└── prints components, hedges, intensities, chosen head units, alpha decay

    NO text encoder, NO diffusion, NO image.
    Every DECISION FLAIR makes is CPU-only; the GPU only turns those
    decisions into pixels.
```

---

# The order across the whole project

```
 1. ./run-local.sh test              full suite           local   4s
 2. verify_env.py                    imports OK           Kaggle  1m
 3. verify_api.py                    19 diffusers checks  Kaggle  1m
 4. smoke_test.py                    images, N_BLOCKS, N_HEADS, T_GEN
 5. (budget arithmetic)                                   local   2m
 6. calibrate.py prefilter           vitality.json        Kaggle  20m (for FLUX)
 7. calibrate.py hasm                hasm.npz, basm.npz   Kaggle  2-6h
 8. (the Week 4 gate: distinct peaks per attribute?)
 9. smoke_test.py --hasm hasm.npz    the first REAL images
10. bundle + download                everything survives the session
```

Steps 4 -> 5 -> 7 cannot be reordered: 4 measures the numbers 5 needs,
and 7 produces the files 9 requires.

See `RUNBOOK.md` for the commands and what to check at each step.
