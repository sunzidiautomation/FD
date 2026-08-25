# FLAIR Runbook — every step, in order, through Task 16b

Follow top to bottom. Do not skip ahead: each phase produces something the
next one needs.

**Legend:** 💻 runs locally (no GPU) · ☁️ runs on Kaggle (GPU) · ⏱ rough time

---

## Phase 0 · Local verification 💻 ⏱ 5 min

Confirm the code is sound before spending any GPU quota.

### 0.1 Build the image (once)

```bash
cd C:/FD
docker build -t flair-test .
```

First build ~2 min (downloads CPU torch + spaCy). Later builds are seconds.

### 0.2 Run the tests

```bash
./run-local.sh test
```

✅ **Expect:** `175 passed`
❌ **If anything fails:** stop. Do not go to Kaggle with a red suite.

### 0.3 Look at a routing plan

```bash
./run-local.sh explain "A small red sports car under warm evening light"
```

✅ **Expect:** 4 parsed components; each routed to a different block; α
decaying to 0 partway through the steps.

This uses a **synthetic** BASM — it shows the machinery, not real
sensitivities. Records are saved to `C:\FD\outputs\`.

---

## Phase 1 · Kaggle setup ☁️ ⏱ 10 min, once only

### 1.1 Accept the model licence

Open the **stabilityai/stable-diffusion-3.5-medium** page on Hugging Face,
sign in, and accept the licence. **SD3.5 is a gated model — every download
fails with 403 until you do this.** This is the single most common way this
phase fails.

### 1.2 Create a Hugging Face token

HF → Settings → Access Tokens → New token, **read** scope. Copy it.

### 1.3 New Kaggle notebook

- **Settings → Accelerator → GPU T4 ×2** (or P100)
- **Settings → Internet → On**
- **Add-ons → Secrets → Add secret**, name it exactly `HF_TOKEN`, paste the token

### 1.4 Open the notebook

Upload `notebooks/flair_kaggle.ipynb`, or paste the cells below by hand.

---

## Phase 2 · Install ☁️ ⏱ 5 min

### 2.1 Token + clone + install

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

!git clone -q https://github.com/sunzidiautomation/FD.git /kaggle/working/FD
%cd /kaggle/working/FD
!pip install -q -r requirements.txt
!python -m spacy download en_core_web_sm -q
```

### 2.2 ⚠️ RESTART THE SESSION

**Run → Restart session.** Not optional.

Newly installed packages are invisible to the already-running kernel. This
is exactly what produced the earlier `ModuleNotFoundError: No module named
'lpips'` / `'skfuzzy'` — the install had worked; the kernel just could not
see it.

### 2.3 Verify

```python
%cd /kaggle/working/FD
!python scripts/verify_env.py
!python scripts/verify_api.py
!python -m pytest -q
```

✅ **Expect:** `Environment OK.` · `All 12 checks passed` · `175 passed`
❌ **`verify_env` fails:** a package is missing — `!pip install -q <name>`,
restart again.
❌ **`verify_api` fails:** diffusers changed its internals. The output names
which assumption broke.

---

## Phase 3 · Smoke test ☁️ ⏱ 15 min

The first images, and the two numbers everything downstream depends on.

```python
!python scripts/smoke_test.py --steps 20 --seed 0 --out /kaggle/working/outputs
```

Downloads ~5GB of weights on first run.

✅ **Expect output like:**
```
N_BLOCKS = 24
BASM     uniform (UNCALIBRATED placeholder)
T_GEN = 12.4s  (one 20-step generation)
routed components: ['c_identity', 'c_color', 'c_size', 'c_lighting']
blocks touched:    [0]
```

📝 **Write down `N_BLOCKS` and `T_GEN`.** Phase 4's budget comes from them.

### How to judge these images

The BASM is uncalibrated — all cells 0.5, ties broken by lowest block id, so
**every attribute routes to block 0**. This proves the *plumbing*, nothing
more.

- ✅ five images exist, none is noise
- ✅ baseline and routed differ
- ✅ log shows 4 components, `blocks touched: [0]`
- ❌ **do not** judge colour or size fidelity — nothing is calibrated yet

❌ **403 / gated repo:** step 1.1 was skipped.
❌ **CUDA out of memory:** confirm the GPU accelerator is actually on.

---

## Phase 4 · Choose the budget 💻 ⏱ 2 min

```python
N_BLOCKS = 24     # from Phase 3
T_GEN    = 12.4   # from Phase 3

A, B, HOURS = 7, 10, 18          # attributes, blocks to keep, GPU-hours
budget = (HOURS * 3600) / (A * (1 + B) * T_GEN)
print(f"P x S must be <= {budget:.1f}")
for P, S in [(5,1), (10,1), (10,2)]:
    print(f"  P={P}, S={S} -> {A*P*S*(1+B)} generations, "
          f"{A*P*S*(1+B)*T_GEN/3600:.1f} GPU-hours")
```

Pick the largest `P × S` that fits. The shipped corpus has **P = 5**; more
pairs means adding them to `data/contrastive_pairs.json`.

**Start with `--seeds 0` (S=1).** Add a second seed only if the budget
allows comfortably.

---

## Phase 5 · Vital-layer prefilter ☁️ ⏱ 20 min

Narrows 24 blocks to ~10. Cheap — about 75 generations.

```python
!python scripts/calibrate.py prefilter --top-n 10 --out calibration_runs/
```

✅ **Expect:**
```
vitality, most vital first:
  B12  0.4821
  B9   0.4033
  ...
--top-n 10 kept: (2, 5, 7, 9, 11, 12, 15, 18, 20, 22)
elbow rule suggests:  (5, 7, 9, 11, 12, 18)
```

**If the elbow suggestion differs from `--top-n`, prefer the elbow** and
re-run with that count — every extra block costs a full column of
generations and adds a noise row to the BASM.

Writes `calibration_runs/vitality.json`.

---

## Phase 6 · BASM calibration ☁️ ⏱ 2-6 hours

The expensive phase, and the one that produces the first real result.

```python
!python scripts/calibrate.py basm \
    --vitality calibration_runs/vitality.json \
    --seeds 0 \
    --out calibration_runs/
```

✅ **Expect a running log:**
```
  identity  B5   raw=0.0312
  identity  B7   raw=0.1847
  color     B5   raw=0.0203
  ...
```

### If the session dies

**Just run the exact same command again.** Every completed
(attribute, block) cell is checkpointed to `calibration_runs/cells/` and
skipped on resume. A 12-hour timeout costs you the current cell, not the
campaign.

```python
!ls calibration_runs/cells/ | wc -l     # cells done so far
```

Total cells = attributes × blocks (7 × 10 = 70).

Writes `calibration_runs/basm.npz`.

---

## Phase 7 · The Week 4 gate ☁️ ⏱ 5 min

**Check this before building anything on the matrix.** A BASM that fails
here is measuring noise, and every downstream number would be meaningless.

```python
import numpy as np
from flair_t2i.basm import BASM

basm = BASM.load("calibration_runs/basm.npz")

print(f"{'attribute':<10} {'top blocks':<28} {'max':>6}")
for attr in basm.attributes:
    top = basm.top_k(attr, 3)
    print(f"{attr.value:<10} {str(top):<28} {basm.matrix[:, basm.attributes.index(attr)].max():.3f}")

peaks = {a: basm.top_k(a, 1)[0][0] for a in basm.attributes}
print(f"\npeak block per attribute: {peaks}")
print(f"distinct peaks: {len(set(peaks.values()))} of {len(peaks)}")
```

**Three checks:**

1. **No dead column.** No attribute may be all-zero after normalisation.
2. **Selectivity.** For the 4 core attributes, the top block should clearly
   exceed the column median — not a flat field.
3. **Distinct peaks.** ⚠️ **If Color, Size, and Lighting all peak on the same
   block, there is no disentanglement to exploit and the paper's premise is
   in trouble.** `calibrate.py` prints a warning when this happens.

Check 3 failing is an early signal of a Week 7 no-go. Raise it then, not
after evaluation.

---

## Phase 8 · Generate with the real BASM ☁️ ⏱ 15 min

Now routing means something.

```python
!python scripts/smoke_test.py --steps 20 --seed 0 \
    --basm calibration_runs/basm.npz \
    --tag calibrated \
    --out /kaggle/working/outputs
```

✅ **Expect** `blocks touched:` to list **several distinct blocks**, not
`[0]`. Compare `calibrated_routed.png` against `smoke_baseline.png`.

This is the first image that demonstrates the method rather than the
plumbing. It is still not paper-quality — that needs Week 5 α₀ / timestep
tuning.

---

## Phase 9 · Save everything ☁️ ⏱ 5 min

Kaggle deletes the session on shutdown.

```python
import shutil
from pathlib import Path
from flair_t2i.artifacts import summarise

print(summarise("/kaggle/working/outputs"))

bundle = Path("/kaggle/working/flair_results")
bundle.mkdir(exist_ok=True)
for src in [Path("/kaggle/working/outputs"),
            Path("/kaggle/working/FD/calibration_runs")]:
    if src.exists():
        shutil.copytree(src, bundle / src.name, dirs_exist_ok=True)

print(shutil.make_archive("/kaggle/working/flair_results", "zip", bundle))
```

Then **Save Version**, and download `flair_results.zip`.

📌 **`basm.npz` is the expensive artefact** — hours of GPU. Commit it:

```bash
# locally, after downloading
cp ~/Downloads/basm.npz C:/FD/calibration_runs/
cd C:/FD && git add calibration_runs/basm.npz
git commit -m "data: calibrated BASM for SD3.5-M"
git push
```

---

## Quick reference

| Phase | Where | Command | Time |
|---|---|---|---|
| 0 | 💻 | `./run-local.sh test` | 5 min |
| 1 | ☁️ | licence + token + GPU on | 10 min |
| 2 | ☁️ | clone, install, **restart**, verify | 5 min |
| 3 | ☁️ | `smoke_test.py` → N_BLOCKS, T_GEN | 15 min |
| 4 | 💻 | budget arithmetic | 2 min |
| 5 | ☁️ | `calibrate.py prefilter` | 20 min |
| 6 | ☁️ | `calibrate.py basm` (resumable) | 2-6 h |
| 7 | ☁️ | **Week 4 gate** — distinct peaks? | 5 min |
| 8 | ☁️ | `smoke_test.py --basm ...` | 15 min |
| 9 | ☁️ | bundle + Save Version | 5 min |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `403` on model download | licence not accepted | Phase 1.1 |
| `ModuleNotFoundError` after install | kernel not restarted | Phase 2.2 |
| `verify_api` fails | diffusers changed | pin `diffusers==0.39.0` |
| `CUDA out of memory` | GPU off, or too many steps | check accelerator; `--steps 16` |
| Session died in Phase 6 | 12-hour cap | re-run the same command — it resumes |
| Every attribute peaks on one block | **the premise may not hold** | Phase 7, check 3 |
| Calibration too slow | budget too large | fewer blocks (`--top-n`), one seed |

## After Task 16b

Tasks 17-19 (evaluation) get planned at Week 6, once the BASM is real —
their design depends on what Phase 7 shows. See
`docs/superpowers/plans/2026-08-25-flair-master-roadmap.md`.
