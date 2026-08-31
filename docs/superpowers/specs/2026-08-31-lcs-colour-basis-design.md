# FLAIR — Latent Colour Subspace (LCS) Basis for SD3.5

**Status:** Design, awaiting approval
**Date:** 2026-08-31
**Sibling:** [`2026-08-31-adain-latent-split-design.md`](./2026-08-31-adain-latent-split-design.md) — the zero-calibration baseline this metric must outperform.
**Amended by:** [`2026-08-31-colour-localization-research.md`](./2026-08-31-colour-localization-research.md) §4 and §8 — the higher-value application of this basis is projecting per-head **V-token** changes, not final-latent changes. The Stage A fitting procedure below is unchanged; the Stage B scoring input is.
**Precondition:** Tasks 1-16b complete, 175 tests passing. The sibling spec's phase-1 test harness is expected to land first and is reused here.

---

## 1. What this builds, in one paragraph

A frozen `16 × 3` projection matrix that maps SD3.5's VAE latent channels onto
**Hue, Saturation and Lightness** axes, fitted once offline from synthetic colour
swatches, and used thereafter to decompose any latent difference `Δz` into a
colour component and a structural residual. Scoring is one matmul and never
touches the VAE. Unlike the sibling AdaIN metric, the axes carry perceptual
labels, so the metric reports not merely *how much* colour moved but **in which
direction** — recovering in latent space the readout that CIELAB gives in pixel
space.

## 2. Why this is possible, and what the literature already settled

Two recent results establish that colour is not chaotically distributed across
VAE latent channels but occupies interpretable structure.

**[The Latent Color Subspace: Emergent Order in High-Dimensional Chaos](https://arxiv.org/abs/2603.12261)**
— Pach, Bader, Bouniot, Belongie, Akata. Interprets colour representation in the
VAE latent space of **FLUX.1 [Dev]**, finding structure that reflects Hue,
Saturation and Lightness, and verifies the interpretation by using it for
training-free, closed-form colour control. FLUX.1 and SD3.5 both use a
**16-channel** VAE, so the setting transfers.

**[Color encoding in Latent Space of Stable Diffusion Models](https://arxiv.org/abs/2512.09477)**
— Arias, Solà, Armengod, Vanrell. Uses controlled synthetic colour datasets plus
PCA to show colour is encoded along **circular, opponent axes**, concentrated in
channels c₃/c₄, while intensity and shape occupy c₁/c₂ (in 4-channel SD).

### 2.1 The constraint this respects

The project constraint is **never use the decoder**. Both papers derive their
basis by pushing known-colour images through the VAE **encoder** and running
PCA. The encoder is a different network from the decoder, it runs **once,
offline, during fitting**, and it never appears in the scoring loop. After
fitting, the metric is a frozen matrix — the decoder is never invoked, at fit
time or at score time.

### 2.2 What is new here

Neither paper does what this spec needs. LCS demonstrates on FLUX.1, not SD3.5,
and uses the subspace for *control*, not *measurement*. Arias et al. work on
4-channel SD. Replicating the LCS derivation on SD3.5-medium and turning it into
a **calibration metric for block sensitivity** is the contribution, and §5's
circularity gate is what makes the replication falsifiable rather than assumed.

## 3. The mechanism

### 3.1 Stage A — fit the basis (offline, once)

**A1. Synthesise the calibration set.** Sample an HSV grid — 24 hues × 5
saturations × 5 values = 600 patches at 512×512. Include both *flat* swatches and
*textured* patches at matched hues. The flat patches isolate colour with no
structure; the textured ones exist to verify the recovered axes are
texture-invariant rather than an artefact of flat images being out of
distribution for the VAE.

**A2. Encode and collapse space.** Push each patch through
`vae.encode(...).latent_dist.mean` → `[16, 64, 64]`, and apply the pipeline's
own convention. Decoding at `scripts/test_latent_color_blocks.py:110` does
`(latent / scaling_factor) + shift_factor`, so encoding is its inverse:

```
z = (raw_latent − shift_factor) * scaling_factor
```

A flat patch has no spatial structure, so its per-channel **spatial mean** *is*
its colour code. Collapse to one 16-vector per patch and stack:
`X ∈ R^{600×16}`.

**A3. Centre and decompose.** Subtract the column mean, then SVD. Expect the
first ~3 components to carry the large majority of variance.

**A4. Gate — verify the axes are actually colour.** Regress each component
against the known H/S/V that generated each patch, and check three things:

1. explained variance of the top 3 components,
2. R² of lightness against PC1,
3. **hue circularity** — projected onto the PC2–PC3 plane, hue angle must be
   monotonic and close a full 360°.

Item 3 is the decisive one: it is the direct test of the "circular, opponent
axes" finding, and it cannot be satisfied by accident. Emit a scatter plot of the
PC2–PC3 plane coloured by true hue as the visual artefact.

**This gate is a hard stop.** If hue does not close a loop, SD3.5's VAE is not
FLUX-like in the relevant sense, this plan is abandoned, and the sibling AdaIN
metric ships instead. The gate costs minutes and de-risks everything downstream.

**A5. Label, scale, freeze.** Assign axis names by correlation with V, hue angle
and saturation radius; scale the axes so a unit step is comparable across them.
Serialise to `calibration_runs/lcs_basis.npz` with a version field.

### 3.2 Stage B — score (in the sweep, no VAE)

For `Δz = z_swap − z_base` at each position `p` inside mask `M`:

```
c_p = P_colourᵀ · Δz_p          ∈ R³      # colour component
r_p = Δz_p − P_colour · c_p     ∈ R¹⁶     # structural residual

colour_delta    = mean_{p∈M} ‖c_p‖
structure_delta = mean_{p∈M} ‖r_p‖
colour_purity   = colour_delta / (colour_delta + structure_delta)
```

Report `‖mean_p c_p‖` alongside `mean_p ‖c_p‖`. Their ratio distinguishes a
**uniform** recolour from a **patchy** one — the first is a coherent shift, the
second averages toward zero.

### 3.3 The directional readout

Because the axes are labelled, the metric additionally returns:

| quantity | computation |
|---|---|
| **ΔLightness** | `mean_p c_p[0]` |
| **ΔHue** | angular difference in the `(c[1], c[2])` plane |
| **ΔSaturation** | radius change in that plane |

This is the readout CIELAB provides in pixel space and that is otherwise lost on
moving to latents. It is the substantive reason this spec exists rather than
stopping at the sibling.

## 4. Files

| Path | Change |
|---|---|
| `flair_t2i/metrics/latent_color.py` | extend — `LatentColorBasis`, `fit_colour_basis()`, `project_delta()`, `latent_colour_delta()` |
| `scripts/fit_latent_color_basis.py` | **new** — Stage A driver, writes `calibration_runs/lcs_basis.npz` |
| `tests/test_latent_color_basis.py` | **new** — swatch and subspace-recovery tests |
| `scripts/test_latent_color_blocks.py` | emit the new columns and the directional readout |

`LatentColorBasis` is a frozen dataclass: `matrix` (16×k), `mean` (16,),
`axis_names`, `explained_variance`, `version`.

Per `CLAUDE.md`, `calibration_runs/` is gitignored; the basis npz is committed
explicitly with `git add -f`, matching the existing treatment of
`calibration_runs/basm.npz`.

## 5. Build steps

1. **Swatch generation** — `make_colour_swatches(n_hue, n_sat, n_val, size, textured)` → `[(image, (h,s,v))]`. Pure CPU. Test count and HSV round-trip.
2. **Encode and collapse** — test against a stub VAE, as `flair_t2i/latents.py` already does for decoding.
3. **Fit** — centre, SVD, top-k. **Test on synthetic data**: construct `X` in which 3 known directions carry HSV and the remaining 13 are noise, then assert the recovered subspace matches the planted one via principal angles. This validates the fitting code independently of whether SD3.5 cooperates.
4. **The A4 gate** — `validate_basis()` returning the three diagnostics plus the PC2–PC3 plot. Hard stop on failure.
5. **Label and scale axes.**
6. **Serialise** with a version field; round-trip test on the loader.
7. **Scoring functions** — `project_delta()` and the scalar set plus directional readout.
8. **Wire into the sweep script.**
9. **Real-pair validation** (§6).

Region gating (cross-attention mask on the 64×64 grid, DAAM-style) is shared
with the sibling spec and is deferred to the same phase-2 position there, for the
same reason: validate the metric core ungated first.

## 6. Acceptance

**Fitting:**
- A4 gate passes — hue closes a circle in the PC2–PC3 plane.
- Synthetic subspace-recovery test green.

**Real pairs**, same seed, no routing:

| pair | expected |
|---|---|
| `"a red car"` → `"a blue car"` | large ΔHue, small `structure_delta` |
| `"a red car"` → `"a red truck"` | small ΔHue, large `structure_delta` |
| `"a red car"` → `"a dark red car"` | **large ΔLightness, ΔHue ≈ 0** |

The third row is the decisive experiment. The sibling AdaIN metric cannot pass
it — it reports the same appearance signature for a lightness change as for a
hue change (sibling spec §7). Passing it is what justifies this spec existing,
and failing it means the extra machinery bought nothing.

## 7. Cost

| | |
|---|---|
| Fitting | ~600 encoder calls, one time, minutes |
| Per-block scoring | one 16×3 matmul — negligible |
| Validation | ~8 generations |

The basis is fitted once and reused across every block, head, prompt and
attribute in the BASM/HASM calibration campaign. The 24-block sweep gets no
slower than it is today.

## 8. Risks

| Risk | Mitigation |
|---|---|
| SD3.5's VAE may not organise colour as cleanly as FLUX.1 | §3.1 A4 is a hard gate costing minutes; failure routes to the sibling spec |
| Flat swatches are out of distribution for the VAE | Textured patches at matched hues included in the fit; axes must agree across both |
| Colour and lightness are never fully separable in any latent | Measure the leakage and report it; do not claim orthogonality the data does not support |
| Ungated metrics mix in background change | Phase-2 cross-attention masking; the same defect was already fixed in pixel space (`fix: gate the region the metric reads`) |

## 9. References

- Pach, Bader, Bouniot, Belongie & Akata, *The Latent Color Subspace: Emergent Order in High-Dimensional Chaos* — [arXiv:2603.12261](https://arxiv.org/abs/2603.12261)
- Arias, Solà, Armengod & Vanrell, *Color encoding in Latent Space of Stable Diffusion Models* — [arXiv:2512.09477](https://arxiv.org/abs/2512.09477)
