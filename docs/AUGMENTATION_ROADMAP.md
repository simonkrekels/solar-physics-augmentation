# Augmentation roadmap — clearing the oversample bar

## Where we are

Rigorous evaluation (`EXPERIMENTS.md` → *Honest baselines*) showed the current
heat-equation augmentation is **statistically indistinguishable from duplicating
real images**, and does **not** improve recall on the rare fault classes it
targets. So the bar any future augmentation must clear is concrete and measured:

> **Beat `oversample` (and `clean`) on rare-class recall, with a paired-bootstrap
> CI that excludes 0, without sacrificing accuracy.**
> Current rare-class recall (3-seed mean): clean **0.612**, oversample 0.574,
> randaugment 0.615, physics 0.569. Nothing yet beats `clean` meaningfully.

Every idea below is measured the same way: `training/verify_seeds.py` with a new
`--conditions` entry, reporting per-class recall CIs and paired deltas.

## Two hypotheses for *why* it fails (diagnose before building)

The negative result has two plausible causes; the cheap diagnostics tell us which
to invest in.

- **H1 — domain gap.** Synthetic images look detectably "fake" (additive RGB
  blend, smooth steady-state blobs), so their features don't transfer to real
  test faults — or the model uses "syntheticness" as a shortcut.
- **H2 — rebalancing ↔ class-weighting interaction.** The loss already uses
  inverse-frequency weights. Topping rare classes up to 500 makes them *less*
  rare, softening their upweighting and shifting the model toward the majority
  class. This is consistent with the observed "accuracy ↑, rare-recall ↓" and
  with oversample behaving identically.

### Diagnostics first (cheap, do these before any generator work)

| # | Experiment | Answers | Cost |
|---|---|---|---|
| D1 | Train a **synthetic-vs-real discriminator** on the synthetic set | H1: if AUC ≈ 1.0, realism is *the* problem | 1 short run |
| D2 | **Decouple rebalance from weighting**: augment with class weights *off* | H2: does the interaction explain the rare-recall drop? | reuse harness |
| D3 | **Quantity/fraction sweep** (target_min ∈ {250, 500, 1000}) | confirms diminishing/negative returns from volume | reuse harness |

If D2 alone recovers rare-class recall, the generator may be fine and the fix is
*how we train with it* — much cheaper than rebuilding the physics.

### Diagnostic results

**D1 — domain gap is LARGE and structural (H1 confirmed).**
`training/diagnose_domain_gap.py` trains an EfficientNet-B0 to separate synthetic
from real images of the same four classes (balanced; real images JPEG-round-tripped
to match the synthetics' extra compression generation).

| Low-pass blur | Best val AUC |
|---|---|
| none | **0.966** |
| radius 2 | 0.934 |
| radius 4 | 0.867 |

The discriminator assigns P(synthetic) ≈ 0.85–0.97 to synthetics vs 0.09–0.24 to
reals. Crucially the gap **survives heavy blur** (AUC 0.87 at radius 4), so it is
not seams or JPEG artifacts — it is the **coarse shape/structure** of the
steady-state heat blobs. This directly explains the negative result: the
synthetic faults don't look like real faults, so their features don't transfer.
→ Realism work (Thrust A, especially #1 time-dependent and #3 structure-aware) is
*required*, and cosmetic fixes (blending/compression) won't be enough.

**D2 — rebalance vs class-weighting: H2 confirmed.** Three `*_nw` conditions
(class weighting OFF), 3 seeds, appended for a unified 7-condition analysis.

| Condition | Accuracy | Macro recall | Rare-class recall |
|---|---|---|---|
| clean (weights on) | 0.782 | 0.699 | **0.612** |
| oversample | 0.807 | 0.697 | 0.574 |
| physics | 0.800 | 0.693 | 0.569 |
| clean_nw | 0.839 | 0.684 | 0.560 |
| oversample_nw | 0.826 | 0.680 | 0.582 |
| physics_nw | 0.837 | 0.686 | 0.568 |

Findings:
1. **Class weighting is the single most effective lever for rare classes.**
   `clean` with weights on (0.612 rare recall) beats *every* other condition,
   weighted or not. Turning weights off drops rare recall (clean→clean_nw
   −0.052, CI excludes 0) — exactly what the weighting is for.
2. **It trades accuracy for rare recall.** Weights off lifts accuracy 0.78→0.84
   and macro-F1 0.66→0.70 (No-Anomaly recall 0.85→0.97), at the cost of rare
   recall. A clean precision/recall knob.
3. **Data rebalancing is a weaker *substitute* for weighting, not a complement.**
   With weights off, rebalancing nudges rare recall up (clean_nw 0.560 →
   oversample_nw 0.582 / physics_nw 0.568, not significant). But *stacking*
   rebalancing on top of weighting **hurts** (clean 0.612 → oversample/physics
   0.574/0.569). The two over-correct together, pushing toward the majority.
4. **Physics still ≈ duplication** in every regime (physics_nw 0.568 ≈
   oversample_nw 0.582) — consistent with D1: the synthetic content carries no
   real rare-fault signal.

**Combined D1 + D2 diagnosis.** The current augmentation fails for *two
independent* reasons: the synthetic images are structurally fake (D1, AUC 0.97),
**and** its rebalancing role is redundant-to-harmful given class weighting
already handles imbalance more effectively (D2). To be useful, a generator must
(a) add *real* rare-fault signal — i.e. close the D1 gap (→ v2 below) — and (b)
be deployed without fighting the class weights (e.g. weight-off, or mixed at a
controlled fraction).

## v2 generator — first iteration

`augmentation/heat_equation_v2.py` (`SyntheticAugmenterV2`, writes to
`data/processed/synthetic_v2/`). Targets the four defects D1 + visual inspection
exposed (see `docs/montage_v2.png`, real | old | v2):

| Defect (v1) | v2 fix |
|---|---|
| Steady-state → over-diffused round blobs | **transient** diffusion, finite sampled time → localized, shape-preserving |
| 64×64 square grid resized to 24×40 (aspect warp) | solve at **native aspect**, super-sampled |
| Implausibly smooth, no sensor noise | **grain calibrated** to real high-freq residual (σ≈3.7) |
| Soiling = smooth *darkening*; Diode = hard bright stripes | Soiling = **bright streaky** edge-weighted patches; Diode = **faint** soft bands; per-class **contrast calibrated** to real (Hot-Spot 22, Soiling 61, …) |

**Status: v2 looked better to the eye but FAILED the discriminator test.**
This is the value of D1-as-a-metric — visual realism was misleading.

| Generator | Discriminator AUC |
|---|---|
| v1 (steady-state blobs) | 0.966 |
| v2 **with** added iid grain | **1.000** |
| v2 **without** grain (ablation) | 0.934 |

1. **The added grain backfired (confirmed).** Removing it dropped AUC 1.00→0.93
   — the predicted double-counting: the base already carries real,
   spatially-correlated sensor noise, so iid grain is a giveaway. `add_grain` is
   now off by default.
2. **The structural fixes didn't close the gap.** v2-no-grain (0.934) ≈ v1
   (0.966), likely within noise — both remain *trivially separable* (AUC > 0.9).
   Per class, Diode-Multi is closest to real (P(syn)≈0.67), the multi-spot
   classes least (Hot-Spot-Multi ≈ 1.0).

**Implication.** Additive blending of a smooth synthetic field onto a real
No-Anomaly base is near a detectability ceiling at 24×40: incremental geometry
tweaks give marginal gains. Closing the gap likely needs a *different paradigm*,
not a better blob — candidates: (a) **perturb real fault images** instead of
synthesising onto clean bases; (b) **match noise spatial-correlation** rather
than adding iid; (c) treat synthetic as **pretraining only**; or (d) accept that
the rigorous negative result — class weighting beats synthetic rebalancing here —
*is* the finding. The expensive downstream test (`--conditions physics_v2`) is
not worth running until a generator first clears the discriminator bar.

## Thrust A — make the physics earn its place (realism)

Ordered by expected leverage. Each is independently A/B-testable.

1. **Time-dependent solve.** Integrate ∂T/∂t = α∇²T + q and sample at several
   times → a *distribution* of signatures (sharp/localized early → diffuse late)
   instead of the single t→∞ equilibrium. Directly attacks the diversity deficit
   that makes "more synthetic data" useless. *High leverage, medium effort.*
2. **Temperature-space + sensor transfer-function blending.** Stop adding a
   normalized field to RGB. Add ΔT in temperature space, then apply the
   radiometric → colormap mapping the camera uses. Removes the additive-RGB
   artifact a discriminator (D1) would key on. *High leverage if D1 says gap is
   large; medium effort.*
3. **Structure-aware sources & conduction.** Encode the module's cell/busbar
   grid into a spatially-varying κ and into source geometry (diode short heats a
   cell *string*; cracks are thermal discontinuities; cell defects align to the
   lattice). Most relevant to Cell / Cell-Multi / Cracking. *High leverage,
   higher effort (needs module geometry model).*
4. **Soiling = correlated noise, not a blob.** Replace the "uniform cooling
   blob" with masked spatially-correlated noise (Perlin/simplex), edge/bottom
   weighted to match real dust/streak deposition. Soiling is the worst class
   everywhere (0.29–0.42); a targeted, physically-motivated win. *Medium leverage,
   low effort — good first realism experiment.*
5. **Robin boundary condition.** Replace pure-Neumann (which forces the
   non-physical per-step mean-subtraction) with a convective loss −h(T−T_amb).
   More physical, better-posed, localizes hot spots. *Low effort, modest leverage.*
6. **Calibrate parameters to real statistics.** `intensity_scale=40`, `kappa=1`,
   source sizes are currently hand-set. Fit hot-spot size/contrast distributions
   from the *real* rare-class images so synthetic parameters match reality.
   *Low effort, de-risks all of the above.*

## Thrust B — training integration

7. **Synthetic-as-pretraining, not training data.** Use synthetic images for
   self-supervised / contrastive *representation* pretraining, then fine-tune on
   real only. Synthetic-as-pretraining transfers better than synthetic-as-labels
   when there's a domain gap.
8. **Copy-paste / Mixup baselines.** Paste *real* rare-fault crops onto real
   backgrounds, and Mixup synthetic↔real. A strong "physics-lite" competitor —
   and the honest bar physics should beat to justify the PDE machinery.
9. **Domain-adversarial alignment** if D1 confirms a large gap: train features to
   be invariant to synthetic-vs-real while classifying the fault.

## Honest framing

It is a live possibility that **on a dataset this small, no generator beats
real-image duplication / copy-paste**, and the correct product conclusion is
"label more real rare faults." The roadmap is structured so the cheap diagnostics
(D1–D3) and the cheap realism win (#4 Soiling) come first — if those don't move
rare-class recall above `clean` on the harness, that negative result is itself a
publishable, portfolio-worthy finding, and we stop before over-investing.

## Suggested first sprint

1. D1 (discriminator) + D2 (decouple rebalance/weighting) — diagnose H1 vs H2.
2. #4 Soiling correlated-noise source — cheapest realism experiment, worst class.
3. Whichever of #1 (time-dependent) / #2 (temperature-space) the diagnosis favors.

All scored against `clean`/`oversample` on rare-class recall with CIs.
