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

**D2 — rebalance vs class-weighting (running).** `verify_seeds.py` with the
`*_nw` conditions (class weighting off). Results pending.

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
