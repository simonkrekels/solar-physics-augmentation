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

## SimGAN refiner — attempted, did not clear the realism gate

`augmentation/refiner.py`: a residual refiner $R$ (physics proposes structure,
$R$ learns appearance) trained adversarially + self-reg $\lambda\lVert R(x)-x\rVert_1$,
then scored by an **independent** discriminator (Stage-1 gate; target AUC $\le 0.75$).

| Variant | What $R$ did | Stage-1 AUC |
|---|---|---|
| v2 no-grain (input) | — | 0.934 |
| refiner, **patch** D | collapsed to near-identity | $\approx$0.93 (no change) |
| refiner, **global** D | added frame borders + banding artifacts | **0.980 (worse)** |

**It failed, for diagnosable reasons:**
1. **The in-loop discriminator never learned to separate** (stayed at chance,
   both patch and global). At $24\times40$, a small from-scratch $D$ cannot
   perceive the gap that the pretrained 224-px EfficientNet scorer sees — so $R$
   trained against an uninformative adversary and introduced *new* artifacts the
   scorer trivially catches.
2. **SimGAN refines appearance, but D1 showed the gap is *structural*** (coarse
   shape, survives blur). A self-reg-constrained refiner cannot change coarse
   structure; freed (low $\lambda$, global $D$) it adds spurious structure.
3. **~612 real images** make the adversarial game unstable.

**Conclusion.** Learned refinement *on top of the additive-blend paradigm* does
not close the gap — the second realism attempt to fail (after v2). The remaining
candidates are genuine paradigm shifts: **perturb real fault images** (not
synthesise onto clean bases); **synthetic-as-pretraining-only**; or accept the
well-characterised negative result. A stronger in-loop adversary (a 224-px
EfficientNet $D$) is the one untried lever, but the structural-gap argument
predicts limited payoff.

## Perturbing real faults — Stage 1 CLEARED the realism gate

`augmentation/perturb.py` (`RealFaultAugmenter`): start from a *real* fault crop
and apply the heat equation to the fault itself. The heat-evolution operator is
Gaussian diffusion, so we decompose `image = background + fault + grain`, then
diffuse (thermal time) and rescale (severity) only the fault, keeping the *real*
background and grain. Realistic by construction; diverse in physically meaningful
axes (severity, spread, orientation).

**Stage 1 (realism) — passed.** Discriminator real-vs-perturbed AUC **0.711**
(target ≤0.75), vs 0.93–0.98 for every synthesise-onto-clean-base approach. The
gap is finally closed (Soiling ≈ chance, 0.52). This is the first generator to
clear the bar — confirming the diagnosis that the gap was *structural* and that
starting from real structure is the fix.

**Stage 2 (utility) — FAILED: realism did not translate to utility.**

| Condition | Rare-class recall (95% CI) |
|---|---|
| clean (weights on) | **0.612** [0.55, 0.68] — best overall |
| oversample | 0.574 |
| perturb | 0.572 |
| oversample_nw | 0.582 |
| perturb_nw | 0.572 |

`perturb − oversample` rare-recall $= -0.0015$, CI $[-0.038, +0.035]$ — dead even.
In **both** weight regimes, physically perturbing real faults is statistically
indistinguishable from simply duplicating them, and below plain `clean`.

**The definitive conclusion.** We made augmentation *realistic* (perturb cleared
the Stage-1 gate at AUC 0.711) and it *still* gave no rare-class benefit over
duplication. So augmentation quality was never the bottleneck. The limiting
factors are (i) ~150 real images per rare class is too little information for any
augmentation to manufacture genuinely new discriminative signal, and (ii) class
weighting already handles the imbalance better than rebalancing (D2). **For
rare-fault detection on this dataset, augmentation — synthetic or real-derived,
crude or realistic — is not the lever; class weighting is.** Every plausible form
of the physics-augmentation thesis has now been tested and bounded.

Productive next directions are off the generator entirely: long-tail losses
(focal/LDAM, which target the rare-class boundary without fighting the weights),
on-domain self-supervised pretraining, or simply collecting more real rare-fault
labels.

## Off-generator lever: logit adjustment works

`verify_seeds --conditions logit_adj` — logit-adjusted loss (Menon et al. 2020):
add $\tau\log(\text{prior})$ to logits in training, infer on raw logits; replaces
class weighting (`train_one(loss_type="logit_adj")`).

| Condition | Accuracy | Macro recall | Rare-class recall |
|---|---|---|---|
| clean (class-weighted CE) | 0.782 | 0.699 | 0.612 |
| **logit_adj** | **0.808** | **0.709** | **0.634** |

`logit_adj` is the **best** condition on rare-class recall *and* macro-recall
*and* accuracy — it **Pareto-dominates** the class-weighting baseline. Where
inverse-frequency weighting trades accuracy for rare recall (D2: 0.84→0.78),
logit adjustment improves both, because it shifts the decision boundary by the
class priors rather than reweighting gradients.

At τ=1.0 the rare-recall gain (+0.022) is within bootstrap noise. **τ is a knob
on the recall/accuracy frontier** (`logit_adj_tNN` → τ=NN/10), and pushing it
makes the gain large and significant (see `docs/fig_logit_adj.png`):

| Condition | Accuracy | Macro recall | Rare-class recall | Δ rare vs clean (95% CI) |
|---|---|---|---|---|
| clean (weighted CE) | 0.782 | 0.699 | 0.612 | — |
| logit_adj τ=1.0 | **0.808** | 0.709 | 0.634 | +0.022 [−0.016, +0.058] |
| logit_adj τ=1.5 | 0.763 | **0.716** | 0.696 | **+0.084 [+0.038, +0.131]** ✓ |
| logit_adj τ=2.0 | 0.683 | 0.703 | **0.728** | **+0.116 [+0.071, +0.162]** ✓ |

Rare-fault recall climbs 0.612 → **0.728** (+0.116, ~19% relative, CI excludes 0)
as τ→2.0, traded against accuracy (0.808 → 0.683). The improvement is now
**statistically significant**, not just a better point estimate.

**Recommended operating points:**
- **τ=1.5 — balanced default.** Best macro-recall of *any* condition (0.716),
  significantly higher rare recall (0.696), only a small accuracy cost
  (0.763 vs 0.782). Pareto-sensible for the imbalanced objective.
- **τ=2.0 — max fault detection.** Highest rare recall (0.728) if false alarms
  are acceptable (accuracy 0.683, precision down).

**Bottom line of the whole investigation:** augmentation (any form) was not the
lever; a principled long-tail *loss* is. Make `logit_adj` (τ≈1.5) the default,
replacing inverse-frequency weighting. Further headroom: LDAM + deferred
reweighting, on-domain SSL pretraining, more real labels.

## Suggested first sprint

1. D1 (discriminator) + D2 (decouple rebalance/weighting) — diagnose H1 vs H2.
2. #4 Soiling correlated-noise source — cheapest realism experiment, worst class.
3. Whichever of #1 (time-dependent) / #2 (temperature-space) the diagnosis favors.

All scored against `clean`/`oversample` on rare-class recall with CIs.
