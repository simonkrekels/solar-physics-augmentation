# Experiments

Supplementary experiments beyond the two headline runs. See `CLAUDE.md` →
*Experiment results* for the primary baseline vs physics-augmented comparison.

## Backbone comparison

Script: `training/sweep_backbones.py`. Fine-tunes several `timm` backbones on
the **clean (un-augmented) splits** with the same class-weighted loss and
warmup→cosine schedule as `training/train.py`, so results are comparable to
each other. **15 epochs**, batch 32, seed 42, no W&B.

```bash
uv run python -m training.sweep_backbones \
  --models efficientnet_b0 resnet50 convnext_tiny mobilenetv3_large_100 \
  --epochs 15 --output sweep_results.csv
```

Results (sorted by macro F1), saved to `sweep_results.csv`:

| Model | Params (M) | Input | Test acc | Macro F1 | Train (min, MPS) |
|---|---|---|---|---|---|
| **efficientnet_b0** | 4.02 | 224 | **0.784** | **0.658** | 59 |
| mobilenetv3_large_100 | 4.22 | 224 | 0.760 | 0.646 | 39 |
| convnext_tiny | 27.83 | 224 | 0.745 | 0.635 | 132 |
| resnet50 | 23.53 | 224 | 0.694 | 0.577 | 92 |

**Conclusion:** EfficientNet-B0 was the right default — best macro F1 *and*
smallest. MobileNetV3 is a strong, faster runner-up worth remembering for the
serving path. The larger backbones (ResNet-50, ConvNeXt-Tiny) did worse despite
~6–7× the parameters: classic small/imbalanced-dataset overfitting under a short
training budget.

## Training-regime reconciliation ⚠️

The clean `efficientnet_b0` sweep run (78.4%) came out far above the recorded
clean baseline (69.0%) — same model, same clean data. Reconciling this exposed
a **training-regime effect, not a backbone or data effect**.

Re-running the project's own configs as-written (both 30 epochs, seed 42)
reproduced the documented numbers exactly:

| Run | Regime | Test acc | Early stop |
|---|---|---|---|
| `baseline.yaml` (clean) | 30 ep cosine | 0.690 | epoch 12 (best ≈7) |
| `augmented.yaml` | 30 ep cosine | 0.788 | epoch 19 (best ≈14) |
| sweep clean `efficientnet_b0` | **15 ep cosine** | 0.784 | ran full 15 |

So the documented results are valid and reproducible. The catch is **how** the
30-epoch baseline trains:

- The cosine schedule uses `T_max = num_epochs − warmup_epochs`, so a 30-epoch
  run anneals LR *slowly*. Val loss plateaus while LR is still high, and the
  (hardcoded, patience-5) early stop fires at **epoch 12** — best checkpoint
  ≈ epoch 7, before the LR anneal can sharpen the model → 69%.
- A 15-epoch cosine anneals *fast*; val loss keeps improving to the end, no early
  stop → **78.4% on identical clean data, no synthetic images**.
- Augmentation partly works *through* this mechanism: +1,388 images (14,000 →
  15,388) keeps val loss improving longer, pushing early stop to **epoch 19** —
  the augmented model simply gets deeper into the LR anneal.

**Takeaway:** the "+9.8 points from physics augmentation" is real *relative to its
matched 30-epoch baseline*, but that baseline is undertrained. A clean model on a
better schedule reaches the same place. The open question — does augmentation add
value *on top of* a well-tuned schedule? — is tested below.

### Augmentation under the better (15-epoch) schedule

`configs/augmented_15ep.yaml` — `augmented.yaml` at 15 epochs, compared to the
sweep's clean 15-epoch `efficientnet_b0` (78.4%).

Full picture (`efficientnet_b0`, seed 42, all on the same clean val/test):

| Run | Schedule | Aug | Test acc |
|---|---|---|---|
| `baseline.yaml` | 30 ep | — | 0.690 |
| `augmented.yaml` | 30 ep | ✓ | 0.788 |
| sweep clean | 15 ep | — | 0.784 |
| **`augmented_15ep.yaml`** | **15 ep** | **✓** | **0.808** |

**Verdict: augmentation adds genuine value, but smaller than headlined.**
augmented @ 15 ep (0.808) beats clean @ 15 ep (0.784) — so physics augmentation
helps *even on a well-tuned schedule*. But the headline "+9.8 over the 30-epoch
baseline" mostly reflects the schedule: fixing the training budget alone closes
~+9.4 of that gap, and augmentation contributes the rest on top.

Bonus: `augmented_15ep` (0.808) is the **best single run found** — it beats the
project's previous best (`augmented.yaml` @ 30 ep, 0.788), because the augmented
model also benefits from the faster anneal. Worth promoting to the default
augmented config.

### 3-seed verification

The single-seed +2.4 pt delta above is seed 42 — the most favourable seed. Run
`training/verify_seeds.py` (clean vs augmented @ 15 ep, paired on the same fixed
test set, seeds 42/1/2; results in `verify_seeds.csv`):

| Seed | Clean acc | Augmented acc | Δ acc |
|---|---|---|---|
| 42 | 0.7837 | 0.8077 | +0.0240 |
| 1 | 0.7820 | 0.8010 | +0.0190 |
| 2 | 0.7800 | 0.7927 | +0.0127 |
| **mean ± std** | **0.7819 ± 0.0019** | **0.8005 ± 0.0075** | **+0.0186 ± 0.0057** |

**The effect holds.** Augmentation beats clean on *every* seed, and the mean
test-acc gain (+1.86 pts) exceeds the run-to-run std (0.57 pts). The verified
gain (+1.86) is a touch below the seed-42 +2.4, as expected.

**Caveat — metric matters.** On **macro F1** the effect is positive on every
seed but within noise: +0.0104 ± 0.0105 (per-seed +0.0073 / +0.0018 / +0.0221).
So the robust, multi-seed claim is on **test accuracy**, not macro F1. The clean
baseline is remarkably stable (acc std 0.0019); augmentation adds both a higher
mean and more variance.

> ⚠️ **This "the effect holds" conclusion is superseded by the honest-baseline
> analysis below.** The +1.9 pt accuracy gain is real but is a *rebalancing*
> effect (cheap duplication matches it) and does **not** come from the rare
> classes. Read the next section.

## Honest baselines & per-class recall — the augmentation does not beat duplication ⚠️

The clean-vs-augmented comparison above asks the wrong question. The right ones:
(1) does physics synthesis beat the *cheap* ways to rebalance (duplicating real
images, or generic strong augmentation)? and (2) does it help the **rare fault
classes** it was designed for — measured with uncertainty, since each has only
~25 test images?

`training/verify_seeds.py` now trains four conditions on the same fixed splits,
3 seeds each, 15-epoch schedule, and bootstraps per-class recall CIs:

- **clean** — real data only (control)
- **oversample** — rare classes *duplicated* (with replacement) to `target_min`
  (= 500); rebalance-only, no new pixels
- **randaugment** — real data + RandAugment; strong generic-aug control
- **physics** — real data + heat-equation synthetic samples to `target_min`

`oversample` and `physics` top up the *same* classes to the *same* target, so
`physics − oversample` isolates the synthetic **signal** from the rebalancing.

```bash
uv run python -m training.verify_seeds --seeds 42 1 2     # ~12h on MPS
uv run python -m training.verify_seeds --analyze-only     # re-report only
```

**Headline (mean ± std across 3 seeds):**

| Condition | Accuracy | Macro F1 | Macro recall | Rare-class recall |
|---|---|---|---|---|
| clean | 0.782 ± 0.002 | 0.658 | 0.699 | **0.612** |
| **oversample** | **0.807 ± 0.006** | **0.678** | 0.697 | 0.574 |
| randaugment | 0.782 ± 0.002 | 0.654 | 0.702 | 0.615 |
| physics | 0.800 ± 0.006 | 0.668 | 0.693 | 0.569 |

**Paired deltas vs physics (bootstrap 95% CI; `verify_deltas.csv`):**

| Comparison | Metric | Δ | 95% CI | Sig? |
|---|---|---|---|---|
| physics − oversample | accuracy/recall | ~0 | spans 0 | no |
| physics − clean | rare-class recall | **−0.043** | [−0.082, −0.003] | **yes** |
| physics − randaugment | rare-class recall | −0.046 | [−0.098, +0.009] | no |

**Findings:**

1. **Physics synthesis does not beat trivial oversampling.** On accuracy, plain
   duplication (0.807) actually *edges out* physics (0.800); on macro F1, macro
   recall, and rare-class recall they are statistically indistinguishable. The
   heat-equation pixels add nothing over copy-paste.
2. **The accuracy gain is a rebalancing artifact, not rare-fault improvement.**
   Physics' +1.9 pt over clean comes from the **majority** class (No-Anomaly
   recall 0.878 vs clean 0.846), not the rare classes. On rare-class recall
   physics is the *worst* of the four conditions, and significantly below doing
   nothing (clean). Per-class, physics is worst-or-near-worst on Hot-Spot (0.523
   vs 0.559), Hot-Spot-Multi (0.477 vs 0.595), and Soiling (0.355 vs
   randaugment's 0.419); only Diode-Multi is easy for all (~0.95).
3. **Likely mechanism.** The loss already uses inverse-frequency class weights.
   Topping rare classes up to 500 — by *any* means — makes them less rare, which
   softens their upweighting and shifts the precision/recall trade toward the
   majority. Net: majority accuracy ↑, rare-class recall flat-to-down. Both
   oversample and physics do this; the synthetic geometry adds no rare-class
   signal on top and may inject slightly misleading structure.

**Verdict.** Under a metric-appropriate, baseline-controlled, CI'd evaluation,
the current heat-equation augmentation **does not support the project's central
claim**. It is a (weaker-than-duplication) rebalancing trick, not a source of
useful rare-fault signal. This indicts *this generator* (steady-state blobs and
strips, additive RGB blend) — not physics-informed augmentation in principle.
See `docs/AUGMENTATION_ROADMAP.md` for the realism upgrades that could clear the
oversample bar, now measurable against this harness.

**Caveats.** Rare-class test support is ~25 images each, so CIs are wide and the
`physics < clean` significance is marginal. The robust, tight-CI finding is the
one that matters most: **physics ≈ oversample**, i.e. the synthesis earns
nothing over duplication.
