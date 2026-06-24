# Physics-Informed Defect Detection for Thermal Solar Inspection

Fine-tuning a vision classifier on thermal IR images of solar modules, then
closing the class-imbalance gap with synthetic training data generated from the
steady-state heat equation.

---

## The problem

Thermal inspection of solar farms produces thousands of IR images per flight.
Each image is a cropped module labelled by defect type — hot spots, shading,
cell cracks, diode failures, soiling, and healthy modules. The challenge is
two-fold:

1. **Heavy class imbalance.** Healthy modules dominate (~50 % of images).
   Rare faults (`Diode-Multi`, `Hot-Spot-Multi`) have only a few hundred
   examples — not enough to learn a reliable decision boundary.
2. **Physics distinguishes fault types.** A localised diode short produces a
   concentrated thermal signature; multi-cell degradation spreads heat across
   a region. A classifier trained on raw pixel patterns risks ignoring these
   structural cues.

The usual answer — collect more labelled fault images — is slow and expensive.
This project asks whether physics can substitute for data.

---

## Approach

### 1. Baseline classifier

EfficientNet-B0 (ImageNet pre-trained, head replaced for 12 classes) trained
with:

- **Inverse-frequency class weights** applied to cross-entropy loss — ensures
  rare faults are penalised as heavily as the majority class.
- **AdamW** with 5-epoch linear warmup → cosine annealing, early stopping.
- Stratified 70 / 15 / 15 splits, `seed=42`.

### 2. Synthetic augmentation via the heat equation

For each rare anomaly class that falls below a sample threshold, the augmenter:

1. **Constructs a class-specific heat source geometry** — a Gaussian blob for a
   hot spot, parallel strips for cell-level degradation, a point load for a
   diode short, uniform cooling for soiling, etc.
2. **Solves ∇²T = −q / κ** on a grid using Jacobi iteration with Neumann
   (zero-flux) boundary conditions — the physically correct BC for a thermally
   insulated module edge.
3. **Blends the resulting temperature field** onto a healthy base image,
   producing a labelled synthetic training sample that inherits real sensor
   noise and lighting while carrying a physically plausible anomaly signature.

Synthetic images top up every under-represented class to a target minimum,
then the augmented training set is used to retrain from the same ImageNet
initialisation. Validation and test sets are always real images.

### 3. Experiment tracking

Both runs log to Weights & Biases (`solar-thermal-cv`). Model weights and split
CSVs are saved as W&B artifacts for full reproducibility.

---

## Results

| Model | Test accuracy | Macro F1 |
|---|---|---|
| Baseline (30-epoch) | 69.0 % | 0.562 |
| Physics-augmented (30-epoch) | 78.8 % | 0.647 |

The augmenter generated **1 388 synthetic images** across four under-represented
classes (`Diode-Multi`, `Hot-Spot`, `Hot-Spot-Multi`, `Soiling`). The rarest
classes gained the most. `Soiling` improved least (+0.066 F1) — its real
signature is spatially diffuse and irregular, which the blob geometry only
approximates.

### Reading these numbers honestly

A +9.8-point headline is easy to over-claim, so the project stress-tests it
(full analysis in `EXPERIMENTS.md`). Under rigorous evaluation the headline does
**not** hold up — and surfacing that is the point:

- **Most of the gap is the training schedule, not the augmentation.** The
  30-epoch baseline is *undertrained*: a slow cosine schedule plus early stopping
  halts it at epoch 12. A clean model on a faster 15-epoch schedule reaches
  **~78 %** with no synthetic data — closing ~+9 of the +9.8 on its own.
- **What's left is a rebalancing artifact, not a physics effect.** With honest
  baselines (3 seeds, paired, bootstrap CIs), physics augmentation is
  **statistically indistinguishable from plain oversampling** — duplicating real
  rare-class images matches it (and slightly beats it on accuracy: 80.7 % vs
  80.0 %). The synthetic pixels add nothing over copy-paste.
- **It does not help the rare faults it was built for.** The accuracy bump comes
  from the *majority* class (No-Anomaly recall 0.88 vs 0.85), not the rare ones.
  On rare-class recall, physics is the **worst** of {clean, oversample,
  randaugment, physics} — and significantly below doing nothing.

| Condition (15-epoch, 3-seed mean) | Accuracy | Rare-class recall |
|---|---|---|
| clean | 78.2 % | 0.612 |
| oversample (duplicate reals) | **80.7 %** | 0.574 |
| randaugment | 78.2 % | **0.615** |
| physics-augmented | 80.0 % | 0.569 |

The takeaway: **this heat-equation generator (steady-state blobs, additive RGB
blend) is a weaker-than-duplication rebalancing trick, not a source of useful
rare-fault signal.** That indicts the current implementation, not physics-informed
augmentation in principle — see `docs/AUGMENTATION_ROADMAP.md` for the realism
upgrades that could clear the oversample bar, now measurable against a
metric-appropriate, baseline-controlled harness (`training/verify_seeds.py`).

### Backbone choice

EfficientNet-B0 isn't assumed — it's the measured winner of a sweep
(`training/sweep_backbones.py`) over four `timm` backbones on identical data:

| Backbone | Params | Test acc | Macro F1 |
|---|---|---|---|
| **EfficientNet-B0** | 4.0 M | **78.4 %** | **0.658** |
| MobileNetV3-Large | 4.2 M | 76.0 % | 0.646 |
| ConvNeXt-Tiny | 27.8 M | 74.5 % | 0.635 |
| ResNet-50 | 23.5 M | 69.4 % | 0.577 |

It wins on macro F1 *and* is the smallest model — the larger backbones overfit
this small, imbalanced dataset. The backbone is now selectable via the config's
`model:` field, so swapping it requires no code change.

---

## What the notebooks show

| Notebook | Contents |
|---|---|
| `01_eda.ipynb` | Class distribution, image grid, dimension statistics |
| `02_error_analysis.ipynb` | Confusion matrices, per-class F1 comparison, failure gallery |
| `03_augmentation_demo.ipynb` | PDE heat fields, augmentation pipeline, diversity and intensity sweeps |

---

## Repository layout

```
├── augmentation/heat_equation.py   # PDE solver, source geometries, SyntheticAugmenter
├── training/
│   ├── dataset.py                  # stratified splits, transforms, DataLoader
│   ├── model.py                    # config-selectable backbone via timm
│   ├── train.py                    # training loop with W&B logging
│   ├── evaluate.py                 # metrics, classification report
│   ├── sweep_backbones.py          # backbone comparison sweep
│   └── verify_seeds.py             # multi-seed clean vs augmented verification
├── configs/
│   ├── baseline.yaml
│   ├── augmented.yaml
│   └── augmented_15ep.yaml         # best config — tuned 15-epoch schedule
├── EXPERIMENTS.md                  # backbone sweep + training-regime reconciliation
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_error_analysis.ipynb
│   └── 03_augmentation_demo.ipynb
└── tests/
    ├── test_dataset.py
    └── test_heat_equation.py
```

## Quick start

```bash
# Clone dataset
git clone https://github.com/RaptorMaps/InfraredSolarModules data/raw/InfraredSolarModules

# Create splits
uv run python training/dataset.py

# Train baseline
uv run python -m training.train --config configs/baseline.yaml

# Train physics-augmented (best config: tuned 15-epoch schedule)
uv run python -m training.train --config configs/augmented_15ep.yaml

# Compare backbones / verify the augmentation effect across seeds
uv run python -m training.sweep_backbones
uv run python -m training.verify_seeds --seeds 42 1 2
```

---

## Stack & tooling

| Layer | Choice |
|---|---|
| Model backbone | EfficientNet-B0 via [timm](https://github.com/huggingface/pytorch-image-models) |
| Training framework | PyTorch |
| Numerics / PDE solver | NumPy, SciPy |
| Experiment tracking | Weights & Biases — runs, metrics, model + data artifacts |
| Data format | Pillow (image I/O), pandas (split CSVs) |
| Package management | [uv](https://github.com/astral-sh/uv) |
| Testing | pytest |
| Notebooks | Jupyter |
| Config | YAML |

---

## Dataset

[InfraredSolarModules](https://github.com/RaptorMaps/InfraredSolarModules) —
RaptorMaps, open licence. ~20 000 thermal IR images across 12 anomaly classes.
