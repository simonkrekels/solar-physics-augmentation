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
| Baseline | 69.0 % | 0.562 |
| Physics-augmented | 78.8 % | 0.647 |

The augmenter generated **1 388 synthetic images** across four under-represented
classes (`Diode-Multi`, `Hot-Spot`, `Hot-Spot-Multi`, `Soiling`). The rarest
classes gained the most. `Soiling` improved least (+0.066 F1) — its real
signature is spatially diffuse and irregular, which the blob geometry only
approximates.

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
│   ├── model.py                    # EfficientNet-B0 via timm
│   ├── train.py                    # training loop with W&B logging
│   └── evaluate.py                 # metrics, classification report
├── configs/
│   ├── baseline.yaml
│   └── augmented.yaml
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

# Train physics-augmented
uv run python -m training.train --config configs/augmented.yaml
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
