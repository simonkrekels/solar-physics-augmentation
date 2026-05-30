# Thermal Solar Panel Defect Detection
### A physics-informed CV project for portfolio

**Goal:** Fine-tune an image classifier on thermal IR images of solar modules, then
improve it using synthetic data augmentation grounded in the 2D heat equation.
Demonstrates: dataset curation, model fine-tuning, experiment tracking (W&B),
error analysis, and physics-informed thinking.

**Estimated time:** 1 focused week.

---

## 0. Repository layout

Set up this structure from the start — keep it clean enough to show.

```
solar-thermal-cv/
├── data/
│   ├── raw/                  # downloaded dataset, untouched
│   └── processed/            # train / val / test splits
├── augmentation/
│   └── heat_equation.py      # physics-based patch generator
├── training/
│   ├── dataset.py            # PyTorch Dataset class
│   ├── model.py              # model definition & loading
│   ├── train.py              # training loop
│   └── evaluate.py           # error analysis helpers
├── notebooks/
│   ├── 01_eda.ipynb          # exploratory data analysis
│   ├── 02_error_analysis.ipynb
│   └── 03_augmentation_demo.ipynb
├── configs/
│   ├── baseline.yaml         # hyperparameters for baseline run
│   └── augmented.yaml        # hyperparameters for augmented run
├── requirements.txt
└── README.md
```

---

## 1. Environment setup

Create a dedicated virtual environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision            # deep learning
pip install timm                         # pretrained backbones
pip install wandb                        # experiment tracking
pip install scipy matplotlib seaborn     # numerics & viz
pip install scikit-learn                 # metrics
pip install Pillow tqdm pyyaml           # utilities
pip install jupyter                      # notebooks
```

Save to `requirements.txt` with pinned versions once the environment is stable.

Create a W&B account (free) at https://wandb.ai and run `wandb login`.

---

## 2. Dataset

### 2.1 Download

The **InfraredSolarModules** dataset (RaptorMaps, open licence) is a collection of
~20 000 thermal IR images of individual solar modules, each labelled with an anomaly
class. It is the most directly relevant public dataset for Sitemark's domain.

```bash
git clone https://github.com/RaptorMaps/InfraredSolarModules data/raw/InfraredSolarModules
```

If the repo is unavailable or very large, a fallback is to search for it on
[Kaggle](https://www.kaggle.com) or [Roboflow Universe](https://universe.roboflow.com)
(search "solar panel thermal"). The class structure should be similar.

### 2.2 Classes

The dataset has ~11 anomaly classes plus a clean class. The key ones:

| Class | Physics cause |
|---|---|
| `No-Anomaly` | Healthy cell |
| `Hot-Spot` | Localised Joule heating from a shunted or cracked cell |
| `Hot-Spot-Multi` | Multiple shunted cells |
| `Diode` | Failed bypass diode — entire string overheats |
| `Diode-Multi` | Multiple failed diodes |
| `Cell` | Single degraded cell |
| `Cell-Multi` | Multiple degraded cells |
| `Shadowing` | Partial shadow causing reverse-bias heating |
| `Offline-Module` | Entire module disconnected |
| `Vegetation` | Shading from plant growth |
| `Cracking` | Physical crack causing localised resistance |
| `Short` | Internal short circuit |

Understanding *why* each class looks different thermally is the physics hook. Write
a brief section on this in the README (see Section 8).

### 2.3 EDA — `notebooks/01_eda.ipynb`

Before any training, explore the data:

- Plot class distribution (expect heavy imbalance — `No-Anomaly` will dominate).
- Show a grid of representative images per class.
- Note image dimensions and whether they are consistent.
- Check for near-duplicate images between splits if pre-split; if not, be careful
  that augmented synthetics don't leak into validation.
- Log the class distribution table to W&B as a table artifact.

### 2.4 Create train / val / test splits

If the dataset does not come pre-split:

```python
# training/dataset.py  (pseudocode — ask Claude Code to implement)
from sklearn.model_selection import train_test_split

# stratified split to preserve class balance
# 70% train, 15% val, 15% test
# save split CSVs to data/processed/ so splits are reproducible
# log the split CSVs as W&B artifacts for full reproducibility
```

Use stratified splitting (`stratify=labels`) because of the class imbalance.

---

## 3. Baseline model

### 3.1 Model choice

Use **EfficientNet-B0** loaded from `timm` with ImageNet pretrained weights.
It is small, fast to fine-tune, and well understood. Replace the classification head
for the number of classes in the dataset.

```python
# training/model.py  (pseudocode)
import timm
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)
```

### 3.2 Training loop — `training/train.py`

Key implementation points:

- **Loss:** cross-entropy with `weight` parameter set to inverse class frequency to
  handle imbalance.
- **Optimiser:** AdamW, initial LR from config.
- **LR schedule:** cosine annealing with warm-up (5 epochs warm-up, then cosine
  decay).
- **Augmentation (standard):** random horizontal/vertical flip, random rotation
  (±15°), colour jitter (brightness/contrast only — thermal images have no colour,
  but intensity variation is meaningful). Use `torchvision.transforms`.
- **Early stopping:** monitor validation loss, patience = 5 epochs.

### 3.3 Config — `configs/baseline.yaml`

```yaml
model: efficientnet_b0
pretrained: true
num_epochs: 30
batch_size: 32
learning_rate: 3.0e-4
weight_decay: 1.0e-4
warmup_epochs: 5
seed: 42
wandb:
  project: solar-thermal-cv
  run_name: baseline
```

### 3.4 W&B integration

Log the following at every epoch:

```python
import wandb
wandb.log({
    "train/loss": train_loss,
    "train/acc": train_acc,
    "val/loss": val_loss,
    "val/acc": val_acc,
    "epoch": epoch,
    "lr": current_lr,
})
```

At the end of training, log:
- The full classification report (per-class precision, recall, F1) as a W&B table.
- The confusion matrix as a W&B plot.
- The trained model weights as a W&B artifact.

---

## 4. Error analysis — `notebooks/02_error_analysis.ipynb`

This step drives the physics-informed augmentation. It is important — do not skip it.

### 4.1 Confusion matrix

Plot the confusion matrix on the **validation set**. Identify which classes are most
confused with each other. Expected findings:

- Rare classes (`Short`, `Cracking`, `Vegetation`) will likely have low recall due
  to few training examples.
- `Hot-Spot` and `Cell` may be confused — they are visually similar at low severity.

### 4.2 Failure case visualisation

For each misclassified class, display a grid of the worst failures (highest
confidence wrong predictions). Ask: *why* did the model fail here? Is the anomaly
subtle, occluded, or does it resemble another class?

### 4.3 Per-class sample count vs F1

Plot a scatter of training sample count vs validation F1 per class. This will almost
certainly show a positive correlation and confirm that rare classes need more data —
which motivates the synthetic augmentation.

Log all of this to the W&B run for the record.

---

## 5. Physics-informed augmentation — `augmentation/heat_equation.py`

This is the differentiating piece. Focus synthetic augmentation on the rare, poorly
performing anomaly classes identified in Step 4.

### 5.1 The physics

A localised defect (shunted cell, failed diode) acts as a point or patch heat source
$q(\mathbf{r})$ embedded in the solar module. In steady state, the temperature
distribution satisfies:

$$\nabla^2 T(\mathbf{r}) = -\frac{q(\mathbf{r})}{\kappa}$$

where $\kappa$ is the thermal conductivity of the module laminate. Boundary conditions
are approximately Neumann (insulating edges) or fixed ambient temperature (Dirichlet).
The resulting $T(\mathbf{r})$ produces the hot patch seen in thermal imagery.

### 5.2 Implementation

Solve on a small 2D grid (e.g. 64×64) using `scipy.sparse.linalg` (sparse linear
solve) or a simple iterative relaxation (Gauss-Seidel). The latter is ~10 lines and
fast enough.

```python
# augmentation/heat_equation.py  (pseudocode — ask Claude Code to implement fully)
import numpy as np
from scipy.ndimage import gaussian_filter

def solve_heat_patch(
    grid_size: int = 64,
    source_center: tuple = (32, 32),
    source_radius: float = 5.0,
    source_strength: float = 1.0,
    n_iter: int = 500,
) -> np.ndarray:
    """
    Solve nabla^2 T = -q/kappa on a grid using Gauss-Seidel iteration.
    Returns normalised temperature array in [0, 1].
    """
    T = np.zeros((grid_size, grid_size))
    q = np.zeros_like(T)

    # Gaussian heat source centred at source_center
    yy, xx = np.ogrid[:grid_size, :grid_size]
    r2 = (xx - source_center[1])**2 + (yy - source_center[0])**2
    q = source_strength * np.exp(-r2 / (2 * source_radius**2))

    # Iterative Gauss-Seidel relaxation
    for _ in range(n_iter):
        T[1:-1, 1:-1] = 0.25 * (
            T[:-2, 1:-1] + T[2:, 1:-1] +
            T[1:-1, :-2] + T[1:-1, 2:] +
            q[1:-1, 1:-1]
        )
        # Neumann BC: zero-flux at edges
        T[0, :] = T[1, :]
        T[-1, :] = T[-2, :]
        T[:, 0] = T[:, 1]
        T[:, -1] = T[:, -2]

    return (T - T.min()) / (T.max() - T.min() + 1e-8)


def generate_augmented_patch(
    base_image: np.ndarray,
    anomaly_class: str,
) -> np.ndarray:
    """
    Blend a synthetic heat patch into a base image to simulate an anomaly.
    - base_image: H x W float array, already a thermal image
    - anomaly_class: controls source geometry (point source for Hot-Spot,
      strip source for Diode, etc.)
    Returns the augmented image as a float array.
    """
    # Vary source geometry by class:
    # Hot-Spot / Cell -> small Gaussian source
    # Diode -> elongated strip source (sum of Gaussians along a row)
    # Shadowing -> large diffuse cool patch (negative source strength)
    ...
```

### 5.3 Augmentation strategy

- For each underrepresented class, generate N synthetic training images by blending
  solved heat patches onto randomly selected `No-Anomaly` base images.
- Control the blending intensity (alpha) randomly in a plausible range.
- Vary source position, size, and strength to maximise diversity.
- Target: bring the rarest classes up to at least 50% of the majority class count.

Show the synthetic patches in `notebooks/03_augmentation_demo.ipynb` — visualise
the heat equation solution alongside the blended result. This notebook is good
portfolio material.

---

## 6. Augmented model training

Re-train with the expanded dataset using `configs/augmented.yaml`:

```yaml
model: efficientnet_b0
pretrained: true
num_epochs: 30
batch_size: 32
learning_rate: 3.0e-4
weight_decay: 1.0e-4
warmup_epochs: 5
seed: 42
augmentation:
  physics_synthetic: true
  target_min_samples: 500   # minimum per class after augmentation
wandb:
  project: solar-thermal-cv
  run_name: physics-augmented
```

Log everything the same way as the baseline so the W&B dashboard shows a direct
comparison between the two runs.

---

## 7. Results comparison

In `notebooks/02_error_analysis.ipynb` (or a new cell section), compare:

- Per-class F1: baseline vs augmented (bar chart, easy to read at a glance).
- Overall macro F1 and weighted F1.
- Confusion matrices side by side.
- Training curves (W&B has this built in — link the run in the README).

The expected result: improved recall on rare anomaly classes. Even a modest improvement
is scientifically honest and tells a clear story.

---

## 8. README.md

The README is half the portfolio value. Structure it as follows:

### Sections

1. **Project summary** (3 sentences: what, why, result)
2. **Physics background** — brief, accessible explanation of why thermal anomalies
   appear in solar panels. Draw the connection between current redistribution → Joule
   heating → thermal signature. This is where your PhD background earns its place.
3. **Dataset** — what it is, how to download it, class distribution figure.
4. **Approach** — describe the two-stage plan (baseline → error analysis → 
   physics augmentation → retrain).
5. **Results** — the comparison table and figures. Include a link to the W&B project
   (make it public).
6. **How to reproduce** — clear install and run instructions.
7. **What I'd do next** — e.g. detection (bounding boxes) instead of classification,
   larger backbone, active learning loop. Shows product thinking.

---

## 9. Definition of done

The project is ready to reference in applications when:

- [ ] Baseline runs cleanly end-to-end
- [ ] W&B dashboard is public and shows both runs
- [ ] Error analysis notebook explains the augmentation motivation
- [ ] Augmentation demo notebook shows synthetic patches with the PDE solution
- [ ] Per-class F1 comparison shows a measurable improvement on rare classes
- [ ] README is written and the physics section is clear to a non-physicist
- [ ] GitHub repo is public with a clean commit history (not one giant commit)

---

## Appendix: useful references

- InfraredSolarModules dataset: https://github.com/RaptorMaps/InfraredSolarModules
- timm library: https://github.com/huggingface/pytorch-image-models
- W&B quickstart: https://docs.wandb.ai/quickstart
- EfficientNet paper: Tan & Le, 2019 (arXiv:1905.11946)
- On hot-spot physics: Köntges et al., *Review of Failures of Photovoltaic Modules*,
  IEA PVPS Task 13 (2014) — good background reading for the README physics section
