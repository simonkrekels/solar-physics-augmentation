# Technical Reference

Full developer reference for the solar thermal defect detection project.
Intended audience: me, future me, or anyone who needs to extend or re-run experiments.

---

## Table of contents

1. [Environment setup](#1-environment-setup)
2. [Dataset & splits](#2-dataset--splits)
3. [Data pipeline — `training/dataset.py`](#3-data-pipeline--trainingdatasetpy)
4. [Model — `training/model.py`](#4-model--trainingmodelpy)
5. [Training loop — `training/train.py`](#5-training-loop--trainingtrainpy)
6. [Evaluation — `training/evaluate.py`](#6-evaluation--trainingevaluatepy)
7. [Physics augmentation — `augmentation/heat_equation.py`](#7-physics-augmentation--augmentationheat_equationpy)
8. [Configs](#8-configs)
9. [Experiment tracking (W&B)](#9-experiment-tracking-wb)
10. [Tests](#10-tests)
11. [Reproducing the published results](#11-reproducing-the-published-results)
12. [Known limitations & next steps](#12-known-limitations--next-steps)

---

## 1. Environment setup

All package management uses `uv`. The venv lives at `.venv/` and is already initialised.

```bash
# Add a new dependency
uv add <package>

# Run anything inside the venv
uv run python <script>
uv run jupyter notebook
uv run pytest tests/ -v
```

Never install into the global Python environment.

---

## 2. Dataset & splits

### Source

```bash
git clone https://github.com/RaptorMaps/InfraredSolarModules data/raw/InfraredSolarModules
```

Metadata lives at `data/raw/InfraredSolarModules/InfraredSolarModules/module_metadata.json`.
Each entry maps an image filename to `{"image_filepath": "...", "anomaly_class": "..."}`.

### Classes (12 total)

```
No-Anomaly       # ~50 % of all images — majority class
Hot-Spot
Hot-Spot-Multi
Diode
Diode-Multi
Cell
Cell-Multi
Shadowing
Offline-Module
Vegetation
Cracking
Soiling
```

Note: the project plan says class 12 is "Short"; it is actually **Soiling**.

### Split sizes

| Split | Images |
|---|---|
| Train | 14 000 |
| Val   | 3 000  |
| Test  | 3 000  |

Stratified 70 / 15 / 15, `seed=42`. CSVs written to `data/processed/{train,val,test}.csv`.
Each CSV has two columns: `path` (absolute), `label` (class string).

Splits are checked in via the W&B `splits` artifact — fetch them with `wandb artifact get`.

### Regenerating splits

```bash
uv run python training/dataset.py
```

`make_splits()` will overwrite existing CSVs. Always use `seed=42` to stay reproducible.

---

## 3. Data pipeline — `training/dataset.py`

### Key constants

| Name | Value |
|---|---|
| `IMG_SIZE` | 224 |
| `IMAGENET_MEAN` | `[0.485, 0.456, 0.406]` |
| `IMAGENET_STD` | `[0.229, 0.224, 0.225]` |
| `RAW_DIR` | `data/raw/InfraredSolarModules/InfraredSolarModules/` |
| `PROCESSED_DIR` | `data/processed/` |

### Transforms

**Train** (in order):
1. `Resize(224, 224)`
2. `RandomHorizontalFlip()`
3. `RandomVerticalFlip()`
4. `RandomRotation(15°)`
5. `ColorJitter(brightness=0.3, contrast=0.3)`
6. `ToTensor()`
7. `Normalize(ImageNet mean/std)`

**Val / Test** — resize, ToTensor, Normalize only (no randomness).

### `SolarModuleDataset`

```python
SolarModuleDataset(df: pd.DataFrame, classes: list[str], split: str = "train")
```

- `classes` must be the full sorted list of 12 class names — derived at runtime from `train_df["label"].unique()` after any augmentation.
- `class_to_idx` maps class string → int in the order `classes` was passed.
- Images are opened with PIL and converted to RGB.

### Public functions

```python
make_splits(seed=42) -> dict[str, pd.DataFrame]   # writes CSVs, returns dfs
load_splits() -> dict[str, pd.DataFrame]           # reads existing CSVs
splits_exist() -> bool                             # checks all three CSVs present
get_transforms(split: str) -> transforms.Compose  # "train" | "val" | "test"
```

---

## 4. Model — `training/model.py`

```python
build_model(num_classes: int, pretrained: bool = True) -> nn.Module
```

Thin wrapper around:

```python
timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
```

`timm` replaces the classification head automatically — ImageNet backbone weights are preserved.
Checkpoint files are saved as `checkpoints/<run_name>_best.pt` (state dict only).
Loading:

```python
model = build_model(n_classes)
model.load_state_dict(torch.load("checkpoints/baseline_best.pt", map_location=device, weights_only=True))
```

---

## 5. Training loop — `training/train.py`

### Running

```bash
uv run python -m training.train --config configs/baseline.yaml
uv run python -m training.train --config configs/augmented.yaml
```

Must be run as a module (not `python training/train.py`) because of relative imports.

### Device selection

Auto-detects: CUDA → MPS (Apple Silicon) → CPU.

### Class weighting

```python
weights[i] = 1.0 / count(class_i)
weights /= weights.mean()   # normalise so mean weight == 1
```

Applied to `nn.CrossEntropyLoss(weight=...)`. This scales the loss for rare classes up,
not their sample count — they are still seen less frequently per epoch.

### Optimiser & scheduler

| Component | Setting |
|---|---|
| Optimiser | AdamW |
| LR | `3e-4` |
| Weight decay | `1e-4` |
| Warmup | Linear 0.1× → 1.0× over 5 epochs |
| After warmup | CosineAnnealingLR, `eta_min=1e-6`, `T_max = num_epochs - warmup_epochs` |
| Early stopping | patience = 5 epochs on val loss |

`SequentialLR` chains the two schedulers at `milestone=[warmup_epochs]`.

### Augmentation hook

If `augmentation.physics_synthetic: true` in the config:

1. `SyntheticAugmenter(train_df, seed=cfg["seed"])` is constructed.
2. `augmenter.augment_split(train_df, target_min, output_dir)` extends `train_df` in memory and
   saves synthetic images to `data/processed/synthetic/<class_name>/syn_NNNNN.jpg`.
3. Only `train_df` is extended — val and test are always untouched.
4. `classes` is derived from the extended `train_df` after augmentation.

### Checkpointing

Best val-loss checkpoint is saved to `checkpoints/<run_name>_best.pt`.
At the end of training the best checkpoint is loaded and evaluated on the test set.

### W&B logging

Per epoch: `train/loss`, `train/acc`, `val/loss`, `val/acc`, `lr`.
End of run: `test/accuracy`, `test/classification_report` (Table), `test/confusion_matrix`.
Artifacts: `splits` (the three CSVs), `model` (best checkpoint `.pt`).

---

## 6. Evaluation — `training/evaluate.py`

```python
run_eval(model, loader, device) -> (avg_loss, accuracy, predictions, true_labels)
```

Runs in `model.eval()` with `torch.no_grad()`. Loss is unweighted `CrossEntropyLoss`
(class weights are a training concern, not an evaluation one).

```python
classification_report_dict(preds, labels, classes) -> dict
```

Thin wrapper around `sklearn.metrics.classification_report(..., output_dict=True)`.
Keys are class names; each value has `precision`, `recall`, `f1-score`, `support`.

```python
get_confusion_matrix(preds, labels, n_classes) -> np.ndarray   # shape (n, n)
```

---

## 7. Physics augmentation — `augmentation/heat_equation.py`

### PDE being solved

Steady-state 2D heat equation:

```
∇²T = −q / κ
```

where `q(y, x)` is the heat source density and `κ` is thermal conductivity.

**Boundary conditions**: Neumann (`∂T/∂n = 0` at all edges) — zero heat flux through module boundaries.
Implemented by edge-padding the grid before each Jacobi update.

**Null-space fix**: pure Neumann problems have T determined only up to a constant. The mean
is subtracted after each iteration to pin the solution.

### `solve_heat_2d(source, kappa=1.0, max_iter=500, tol=1e-3) -> np.ndarray`

Jacobi iteration (vectorised NumPy). Converges when `max(|T - T_prev|) < tol`.
Returns a zero-mean float64 array of the same shape as `source`.

### `make_source(class_name, grid_h=64, grid_w=64, rng=None) -> np.ndarray`

Returns a class-specific heat source `q(y, x)`. Positive = Joule heating, negative = cooling.

| Class | Geometry |
|---|---|
| `Hot-Spot` | Single Gaussian blob, random centre, σ ∈ [1.5, 3.5] |
| `Hot-Spot-Multi` | 2–4 Gaussian blobs with smaller σ |
| `Cell` | Single rectangle, ~1/6 of grid |
| `Cell-Multi` | 2–3 rectangles |
| `Diode` | Full-height vertical strip (one bypass-diode column) |
| `Diode-Multi` | 2–3 full-height vertical strips |
| `Shadowing` | Single large rectangle with **negative** amplitude |
| `Vegetation` | Single large negative Gaussian |
| `Soiling` | 2–5 small negative Gaussian blobs |
| `Offline-Module` | Uniform field of −5.0 |
| `Cracking` | Thin resistive line at random angle |

`No-Anomaly` is not in `ANOMALY_CLASSES` and is never synthesised — it serves as the base image pool.

### `blend_heat_patch(base_img, heat, intensity_scale=40.0) -> Image`

1. Resizes `heat` to the image size with bilinear interpolation (`scipy.ndimage.zoom`).
2. Normalises: `delta = (heat / |heat|_max) * intensity_scale`.
3. Adds `delta` to all three RGB channels; clips to `[0, 255]`.

`intensity_scale=40.0` means a peak heat anomaly shifts pixel values by ±40 out of 255.

### `synthesise(class_name, base_img, grid_size=64, kappa=1.0, intensity_scale=40.0, rng=None)`

Convenience wrapper: `make_source → solve_heat_2d → blend_heat_patch`.

### `SyntheticAugmenter`

```python
SyntheticAugmenter(train_df, grid_size=64, kappa=1.0, intensity_scale=40.0, seed=42)
```

- Requires at least one `No-Anomaly` row in `train_df` (used as base image pool).
- `self.rng` is a seeded `np.random.default_rng` — all randomness flows through it.

```python
augmenter.generate(class_name, n, output_dir) -> list[Path]
```

Saves `n` images as `output_dir/<class_name>/syn_NNNNN.jpg` (quality=90).

```python
augmenter.augment_split(train_df, target_min, output_dir) -> pd.DataFrame
```

For each class in `ANOMALY_CLASSES` with fewer than `target_min` samples:
generates `target_min - current_count` images and appends rows to the DataFrame.
Returns a new DataFrame (original rows first, then synthetic rows).

---

## 8. Configs

### `configs/baseline.yaml`

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

### `configs/augmented.yaml`

Same as baseline plus:

```yaml
augmentation:
  physics_synthetic: true
  target_min_samples: 500
```

---

## 9. Experiment tracking (W&B)

Project: `solar-thermal-cv`.

Both runs are logged there with full configs, per-epoch curves, confusion matrices,
per-class F1 tables, and model + split artifacts.

To re-download a checkpoint:

```bash
wandb artifact get solar-thermal-cv/model:latest --root checkpoints/
```

---

## 10. Tests

```bash
uv run pytest tests/ -v
```

### `tests/test_heat_equation.py`

| Test | What it checks |
|---|---|
| `test_solver_zero_source` | Zero source → zero field |
| `test_solver_shape` (parametrized) | Output shape matches input |
| `test_solver_positive_source_is_hot` | Localised positive source → positive T at that point |
| `test_solver_symmetric_source` | Symmetric source → symmetric field |
| `test_solver_negative_source_is_cool` | Negative source → negative T |
| `test_make_source_all_classes` (parametrized) | All 11 classes produce finite, non-zero arrays |
| `test_make_source_hot_classes_positive` | Hot-Spot, Diode, Cell → `q.max() > 0` |
| `test_make_source_cool_classes_negative` | Shadowing, Vegetation, Offline → `q.min() < 0` |
| `test_blend_*` | Size/mode preservation, brightening, darkening, clipping |
| `test_synthesise_*` | Returns PIL Image of correct size |
| `test_augmenter_generate` | Saves correct number of `.jpg` files |
| `test_augmenter_augment_split` | Counts reach target_min after augmentation |
| `test_augmenter_no_generation_when_above_target` | No rows added when already at target |
| `test_augmenter_raises_without_no_anomaly` | ValueError if base pool is missing |

### `tests/test_dataset.py`

Requires the real dataset to be cloned (or existing split CSVs).

| Test | What it checks |
|---|---|
| `test_splits_sizes` | 14 000 / 3 000 / 3 000 |
| `test_splits_stratification` | All 12 classes present in each split |
| `test_splits_no_overlap` | No image path appears in more than one split |
| `test_dataset_item_shape` | Returns `(3, 224, 224)` tensor + int label |
| `test_dataset_train_transform_augments` | Train pipeline longer than val pipeline |

---

## 11. Reproducing the published results

```bash
# 1. Clone dataset
git clone https://github.com/RaptorMaps/InfraredSolarModules data/raw/InfraredSolarModules

# 2. Baseline
uv run python -m training.train --config configs/baseline.yaml
# → checkpoints/baseline_best.pt

# 3. Physics-augmented  (generates synthetic images first)
uv run python -m training.train --config configs/augmented.yaml
# → checkpoints/physics-augmented_best.pt
```

Expected results:

| Run | Test acc | Macro F1 |
|---|---|---|
| baseline | 69.0 % | 0.562 |
| physics-augmented | 78.8 % | 0.647 |

The augmenter generated **1 388 synthetic images** across `Diode-Multi`, `Hot-Spot`,
`Hot-Spot-Multi`, and `Soiling` (the four classes below 500 in the train split).

---

## 12. Known limitations & next steps

### Soiling

Blob geometry is a poor approximation of real soiling (irregular, spatially diffuse).
F1 improvement was the weakest of the four augmented classes (+0.066).
Better geometry: stochastic fractal patches or real soiling masks from an image segmentation step.

### Solver speed

Jacobi iteration on a 64×64 grid converges in ~100–200 iterations, which is fast enough
for offline augmentation. If online augmentation is needed (per-batch generation),
switch to a FFT-based Poisson solver for O(n log n) instead of O(n²) per iteration.

### Backbone

EfficientNet-B0 is small and fast. If compute allows, B3/B5 would likely push
macro F1 further without any change to the augmentation pipeline.

### Multi-label

The dataset is single-label per module. Real inspections often have co-occurring
faults (e.g. hot spot + cracking). Multi-label reformulation would need a re-annotation pass.

### Deployment

No inference script exists yet. Next step would be a `training/infer.py` that accepts
a directory of JPEG images and outputs a CSV of `(filename, predicted_class, confidence)`.
