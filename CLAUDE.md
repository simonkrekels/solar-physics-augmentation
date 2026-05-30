# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This project is a physics-informed computer vision portfolio project: fine-tune an EfficientNet-B0 image classifier on thermal IR images of solar modules, then improve it using synthetic data augmentation grounded in the 2D heat equation. Full plan is in `project_plan_thermal_solar.md`.

## Tooling

Use `uv` for all package management. The venv is at `.venv/` and is already initialised. Never install packages in the global environment.

```bash
# Add a dependency
uv add <package>

# Run a script inside the venv
uv run python training/train.py --config configs/baseline.yaml

# Launch Jupyter
uv run jupyter notebook

# Run a single test file
uv run pytest tests/test_heat_equation.py -v
```

Commit after every significant change. Repo has no remote yet.

## Planned repository layout

```
solar-physics-project/
├── data/
│   ├── raw/InfraredSolarModules/   # cloned dataset, never modified
│   └── processed/                  # stratified train/val/test split CSVs
├── augmentation/
│   └── heat_equation.py            # 2D PDE solver + patch blending
├── training/
│   ├── dataset.py                  # PyTorch Dataset
│   ├── model.py                    # EfficientNet-B0 via timm
│   ├── train.py                    # training loop with W&B logging
│   └── evaluate.py                 # confusion matrix, per-class F1 helpers
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_error_analysis.ipynb
│   └── 03_augmentation_demo.ipynb
├── configs/
│   ├── baseline.yaml
│   └── augmented.yaml
└── tests/
```

## Architecture overview

### Data pipeline
`training/dataset.py` loads from the split CSVs in `data/processed/`. Splits are stratified (70/15/15) to handle class imbalance. Split CSVs are the source of truth for reproducibility — regenerating them must use `seed=42`.

### Model
`training/model.py` wraps `timm.create_model('efficientnet_b0', pretrained=True, num_classes=N)`. The classification head is replaced; ImageNet backbone weights are kept for fine-tuning.

### Training loop
`training/train.py` reads a YAML config, instantiates the model, applies inverse-frequency class weights to the cross-entropy loss, uses AdamW + cosine annealing with 5-epoch warm-up, and logs train/val metrics to W&B every epoch.

### Physics augmentation
`augmentation/heat_equation.py` solves ∇²T = −q/κ on a 2D grid via Gauss-Seidel iteration with Neumann boundary conditions. The resulting heat patch is blended onto a `No-Anomaly` base image to synthesise rare anomaly classes. Source geometry varies by class: point/Gaussian for `Hot-Spot`/`Cell`, elongated strip for `Diode`, large cool patch for `Shadowing`.

### Experiment tracking
Both the baseline and augmented runs log to the same W&B project (`solar-thermal-cv`) so the dashboard shows a direct comparison. Model weights and split CSVs are logged as W&B artifacts.

## Dataset

```bash
git clone https://github.com/RaptorMaps/InfraredSolarModules data/raw/InfraredSolarModules
```

12 classes: `No-Anomaly`, `Hot-Spot`, `Hot-Spot-Multi`, `Diode`, `Diode-Multi`, `Cell`, `Cell-Multi`, `Shadowing`, `Offline-Module`, `Vegetation`, `Cracking`, `Short`. Expect heavy imbalance toward `No-Anomaly`.
