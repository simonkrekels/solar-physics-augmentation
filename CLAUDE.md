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

## Repository layout

```
solar-physics-project/
├── data/
│   ├── raw/InfraredSolarModules/   # cloned dataset, never modified
│   └── processed/                  # train/val/test CSVs + synthetic/ subdir
├── augmentation/
│   └── heat_equation.py            # PDE solver, source factory, SyntheticAugmenter
├── training/
│   ├── dataset.py                  # PyTorch Dataset, make_splits, get_transforms
│   ├── model.py                    # EfficientNet-B0 via timm
│   ├── train.py                    # training loop with W&B logging
│   └── evaluate.py                 # run_eval, classification_report_dict
├── notebooks/
│   ├── 01_eda.ipynb                # class distribution, image grid, dimension stats
│   ├── 02_error_analysis.ipynb     # confusion matrices, per-class F1, fixed-failure gallery
│   └── 03_augmentation_demo.ipynb  # PDE fields, pipeline, gallery, diversity, intensity sweep
├── configs/
│   ├── baseline.yaml               # no augmentation
│   └── augmented.yaml              # physics_synthetic: true, target_min_samples: 500
├── checkpoints/                    # gitignored — baseline_best.pt, physics-augmented_best.pt
└── tests/
    ├── test_dataset.py
    └── test_heat_equation.py
```

## Architecture overview

### Data pipeline
`training/dataset.py` loads from the split CSVs in `data/processed/`. Splits are stratified (70/15/15) — regenerating must use `seed=42`. The 12th class is `Soiling` (not `Short` as the project plan says).

### Model
`training/model.py` wraps `timm.create_model('efficientnet_b0', pretrained=True, num_classes=N)`. Classification head replaced; ImageNet backbone kept for fine-tuning.

### Training loop
`training/train.py` reads a YAML config, applies inverse-frequency class weights to CrossEntropyLoss, uses AdamW + 5-epoch linear warmup → CosineAnnealingLR, early stopping patience=5, and logs to W&B. Run as a module: `uv run python -m training.train --config configs/baseline.yaml`.

When `augmentation.physics_synthetic: true` in config, `SyntheticAugmenter.augment_split` extends only `train_df` before the DataLoader is built — val and test are always clean real images.

### Physics augmentation
`augmentation/heat_equation.py` exports:
- `solve_heat_2d(source, kappa, max_iter, tol)` — vectorised Jacobi solver, Neumann BC via edge-padding, mean-subtracted each step
- `make_source(class_name, grid_h, grid_w, rng)` — 11 class-specific geometries (Gaussian blobs, strips, rectangles, lines, uniform cooling)
- `blend_heat_patch(base_img, heat, intensity_scale)` — resizes T to image size, normalises, adds to all RGB channels
- `synthesise(class_name, base_img, ...)` — single-image convenience wrapper
- `SyntheticAugmenter` — bulk generator; `augment_split(train_df, target_min, output_dir)` tops up every anomaly class below `target_min`

Synthetic images are saved to `data/processed/synthetic/<class_name>/syn_NNNNN.jpg`.

### Experiment results
| Model | Test acc | Macro F1 |
|---|---|---|
| Baseline | 69.0% | 0.562 |
| Physics-augmented | 78.8% | 0.647 |

Augmenter generated 1,388 synthetic images (4 classes below 500: `Diode-Multi`, `Hot-Spot`, `Hot-Spot-Multi`, `Soiling`). Rarest classes gained most. `Soiling` improved least (+0.066 F1) — blob geometry is a weak model for real soiling.

Both numbers reproduce exactly, but **the +9.8 does not survive rigorous evaluation** — treat the table above as the *as-reported* headline, not a validated claim. Two stages of scrutiny (see `EXPERIMENTS.md`):
1. **Schedule.** The 30-epoch baseline is undertrained (slow cosine + patience-5 early stop cuts it at epoch 12). A clean 15-epoch run reaches ~78% with no augmentation, closing ~+9 of the +9.8.
2. **Honest baselines (`training/verify_seeds.py`, 4 conditions × 3 seeds, bootstrap CIs).** The remaining ~+1.9 acc is a *rebalancing* artifact: physics is statistically indistinguishable from plain **oversampling** (duplicating real images — which slightly beats it: 80.7% vs 80.0%), the gain comes from the **majority** class not the rare ones, and on **rare-class recall** physics is the worst of {clean, oversample, randaugment, physics} and significantly below clean. So the current heat-equation generator (steady-state blobs, additive RGB blend) is a weaker-than-duplication rebalancing trick, not useful rare-fault signal. This indicts the implementation, not physics-augmentation in principle; `docs/AUGMENTATION_ROADMAP.md` tracks realism upgrades to test against this harness. EfficientNet-B0 is the measured backbone-sweep winner (`training/sweep_backbones.py`).

### Experiment tracking
Both runs log to W&B project `solar-thermal-cv`. Model weights and split CSVs logged as artifacts.

## Dataset

```bash
git clone https://github.com/RaptorMaps/InfraredSolarModules data/raw/InfraredSolarModules
```

12 classes: `No-Anomaly`, `Hot-Spot`, `Hot-Spot-Multi`, `Diode`, `Diode-Multi`, `Cell`, `Cell-Multi`, `Shadowing`, `Offline-Module`, `Vegetation`, `Cracking`, `Soiling`. Heavy imbalance: `No-Anomaly` is ~50% of data.
