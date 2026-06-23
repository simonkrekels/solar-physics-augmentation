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

<!-- RESULT-PENDING: filled in once the run completes -->

- augmented @ 15 ep **>** 78.4% → augmentation adds value beyond the schedule fix.
- augmented @ 15 ep **≈** 78%  → augmentation was largely substituting for a
  too-short training budget.
