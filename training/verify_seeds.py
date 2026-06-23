"""Multi-seed verification of the augmentation effect on the tuned (15-epoch)
schedule.

For each seed, fine-tunes EfficientNet-B0 twice on the *same fixed splits* —
once on clean data, once with physics-synthetic augmentation — and reports the
paired clean→augmented delta plus mean±std across seeds. Same training harness
as ``training/sweep_backbones.py`` (no W&B). The test set is identical across
every run, so the per-seed delta is a paired comparison.

Usage:
    uv run python -m training.verify_seeds --seeds 42 1 2 --epochs 15
"""

import argparse
import statistics

import pandas as pd

import training.dataset as dataset
from training.dataset import PROCESSED_DIR, load_splits, make_splits, splits_exist
from training.sweep_backbones import train_one
from training.train import get_device
from augmentation.heat_equation import SyntheticAugmenter


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--learning_rate", type=float, default=3.0e-4)
    ap.add_argument("--weight_decay", type=float, default=1.0e-4)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--target_min_samples", type=int, default=500)
    ap.add_argument("--output", default="verify_seeds.csv")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")

    splits = load_splits() if splits_exist() else make_splits(seed=42)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    classes = sorted(train_df["label"].unique().tolist())
    print(f"{len(classes)} classes, {len(train_df)} clean train images")
    print(f"Seeds: {args.seeds}  |  model: {args.model}  |  {args.epochs} epochs\n")

    common = dict(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate,
        weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
        patience=args.patience,
    )

    rows = []
    for seed in args.seeds:
        print(f"===== seed {seed} :: clean =====")
        clean = train_one(args.model, train_df, val_df, test_df, classes, device,
                          seed=seed, **common)
        clean.update(condition="clean", seed=seed)
        rows.append(clean)

        print(f"===== seed {seed} :: augmented =====")
        aug_df = SyntheticAugmenter(train_df, seed=seed).augment_split(
            train_df, target_min=args.target_min_samples, output_dir=PROCESSED_DIR / "synthetic",
        )
        aug = train_one(args.model, aug_df, val_df, test_df, classes, device,
                        seed=seed, **common)
        aug.update(condition="augmented", seed=seed)
        rows.append(aug)
        print()

    df = pd.DataFrame(rows)[["seed", "condition", "test_acc", "macro_f1", "train_min"]]
    df.to_csv(args.output, index=False)

    print("=" * 60)
    print("Per-run results:\n")
    print(df.to_string(index=False))

    # Paired deltas (same seed, same fixed test set)
    pivot = df.pivot(index="seed", columns="condition", values="test_acc")
    f1piv = df.pivot(index="seed", columns="condition", values="macro_f1")
    acc_deltas = (pivot["augmented"] - pivot["clean"]).tolist()
    f1_deltas = (f1piv["augmented"] - f1piv["clean"]).tolist()

    def ms(xs: list[float]) -> str:
        if len(xs) > 1:
            return f"{statistics.mean(xs):.4f} ± {statistics.stdev(xs):.4f}"
        return f"{xs[0]:.4f}"

    print("\n" + "=" * 60)
    print("Summary (mean ± std across seeds):\n")
    for cond in ("clean", "augmented"):
        sub = df[df.condition == cond]
        print(f"  {cond:10s}  acc {ms(sub.test_acc.tolist())}   "
              f"macroF1 {ms(sub.macro_f1.tolist())}")
    print(f"\n  augmentation effect (paired, augmented − clean):")
    print(f"    test acc  {ms(acc_deltas)}   per-seed {[round(d, 4) for d in acc_deltas]}")
    print(f"    macro F1  {ms(f1_deltas)}   per-seed {[round(d, 4) for d in f1_deltas]}")
    holds = all(d > 0 for d in acc_deltas)
    print(f"\n  augmentation helps on every seed: {holds}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
