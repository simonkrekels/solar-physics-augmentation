"""Multi-seed, multi-condition verification of the augmentation effect, with
per-class recall bootstrap confidence intervals.

Trains EfficientNet-B0 on the *same fixed splits* under several training-data
conditions, across several seeds, on the tuned 15-epoch schedule:

  clean        real data only (control)
  oversample   rare anomaly classes duplicated (with replacement) up to
               target_min — a rebalance-only control with no new pixels
  randaugment  real data + RandAugment — a strong generic-augmentation control
  physics      real data + physics-synthetic samples up to target_min

`oversample` and `physics` top up exactly the same classes to the same target,
so `physics − oversample` isolates the synthetic *signal* from mere rebalancing.
The test set is identical across every run, so comparisons are paired.

Raw per-image test predictions are saved to `verify_preds.csv`; the analysis
(per-class recall + bootstrap CIs + paired condition deltas) is recomputable
from that file alone via `--analyze-only`.

Usage:
    uv run python -m training.verify_seeds --seeds 42 1 2
    uv run python -m training.verify_seeds --analyze-only        # re-report only
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from torchvision import transforms

import training.dataset as dataset
from training.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PROCESSED_DIR,
    load_splits,
    make_splits,
    splits_exist,
)
from training.sweep_backbones import input_size_for, train_one
from training.train import get_device
from augmentation.heat_equation import ANOMALY_CLASSES, SyntheticAugmenter

CONDITIONS = ["clean", "oversample", "randaugment", "physics"]
PREDS_PATH = Path("verify_preds.csv")
SUMMARY_PATH = Path("verify_summary.csv")


# --------------------------------------------------------------------------- #
# Condition builders
# --------------------------------------------------------------------------- #
def make_strong_transform(img_size: int) -> transforms.Compose:
    """Train transform with RandAugment on top of domain-valid flips."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def oversample_rare(train_df: pd.DataFrame, target_min: int, rng: np.random.Generator) -> pd.DataFrame:
    """Duplicate real images (with replacement) so every anomaly class below
    *target_min* reaches it — same classes/targets as SyntheticAugmenter."""
    counts = train_df["label"].value_counts()
    extra = []
    for cls in ANOMALY_CLASSES:
        needed = target_min - int(counts.get(cls, 0))
        if needed <= 0:
            continue
        pool = train_df[train_df["label"] == cls]
        extra.append(pool.iloc[rng.integers(0, len(pool), size=needed)])
    if not extra:
        return train_df
    return pd.concat([train_df, *extra], ignore_index=True)


def build_condition(name: str, train_df: pd.DataFrame, seed: int, target_min: int, img_size: int):
    """Return (train_df_for_condition, train_transform_or_None)."""
    if name == "clean":
        return train_df, None
    if name == "oversample":
        return oversample_rare(train_df, target_min, np.random.default_rng(seed)), None
    if name == "randaugment":
        return train_df, make_strong_transform(img_size)
    if name == "physics":
        aug = SyntheticAugmenter(train_df, seed=seed).augment_split(
            train_df, target_min=target_min, output_dir=PROCESSED_DIR / "synthetic")
        return aug, None
    raise ValueError(f"unknown condition: {name}")


def append_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


# --------------------------------------------------------------------------- #
# Training phase
# --------------------------------------------------------------------------- #
def run_training(args, train_df, val_df, test_df, classes, device) -> None:
    if not args.append:
        for stale in (PREDS_PATH, SUMMARY_PATH):
            stale.unlink(missing_ok=True)

    done = set()
    if args.append and PREDS_PATH.exists():
        ex = pd.read_csv(PREDS_PATH, usecols=["seed", "condition"]).drop_duplicates()
        done = {(int(s), c) for s, c in ex.itertuples(index=False)}

    img_size = input_size_for(args.model)
    common = dict(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate,
        weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
        patience=args.patience,
    )

    for seed in args.seeds:
        for cond in args.conditions:
            if (seed, cond) in done:
                print(f"skip seed {seed} :: {cond} (already in {PREDS_PATH})")
                continue
            # A trailing "_nw" means: same data condition, but class weighting OFF.
            base = cond[:-3] if cond.endswith("_nw") else cond
            use_weights = not cond.endswith("_nw")
            print(f"===== seed {seed} :: {cond}  (class_weights={use_weights}) =====")
            cond_df, tfm = build_condition(base, train_df, seed, args.target_min_samples, img_size)
            try:
                res = train_one(args.model, cond_df, val_df, test_df, classes, device,
                                seed=seed, train_transform=tfm, return_preds=True,
                                class_weights=use_weights, **common)
            except Exception as exc:  # noqa: BLE001 — keep the long sweep alive
                print(f"  !! {cond} (seed {seed}) failed: {exc}")
                continue

            append_csv(pd.DataFrame({
                "seed": seed, "condition": cond,
                "idx": np.arange(len(res["labels"])),
                "y_true": [classes[i] for i in res["labels"]],
                "y_pred": [classes[i] for i in res["preds"]],
            }), PREDS_PATH)
            append_csv(pd.DataFrame([{
                "seed": seed, "condition": cond, "test_acc": res["test_acc"],
                "macro_f1": res["macro_f1"], "macro_recall": res["macro_recall"],
                "train_min": res["train_min"],
            }]), SUMMARY_PATH)
            print(f"  acc {res['test_acc']:.4f}  macroF1 {res['macro_f1']:.4f}  "
                  f"macroRecall {res['macro_recall']:.4f}\n")


# --------------------------------------------------------------------------- #
# Analysis phase — per-class recall bootstrap CIs + paired condition deltas
# --------------------------------------------------------------------------- #
def _recall_vec(true_codes: np.ndarray, correct: np.ndarray, k: int) -> np.ndarray:
    sums = np.bincount(true_codes, weights=correct, minlength=k)
    cnts = np.bincount(true_codes, minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnts > 0, sums / cnts, np.nan)


def analyze(target_min: int, n_boot: int, baseline: str = "physics") -> None:
    if not PREDS_PATH.exists():
        print(f"No {PREDS_PATH}; nothing to analyze.")
        return
    preds = pd.read_csv(PREDS_PATH)
    present = list(preds["condition"].unique())
    order = CONDITIONS + [f"{c}_nw" for c in CONDITIONS]
    conditions = [c for c in order if c in present] + [c for c in present if c not in order]
    seeds = sorted(int(s) for s in preds["seed"].unique())
    classes = sorted(preds["y_true"].unique())
    code = {c: i for i, c in enumerate(classes)}
    k = len(classes)

    counts = load_splits()["train"]["label"].value_counts()
    rare = [c for c in ANOMALY_CLASSES if int(counts.get(c, 0)) < target_min]
    rare_codes = [code[c] for c in rare]

    # Seed-averaged per-example correctness, aligned by idx (shared test order).
    preds["correct"] = (preds["y_true"] == preds["y_pred"]).astype(float)
    true_by_idx = preds.groupby("idx")["y_true"].first()
    idxs = true_by_idx.index.to_numpy()
    true_codes = np.array([code[true_by_idx[i]] for i in idxs])
    corr = {c: preds[preds.condition == c].groupby("idx")["correct"].mean()
            .reindex(idxs).to_numpy() for c in conditions}
    n = len(idxs)

    # Point estimates and bootstrap distributions.
    rng = np.random.default_rng(0)
    point = {c: _recall_vec(true_codes, corr[c], k) for c in conditions}
    boot = {c: np.empty((n_boot, k)) for c in conditions}
    for b in range(n_boot):
        s = rng.integers(0, n, n)
        tc = true_codes[s]
        for c in conditions:
            boot[c][b] = _recall_vec(tc, corr[c][s], k)

    def ci(arr: np.ndarray) -> tuple[float, float]:
        return float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))

    # ---- Headline summary (per-seed mean ± std) ----
    print("\n" + "=" * 78)
    print(f"Conditions: {conditions}   seeds: {seeds}   bootstrap: {n_boot}")
    print("Headline metrics (mean ± std across seeds):\n")
    print(f"  {'condition':12s} {'acc':>16s} {'macroF1':>16s} {'macroRecall':>16s}")
    per_seed = []
    for c in conditions:
        accs, f1s, recs = [], [], []
        for seed in seeds:
            g = preds[(preds.condition == c) & (preds.seed == seed)]
            if g.empty:
                continue
            accs.append((g.y_true == g.y_pred).mean())
            f1s.append(f1_score(g.y_true, g.y_pred, labels=classes, average="macro", zero_division=0))
            recs.append(recall_score(g.y_true, g.y_pred, labels=classes, average="macro", zero_division=0))
        per_seed.append((c, accs, f1s, recs))

        def ms(xs):
            return f"{np.mean(xs):.4f}±{np.std(xs):.4f}" if len(xs) > 1 else f"{xs[0]:.4f}"
        print(f"  {c:12s} {ms(accs):>16s} {ms(f1s):>16s} {ms(recs):>16s}")

    # ---- Per-class recall with bootstrap 95% CI ----
    print("\n" + "=" * 78)
    print("Per-class recall  (point [95% CI]);  * = rare class topped up to "
          f"{target_min}\n")
    header = "  " + f"{'class':16s}" + "".join(f"{c:>22s}" for c in conditions)
    print(header)
    rows_out = []
    for cls in classes:
        ci_idx = code[cls]
        line = "  " + f"{('* ' + cls) if cls in rare else cls:16s}"
        for c in conditions:
            lo, hi = ci(boot[c][:, ci_idx])
            pt = point[c][ci_idx]
            line += f"{pt:6.3f} [{lo:.2f},{hi:.2f}]".rjust(22)
            rows_out.append({"class": cls, "rare": cls in rare, "condition": c,
                             "recall": round(pt, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
        print(line)
    pd.DataFrame(rows_out).to_csv("verify_perclass.csv", index=False)

    # ---- Aggregate recalls + paired deltas vs baseline ----
    macro_boot = {c: np.nanmean(boot[c], axis=1) for c in conditions}
    rare_boot = {c: np.nanmean(boot[c][:, rare_codes], axis=1) for c in conditions}
    macro_pt = {c: float(np.nanmean(point[c])) for c in conditions}
    rare_pt = {c: float(np.nanmean(point[c][rare_codes])) for c in conditions}

    print("\n" + "=" * 78)
    print(f"Aggregate recall  (point [95% CI]):\n")
    print(f"  {'condition':12s} {'macro-recall':>24s} {'rare-class recall':>26s}")
    for c in conditions:
        ml, mh = ci(macro_boot[c]); rl, rh = ci(rare_boot[c])
        print(f"  {c:12s} {f'{macro_pt[c]:.3f} [{ml:.2f},{mh:.2f}]':>24s} "
              f"{f'{rare_pt[c]:.3f} [{rl:.2f},{rh:.2f}]':>26s}")

    print("\n" + "=" * 78)
    print(f"Paired deltas vs '{baseline}'  (positive ⇒ {baseline} better; "
          "CI excluding 0 ⇒ significant):\n")
    deltas_out = []
    for c in conditions:
        if c == baseline:
            continue
        for label, bdist, bpt in (("macro-recall", macro_boot, macro_pt),
                                   ("rare-recall", rare_boot, rare_pt)):
            d = bdist[baseline] - bdist[c]
            lo, hi = ci(d)
            sig = "  *" if (lo > 0 or hi < 0) else ""
            pt = bpt[baseline] - bpt[c]
            print(f"  {baseline} − {c:12s} {label:14s} "
                  f"{pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]{sig}")
            deltas_out.append({"baseline": baseline, "vs": c, "metric": label,
                               "delta": round(pt, 4), "ci_lo": round(lo, 4),
                               "ci_hi": round(hi, 4), "significant": bool(lo > 0 or hi < 0)})
        print()
    pd.DataFrame(deltas_out).to_csv("verify_deltas.csv", index=False)
    print("Saved: verify_perclass.csv, verify_deltas.csv "
          "(+ verify_preds.csv, verify_summary.csv)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS,
                    choices=CONDITIONS + [f"{c}_nw" for c in CONDITIONS],
                    help="'_nw' suffix = same data, class weighting OFF")
    ap.add_argument("--append", action="store_true",
                    help="append to existing verify_preds.csv instead of overwriting")
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--learning_rate", type=float, default=3.0e-4)
    ap.add_argument("--weight_decay", type=float, default=1.0e-4)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--target_min_samples", type=int, default=500)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--baseline", default="physics",
                    help="condition to compute paired deltas against")
    ap.add_argument("--analyze-only", action="store_true",
                    help="skip training; recompute the report from verify_preds.csv")
    args = ap.parse_args()

    if not args.analyze_only:
        device = get_device()
        print(f"Device: {device}")
        splits = load_splits() if splits_exist() else make_splits(seed=42)
        train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
        classes = sorted(train_df["label"].unique().tolist())
        print(f"{len(classes)} classes | seeds {args.seeds} | conditions {args.conditions}\n")
        run_training(args, train_df, val_df, test_df, classes, device)

    analyze(args.target_min_samples, args.n_boot, baseline=args.baseline)


if __name__ == "__main__":
    main()
