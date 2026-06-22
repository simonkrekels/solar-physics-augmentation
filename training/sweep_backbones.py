"""Fine-tune several timm backbones on the clean (un-augmented) splits and
print a comparison table against the EfficientNet-B0 baseline.

This isolates the effect of the backbone choice: all models see the same
stratified splits, class-weighted loss, optimiser, and schedule as
``training/train.py`` — only ``model_name`` changes. No W&B, no synthetic
augmentation, so numbers are directly comparable to the 69.0% baseline.

Usage:
    uv run python -m training.sweep_backbones \
        --models efficientnet_b0 resnet50 convnext_tiny mobilenetv3_large_100 \
        --epochs 15 --output sweep_results.csv
"""

import argparse
import time

import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

import training.dataset as dataset
from training.dataset import (
    SolarModuleDataset,
    load_splits,
    make_splits,
    splits_exist,
)
from training.evaluate import classification_report_dict, run_eval
from training.model import build_model
from training.train import compute_class_weights, get_device, set_seed

DEFAULT_MODELS = [
    "efficientnet_b0",       # the current baseline
    "resnet50",
    "convnext_tiny",
    "mobilenetv3_large_100",
]


def input_size_for(model_name: str) -> int:
    """The square input resolution the pretrained weights expect."""
    cfg = timm.get_pretrained_cfg(model_name, allow_unregistered=True)
    if cfg is not None and getattr(cfg, "input_size", None):
        return cfg.input_size[-1]
    return 224


def train_one(
    model_name: str,
    train_df,
    val_df,
    test_df,
    classes: list[str],
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    patience: int,
    seed: int,
) -> dict:
    """Fine-tune one backbone and return its test metrics."""
    set_seed(seed)  # identical init/order for every model

    # Match the transform resolution to what the pretrained weights expect.
    dataset.IMG_SIZE = input_size_for(model_name)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        SolarModuleDataset(train_df, classes, split="train"),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        SolarModuleDataset(val_df, classes, split="val"),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        SolarModuleDataset(test_df, classes, split="test"),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory,
    )

    model = build_model(len(classes), model_name=model_name, pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_df, classes).to(device))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6),
        ],
        milestones=[warmup_epochs],
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_loss, val_acc, _, _ = run_eval(model, val_loader, device)
        print(f"  [{model_name}] epoch {epoch+1:2d}/{epochs}  val {val_loss:.4f}/{val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [{model_name}] early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_acc, preds, labels = run_eval(model, test_loader, device)
    report = classification_report_dict(preds, labels, classes)

    return {
        "model": model_name,
        "params_M": round(n_params / 1e6, 2),
        "input_size": dataset.IMG_SIZE,
        "test_acc": round(test_acc, 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "train_min": round((time.time() - t0) / 60, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="timm model names to compare")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--learning_rate", type=float, default=3.0e-4)
    ap.add_argument("--weight_decay", type=float, default=1.0e-4)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="sweep_results.csv")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")

    set_seed(args.seed)
    splits = load_splits() if splits_exist() else make_splits(seed=args.seed)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    classes = sorted(train_df["label"].unique().tolist())
    print(f"{len(classes)} classes, {len(train_df)} train images\n")

    results = []
    for model_name in args.models:
        print(f"=== {model_name} ===")
        try:
            results.append(train_one(
                model_name, train_df, val_df, test_df, classes, device,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate,
                weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
                patience=args.patience, seed=args.seed,
            ))
        except Exception as exc:  # noqa: BLE001 — keep sweeping if one backbone fails
            print(f"  !! {model_name} failed: {exc}")
        print()

    if not results:
        print("No models trained successfully.")
        return

    df = pd.DataFrame(results).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    df.to_csv(args.output, index=False)
    print("=" * 64)
    print("Backbone comparison (sorted by macro F1):\n")
    print(df.to_string(index=False))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
