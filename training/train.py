import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from training.dataset import (
    PROCESSED_DIR,
    SolarModuleDataset,
    load_splits,
    make_splits,
    splits_exist,
)
from training.evaluate import classification_report_dict, run_eval
from training.model import build_model
from augmentation.heat_equation import SyntheticAugmenter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_class_weights(df, classes: list[str]) -> torch.Tensor:
    counts = df["label"].value_counts()
    weights = torch.tensor([1.0 / counts[c] for c in classes], dtype=torch.float32)
    return weights / weights.mean()


def train(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device()
    print(f"Device: {device}")

    if not splits_exist():
        print("Creating stratified splits...")
        splits = make_splits(seed=cfg["seed"])
    else:
        splits = load_splits()

    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    aug_cfg = cfg.get("augmentation", {})
    if aug_cfg.get("physics_synthetic", False):
        print("Generating physics-informed synthetic samples …")
        augmenter = SyntheticAugmenter(train_df, seed=cfg["seed"])
        train_df = augmenter.augment_split(
            train_df,
            target_min=aug_cfg.get("target_min_samples", 500),
            output_dir=PROCESSED_DIR / "synthetic",
        )
        print(f"  Extended train set: {len(train_df)} images")

    classes = sorted(train_df["label"].unique().tolist())
    n_classes = len(classes)
    print(f"{n_classes} classes")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        SolarModuleDataset(train_df, classes, split="train"),
        batch_size=cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        SolarModuleDataset(val_df, classes, split="val"),
        batch_size=cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        SolarModuleDataset(test_df, classes, split="test"),
        batch_size=cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=pin_memory,
    )

    model = build_model(
        n_classes, model_name=cfg.get("model", "efficientnet_b0"), pretrained=cfg["pretrained"]
    ).to(device)
    # Default loss: logit adjustment (Menon et al. 2020). Our audit showed this
    # beats inverse-frequency class weighting on rare-fault recall (τ=1.5:
    # +0.084 rare recall, CI excludes 0); see docs/findings.pdf. Set
    # `loss.type: ce` to fall back to class-weighted cross-entropy.
    loss_cfg = cfg.get("loss", {})
    if loss_cfg.get("type", "logit_adj") == "logit_adj":
        tau = loss_cfg.get("tau", 1.5)
        counts = train_df["label"].value_counts()
        prior = torch.tensor([counts.get(c, 0) for c in classes],
                             dtype=torch.float32, device=device)
        log_prior = tau * torch.log(prior / prior.sum() + 1e-12)
        _ce = nn.CrossEntropyLoss()
        def criterion(out, lab):  # logit-adjusted; infer on raw logits
            return _ce(out + log_prior, lab)
        print(f"Loss: logit-adjusted cross-entropy (τ={tau})")
    else:
        criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_df, classes).to(device))
        print("Loss: inverse-frequency class-weighted cross-entropy")
    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=cfg["warmup_epochs"])
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=cfg["num_epochs"] - cfg["warmup_epochs"], eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[cfg["warmup_epochs"]]
    )

    wandb_cfg = cfg.get("wandb", {})
    run = wandb.init(
        project=wandb_cfg.get("project", "solar-thermal-cv"),
        name=wandb_cfg.get("run_name", "run"),
        config=cfg,
    )
    artifact = wandb.Artifact("splits", type="dataset")
    for name in ("train", "val", "test"):
        artifact.add_file(str(PROCESSED_DIR / f"{name}.csv"))
    run.log_artifact(artifact)

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"{wandb_cfg.get('run_name', 'run')}_best.pt"

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg["num_epochs"]):
        model.train()
        train_loss = train_correct = train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * len(labels)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total   += len(labels)

        scheduler.step()
        train_loss /= train_total
        train_acc   = train_correct / train_total
        val_loss, val_acc, _, _ = run_eval(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:3d}/{cfg['num_epochs']}  "
            f"train {train_loss:.4f}/{train_acc:.3f}  "
            f"val {val_loss:.4f}/{val_acc:.3f}  lr {current_lr:.2e}"
        )
        wandb.log({
            "train/loss": train_loss, "train/acc": train_acc,
            "val/loss":   val_loss,   "val/acc":   val_acc,
            "epoch": epoch + 1,       "lr":        current_lr,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Final evaluation on test set
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    _, test_acc, test_preds, test_labels = run_eval(model, test_loader, device)
    print(f"\nTest accuracy: {test_acc:.4f}")

    report = classification_report_dict(test_preds, test_labels, classes)
    tbl = wandb.Table(columns=["class", "precision", "recall", "f1", "support"])
    for cls in classes:
        r = report[cls]
        tbl.add_data(cls, round(r["precision"], 4), round(r["recall"], 4),
                     round(r["f1-score"], 4), int(r["support"]))
    wandb.log({
        "test/accuracy": test_acc,
        "test/classification_report": tbl,
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, y_true=test_labels, preds=test_preds, class_names=classes
        ),
    })

    model_artifact = wandb.Artifact("model", type="model")
    model_artifact.add_file(str(ckpt_path))
    run.log_artifact(model_artifact)
    run.finish()
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    train(ap.parse_args().config)
