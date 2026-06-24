"""D1 — synthetic-vs-real domain-gap discriminator.

Tests hypothesis H1 (the synthetic images look detectably "fake", so their
features don't transfer to real faults). Trains a binary classifier to separate
physics-synthetic images from *real images of the same anomaly classes*. A high
validation AUC means the synthetics are trivially separable — a domain gap a
fault classifier can shortcut on.

Confound control: synthetic images have been through one extra JPEG
decode→process→encode generation vs the real ones. To stop the discriminator
keying on compression artifacts instead of content, real images get one matching
in-memory JPEG round-trip. `--blur` additionally low-passes both sets, so a
still-high AUC implies the gap is in coarse *content/shape* (i.e. the physics),
not high-frequency artifacts.

Usage:
    uv run python -m training.diagnose_domain_gap
    uv run python -m training.diagnose_domain_gap --blur 3
"""

import argparse
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

import training.dataset as dataset
from training.dataset import IMAGENET_MEAN, IMAGENET_STD, PROCESSED_DIR, load_splits
from training.evaluate import run_eval  # noqa: F401 (kept for parity; not required)
from training.model import build_model
from training.train import get_device, set_seed
from torchvision import transforms

# The four classes that received synthetic top-ups.
AUG_CLASSES = ["Diode-Multi", "Hot-Spot", "Hot-Spot-Multi", "Soiling"]


def jpeg_roundtrip(img: Image.Image, quality: int = 90) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class DiscDataset(Dataset):
    def __init__(self, paths, labels, clss, blur: float):
        self.paths, self.labels, self.clss, self.blur = paths, labels, clss, blur
        self.t = transforms.Compose([
            transforms.Resize((dataset.IMG_SIZE, dataset.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.labels[i] == 0:            # real: match synthetic's extra JPEG generation
            img = jpeg_roundtrip(img)
        if self.blur > 0:
            img = img.filter(ImageFilter.GaussianBlur(self.blur))
        return self.t(img), self.labels[i], self.clss[i]


def gather(seed: int, syn_root: Path):
    rng = np.random.default_rng(seed)
    real = load_splits()["train"]
    real = real[real["label"].isin(AUG_CLASSES)]
    real_paths = real["path"].tolist()
    real_cls = real["label"].tolist()

    syn_paths, syn_cls = [], []
    for cls in AUG_CLASSES:
        for p in sorted((syn_root / cls).glob("*.jpg")):
            syn_paths.append(str(p)); syn_cls.append(cls)

    n = min(len(real_paths), len(syn_paths))            # balance the two sides
    ri = rng.permutation(len(real_paths))[:n]
    si = rng.permutation(len(syn_paths))[:n]
    paths = [real_paths[i] for i in ri] + [syn_paths[i] for i in si]
    labels = [0] * n + [1] * n
    clss = [real_cls[i] for i in ri] + [syn_cls[i] for i in si]

    idx = rng.permutation(len(paths))
    cut = int(0.8 * len(idx))
    tr, va = idx[:cut], idx[cut:]
    pick = lambda arr, ix: [arr[i] for i in ix]
    print(f"real {len(real_paths)} | synthetic {len(syn_paths)} | balanced n={n} per side")
    return (pick(paths, tr), pick(labels, tr), pick(clss, tr),
            pick(paths, va), pick(labels, va), pick(clss, va))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--blur", type=float, default=0.0,
                    help="Gaussian blur radius applied to both sets (low-pass test)")
    ap.add_argument("--synthetic-dir", default=str(PROCESSED_DIR / "synthetic"),
                    help="directory of synthetic images to test (e.g. .../synthetic_v2)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}  | blur={args.blur}  | synthetic={args.synthetic_dir}")

    trp, trl, trc, vap, val, vac = gather(args.seed, Path(args.synthetic_dir))
    pin = device.type == "cuda"
    train_loader = DataLoader(DiscDataset(trp, trl, trc, args.blur),
                              batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=pin)
    val_loader = DataLoader(DiscDataset(vap, val, vac, args.blur),
                            batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin)

    model = build_model(2, pretrained=True).to(device)
    opt = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_auc = 0.0
    for epoch in range(args.epochs):
        model.train()
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()

        model.eval()
        probs, ys, cs = [], [], []
        with torch.no_grad():
            for x, y, c in val_loader:
                p = model(x.to(device)).softmax(1)[:, 1].cpu().numpy()
                probs.extend(p); ys.extend(y.numpy()); cs.extend(c)
        ys = np.array(ys); probs = np.array(probs)
        auc = roc_auc_score(ys, probs)
        acc = ((probs > 0.5).astype(int) == ys).mean()
        best_auc = max(best_auc, auc)
        print(f"  epoch {epoch+1}/{args.epochs}  val AUC {auc:.3f}  acc {acc:.3f}")

    # Per-class detectability: how confidently is each class's synthetic flagged?
    print("\nMean P(synthetic) on validation, by class:")
    cs = np.array(cs)
    for cls in AUG_CLASSES:
        m_syn = (cs == cls) & (ys == 1)
        m_real = (cs == cls) & (ys == 0)
        if m_syn.any():
            print(f"  {cls:16s} synthetic {probs[m_syn].mean():.3f}   "
                  f"real {probs[m_real].mean() if m_real.any() else float('nan'):.3f}")

    print(f"\nBest val AUC: {best_auc:.3f}")
    verdict = ("LARGE domain gap — synthetics are trivially separable (H1 likely)"
               if best_auc > 0.9 else
               "MODERATE gap" if best_auc > 0.75 else
               "SMALL gap — synthetics are hard to tell from real")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
