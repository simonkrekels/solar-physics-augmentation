"""SimGAN-style learned refiner for physics-synthetic thermal faults.

Physics (heat_equation_v2) supplies the fault *structure* (and the label); this
network learns the *appearance* — the sensor texture / statistics the D1
discriminator keys on — without erasing the fault. Trained adversarially against
a small patch discriminator plus a self-regularization term λ‖R(x)−x‖₁ that
preserves content (Shrivastava et al., 2017).

Pipeline (24×40 grayscale, native resolution):
  physics-v2 (grain-off) image  →  R  →  refined image, saved to a new dir.

Validation is staged elsewhere (honest gates):
  Stage 1 — score refined-vs-real with an *independent* discriminator
            (`training/diagnose_domain_gap.py --synthetic-dir <out>`); target
            AUC ≤ 0.75 (baseline 0.93).
  Stage 2 — only if Stage 1 passes: `physics_refined` condition in verify_seeds.

Usage:
    uv run python -m augmentation.refiner --steps 3000 --lam 2.0 \
        --src-dir data/processed/synthetic_v2_nograin \
        --out-dir data/processed/synthetic_refined
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from training.dataset import PROCESSED_DIR, load_splits
from training.train import get_device, set_seed

AUG_CLASSES = ["Diode-Multi", "Hot-Spot", "Hot-Spot-Multi", "Soiling"]
H, W = 40, 24  # native crop size


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #
class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class Refiner(nn.Module):
    """Fully-conv residual net; output = x + bounded residual (size-preserving)."""

    def __init__(self, ch: int = 64, n_blocks: int = 4):
        super().__init__()
        self.inp = nn.Conv2d(1, ch, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(n_blocks)])
        self.out = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(self, x):
        h = self.blocks(F.relu(self.inp(x)))
        return torch.clamp(x + torch.tanh(self.out(h)), -1.0, 1.0)


class PatchD(nn.Module):
    """Local (PatchGAN-style) discriminator → per-patch real/fake logits."""

    def __init__(self, ch: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, 64, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class GlobalD(nn.Module):
    """Whole-image discriminator → one logit. D1 showed the domain gap is
    *global* (it survives blur), which a patch discriminator cannot perceive."""

    def __init__(self, ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True),       # 20x12
            nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True),  # 10x6
            nn.Conv2d(ch * 2, ch * 4, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True),  # 5x3
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch * 4, 1),
        )

    def forward(self, x):
        return self.net(x)


class History:
    """SimGAN replay buffer of refined images, to stabilise D."""

    def __init__(self, cap: int = 512):
        self.cap, self.data = cap, []

    def query(self, imgs: torch.Tensor) -> torch.Tensor:
        out = []
        for im in imgs:
            im = im.unsqueeze(0)
            if len(self.data) < self.cap:
                self.data.append(im.detach().clone()); out.append(im)
            elif random.random() < 0.5:
                i = random.randint(0, self.cap - 1)
                out.append(self.data[i].clone()); self.data[i] = im.detach().clone()
            else:
                out.append(im)
        return torch.cat(out, 0)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _load(path: str) -> np.ndarray:
    img = Image.open(path).convert("L").resize((W, H))
    return np.asarray(img, dtype=np.float32) / 127.5 - 1.0   # → [-1, 1]


def _stack(paths) -> torch.Tensor:
    return torch.from_numpy(np.stack([_load(p) for p in paths])[:, None])  # (N,1,H,W)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_refiner(real: torch.Tensor, syn: torch.Tensor, device, *,
                  steps: int, batch: int, lam: float, lr: float,
                  d_type: str = "global", k_g: int = 2, k_d: int = 1) -> Refiner:
    R = Refiner().to(device)
    D = (GlobalD() if d_type == "global" else PatchD()).to(device)
    print(f"  adversary: {d_type} discriminator")
    optR = torch.optim.Adam(R.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    buf = History()
    real, syn = real.to(device), syn.to(device)

    def rand(t, n):
        return t[torch.randint(0, len(t), (n,), device=device)]

    for step in range(1, steps + 1):
        # --- update refiner R ---
        for _ in range(k_g):
            x = rand(syn, batch)
            xr = R(x)
            adv = bce(D(xr), torch.ones_like(D(xr)))
            reg = lam * F.l1_loss(xr, x)
            optR.zero_grad(); (adv + reg).backward(); optR.step()
        # --- update discriminator D ---
        for _ in range(k_d):
            xr_real = rand(real, batch)
            fake = buf.query(R(rand(syn, batch)).detach())
            d_real = bce(D(xr_real), torch.ones_like(D(xr_real)))
            d_fake = bce(D(fake), torch.zeros_like(D(fake)))
            optD.zero_grad(); (0.5 * (d_real + d_fake)).backward(); optD.step()

        if step % 500 == 0 or step == 1:
            with torch.no_grad():
                x = rand(syn, 256); xr = R(x)
                p_fake = torch.sigmoid(D(xr)).mean().item()      # →0.5 = fooling D
                p_real = torch.sigmoid(D(rand(real, 256))).mean().item()
                delta = (xr - x).abs().mean().item() * 127.5     # mean edit, grey levels
            print(f"  step {step:5d}  D(real)={p_real:.2f} D(refined)={p_fake:.2f} "
                  f"  mean|Δ|={delta:.1f} gl  reg_lam={lam}")
    return R


@torch.no_grad()
def refine_dir(R: Refiner, src: Path, dst: Path, device) -> int:
    R.eval(); n = 0
    for cls in AUG_CLASSES:
        out = dst / cls; out.mkdir(parents=True, exist_ok=True)
        for p in sorted((src / cls).glob("*.jpg")):
            x = torch.from_numpy(_load(str(p)))[None, None].to(device)
            y = R(x)[0, 0].cpu().numpy()
            arr = np.clip((y + 1.0) * 127.5, 0, 255).astype(np.uint8)
            Image.fromarray(arr).convert("RGB").save(out / p.name, quality=90)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lam", type=float, default=1.0, help="self-regularization weight")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d-type", default="global", choices=["global", "patch"],
                    help="adversary: global (coarse-shape gap) or patch (local texture)")
    ap.add_argument("--src-dir", default=str(PROCESSED_DIR / "synthetic_v2_nograin"))
    ap.add_argument("--out-dir", default=str(PROCESSED_DIR / "synthetic_refined"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    tr = load_splits()["train"]
    real_paths = tr[tr["label"].isin(AUG_CLASSES)]["path"].tolist()
    src = Path(args.src_dir)
    syn_paths = [str(p) for cls in AUG_CLASSES for p in sorted((src / cls).glob("*.jpg"))]
    if not syn_paths:
        raise SystemExit(f"No synthetics in {src}; generate the v2 grain-off set first.")
    print(f"real fault images: {len(real_paths)} | synthetic inputs: {len(syn_paths)}")

    real, syn = _stack(real_paths), _stack(syn_paths)
    R = train_refiner(real, syn, device, steps=args.steps, batch=args.batch,
                      lam=args.lam, lr=args.lr, d_type=args.d_type)

    ckpt = Path("checkpoints"); ckpt.mkdir(exist_ok=True)
    torch.save(R.state_dict(), ckpt / "refiner.pt")
    n = refine_dir(R, src, Path(args.out_dir), device)
    print(f"Saved refiner → {ckpt/'refiner.pt'}; refined {n} images → {args.out_dir}")


if __name__ == "__main__":
    main()
