"""Improved (v2) thermal-anomaly synthesiser.

Motivated by the D1 domain-gap diagnostic: an EfficientNet separates the v1
(steady-state) synthetics from real faults at AUC 0.97, and the gap survives
heavy blur — i.e. the *coarse shape* is wrong. Visual inspection of real 24×40
thermal crops vs v1 synthetics shows four concrete defects this module fixes:

1. **Over-diffused, symmetric blobs.** v1 solves the *steady-state* equation, so
   every source relaxes to a smooth round blob. → v2 solves the **transient**
   equation and samples a finite diffusion time, keeping signatures localized and
   shape-preserving (a line stays a line, a spot stays a spot).
2. **Aspect warp.** v1 solves on a 64×64 *square* grid then resizes to 24×40. →
   v2 works at the **native image aspect** (super-sampled), so nothing is squished.
3. **No sensor grain.** Real crops have visible thermal-sensor noise; v1 blobs are
   implausibly smooth. → v2 adds **grain calibrated to real images**.
4. **Wrong class models.** Real "Soiling" is *bright/streaky*, not a smooth
   darkening; real Diode bands are *subtle*. → v2 fixes per-class geometry and
   **calibrates contrast** to the real per-class statistic.

This module is independent of `heat_equation.py` (which the running experiments
depend on) and writes to `data/processed/synthetic_v2/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

from augmentation.heat_equation import ANOMALY_CLASSES  # re-export, read-only


# --------------------------------------------------------------------------- #
# Transient diffusion solver
# --------------------------------------------------------------------------- #
def diffuse(field: np.ndarray, alpha: float = 0.2, steps: int = 20) -> np.ndarray:
    """Diffuse an initial field for *steps* explicit time-steps (Neumann BC).

    ∂T/∂t = α∇²T, forward Euler. `alpha ≤ 0.25` is stable. Unlike the v1
    steady-state solve, a *small* step count keeps the source shape localized;
    a larger count spreads it — a physically-meaningful diversity knob.
    """
    T = field.astype(np.float64).copy()
    for _ in range(steps):
        Tp = np.pad(T, 1, mode="edge")
        lap = (Tp[:-2, 1:-1] + Tp[2:, 1:-1] + Tp[1:-1, :-2] + Tp[1:-1, 2:] - 4.0 * T)
        T += alpha * lap
    return T


# --------------------------------------------------------------------------- #
# Source geometry — native aspect, corrected per class
# --------------------------------------------------------------------------- #
def _spot(H, W, cy, cx, sigma, amp):
    y, x = np.ogrid[:H, :W]
    return amp * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma ** 2))


def make_source_v2(class_name: str, H: int, W: int, rng: np.random.Generator):
    """Return (source, steps): a heat source on an H×W (native-aspect) grid and
    the diffusion-time (step count) to apply. Positive heats, negative cools."""
    q = np.zeros((H, W), dtype=np.float64)
    s = min(H, W)

    if class_name == "Hot-Spot":
        # one *small, sharp* spot — real hot spots are a few pixels across
        cy, cx = rng.uniform(0.2, 0.8, 2) * (H, W)
        q += _spot(H, W, cy, cx, sigma=rng.uniform(0.025, 0.05) * s, amp=10.0)
        steps = int(rng.integers(4, 14))

    elif class_name == "Hot-Spot-Multi":
        for _ in range(int(rng.integers(2, 5))):
            cy, cx = rng.uniform(0.1, 0.9, 2) * (H, W)
            q += _spot(H, W, cy, cx, sigma=rng.uniform(0.02, 0.045) * s, amp=9.0)
        steps = int(rng.integers(4, 14))

    elif class_name == "Diode-Multi":
        # subtle vertical sub-string bands — real Diode bands are faint, not stripes
        width = max(2, int(W * rng.uniform(0.14, 0.24)))
        for col in rng.choice(W - width, size=int(rng.integers(2, 4)), replace=False):
            band = rng.uniform(0.0, 0.3) + 0.85     # mild, slightly varying
            q[:, int(col):int(col) + width] += 1.6 * band
        steps = int(rng.integers(20, 40))           # diffuse more → soft, not striped

    elif class_name == "Soiling":
        # real soiling: irregular *bright* streaky patches, often edge-weighted
        for _ in range(int(rng.integers(2, 5))):
            cy = rng.uniform(0.0, 1.0) * H
            cx = rng.choice([rng.uniform(0, 0.25), rng.uniform(0.75, 1.0)]) * W  # edges
            sy, sx = rng.uniform(0.08, 0.2) * H, rng.uniform(0.03, 0.08) * W
            y, x = np.ogrid[:H, :W]
            q += 3.0 * np.exp(-((y - cy) ** 2 / (2 * sy ** 2) + (x - cx) ** 2 / (2 * sx ** 2)))
        # correlated texture so it isn't a clean gaussian
        q += 1.2 * gaussian_filter(rng.standard_normal((H, W)), sigma=0.04 * s)
        steps = int(rng.integers(6, 16))

    else:  # generic fallback (other classes aren't currently topped up)
        cy, cx = rng.uniform(0.2, 0.8, 2) * (H, W)
        q += _spot(H, W, cy, cx, sigma=0.1 * s, amp=6.0)
        steps = int(rng.integers(10, 30))

    return q, steps


# --------------------------------------------------------------------------- #
# Blending — calibrated contrast + sensor grain
# --------------------------------------------------------------------------- #
def synthesise_v2(
    class_name: str,
    base_img: Image.Image,
    rng: Optional[np.random.Generator] = None,
    *,
    supersample: int = 3,
    contrast: float = 35.0,
    grain_std: float = 5.0,
) -> Image.Image:
    """Generate one v2 synthetic image.

    `contrast` is the target peak fault amplitude in grey levels (calibrate per
    class from real data). `grain_std` is the sensor-noise std to inject.
    """
    if rng is None:
        rng = np.random.default_rng()
    base = np.array(base_img.convert("L"), dtype=np.float64)   # work in 1 channel
    H, W = base.shape
    gh, gw = H * supersample, W * supersample

    source, steps = make_source_v2(class_name, gh, gw, rng)
    heat = diffuse(source, alpha=0.2, steps=steps * supersample)
    heat = zoom(heat, (H / gh, W / gw), order=1)               # native aspect → no warp

    peak = max(float(np.abs(heat).max()), 1e-9)
    delta = (heat / peak) * contrast                           # calibrated contrast
    out = base + delta
    out += rng.normal(0.0, grain_std, size=out.shape)          # sensor grain
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out).convert("RGB")


# --------------------------------------------------------------------------- #
# Bulk augmenter — drop-in alternative to SyntheticAugmenter, calibrated to real
# --------------------------------------------------------------------------- #
def _load_L(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float64)


class SyntheticAugmenterV2:
    """Like `SyntheticAugmenter` but uses the v2 (transient, grained, calibrated)
    synthesiser. Grain and per-class contrast are measured from the real train
    images, so the synthetics match real sensor statistics.
    """

    def __init__(self, train_df, seed: int = 42, supersample: int = 3):
        self.rng = np.random.default_rng(seed)
        self.supersample = supersample
        self._base_paths = train_df.loc[train_df["label"] == "No-Anomaly", "path"].tolist()
        if not self._base_paths:
            raise ValueError("train_df must contain at least one No-Anomaly image")

        # Calibrate sensor grain (global) and target contrast (per class) from real data.
        sample = train_df["path"].sample(min(60, len(train_df)), random_state=seed)
        self.grain_std = float(np.mean([(im - gaussian_filter(im, 1.2)).std()
                                        for im in (_load_L(p) for p in sample)]))
        self.contrast: dict[str, float] = {}
        for cls in ANOMALY_CLASSES:
            paths = train_df.loc[train_df["label"] == cls, "path"].tolist()[:80]
            self.contrast[cls] = (float(np.mean([np.percentile(im, 99) - np.median(im)
                                                 for im in map(_load_L, paths)]))
                                  if paths else 35.0)

    def _random_base(self) -> Image.Image:
        return Image.open(self._base_paths[int(self.rng.integers(0, len(self._base_paths)))])

    def generate(self, class_name: str, n: int, output_dir: Path) -> list[Path]:
        out_cls = output_dir / class_name
        out_cls.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(n):
            img = synthesise_v2(class_name, self._random_base(), self.rng,
                                supersample=self.supersample,
                                contrast=self.contrast.get(class_name, 35.0),
                                grain_std=self.grain_std)
            p = out_cls / f"syn_{i:05d}.jpg"
            img.save(p, quality=90)
            paths.append(p)
        return paths

    def augment_split(self, train_df, target_min: int, output_dir: Path):
        import pandas as pd
        counts = train_df["label"].value_counts()
        new_rows = []
        for cls in ANOMALY_CLASSES:
            needed = target_min - int(counts.get(cls, 0))
            if needed <= 0:
                continue
            print(f"  [v2] generating {needed:4d} '{cls}' images …")
            new_rows.extend({"path": str(p), "label": cls}
                            for p in self.generate(cls, needed, output_dir))
        if not new_rows:
            return train_df
        return pd.concat([train_df, pd.DataFrame(new_rows)], ignore_index=True)
