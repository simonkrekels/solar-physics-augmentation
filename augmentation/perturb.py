"""Physics-grounded augmentation by perturbing *real* fault images.

The previous generators synthesised faults onto clean bases and hit a structural
domain gap (D1: AUC 0.93–0.97). This module instead starts from a *real* fault
crop — so the structure, background and sensor grain are real by construction —
and applies the heat equation to the fault itself.

Key fact: the heat equation's time-evolution operator is convolution with a
Gaussian ($\\sigma=\\sqrt{2Dt}$). So "advance the fault by a small thermal time"
is exactly a small Gaussian blur of the fault component. We decompose

    image = background  +  fault  +  grain

(background = large-σ blur; fault = coarse detail; grain = fine residual),
then physically perturb only the fault — diffuse it (thermal time) and scale its
amplitude (fault severity) — while keeping the *real* background and grain. The
result is a new, realistic, label-preserving variant whose diversity is in the
physically meaningful axes (severity, thermal spread, orientation).

Writes to ``data/processed/perturbed/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

from augmentation.heat_equation import ANOMALY_CLASSES


def perturb_fault(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma_bg: float = 6.0,
    sigma_fault: float = 1.0,
    alpha_range: tuple[float, float] = (0.5, 1.6),
    sigma_t_range: tuple[float, float] = (0.0, 1.8),
) -> np.ndarray:
    """Return one physics-perturbed variant of a real grayscale fault image.

    ``alpha`` scales fault severity; ``sigma_t`` is the extra heat-diffusion
    (thermal time). Background and grain are taken unchanged from the real image.
    """
    bg = gaussian_filter(img, sigma_bg)          # coarse module thermal field (real)
    detail = img - bg
    fault = gaussian_filter(detail, sigma_fault)  # coarse fault component
    grain = detail - fault                        # fine sensor grain (real)

    alpha = rng.uniform(*alpha_range)
    sigma_t = rng.uniform(*sigma_t_range)
    out = bg + alpha * gaussian_filter(fault, sigma_t) + grain   # diffuse + rescale fault

    if rng.random() < 0.5:
        out = out[:, ::-1]
    if rng.random() < 0.5:
        out = out[::-1, :]
    return np.clip(out, 0, 255)


class RealFaultAugmenter:
    """Top up rare anomaly classes by physically perturbing their *real* images.

    Same interface as ``SyntheticAugmenter`` — drops into ``verify_seeds`` as a
    new condition — but every output derives from a real crop of the target class.
    """

    def __init__(self, train_df: pd.DataFrame, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.by_class = {
            c: train_df.loc[train_df["label"] == c, "path"].tolist()
            for c in ANOMALY_CLASSES
        }

    def generate(self, class_name: str, n: int, output_dir: Path) -> list[Path]:
        pool = self.by_class.get(class_name, [])
        if not pool:
            return []
        out_cls = output_dir / class_name
        out_cls.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(n):
            src = pool[int(self.rng.integers(0, len(pool)))]
            img = np.asarray(Image.open(src).convert("L"), dtype=np.float64)
            arr = perturb_fault(img, self.rng).astype(np.uint8)
            p = out_cls / f"pert_{i:05d}.jpg"
            Image.fromarray(arr).convert("RGB").save(p, quality=90)
            paths.append(p)
        return paths

    def augment_split(self, train_df: pd.DataFrame, target_min: int, output_dir: Path) -> pd.DataFrame:
        counts = train_df["label"].value_counts()
        new_rows = []
        for cls in ANOMALY_CLASSES:
            needed = target_min - int(counts.get(cls, 0))
            if needed <= 0:
                continue
            print(f"  [perturb] generating {needed:4d} '{cls}' images …")
            new_rows.extend({"path": str(p), "label": cls}
                            for p in self.generate(cls, needed, output_dir))
        if not new_rows:
            return train_df
        return pd.concat([train_df, pd.DataFrame(new_rows)], ignore_index=True)
