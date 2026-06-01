"""
Synthesise thermal anomaly patches by solving the 2-D steady-state heat equation

    ∇²T = −q / κ

with Neumann (zero-flux) boundary conditions via Jacobi iteration.
The temperature field is blended onto a No-Anomaly base image to produce
labelled training samples for rare anomaly classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import zoom


# ---------------------------------------------------------------------------
# PDE solver
# ---------------------------------------------------------------------------

def solve_heat_2d(
    source: np.ndarray,
    kappa: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-3,
) -> np.ndarray:
    """Solve ∇²T = −q/κ via Jacobi iteration with Neumann boundary conditions.

    Neumann BC (∂T/∂n = 0) is enforced by edge-padding before each update.
    The mean is subtracted each step to anchor the otherwise undetermined
    constant that arises from a pure-Neumann problem.

    Returns:
        T: float64 temperature field, shape == source.shape, zero-mean.
    """
    rhs = source.astype(np.float64) / kappa
    T = np.zeros_like(rhs)

    for _ in range(max_iter):
        T_prev = T
        Tp = np.pad(T, 1, mode="edge")          # Neumann BC at all edges
        T = (
            Tp[:-2, 1:-1] + Tp[2:, 1:-1] +
            Tp[1:-1, :-2] + Tp[1:-1, 2:] +
            rhs
        ) / 4.0
        T -= T.mean()                            # remove null-space drift
        if np.max(np.abs(T - T_prev)) < tol:
            break

    return T


# ---------------------------------------------------------------------------
# Source-geometry primitives
# ---------------------------------------------------------------------------

def _gauss(H: int, W: int, cy: float, cx: float, sigma: float,
           amplitude: float = 1.0) -> np.ndarray:
    y, x = np.ogrid[:H, :W]
    return amplitude * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma ** 2))


def _line(H: int, W: int, row: float, col: float, angle: float,
          thickness: float, amplitude: float = 1.0) -> np.ndarray:
    """Fill pixels within *thickness* of a line through (row, col) at *angle* rad."""
    y, x = np.ogrid[:H, :W]
    dist = np.abs((x - col) * np.sin(angle) - (y - row) * np.cos(angle))
    return amplitude * (dist <= thickness).astype(np.float64)


def _rect(H: int, W: int, r0: int, c0: int, rh: int, rw: int,
          amplitude: float = 1.0) -> np.ndarray:
    q = np.zeros((H, W))
    q[r0: min(r0 + rh, H), c0: min(c0 + rw, W)] = amplitude
    return q


# ---------------------------------------------------------------------------
# Source factory — one geometry per anomaly class
# ---------------------------------------------------------------------------

ANOMALY_CLASSES = [
    "Cell", "Cell-Multi", "Cracking", "Diode", "Diode-Multi",
    "Hot-Spot", "Hot-Spot-Multi", "Offline-Module",
    "Shadowing", "Soiling", "Vegetation",
]


def make_source(
    class_name: str,
    grid_h: int = 64,
    grid_w: int = 64,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Return a 2-D heat-source array q(y, x) shaped (grid_h, grid_w).

    Positive values → Joule heating; negative values → cooling below ambient.
    """
    if rng is None:
        rng = np.random.default_rng()

    q = np.zeros((grid_h, grid_w), dtype=np.float64)

    if class_name == "Hot-Spot":
        cy = rng.uniform(grid_h * 0.25, grid_h * 0.75)
        cx = rng.uniform(grid_w * 0.25, grid_w * 0.75)
        q += _gauss(grid_h, grid_w, cy, cx,
                    sigma=rng.uniform(1.5, 3.5), amplitude=10.0)

    elif class_name == "Hot-Spot-Multi":
        for _ in range(int(rng.integers(2, 5))):
            cy = rng.uniform(grid_h * 0.1, grid_h * 0.9)
            cx = rng.uniform(grid_w * 0.1, grid_w * 0.9)
            q += _gauss(grid_h, grid_w, cy, cx,
                        sigma=rng.uniform(1.0, 2.5), amplitude=8.0)

    elif class_name == "Cell":
        rh, rw = grid_h // 6, grid_w // 6
        r0 = int(rng.integers(grid_h // 4, grid_h // 2))
        c0 = int(rng.integers(grid_w // 4, grid_w // 2))
        q += _rect(grid_h, grid_w, r0, c0, rh, rw, amplitude=5.0)

    elif class_name == "Cell-Multi":
        rh, rw = grid_h // 6, grid_w // 6
        for _ in range(int(rng.integers(2, 4))):
            r0 = int(rng.integers(0, grid_h - rh))
            c0 = int(rng.integers(0, grid_w - rw))
            q += _rect(grid_h, grid_w, r0, c0, rh, rw, amplitude=5.0)

    elif class_name == "Diode":
        # Failed bypass diode → entire vertical string overheats
        col = int(rng.integers(grid_w // 4, 3 * grid_w // 4))
        width = max(2, grid_w // 8)
        q += _rect(grid_h, grid_w, 0, col, grid_h, width, amplitude=6.0)

    elif class_name == "Diode-Multi":
        width = max(2, grid_w // 8)
        cols = rng.choice(grid_w - width, size=int(rng.integers(2, 4)), replace=False)
        for col in cols:
            q += _rect(grid_h, grid_w, 0, int(col), grid_h, width, amplitude=6.0)

    elif class_name == "Shadowing":
        rh = int(rng.integers(grid_h // 3, grid_h // 2))
        rw = int(rng.integers(grid_w // 3, grid_w // 2))
        r0 = int(rng.integers(0, grid_h - rh))
        c0 = int(rng.integers(0, grid_w - rw))
        q += _rect(grid_h, grid_w, r0, c0, rh, rw, amplitude=-4.0)

    elif class_name == "Vegetation":
        cy = rng.uniform(grid_h * 0.2, grid_h * 0.8)
        cx = rng.uniform(grid_w * 0.2, grid_w * 0.8)
        sigma = rng.uniform(grid_h / 8, grid_h / 4)
        q -= _gauss(grid_h, grid_w, cy, cx, sigma, amplitude=3.0)

    elif class_name == "Soiling":
        for _ in range(int(rng.integers(2, 6))):
            cy = rng.uniform(0, grid_h)
            cx = rng.uniform(0, grid_w)
            q -= _gauss(grid_h, grid_w, cy, cx,
                        sigma=rng.uniform(2.0, 6.0), amplitude=2.5)

    elif class_name == "Offline-Module":
        q -= 5.0

    elif class_name == "Cracking":
        # Thin resistive crack → localised heating along a line
        row = rng.uniform(grid_h * 0.2, grid_h * 0.8)
        col = rng.uniform(grid_w * 0.2, grid_w * 0.8)
        angle = rng.uniform(0, np.pi)
        q += _line(grid_h, grid_w, row, col, angle, thickness=1.0, amplitude=4.0)

    return q


# ---------------------------------------------------------------------------
# Image blending
# ---------------------------------------------------------------------------

def blend_heat_patch(
    base_img: Image.Image,
    heat: np.ndarray,
    intensity_scale: float = 40.0,
) -> Image.Image:
    """Add a heat field to a base RGB image.

    The heat map is resized to match the image, normalised to [−1, +1], then
    scaled by *intensity_scale* pixel units and added to all three channels.
    """
    arr = np.array(base_img.convert("RGB"), dtype=np.float32)
    H, W = arr.shape[:2]

    if heat.shape != (H, W):
        zy, zx = H / heat.shape[0], W / heat.shape[1]
        heat = zoom(heat, (zy, zx), order=1)

    peak = max(float(np.abs(heat).max()), 1e-9)
    delta = (heat / peak) * intensity_scale
    arr += delta[:, :, np.newaxis]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# High-level synthesis
# ---------------------------------------------------------------------------

def synthesise(
    class_name: str,
    base_img: Image.Image,
    grid_size: int = 64,
    kappa: float = 1.0,
    intensity_scale: float = 40.0,
    rng: Optional[np.random.Generator] = None,
) -> Image.Image:
    """Generate one synthetic thermal image for *class_name*.

    Solves ∇²T = −q/κ on a (grid_size × grid_size) grid and blends the
    resulting temperature rise onto *base_img* (a No-Anomaly frame).
    """
    if rng is None:
        rng = np.random.default_rng()
    source = make_source(class_name, grid_size, grid_size, rng)
    heat = solve_heat_2d(source, kappa=kappa)
    return blend_heat_patch(base_img, heat, intensity_scale)


# ---------------------------------------------------------------------------
# Bulk augmenter
# ---------------------------------------------------------------------------

class SyntheticAugmenter:
    """Generate physics-informed synthetic samples for under-represented classes.

    Usage::

        aug = SyntheticAugmenter(train_df, seed=42)
        extended_df = aug.augment_split(
            train_df, target_min=500,
            output_dir=Path("data/processed/synthetic"),
        )
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        grid_size: int = 64,
        kappa: float = 1.0,
        intensity_scale: float = 40.0,
        seed: int = 42,
    ) -> None:
        self.grid_size = grid_size
        self.kappa = kappa
        self.intensity_scale = intensity_scale
        self.rng = np.random.default_rng(seed)
        self._base_paths = train_df.loc[
            train_df["label"] == "No-Anomaly", "path"
        ].tolist()
        if not self._base_paths:
            raise ValueError("train_df must contain at least one No-Anomaly image")

    def _random_base(self) -> Image.Image:
        path = self._base_paths[int(self.rng.integers(0, len(self._base_paths)))]
        return Image.open(path).convert("RGB")

    def generate(self, class_name: str, n: int, output_dir: Path) -> list[Path]:
        """Save *n* synthetic images for *class_name* to *output_dir/<class_name>/*.

        Returns the list of saved file paths.
        """
        out_cls = output_dir / class_name
        out_cls.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for i in range(n):
            img = synthesise(
                class_name,
                self._random_base(),
                grid_size=self.grid_size,
                kappa=self.kappa,
                intensity_scale=self.intensity_scale,
                rng=self.rng,
            )
            p = out_cls / f"syn_{i:05d}.jpg"
            img.save(p, quality=90)
            paths.append(p)
        return paths

    def augment_split(
        self,
        train_df: pd.DataFrame,
        target_min: int,
        output_dir: Path,
    ) -> pd.DataFrame:
        """Extend *train_df* so every anomaly class has ≥ *target_min* samples.

        Synthetic images are saved under *output_dir/<class_name>/*.
        Returns a new DataFrame with original rows first, then synthetic rows.
        """
        counts = train_df["label"].value_counts()
        new_rows: list[dict] = []

        for cls in ANOMALY_CLASSES:
            needed = target_min - int(counts.get(cls, 0))
            if needed <= 0:
                continue
            print(f"  Generating {needed:4d} synthetic '{cls}' images …")
            saved = self.generate(cls, needed, output_dir)
            new_rows.extend({"path": str(p), "label": cls} for p in saved)

        if not new_rows:
            return train_df

        return pd.concat(
            [train_df, pd.DataFrame(new_rows)], ignore_index=True
        )
