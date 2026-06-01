import numpy as np
import pandas as pd
import pytest
from PIL import Image

from augmentation.heat_equation import (
    ANOMALY_CLASSES,
    SyntheticAugmenter,
    blend_heat_patch,
    make_source,
    solve_heat_2d,
    synthesise,
)


# ── solver ────────────────────────────────────────────────────────────────────

def test_solver_zero_source():
    T = solve_heat_2d(np.zeros((32, 32)))
    assert T.shape == (32, 32)
    np.testing.assert_allclose(T, 0.0, atol=1e-6)


@pytest.mark.parametrize("shape", [(16, 16), (32, 48), (64, 64)])
def test_solver_shape(shape):
    assert solve_heat_2d(np.ones(shape)).shape == shape


def test_solver_positive_source_is_hot():
    q = np.zeros((32, 32))
    q[16, 16] = 10.0
    T = solve_heat_2d(q, max_iter=1000, tol=1e-4)
    assert T[16, 16] > 0, "localised source should be warmer than the mean"


def test_solver_symmetric_source():
    H, W = 32, 32
    q = np.zeros((H, W))
    for r, c in [(8, 8), (8, W - 9), (H - 9, 8), (H - 9, W - 9)]:
        q[r, c] = 5.0
    T = solve_heat_2d(q, max_iter=1000, tol=1e-4)
    np.testing.assert_allclose(T, T[:, ::-1], atol=1e-3)  # left-right
    np.testing.assert_allclose(T, T[::-1, :], atol=1e-3)  # top-bottom


def test_solver_negative_source_is_cool():
    q = np.zeros((32, 32))
    q[16, 16] = -10.0
    T = solve_heat_2d(q, max_iter=1000, tol=1e-4)
    assert T[16, 16] < 0, "cooling source should be below mean temperature"


# ── source factory ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cls", ANOMALY_CLASSES)
def test_make_source_all_classes(cls):
    rng = np.random.default_rng(0)
    q = make_source(cls, grid_h=32, grid_w=32, rng=rng)
    assert q.shape == (32, 32), f"{cls}: wrong shape"
    assert np.isfinite(q).all(), f"{cls}: non-finite values"
    assert not np.all(q == 0), f"{cls}: source is identically zero"


def test_make_source_hot_classes_positive():
    rng = np.random.default_rng(1)
    for cls in ("Hot-Spot", "Hot-Spot-Multi", "Diode", "Cell"):
        q = make_source(cls, rng=rng)
        assert q.max() > 0, f"{cls}: expected positive heat source"


def test_make_source_cool_classes_negative():
    rng = np.random.default_rng(2)
    for cls in ("Shadowing", "Vegetation", "Offline-Module"):
        q = make_source(cls, rng=rng)
        assert q.min() < 0, f"{cls}: expected negative (cooling) source"


# ── blending ──────────────────────────────────────────────────────────────────

def test_blend_preserves_size_and_mode():
    base = Image.fromarray(np.full((128, 96, 3), 128, dtype=np.uint8))
    heat = np.zeros((64, 64))
    heat[32, 32] = 1.0
    result = blend_heat_patch(base, heat, intensity_scale=30.0)
    assert result.size == base.size
    assert result.mode == "RGB"


def test_blend_hot_brightens_pixel():
    arr = np.full((64, 64, 3), 100, dtype=np.uint8)
    base = Image.fromarray(arr)
    heat = np.zeros((64, 64))
    heat[32, 32] = 5.0
    result = blend_heat_patch(base, heat, intensity_scale=50.0)
    assert np.array(result)[32, 32, 0] > 100


def test_blend_cool_darkens_pixel():
    arr = np.full((64, 64, 3), 150, dtype=np.uint8)
    base = Image.fromarray(arr)
    heat = np.zeros((64, 64))
    heat[32, 32] = -5.0
    result = blend_heat_patch(base, heat, intensity_scale=50.0)
    assert np.array(result)[32, 32, 0] < 150


def test_blend_output_clipped():
    base = Image.fromarray(np.full((32, 32, 3), 250, dtype=np.uint8))
    heat = np.ones((32, 32)) * 100.0
    result = blend_heat_patch(base, heat, intensity_scale=100.0)
    assert np.array(result).max() <= 255


# ── synthesise ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cls", ANOMALY_CLASSES[:6])
def test_synthesise_returns_pil_correct_size(cls):
    base = Image.fromarray(
        np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
    )
    result = synthesise(cls, base, grid_size=32, rng=np.random.default_rng(42))
    assert isinstance(result, Image.Image)
    assert result.size == base.size
    assert result.mode == "RGB"


# ── SyntheticAugmenter ────────────────────────────────────────────────────────

@pytest.fixture
def no_anomaly_df(tmp_path):
    img_dir = tmp_path / "No-Anomaly"
    img_dir.mkdir()
    for i in range(4):
        img = Image.fromarray(
            np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        )
        img.save(img_dir / f"img_{i}.jpg")
    df = pd.DataFrame({
        "path": [str(img_dir / f"img_{i}.jpg") for i in range(4)],
        "label": ["No-Anomaly"] * 4,
    })
    return df


def test_augmenter_generate(no_anomaly_df, tmp_path):
    aug = SyntheticAugmenter(no_anomaly_df, grid_size=16, seed=0)
    paths = aug.generate("Hot-Spot", n=5, output_dir=tmp_path / "synthetic")
    assert len(paths) == 5
    assert all(p.exists() for p in paths)
    assert all(p.suffix == ".jpg" for p in paths)


def test_augmenter_augment_split(no_anomaly_df, tmp_path):
    # Start with a DataFrame that has very few samples per anomaly class
    sparse_rows = []
    for cls in ("Hot-Spot", "Cell"):
        sparse_rows.append({"path": no_anomaly_df["path"].iloc[0], "label": cls})
    train_df = pd.concat(
        [no_anomaly_df, pd.DataFrame(sparse_rows)], ignore_index=True
    )

    aug = SyntheticAugmenter(train_df, grid_size=16, seed=1)
    extended = aug.augment_split(train_df, target_min=3,
                                 output_dir=tmp_path / "syn")

    # Hot-Spot and Cell each had 1 sample → should now have ≥ 3
    counts = extended["label"].value_counts()
    assert counts.get("Hot-Spot", 0) >= 3
    assert counts.get("Cell", 0) >= 3
    # Total rows increased
    assert len(extended) > len(train_df)


def test_augmenter_no_generation_when_above_target(no_anomaly_df, tmp_path):
    # All anomaly classes already have enough samples → no new rows added
    rows = [{"path": no_anomaly_df["path"].iloc[0], "label": cls}
            for cls in ANOMALY_CLASSES for _ in range(3)]
    train_df = pd.concat([no_anomaly_df, pd.DataFrame(rows)], ignore_index=True)
    aug = SyntheticAugmenter(train_df, grid_size=16, seed=2)
    result = aug.augment_split(train_df, target_min=3, output_dir=tmp_path / "syn")
    assert len(result) == len(train_df)


def test_augmenter_raises_without_no_anomaly():
    df = pd.DataFrame({"path": ["x.jpg"], "label": ["Hot-Spot"]})
    with pytest.raises(ValueError, match="No-Anomaly"):
        SyntheticAugmenter(df)
