import pytest
import torch

from training.dataset import (
    SolarModuleDataset,
    get_transforms,
    load_splits,
    make_splits,
    splits_exist,
)


CLASSES = ["Cell", "Cell-Multi", "Cracking", "Diode", "Diode-Multi",
           "Hot-Spot", "Hot-Spot-Multi", "No-Anomaly", "Offline-Module",
           "Shadowing", "Soiling", "Vegetation"]


@pytest.fixture(scope="module")
def splits():
    if not splits_exist():
        return make_splits(seed=42)
    return load_splits()


def test_splits_sizes(splits):
    total = sum(len(df) for df in splits.values())
    assert total == 20000
    assert len(splits["train"]) == 14000
    assert len(splits["val"])   == 3000
    assert len(splits["test"])  == 3000


def test_splits_stratification(splits):
    """Each split should contain all 12 classes."""
    for name, df in splits.items():
        assert set(df["label"].unique()) == set(CLASSES), f"{name} split missing classes"


def test_splits_no_overlap(splits):
    train_paths = set(splits["train"]["path"])
    val_paths   = set(splits["val"]["path"])
    test_paths  = set(splits["test"]["path"])
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)


def test_dataset_item_shape(splits):
    ds = SolarModuleDataset(splits["train"].head(4), CLASSES, split="val")
    img, label = ds[0]
    assert img.shape == (3, 224, 224)
    assert isinstance(label, int)
    assert 0 <= label < len(CLASSES)


def test_dataset_train_transform_augments():
    """Train and val transforms should differ (train has random ops)."""
    t_train = get_transforms("train")
    t_val   = get_transforms("val")
    assert len(t_train.transforms) > len(t_val.transforms)
