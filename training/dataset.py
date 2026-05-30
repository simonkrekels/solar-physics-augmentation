import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "InfraredSolarModules" / "InfraredSolarModules"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
META_PATH = RAW_DIR / "module_metadata.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def make_splits(seed: int = 42) -> dict[str, pd.DataFrame]:
    """Build stratified 70/15/15 splits and write CSVs to data/processed/."""
    with open(META_PATH) as f:
        meta = json.load(f)

    paths  = [str(RAW_DIR / v["image_filepath"]) for v in meta.values()]
    labels = [v["anomaly_class"] for v in meta.values()]

    X_train, X_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    splits: dict[str, pd.DataFrame] = {}
    for name, (xs, ys) in [("train", (X_train, y_train)),
                            ("val",   (X_val,   y_val)),
                            ("test",  (X_test,  y_test))]:
        df = pd.DataFrame({"path": xs, "label": ys})
        df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        splits[name] = df
        print(f"  {name:5s}: {len(df):5d} images")

    return splits


def load_splits() -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(PROCESSED_DIR / f"{name}.csv") for name in ("train", "val", "test")}


def splits_exist() -> bool:
    return all((PROCESSED_DIR / f"{name}.csv").exists() for name in ("train", "val", "test"))


def get_transforms(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class SolarModuleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, classes: list[str], split: str = "train"):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = get_transforms(split)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        return self.transform(img), self.class_to_idx[row["label"]]
