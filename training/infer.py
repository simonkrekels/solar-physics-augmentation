"""
Batch inference on thermal IR solar module images.

Usage:
    uv run python -m training.infer \
        --checkpoint checkpoints/physics-augmented_best.pt \
        --input path/to/images/ \
        --output predictions.csv

Outputs a CSV with columns: filename, predicted_class, confidence
Accepts a single image file or a directory (searched recursively).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from training.dataset import CLASSES, get_transforms
from training.model import build_model

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class _ImageFolder(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.transform = get_transforms("val")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.paths[idx].name


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_inference(
    checkpoint: Path,
    input_path: Path,
    output: Path,
    batch_size: int = 32,
) -> pd.DataFrame:
    device = _get_device()
    print(f"Device: {device}")

    if input_path.is_file():
        paths = [input_path]
    else:
        paths = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS)

    if not paths:
        sys.exit(f"No supported images found in {input_path}")
    print(f"Found {len(paths)} image(s)")

    model = build_model(num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device).eval()

    loader = DataLoader(
        _ImageFolder(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    rows: list[dict] = []
    with torch.no_grad():
        for images, filenames in loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1)
            confs, idxs = probs.max(dim=1)
            for fname, idx, conf in zip(filenames, idxs.cpu(), confs.cpu()):
                rows.append({
                    "filename": fname,
                    "predicted_class": CLASSES[idx.item()],
                    "confidence": round(conf.item(), 4),
                })

    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} predictions → {output}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch inference on thermal IR solar module images")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--input", required=True, type=Path, help="Image file or directory")
    ap.add_argument("--output", default=Path("predictions.csv"), type=Path)
    ap.add_argument("--batch-size", default=32, type=int)
    args = ap.parse_args()
    run_inference(args.checkpoint, args.input, args.output, args.batch_size)
