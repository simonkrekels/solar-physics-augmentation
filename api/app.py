"""
FastAPI inference service for thermal IR solar module defect detection.

Run locally:
    uv run uvicorn api.app:app --reload

Or via Docker (see Dockerfile at project root).

Environment variables:
    CHECKPOINT   Path to .pt checkpoint file (default: checkpoints/physics-augmented_best.pt)
    DEVICE       Override device selection: cuda | mps | cpu
"""

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from training.dataset import CLASSES, get_transforms
from training.model import build_model

_DEFAULT_CHECKPOINT = "checkpoints/physics-augmented_best.pt"
_TOP_K = 3

_state: dict = {}


def _get_device() -> torch.device:
    override = os.getenv("DEVICE", "").lower()
    if override in ("cuda", "mps", "cpu"):
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt_path = Path(os.getenv("CHECKPOINT", _DEFAULT_CHECKPOINT))
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            "Set the CHECKPOINT env var or mount the file into the container."
        )

    device = _get_device()
    model = build_model(num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device).eval()

    _state["model"] = model
    _state["device"] = device
    _state["transform"] = get_transforms("val")
    print(f"Model loaded from {ckpt_path} on {device}")
    yield
    _state.clear()


app = FastAPI(
    title="Solar Thermal Defect Detector",
    description="Classifies thermal IR images of solar modules into 12 anomaly classes.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(_state.get("device", "unknown"))}


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File(description="JPEG/PNG thermal IR image")]):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="Could not decode image file.")

    tensor = _state["transform"](img).unsqueeze(0).to(_state["device"])

    with torch.no_grad():
        probs = torch.softmax(_state["model"](tensor), dim=1)[0]

    top_k = probs.topk(_TOP_K)
    predictions = [
        {"class": CLASSES[idx], "confidence": round(conf, 4)}
        for conf, idx in zip(top_k.values.cpu().tolist(), top_k.indices.cpu().tolist())
    ]

    return JSONResponse({
        "filename": file.filename,
        "predicted_class": predictions[0]["class"],
        "confidence": predictions[0]["confidence"],
        "top_k": predictions,
    })
