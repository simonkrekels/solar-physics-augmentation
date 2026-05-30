import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader


def run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    """Returns (avg_loss, accuracy, predictions, true_labels)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * len(labels)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def classification_report_dict(
    preds: list[int], labels: list[int], classes: list[str]
) -> dict:
    return classification_report(labels, preds, target_names=classes, output_dict=True)


def get_confusion_matrix(preds: list[int], labels: list[int], n_classes: int) -> np.ndarray:
    return confusion_matrix(labels, preds, labels=list(range(n_classes)))
