"""Training, evaluation and metric helpers shared by train.py and evaluate.py."""

import random

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so a CPU run repeats exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    """Return the device named by name ("cpu", "cuda", "cuda:1", "mps"), or the CPU if it is not available."""
    device = torch.device(name)
    available = {"cuda": torch.cuda.is_available, "mps": torch.backends.mps.is_available}
    if device.type in available and not available[device.type]():
        return torch.device("cpu")
    return device


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    """Overall accuracy (OA), average per-class accuracy (AA), Cohen's kappa and the confusion matrix.

    AA averages over the classes present in y_true, so a class missing from a small
    validation split does not turn AA into NaN.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    support = cm.sum(axis=1)
    present = support > 0
    return {
        "OA": float(np.mean(y_true == y_pred)),
        "AA": float(np.mean(np.diag(cm)[present] / support[present])),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": cm,
    }


def train_one_epoch(model, criterion, optimizer, loader, device) -> float:
    """Train for one pass over loader and return the mean loss."""
    model.train()
    running_loss = 0.0
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, criterion, loader, device, num_classes):
    """Return the mean loss and the metrics of model on loader."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            running_loss += criterion(outputs, labels).item() * data.size(0)
            all_preds.append(outputs.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    return running_loss / len(loader.dataset), compute_metrics(y_true, y_pred, num_classes)
