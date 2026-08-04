from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    calibration_bins: int = 10,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(y_true), len(classes)):
        raise ValueError("Probability matrix shape does not match samples and classes.")
    predicted = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(y_true, predicted, labels=np.arange(len(classes)))
    result = {
        "n_samples": int(len(y_true)),
        "classes": classes,
        "accuracy": accuracy_score(y_true, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "macro_precision": precision_score(y_true, predicted, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, predicted, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, predicted, average="macro", zero_division=0),
        "confusion_matrix": matrix.tolist(),
        "per_class": {},
    }
    for index, label in enumerate(classes):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = int(matrix.sum() - tp - fn - fp)
        result["per_class"][label] = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall_sensitivity": tp / (tp + fn) if tp + fn else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "support": int(matrix[index, :].sum()),
        }
    one_hot = label_binarize(y_true, classes=np.arange(len(classes)))
    if len(classes) == 2:
        one_hot = np.column_stack([1 - one_hot[:, 0], one_hot[:, 0]])
    try:
        result["macro_roc_auc_ovr"] = roc_auc_score(
            one_hot, probabilities, average="macro", multi_class="ovr"
        )
    except ValueError:
        result["macro_roc_auc_ovr"] = None
    result["multiclass_brier"] = float(np.mean(np.sum((one_hot - probabilities) ** 2, axis=1)))
    result["expected_calibration_error"] = expected_calibration_error(
        y_true, probabilities, calibration_bins
    )
    return result


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    correctness = predictions == y_true
    boundaries = np.linspace(0, 1, bins + 1)
    error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        selected = (confidences > lower) & (confidences <= upper)
        if selected.any():
            error += selected.mean() * abs(correctness[selected].mean() - confidences[selected].mean())
    return float(error)


def patient_bootstrap_interval(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    patient_ids: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    patient_ids = np.asarray(patient_ids)
    unique_patients = np.unique(patient_ids)
    if unique_patients.size < 2:
        raise ValueError("At least two patients are required for bootstrap uncertainty.")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        indices = np.concatenate([np.flatnonzero(patient_ids == patient) for patient in sampled])
        try:
            estimates.append(float(metric(y_true[indices], probabilities[indices])))
        except ValueError:
            continue
    if not estimates:
        raise ValueError("No valid bootstrap replicates were produced.")
    alpha = (1 - confidence) / 2
    return {
        "estimate": float(metric(y_true, probabilities)),
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1 - alpha)),
        "valid_iterations": len(estimates),
    }

