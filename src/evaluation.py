from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .metrics import classification_metrics, patient_bootstrap_interval
from .reproducibility import write_json
from .training import MRISequence


def evaluate_model(model, manifest: pd.DataFrame, classes: list[str], config: dict[str, Any]) -> dict:
    test = manifest[manifest["split"] == "test"].reset_index(drop=True)
    sequence = MRISequence(test, classes, config, training=False)
    started = time.perf_counter()
    probabilities = np.asarray(model.predict(sequence, verbose=0))
    elapsed = time.perf_counter() - started
    class_to_index = {label: index for index, label in enumerate(classes)}
    y_true = test["label"].map(class_to_index).to_numpy(dtype=int)
    predictions = np.argmax(probabilities, axis=1)
    metrics = classification_metrics(
        y_true, probabilities, classes, int(config["evaluation"]["calibration_bins"])
    )
    metrics["latency"] = {
        "total_seconds": elapsed,
        "milliseconds_per_sample": 1000 * elapsed / max(len(test), 1),
    }
    boot_cfg = config["evaluation"]
    metric_functions = {
        "accuracy": lambda truth, prob: accuracy_score(truth, np.argmax(prob, axis=1)),
        "balanced_accuracy": lambda truth, prob: balanced_accuracy_score(truth, np.argmax(prob, axis=1)),
        "macro_f1": lambda truth, prob: f1_score(truth, np.argmax(prob, axis=1), average="macro"),
    }
    metrics["patient_bootstrap_95ci"] = {
        name: patient_bootstrap_interval(
            y_true,
            probabilities,
            test["patient_id"].to_numpy(),
            function,
            int(boot_cfg["bootstrap_iterations"]),
            float(boot_cfg["confidence_level"]),
            int(config["project"]["seed"]),
        )
        for name, function in metric_functions.items()
    }
    output = test[["sample_id", "patient_id", "label", "source", "modality"]].copy()
    output["predicted_label"] = [classes[index] for index in predictions]
    for index, label in enumerate(classes):
        output[f"probability_{label}"] = probabilities[:, index]
    predictions_path = Path(config["artifacts"]["predictions"])
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(predictions_path, index=False)
    write_json(metrics, config["artifacts"]["metrics"])
    return metrics

