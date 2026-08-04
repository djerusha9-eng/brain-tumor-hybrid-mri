from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .training import MRISequence


def explain_model(model, manifest: pd.DataFrame, classes: list[str], config: dict[str, Any]) -> Path:
    """Generate GradientExplainer SHAP arrays for prespecified test samples."""
    import shap

    train = manifest[manifest["split"] == "train"].head(
        int(config["explainability"]["background_samples"])
    )
    test = manifest[manifest["split"] == "test"].head(
        int(config["explainability"]["explanation_samples"])
    )
    background = _collect_images(MRISequence(train, classes, config, training=False))
    explained = _collect_images(MRISequence(test, classes, config, training=False))
    explainer = shap.GradientExplainer(model, background)
    values = explainer.shap_values(explained)
    output = Path(config["artifacts"]["explanations"])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        shap_values=np.asarray(values),
        images=explained,
        sample_ids=test["sample_id"].to_numpy(),
        classes=np.asarray(classes),
    )
    return output


def _collect_images(sequence) -> np.ndarray:
    batches = [sequence[index][0] for index in range(len(sequence))]
    if not batches:
        raise ValueError("No images are available for explanation.")
    return np.concatenate(batches, axis=0)

