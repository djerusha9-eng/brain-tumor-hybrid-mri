from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .data import load_image
from .preprocessing import preprocess_image


def predict_path(model, path: str | Path, classes: list[str], config: dict[str, Any]) -> dict[str, Any]:
    image = load_image(path, config["data"]["representative_slice"])
    processed, mask = preprocess_image(
        image,
        config["preprocessing"],
        config["segmentation"],
        int(config["project"]["seed"]),
    )
    probabilities = np.asarray(model.predict(processed[None, ...], verbose=0))[0]
    index = int(np.argmax(probabilities))
    return {
        "predicted_label": classes[index],
        "probabilities": {label: float(probabilities[i]) for i, label in enumerate(classes)},
        "fcm_roi_fraction": float(mask.mean()) if mask is not None else None,
    }

