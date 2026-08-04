from pathlib import Path

import numpy as np

from src.config import load_config
from src.preprocessing import preprocess_image, robust_normalize


ROOT = Path(__file__).resolve().parents[1]


def test_robust_normalization_is_finite_and_bounded():
    image = np.array([[np.nan, -100.0], [5.0, np.inf]], dtype=np.float32)
    result = robust_normalize(image)
    assert np.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 1


def test_preprocessing_returns_model_shape():
    config = load_config(ROOT / "config.yaml")
    yy, xx = np.ogrid[:64, :64]
    image = np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / 300).astype(np.float32)
    processed, mask = preprocess_image(
        image, config["preprocessing"], config["segmentation"], config["project"]["seed"]
    )
    assert processed.shape == (299, 299, 3)
    assert processed.dtype == np.float32
    assert mask is None or mask.shape == (299, 299)

