import numpy as np

from src.inference import predict_path


class FixedModel:
    def predict(self, batch, verbose=0):
        assert batch.shape == (1, 32, 32, 3)
        return np.array([[0.1, 0.8, 0.1]], dtype=np.float32)


def test_predict_path_returns_probabilities(monkeypatch):
    monkeypatch.setattr("src.inference.load_image", lambda *args: np.ones((16, 16), dtype=np.float32))
    monkeypatch.setattr(
        "src.inference.preprocess_image",
        lambda *args: (np.ones((32, 32, 3), dtype=np.float32), np.ones((32, 32), dtype=bool)),
    )
    config = {
        "data": {"representative_slice": "central"},
        "preprocessing": {},
        "segmentation": {},
        "project": {"seed": 42},
    }
    result = predict_path(FixedModel(), "unused", ["A", "B", "C"], config)
    assert result["predicted_label"] == "B"
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-6

