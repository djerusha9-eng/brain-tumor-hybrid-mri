import numpy as np

from src.metrics import classification_metrics, expected_calibration_error, patient_bootstrap_interval


def test_perfect_predictions_have_perfect_core_metrics():
    truth = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[truth]
    result = classification_metrics(truth, probabilities, ["A", "B", "C"])
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["expected_calibration_error"] == 0.0


def test_patient_bootstrap_is_reproducible():
    truth = np.array([0, 1, 0, 1, 0, 1])
    probabilities = np.eye(2)[truth]
    patients = np.array(["a", "b", "c", "d", "e", "f"])
    metric = lambda y, p: float(np.mean(y == np.argmax(p, axis=1)))
    first = patient_bootstrap_interval(truth, probabilities, patients, metric, 100, seed=7)
    second = patient_bootstrap_interval(truth, probabilities, patients, metric, 100, seed=7)
    assert first == second
    assert first["lower"] == first["upper"] == 1.0

