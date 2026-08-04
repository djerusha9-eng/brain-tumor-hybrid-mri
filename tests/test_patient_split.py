import pandas as pd
import pytest

from src.data import DataIntegrityError, assert_patient_disjoint, assign_patient_splits


def _frame():
    rows = []
    for label in ("A", "B", "C"):
        for patient in range(20):
            patient_id = f"{label}{patient:02d}"
            for sample in range(2):
                rows.append(
                    {"sample_id": f"{patient_id}_{sample}", "patient_id": patient_id,
                     "path": "/unused", "label": label, "split": ""}
                )
    return pd.DataFrame(rows)


def test_patient_split_is_deterministic_and_disjoint():
    ratios = {"train": 0.7, "validation": 0.15, "test": 0.15}
    first = assign_patient_splits(_frame(), ratios, 42)
    second = assign_patient_splits(_frame(), ratios, 42)
    assert first["split"].tolist() == second["split"].tolist()
    assert_patient_disjoint(first)
    assert set(first["split"]) == {"train", "validation", "test"}


def test_cross_split_patient_is_rejected():
    frame = pd.DataFrame({"patient_id": ["p1", "p1"], "split": ["train", "test"]})
    with pytest.raises(DataIntegrityError, match="multiple splits"):
        assert_patient_disjoint(frame)

