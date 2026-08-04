from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = {"sample_id", "patient_id", "path", "label"}
OPTIONAL_COLUMNS = {"split", "mask_path", "source", "modality"}
VALID_SPLITS = {"train", "validation", "test"}


class DataIntegrityError(ValueError):
    """Raised when data would make an experiment invalid or non-reproducible."""


def read_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path).resolve()
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise DataIntegrityError(f"Manifest is missing columns: {sorted(missing)}")
    for column in OPTIONAL_COLUMNS.difference(frame.columns):
        frame[column] = ""
    for column in REQUIRED_COLUMNS | OPTIONAL_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip()
    for column in ("path", "mask_path"):
        frame[column] = frame[column].map(lambda value: _resolve_manifest_path(path, value))
    return frame


def _resolve_manifest_path(manifest_path: Path, value: str) -> str:
    if not value:
        return ""
    item = Path(value)
    return str(item if item.is_absolute() else (manifest_path.parent / item).resolve())


def validate_manifest(
    frame: pd.DataFrame,
    required_labels: Iterable[str] | None = None,
    allowed_modalities: Iterable[str] | None = None,
    check_files: bool = True,
) -> None:
    if frame.empty:
        raise DataIntegrityError("Manifest contains no samples.")
    for column in REQUIRED_COLUMNS:
        if frame[column].eq("").any():
            raise DataIntegrityError(f"Column {column!r} contains empty values.")
    duplicates = frame.loc[frame["sample_id"].duplicated(keep=False), "sample_id"].unique()
    if len(duplicates):
        raise DataIntegrityError(f"Duplicate sample_id values: {duplicates[:10].tolist()}")
    bad_splits = sorted(set(frame.loc[frame["split"].ne(""), "split"]) - VALID_SPLITS)
    if bad_splits:
        raise DataIntegrityError(f"Unsupported split values: {bad_splits}")
    if required_labels:
        missing = sorted(set(required_labels) - set(frame["label"]))
        if missing:
            raise DataIntegrityError(f"Required labels have no samples: {missing}")
    if allowed_modalities:
        observed = set(frame.loc[frame["modality"].ne(""), "modality"])
        invalid = sorted(observed - set(allowed_modalities))
        if invalid:
            raise DataIntegrityError(f"Unsupported modalities: {invalid}")
    if check_files:
        missing_files = [value for value in frame["path"] if not Path(value).is_file()]
        if missing_files:
            raise DataIntegrityError(f"Missing image files, first entries: {missing_files[:5]}")
        missing_masks = [value for value in frame["mask_path"] if value and not Path(value).is_file()]
        if missing_masks:
            raise DataIntegrityError(f"Missing mask files, first entries: {missing_masks[:5]}")
    assert_patient_disjoint(frame)


def assert_patient_disjoint(frame: pd.DataFrame) -> None:
    assigned = frame.loc[frame["split"].isin(VALID_SPLITS), ["patient_id", "split"]]
    counts = assigned.groupby("patient_id")["split"].nunique()
    leaked = counts[counts > 1].index.tolist()
    if leaked:
        raise DataIntegrityError(f"Patients occur in multiple splits: {leaked[:10]}")


def assign_patient_splits(
    frame: pd.DataFrame,
    ratios: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    result = frame.copy()
    if result["split"].isin(VALID_SPLITS).all():
        assert_patient_disjoint(result)
        return result
    if result["split"].ne("").any():
        raise DataIntegrityError("Use either complete predefined splits or leave every split empty.")

    patients = result[["patient_id", "label"]].drop_duplicates()
    conflicts = patients.groupby("patient_id")["label"].nunique()
    if (conflicts > 1).any():
        raise DataIntegrityError("Patient-level stratification requires one target label per patient.")
    min_class = patients["label"].value_counts().min()
    if min_class < 3:
        raise DataIntegrityError("Each class requires at least three patients for a three-way split.")

    train_patients, remainder = train_test_split(
        patients,
        train_size=float(ratios["train"]),
        random_state=seed,
        stratify=patients["label"],
    )
    relative_validation = float(ratios["validation"]) / (
        float(ratios["validation"]) + float(ratios["test"])
    )
    validation_patients, test_patients = train_test_split(
        remainder,
        train_size=relative_validation,
        random_state=seed + 1,
        stratify=remainder["label"],
    )
    mapping = {patient: "train" for patient in train_patients["patient_id"]}
    mapping.update({patient: "validation" for patient in validation_patients["patient_id"]})
    mapping.update({patient: "test" for patient in test_patients["patient_id"]})
    result["split"] = result["patient_id"].map(mapping)
    assert_patient_disjoint(result)
    return result


def add_content_hashes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["content_sha256"] = result["path"].map(_hash_file)
    crossing = result.groupby("content_sha256")["split"].nunique()
    if (crossing > 1).any():
        hashes = crossing[crossing > 1].index.tolist()
        raise DataIntegrityError(f"Identical image content crosses splits: {hashes[:5]}")
    return result


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image(path: str | Path, strategy: str = "largest_foreground") -> np.ndarray:
    path = Path(path)
    lower_name = path.name.lower()
    if lower_name.endswith((".nii", ".nii.gz")):
        import nibabel as nib

        volume = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))
        return _select_slice(volume, strategy)
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise DataIntegrityError(f"Unable to decode image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float32)


def _select_slice(volume: np.ndarray, strategy: str) -> np.ndarray:
    volume = np.squeeze(volume)
    if volume.ndim == 2:
        return volume.astype(np.float32)
    if volume.ndim != 3:
        raise DataIntegrityError(f"Expected 2-D or 3-D MRI, received shape {volume.shape}")
    if strategy == "central":
        index = volume.shape[-1] // 2
    elif strategy == "largest_foreground":
        threshold = np.percentile(volume[volume != 0], 20) if np.any(volume != 0) else 0
        areas = np.sum(volume > threshold, axis=(0, 1))
        index = int(np.argmax(areas))
    else:
        raise DataIntegrityError(f"Unknown representative-slice strategy: {strategy}")
    return volume[:, :, index].astype(np.float32)
