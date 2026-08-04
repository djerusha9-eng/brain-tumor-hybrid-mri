from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import resolve_path
from .data import add_content_hashes, assign_patient_splits, read_manifest, validate_manifest
from .reproducibility import runtime_metadata, sha256_file, write_json


def prepare_experiment(config: dict[str, Any], check_files: bool = True) -> pd.DataFrame:
    source = resolve_path(config, config["data"]["manifest"])
    frame = read_manifest(source)
    validate_manifest(
        frame,
        config["data"].get("required_labels"),
        config["data"].get("allowed_modalities"),
        check_files=check_files,
    )
    frame = assign_patient_splits(
        frame, config["data"]["split"], int(config["project"]["seed"])
    )
    if check_files and config["data"].get("reject_duplicate_hashes", True):
        frame = add_content_hashes(frame)
    validate_manifest(
        frame,
        config["data"].get("required_labels"),
        config["data"].get("allowed_modalities"),
        check_files=check_files,
    )
    output = resolve_path(config, config["data"]["prepared_manifest"])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    metadata = runtime_metadata(config)
    metadata.update(
        {
            "source_manifest_sha256": sha256_file(source),
            "prepared_manifest_sha256": sha256_file(output),
            "samples": len(frame),
            "patients": frame["patient_id"].nunique(),
            "split_counts": frame["split"].value_counts().to_dict(),
            "class_counts": frame["label"].value_counts().to_dict(),
        }
    )
    write_json(metadata, output.parent / "run_metadata.json")
    return frame

