from pathlib import Path

import pandas as pd
import pytest

from src.data import DataIntegrityError, read_manifest, validate_manifest


def test_duplicate_sample_ids_are_rejected(tmp_path):
    image = tmp_path / "image.bin"
    image.write_bytes(b"mri")
    frame = pd.DataFrame(
        {
            "sample_id": ["same", "same"],
            "patient_id": ["p1", "p2"],
            "path": [str(image), str(image)],
            "label": ["A", "B"],
            "split": ["train", "test"],
            "mask_path": ["", ""],
            "source": ["x", "x"],
            "modality": ["T1", "T1"],
        }
    )
    with pytest.raises(DataIntegrityError, match="Duplicate sample_id"):
        validate_manifest(frame, check_files=False)


def test_relative_paths_are_resolved(tmp_path):
    (tmp_path / "scan.png").write_bytes(b"placeholder")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("sample_id,patient_id,path,label\ns1,p1,scan.png,A\n", encoding="utf-8")
    frame = read_manifest(manifest)
    assert Path(frame.loc[0, "path"]).is_absolute()

