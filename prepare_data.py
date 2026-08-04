from __future__ import annotations

import argparse
import os

from src.config import load_config
from src.reproducibility import seed_everything
from src.validation import prepare_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MRI data and freeze patient-level splits.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-file-checks", action="store_true", help="Validate metadata only.")
    args = parser.parse_args()
    config = load_config(args.config)
    os.chdir(config["_base_dir"])
    seed_everything(int(config["project"]["seed"]), bool(config["project"]["deterministic_ops"]))
    frame = prepare_experiment(config, check_files=not args.skip_file_checks)
    print(frame.groupby(["split", "label"])["patient_id"].nunique().unstack(fill_value=0))
    print(f"Prepared {len(frame)} samples from {frame['patient_id'].nunique()} patients.")


if __name__ == "__main__":
    main()

