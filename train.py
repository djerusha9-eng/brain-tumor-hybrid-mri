from __future__ import annotations

import argparse
import os

import pandas as pd

from src.config import load_config, resolve_path
from src.models import build_hybrid_model, compile_model
from src.reproducibility import seed_everything
from src.training import fit_model
from src.validation import prepare_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the hybrid MRI classifier.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--prepared", action="store_true", help="Reuse an already validated manifest.")
    args = parser.parse_args()
    config = load_config(args.config)
    os.chdir(config["_base_dir"])
    seed_everything(int(config["project"]["seed"]), bool(config["project"]["deterministic_ops"]))
    if args.prepared:
        manifest = pd.read_csv(resolve_path(config, config["data"]["prepared_manifest"]), dtype=str)
    else:
        manifest = prepare_experiment(config)
    classes = list(config["data"]["required_labels"])
    model = compile_model(build_hybrid_model(config, classes), config)
    model.summary()
    fit_model(model, manifest, classes, config)
    print(f"Best model saved to {config['artifacts']['model']}")


if __name__ == "__main__":
    main()

