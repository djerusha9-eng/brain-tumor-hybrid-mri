from __future__ import annotations

import argparse
import os

import pandas as pd

from src.config import load_config, resolve_path
from src.explainability import explain_model
from src.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SHAP explanations for the test set.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    os.chdir(config["_base_dir"])
    seed_everything(int(config["project"]["seed"]), bool(config["project"]["deterministic_ops"]))
    import tensorflow as tf

    model = tf.keras.models.load_model(config["artifacts"]["model"], safe_mode=False)
    manifest = pd.read_csv(resolve_path(config, config["data"]["prepared_manifest"]), dtype=str)
    output = explain_model(model, manifest, list(config["data"]["required_labels"]), config)
    print(f"SHAP arrays saved to {output}")


if __name__ == "__main__":
    main()

