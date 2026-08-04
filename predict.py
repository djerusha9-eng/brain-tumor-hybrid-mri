from __future__ import annotations

import argparse
import json
import os

from src.config import load_config
from src.inference import predict_path
from src.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run research inference on one MRI file.")
    parser.add_argument("image")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    image_path = os.path.abspath(args.image)
    config = load_config(args.config)
    os.chdir(config["_base_dir"])
    seed_everything(int(config["project"]["seed"]), bool(config["project"]["deterministic_ops"]))
    import tensorflow as tf

    model = tf.keras.models.load_model(config["artifacts"]["model"], safe_mode=False)
    result = predict_path(model, image_path, list(config["data"]["required_labels"]), config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

