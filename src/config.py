from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError("Configuration must be a YAML mapping.")
    _validate(config)
    config["_config_path"] = str(path)
    config["_base_dir"] = str(path.parent)
    return config


def _validate(config: dict[str, Any]) -> None:
    required = {
        "project", "data", "preprocessing", "segmentation", "model",
        "training", "evaluation", "explainability", "artifacts",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ConfigError(f"Missing configuration sections: {missing}")

    ratios = config["data"]["split"]
    values = [float(ratios[name]) for name in ("train", "validation", "test")]
    if any(value <= 0 for value in values) or abs(sum(values) - 1.0) > 1e-8:
        raise ConfigError("Train, validation, and test ratios must be positive and sum to 1.")

    if int(config["project"]["seed"]) < 0:
        raise ConfigError("Random seed must be non-negative.")
    if int(config["segmentation"]["clusters"]) < 2:
        raise ConfigError("FCM requires at least two clusters.")
    if float(config["segmentation"]["fuzziness"]) <= 1:
        raise ConfigError("FCM fuzziness must be greater than one.")
    if not 0 <= float(config["model"]["dropout"]) < 1:
        raise ConfigError("Dropout must be in [0, 1).")


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(config["_base_dir"]) / path


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(public_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

