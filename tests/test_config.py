from pathlib import Path

import pytest

from src.config import ConfigError, config_hash, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_default_config_is_valid():
    config = load_config(ROOT / "config.yaml")
    assert config["model"]["dense_units"] == [1024, 512]
    assert len(config_hash(config)) == 64


def test_split_ratios_are_checked(tmp_path):
    text = (ROOT / "config.yaml").read_text(encoding="utf-8").replace("test: 0.15", "test: 0.25")
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="sum to 1"):
        load_config(path)

