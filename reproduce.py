from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.config import load_config


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute the complete reproducibility workflow.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-explanations", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    root = Path(config["_base_dir"])
    python = sys.executable
    if not args.skip_tests:
        run([python, "-m", "pytest"], root)
    run([python, "prepare_data.py", "--config", str(config_path)], root)
    run([python, "train.py", "--config", str(config_path), "--prepared"], root)
    run([python, "evaluate.py", "--config", str(config_path)], root)
    if not args.skip_explanations:
        run([python, "explain.py", "--config", str(config_path)], root)
    print("Reproduction completed. Inspect artifacts/test_metrics.json before reporting results.")


if __name__ == "__main__":
    main()

