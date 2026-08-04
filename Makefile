.PHONY: validate test train evaluate explain reproduce clean

validate:
	python prepare_data.py --config config.yaml

test:
	python -m pytest

train:
	python train.py --config config.yaml

evaluate:
	python evaluate.py --config config.yaml

explain:
	python explain.py --config config.yaml

reproduce:
	python reproduce.py --config config.yaml

clean:
	python -c "from pathlib import Path; import shutil; p=Path('artifacts'); shutil.rmtree(p) if p.exists() else None"

