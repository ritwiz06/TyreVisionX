PYTHON ?= python
CONFIG ?= configs/train/train_resnet18.yaml
CHECKPOINT ?= artifacts/experiments/resnet18_tyrenet_v1/best.pt

.PHONY: setup download manifests folds train eval export serve app test fmt lint

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

download:
	$(PYTHON) scripts/download_datasets.py

manifests:
	$(PYTHON) scripts/prepare_manifests.py

folds:
	$(PYTHON) scripts/prepare_folds.py --config configs/data/datasets.yaml

train:
	$(PYTHON) -m src.train --config $(CONFIG)

eval:
	$(PYTHON) -m src.evaluate --checkpoint $(CHECKPOINT) --split test

export:
	$(PYTHON) -m src.export --checkpoint $(CHECKPOINT)

serve:
	uvicorn src.service_fastapi:app --reload

app:
	streamlit run src/app_streamlit.py

test:
	pytest -q

fmt:
	$(PYTHON) -m black src scripts tests

lint:
	ruff check src scripts tests
