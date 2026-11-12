# Accident Risk System

This repository is a Python research project for building, training, replaying, and serving vehicle-risk models from multi-source sensor telemetry. It combines pipeline code, dataset adapters, training scripts, API serving, and a lightweight Android client for local data capture and experimentation.

Important note: this is a research and prototyping codebase, not a certified safety system. The project is designed to estimate risk from telemetry windows, not to claim a calibrated probability of a future crash.

## What is in this repo

- `src/accident_risk/`: the core Python package
  - `datasets/`: dataset adapters and schema inspection for real data sources
  - `processing/`: feature extraction, synchronization, validation, and windowing
  - `models/`: classical baselines, fusion, and optional deep learning models
  - `training/`: split logic, training entry points, and evaluation helpers
  - `inference/`: shared replay and inference runtime
  - `sources/`: live and replay data sources
  - `api/`: FastAPI service for ingestion, scoring, and dashboard backend
- `scripts/`: runnable workflows for dataset prep, training, replay, and live service launch
- `configs/`: YAML configs for model training, replay, and data preparation
- `data/`: raw, interim, processed datasets, and recording capture samples
- `android/`: Android project for mobile telemetry gathering / app integration
- `tests/`: unit and integration tests using synthetic fixtures
- `docs/`: architecture, data, hardware, verification, and repo tree notes

## Repository tree

```text
accident-risk-system/
├── android/
│   ├── app/
│   ├── gradle/
│   ├── build.gradle.kts
│   ├── settings.gradle.kts
│   ├── gradlew
│   └── gradlew.bat
├── configs/
│   ├── base.yaml
│   ├── physiology.yaml
│   ├── polidriving.yaml
│   ├── replay.yaml
│   └── vzcrash.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── recordings/
│   └── app.sqlite3
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATASETS.md
│   ├── INTERFACE.md
│   ├── LIVE_SETUP.md
│   ├── MODEL_LIMITATIONS.md
│   ├── REPOSITORY_TREE.md
│   └── VERIFICATION.md
├── scripts/
│   ├── evaluate.py
│   ├── harvest_whoop.py
│   ├── inspect_dataset.py
│   ├── prepare_polidriving.py
│   ├── prepare_vzcrash.py
│   ├── replay.py
│   ├── serve.py
│   ├── stream_whoop_ble_csv.py
│   ├── test_obd.py
│   ├── train_physiology.py
│   ├── train_vehicle.py
│   └── ...
├── src/
│   └── accident_risk/
│       ├── api/
│       ├── datasets/
│       ├── inference/
│       ├── models/
│       ├── processing/
│       ├── sources/
│       ├── training/
│       ├── __init__.py
│       ├── cli.py
│       ├── logging_utils.py
│       ├── schemas.py
│       ├── settings.py
│       └── synthetic.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_datasets.py
│   ├── test_features.py
│   ├── test_inference.py
│   ├── test_models.py
│   ├── test_processing.py
│   ├── test_replay.py
│   ├── test_schemas.py
│   ├── test_splits.py
│   ├── test_training.py
│   └── test_windows.py
├── .env.example
├── .gitignore
├── AGENTS.md
├── README.md
├── pyproject.toml
└── repomix-output.xml
```

## Quick start

### 1) Create the environment

```bash
cd /path/to/accident-risk-system
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

### 2) Run validation checks

```bash
ruff check .
python -m pytest
```

### 3) Run the smoke test / replay workflow

```bash
python scripts/serve.py
# or a CLI replay workflow using the generated synthetic artifacts
```

The repo contains synthetic test fixtures and smoke artifacts so the project can be exercised without downloading external datasets.

## Data and model workflow

This project is designed around a common pattern:

1. Inspect raw dataset files and confirm their schema.
2. Prepare data into a normalized format in `data/interim` or `data/processed`.
3. Train vehicle / physiology models from controlled splits.
4. Replay or serve predictions through the API and live dashboard.
5. Evaluate metrics and save artifacts under `artifacts/`.

Use the config files in `configs/` to define dataset mappings, logical timestamps, features, and model settings.

## Important repo notes

- Real datasets are not bundled in the repo; you must obtain/licence them separately.
- `data/raw/` and any proprietary source data are kept read-only.
- The data and model pipeline prioritizes proper grouping by participant, driver, vehicle, trip, and event.
- Missing sensors should not crash inference, and physiology updates should not block faster vehicle/IMU processing.
- Scores are experimental model outputs; they are not safety or crash-probability guarantees.

## More documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/DATASETS.md](docs/DATASETS.md)
- [docs/LIVE_SETUP.md](docs/LIVE_SETUP.md)
- [docs/MODEL_LIMITATIONS.md](docs/MODEL_LIMITATIONS.md)
- [docs/REPOSITORY_TREE.md](docs/REPOSITORY_TREE.md)
- [docs/VERIFICATION.md](docs/VERIFICATION.md)

The repository tree file is kept in sync with the current workspace layout and is the best source for an exact file-by-file snapshot.
