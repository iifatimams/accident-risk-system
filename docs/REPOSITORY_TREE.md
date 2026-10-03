# Repository tree

Snapshot of the local workspace on 2026-10-03, including tracked, untracked, and
ignored project files, datasets, logs, and generated model artifacts.
Virtual environment, Git metadata, package metadata, and cache directories are
shown with their contents collapsed. File entries show names only.

```text
accident-risk-system/
├── .git/ [contents collapsed]
├── .pytest_cache/ [contents collapsed]
├── .ruff_cache/ [contents collapsed]
├── .venv/ [contents collapsed]
├── artifacts/
│   ├── synthetic_smoke/
│   │   ├── input/
│   │   │   └── SYNTHETIC_ONLY.csv
│   │   ├── models/
│   │   │   ├── vehicle_logistic_regression.config.json
│   │   │   ├── vehicle_logistic_regression.features.json
│   │   │   ├── vehicle_logistic_regression.joblib
│   │   │   ├── vehicle_logistic_regression.metadata.json
│   │   │   ├── vehicle_metrics.json
│   │   │   ├── vehicle_model.config.json
│   │   │   ├── vehicle_model.features.json
│   │   │   ├── vehicle_model.joblib
│   │   │   ├── vehicle_model.metadata.json
│   │   │   ├── vehicle_random_forest.config.json
│   │   │   ├── vehicle_random_forest.features.json
│   │   │   ├── vehicle_random_forest.joblib
│   │   │   ├── vehicle_random_forest.metadata.json
│   │   │   ├── vehicle_split_manifest.csv
│   │   │   ├── vehicle_test_features.parquet
│   │   │   ├── vehicle_xgboost.config.json
│   │   │   ├── vehicle_xgboost.features.json
│   │   │   ├── vehicle_xgboost.joblib
│   │   │   └── vehicle_xgboost.metadata.json
│   │   ├── cli_evaluation.json
│   │   ├── cli_evaluation_stdout.json
│   │   ├── cli_predictions.csv
│   │   ├── cli_predictions.stats.json
│   │   ├── cli_prepared.parquet
│   │   ├── cli_prepared.provenance.json
│   │   ├── config.yaml
│   │   ├── predictions.csv
│   │   ├── predictions.stats.json
│   │   └── prepared.parquet
│   └── .gitkeep
├── configs/
│   ├── base.yaml
│   ├── physiology.yaml
│   ├── polidriving.yaml
│   ├── replay.yaml
│   └── vzcrash.yaml
├── data/
│   ├── interim/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       ├── .gitkeep
│       ├── whoop_ble_30s_vitals.csv
│       └── whoop_vitals_continuous.csv
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATASETS.md
│   ├── LIVE_HARDWARE.md
│   ├── MODEL_LIMITATIONS.md
│   ├── REPOSITORY_TREE.md
│   └── VERIFICATION.md
├── scripts/
│   ├── __pycache__/ [contents collapsed]
│   ├── evaluate.py
│   ├── harvest_whoop.py
│   ├── inspect_dataset.py
│   ├── prepare_polidriving.py
│   ├── prepare_vzcrash.py
│   ├── replay.py
│   ├── serve.py
│   ├── smoke_test.py
│   ├── stream_whoop_ble_csv.py
│   ├── test_obd.py
│   ├── train_physiology.py
│   └── train_vehicle.py
├── src/
│   ├── accident_risk/
│   │   ├── __pycache__/ [contents collapsed]
│   │   ├── api/
│   │   │   ├── __pycache__/ [contents collapsed]
│   │   │   ├── static/
│   │   │   │   ├── app.js
│   │   │   │   ├── dashboard.css
│   │   │   │   ├── index.html
│   │   │   │   └── styles.css
│   │   │   ├── __init__.py
│   │   │   └── main.py
│   │   ├── datasets/
│   │   │   ├── __pycache__/ [contents collapsed]
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── polidriving.py
│   │   │   ├── registry.py
│   │   │   └── vzcrash.py
│   │   ├── inference/
│   │   │   ├── __pycache__/ [contents collapsed]
│   │   │   ├── __init__.py
│   │   │   ├── engine.py
│   │   │   └── replay_runner.py
│   │   ├── models/
│   │   │   ├── __pycache__/ [contents collapsed]
│   │   │   ├── __init__.py
│   │   │   ├── baselines.py
│   │   │   ├── fusion.py
│   │   │   ├── physiology_model.py
│   │   │   ├── temporal_torch.py
│   │   │   └── vehicle_model.py
│   │   ├── processing/
│   │   │   ├── __pycache__/ [contents collapsed]
│   │   │   ├── __init__.py
│   │   │   ├── physiology_features.py
│   │   │   ├── pipeline.py
│   │   │   ├── synchronization.py
│   │   │   ├── validation.py
│   │   │   ├── vehicle_features.py
│   │   │   └── windows.py
│   │   ├── sources/
│   │   │   ├── __pycache__/ [contents collapsed]
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── live_buffer.py
│   │   │   └── replay.py
│   │   ├── training/
│   │   │   ├── __pycache__/ [contents collapsed]
│   │   │   ├── __init__.py
│   │   │   ├── evaluate.py
│   │   │   ├── splits.py
│   │   │   ├── train_physiology.py
│   │   │   └── train_vehicle.py
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── logging_utils.py
│   │   ├── schemas.py
│   │   ├── settings.py
│   │   └── synthetic.py
│   ├── accident_risk.egg-info/ [contents collapsed]
│   └── .DS_Store
├── tests/
│   ├── __pycache__/ [contents collapsed]
│   ├── fixtures/
│   │   └── .gitkeep
│   ├── __init__.py
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
├── .env
├── .env.example
├── .gitignore
├── .whoop_tokens.json
├── Accident_Risk_Model_Explained.pdf
├── AGENTS.md
├── harvest.log
├── preview-9780273775324_A37747616.pdf
├── pyproject.toml
├── README.md
└── research-writing-style.md
```
