# Repository tree

Project files; generated datasets, models, caches and the virtual environment are omitted.

```text
accident-risk-system/
├── artifacts/
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
│       └── .gitkeep
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATASETS.md
│   ├── LIVE_HARDWARE.md
│   ├── MODEL_LIMITATIONS.md
│   ├── REPOSITORY_TREE.md
│   └── VERIFICATION.md
├── scripts/
│   ├── evaluate.py
│   ├── inspect_dataset.py
│   ├── prepare_polidriving.py
│   ├── prepare_vzcrash.py
│   ├── replay.py
│   ├── serve.py
│   ├── smoke_test.py
│   ├── train_physiology.py
│   └── train_vehicle.py
├── src/
│   └── accident_risk/
│       ├── api/
│       │   ├── __init__.py
│       │   ├── static/
│       │   │   ├── app.js
│       │   │   ├── index.html
│       │   │   └── styles.css
│       │   └── main.py
│       ├── datasets/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── polidriving.py
│       │   ├── registry.py
│       │   └── vzcrash.py
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── engine.py
│       │   └── replay_runner.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── baselines.py
│       │   ├── fusion.py
│       │   ├── physiology_model.py
│       │   ├── temporal_torch.py
│       │   └── vehicle_model.py
│       ├── processing/
│       │   ├── __init__.py
│       │   ├── physiology_features.py
│       │   ├── pipeline.py
│       │   ├── synchronization.py
│       │   ├── validation.py
│       │   ├── vehicle_features.py
│       │   └── windows.py
│       ├── sources/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── live_buffer.py
│       │   └── replay.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── evaluate.py
│       │   ├── splits.py
│       │   ├── train_physiology.py
│       │   └── train_vehicle.py
│       ├── __init__.py
│       ├── cli.py
│       ├── logging_utils.py
│       ├── schemas.py
│       ├── settings.py
│       └── synthetic.py
├── tests/
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
├── .env.example
├── .gitignore
├── AGENTS.md
├── README.md
└── pyproject.toml
```
