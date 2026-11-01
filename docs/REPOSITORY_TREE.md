# Repository tree

This is a snapshot of the current repository layout as it exists in this workspace. Hidden infrastructure folders such as `.git`, `.venv`, cache directories, and generated build output are collapsed for readability.

```text
accident-risk-system/
├── .git/ [contents collapsed]
├── .pytest_cache/ [contents collapsed]
├── .ruff_cache/ [contents collapsed]
├── .venv/ [contents collapsed]
├── android/
│   ├── .gradle/ [contents collapsed]
│   ├── app/
│   │   ├── build/ [contents collapsed]
│   │   ├── src/
│   │   ├── .DS_Store
│   │   ├── build.gradle.kts
│   │   └── proguard-rules.pro
│   ├── gradle/
│   ├── .gitignore
│   ├── build.gradle.kts
│   ├── gradle.properties
│   ├── gradlew
│   ├── gradlew.bat
│   ├── local.properties
│   └── settings.gradle.kts
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
│   ├── app.sqlite3
│   ├── interim/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   ├── raw/
│   │   ├── .gitkeep
│   │   ├── obd_entries.csv
│   │   ├── polidriving/
│   │   ├── whoop_ble_30s_vitals.csv
│   │   └── ...
│   └── recordings/
│       ├── realtime_telemetry_20261026_163313.csv
│       ├── realtime_telemetry_20261026_163434.csv
│       ├── realtime_telemetry_20261027_113941.csv
│       ├── realtime_telemetry_20261027_115220.csv
│       ├── realtime_telemetry_20261027_120405.csv
│       ├── realtime_telemetry_20261027_121817.csv
│       ├── realtime_telemetry_20261027_123124.csv
│       ├── realtime_telemetry_20261027_124228.csv
│       ├── realtime_telemetry_20261027_125949.csv
│       ├── realtime_telemetry_20261027_132722.csv
│       ├── realtime_telemetry_20261027_155532.csv
│       ├── realtime_telemetry_20261027_160940.csv
│       └── realtime_telemetry_20261027_161806.csv
├── docs/
│   ├── ANDROID.md
│   ├── ARCHITECTURE.md
│   ├── DATASETS.md
│   ├── INTERFACE.md
│   ├── LIVE_HARDWARE.md
│   ├── LIVE_SETUP.md
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
│   ├── run_advanced_terminal.py
│   ├── run_live_terminal.py
│   ├── serve.py
│   ├── stream_whoop_ble_csv.py
│   ├── test_obd.py
│   ├── train_physiology.py
│   ├── train_vehicle.py
│   ├── train_vzcrash_3class.py
│   ├── with_openmp.py
│   └── ...
├── src/
│   ├── accident_risk/
│   │   ├── __pycache__/ [contents collapsed]
│   │   ├── api/
│   │   │   ├── __pycache__/ [contents collapsed]
│   │   │   ├── static/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py
│   │   │   ├── db.py
│   │   │   ├── main.py
│   │   │   ├── recorder.py
│   │   │   └── routes_app.py
│   │   ├── datasets/
│   │   │   ├── __pycache__/ [contents collapsed]
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── physiology.py
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
│   │   │   ├── map_context.py
│   │   │   ├── physiology_features.py
│   │   │   ├── pipeline.py
│   │   │   ├── synchronization.py
│   │   │   ├── validation.py
│   │   │   ├── vehicle_features.py
│   │   │   ├── weather_context.py
│   │   │   └── windows.py
│   │   ├── sources/
│   │   │   ├── __pycache__/ [contents collapsed]
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── hardware.py
│   │   │   ├── live_buffer.py
│   │   │   ├── live_hardware.py
│   │   │   ├── recording.py
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
│   ├── accident_risk.egg-info/
│   └── .DS_Store
├── tests/
│   ├── __pycache__/ [contents collapsed]
│   ├── fixtures/
│   │   └── .gitkeep
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_app_api.py
│   ├── test_app_store.py
│   ├── test_datasets.py
│   ├── test_features.py
│   ├── test_hardware.py
│   ├── test_inference.py
│   ├── test_live_display.py
│   ├── test_models.py
│   ├── test_processing.py
│   ├── test_recording.py
│   ├── test_replay.py
│   ├── test_schemas.py
│   ├── test_splits.py
│   ├── test_training.py
│   └── test_windows.py
├── .env
├── .env.example
├── .gitignore
├── .whoop_tokens.json
├── AGENTS.md
├── README.md
├── pyproject.toml
├── package.json
├── package-lock.json
├── repomix-output.xml
├── test 18-10-2026.csv
├── Accident_Risk_Model_Explained.pdf
├── harvest.log
├── preview-9780273775324_A37747616.pdf
├── research-writing-style.md
└── ...
```

This tree reflects the current workspace state and includes the Android app, dataset recordings, core Python package, scripts, docs, and test suite.
```
