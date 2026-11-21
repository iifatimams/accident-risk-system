# RoadSense: An Edge-Based Accident Risk Prediction System

RoadSense estimates how dangerous a drive is becoming, roughly five seconds before the fact, by combining vehicle dynamics, driver attention, and road hazards observed from a phone mounted inside the car. The phone reads vehicle telemetry over OBD-II, its own accelerometer and gyroscope, GPS, a front camera that watches the driver and a rear camera that watches the road, and, optionally, heart rate and RR intervals from a Bluetooth wearable. Four specialist models, each trained on its own dataset, turn these signals into one risk index between 0 and 99.9: a random forest trained on POLIDriving for the vehicle-dynamics forecast, a MobileNetV3 network trained on the Driver Monitoring Dataset for gaze and attention, a COCO-pretrained YOLOv8n detector for road objects, and a separate classifier trained on VZCrash that watches for an impact already in progress. The first three are combined through confidence-weighted late fusion into the live risk index; the fourth runs on an independent pathway that raises a collision alert and notifies a driver's family regardless of what the risk index was doing just before the event.

This repository holds both halves of that system: the iOS application that runs on the phone, and the Python backend that trains the models, validates the data pipeline, and serves recorded sessions to a driver, family, and administrator interface. It began as a capstone project, and the code still carries that history, most visibly in a `docs/` folder that records design decisions, checkpoints, and the limitations of each experiment as they were discovered rather than smoothed over afterward.

A caveat worth stating plainly: the fused risk index is a weighted combination of evidence, not a calibrated probability of a crash. One of this project's own findings was that no dataset exists linking vehicle dynamics, driver attention, and road hazards to real crash outcomes on the same drives, so the fused number cannot honestly be read as a percentage chance of an accident. Treat it as a warning signal, the way a cardiovascular risk score is a warning signal rather than a diagnosis.

## Layout

The `ios/` directory contains the deployed application, a Swift package with its own Xcode project. The risk fusion logic, attention-state hysteresis, and collision heuristics live under `ios/RoadSense/Services` and are deliberately kept free of UIKit or CoreML dependencies wherever possible, so that the files implementing the actual decision logic can be unit tested with `swift test` on a plain Mac, separately from the parts of the app that only run inside a full Xcode build.

The Python package under `src/accident_risk/` is the research and serving side: `schemas.py` defines the wire contract every sensor reading must satisfy; `processing/` holds the shared feature pipeline used identically in training, replay, and live inference, along with the validation and windowing logic that keeps a driver's windows from leaking across train and test splits; `models/` holds the classical baselines and the fusion logic; `training/` and `inference/` hold the entry points that fit and run those models; `datasets/` holds adapters for the public datasets the project is built on; and `api/` is the FastAPI service that ingests sensor frames, records sessions, and serves the driver, family, and administrator dashboards.

`scripts/` holds the runnable workflows: dataset preparation and inspection, training for each specialist model, CoreML export for on-device deployment, session replay, and a tool for pulling one recorded driving session back out of the database as a readable CSV. `configs/` holds the YAML files that describe each pipeline's windows, feature set, and training targets, with inheritance so a dataset-specific config only states what differs from the shared base. `data/` holds the SQLite database behind the live service, recorded telemetry, and the raw and processed dataset files, none of which are bundled with the repository since the underlying datasets are licensed separately. `artifacts/` holds trained models and their evaluation metrics. `synthetic_data/` holds a synthetic scenario generator used only to exercise the software's own logic in tests, never as evidence of real-world model performance. `tests/` covers both the Python package and, through the `ios/Tests` target, the parts of the Swift app that can run outside Xcode.

## Getting started

To work on the Python side:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
ruff check .
pytest
```

The backend can be run locally with `python scripts/serve.py`, which starts the FastAPI service that the iOS app talks to during a live session; it also serves the driver, family, and administrator web dashboards.

To work on the iOS side, open `ios/RoadSense.xcodeproj` in Xcode and build onto a device or simulator as usual. The subset of the app's logic that has no UIKit, AVFoundation, or generated CoreML dependency can be tested on its own with `swift test` from inside `ios/`, which is the fastest way to check a change to the fusion or attention logic without a full device build.

## Working with the data and models

Each specialist model is trained from its own public dataset through the same shape of pipeline: inspect the raw schema before writing any column mapping, since a dataset's actual columns are never assumed from its documentation alone; split by driver, vehicle, trip, or event rather than by row, so that no window from a held-out participant leaks into training; and apply the same feature definitions whether the data is coming from a training split, a replayed session, or a live stream, so a model never sees features at inference time that differ from what it was trained on. The configs in `configs/` describe these windows and feature sets explicitly, and `docs/DATASETS.md` records what is and is not known about each dataset's labeling and licensing before any mapping is written.

`docs/MODEL_LIMITATIONS.md` and `docs/ARCHITECTURE.md` are worth reading before extending either the fusion logic or the training pipeline: they record, in plain terms, where this system's evidence is thin, which components are genuinely validated against held-out drivers and which are still experimental, and why certain design choices, such as keeping the collision-detection pathway entirely separate from the predictive risk index, were made deliberately rather than incidentally.
