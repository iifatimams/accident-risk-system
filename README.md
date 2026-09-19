# Accident-risk research system

A Python 3.11 research repository for vehicle-event classification, vehicle-risk modeling,
and optional driver-state estimation. Historical replay and future hardware ingestion use
the **same `SensorFrame` and `FeaturePipeline`**, including windowing, staleness handling,
feature extraction, and fitted model preprocessing.

`event_detection_score`, `vehicle_risk_score`, and `driver_state_risk` are separate model
outputs. `overall_risk_index` is a configurable weighted display index. None is an established
probability of a future crash. No real dataset is included, and no real-data performance or
probability calibration has been demonstrated. Missing models produce null scores, not safe-driving claims.

## Install on Apple Silicon

Run from the repository root. Python 3.11 is recommended; Python 3.12 is also allowed.

```bash
brew install python@3.11 libomp
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

`libomp` is XGBoost's native OpenMP dependency on macOS. XGBoost explicitly uses CPU with
histogram trees. The optional `TemporalCNN` selects MPS, then CUDA if present, then CPU;
classical training and serving do not import PyTorch. No CUDA installation is required.
On macOS, run optional PyTorch experiments in a **separate Python process** from XGBoost.
An OpenMP runtime collision was reproduced when both were loaded together; the package
guards against that combination. Temporal tests run in a subprocess and still exercise
forward/backward computation and device selection. See the [upstream issue](https://github.com/dmlc/xgboost/issues/11500).
See [XGBoost CPU settings](https://xgboost.readthedocs.io/en/stable/parameter.html) and
[PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).

## Verify without downloading data

```bash
ruff check .
ruff format --check .
python -m pytest
python scripts/smoke_test.py
python scripts/replay.py --prepared artifacts/synthetic_smoke/prepared.parquet \
  --model artifacts/synthetic_smoke/models/vehicle_model.joblib --speed max \
  --output artifacts/synthetic_smoke/cli_predictions.csv
```

The smoke script generates **SYNTHETIC SOFTWARE TEST DATA**, exercises all three baselines,
saves/loads models, replays observations, measures latency, and tests API ingestion. Its
artifacts and test metrics are not research results. Tests require no network access.

## Place and inspect real datasets

Obtain licensed files yourself, including any access-condition acceptance, and place them at:

```text
data/raw/vzcrash/
data/raw/polidriving/
```

The [official VZCrash dataset](https://huggingface.co/datasets/vzc-research-chapter/VZCrash)
requires access-condition acceptance. This repository never downloads it automatically.
POLIDriving's source is [laboratorioAI/polidriving](https://github.com/laboratorioAI/polidriving).
Read [dataset guidance](docs/DATASETS.md) before selecting files or labels.

```bash
python scripts/inspect_dataset.py --dataset vzcrash --path data/raw/vzcrash
python scripts/inspect_dataset.py --dataset polidriving --path data/raw/polidriving
```

Inspection lists actual files, shape, columns, dtypes, missingness, candidate identifiers,
and candidate label distributions. Cadence is reported after a timestamp column and unit
have been supplied using `--config`. Column-name heuristics are inspection hints only.

Edit `configs/vzcrash.yaml` or `configs/polidriving.yaml` using the inspected schema:

- Select `mapping.files` explicitly; do not select multiple processed versions of one recording.
- Set `timestamp_column`, `timestamp_unit`, and a unique `stream_column` per trip/event.
- Map common sensor names to actual scalar columns under `columns`; declare each source unit under `units`.
- Supply complete, globally unique identifiers under `groups` (participant, driver, vehicle, trip, event).
- Set the actual `label_column`, `label_map`, and written `label_interpretation`.
- Select `training.score_labels` explicitly, including only labels present in the chosen data.
- Set `mapping.confirmed: true` only after review.

The scalar-table adapter supports CSV, Parquet, and JSONL. Nested event arrays, separate
metadata files, or vendor archives require a reader based on inspection of those actual
files. Flatten verified observations into `data/interim` with original observation times
and identifiers; never modify `data/raw`. No undocumented native VZCrash file layout is assumed.
An event-relative clock must be converted with a documented reference; do not pretend it
establishes wall-clock dates or continuous driving exposure.

## Exact next commands after obtaining VZCrash

```bash
source .venv/bin/activate
python scripts/inspect_dataset.py --dataset vzcrash --path data/raw/vzcrash
```

Confirm the mappings and file representation as described above, then run:

```bash
python scripts/inspect_dataset.py --dataset vzcrash --path data/raw/vzcrash --config configs/vzcrash.yaml
python scripts/prepare_vzcrash.py --config configs/vzcrash.yaml
python scripts/train_vehicle.py --config configs/vzcrash.yaml --prepared data/processed/vzcrash.parquet
python scripts/evaluate.py --model artifacts/vehicle_model.joblib --features artifacts/vehicle_test_features.parquet
python scripts/replay.py --dataset vzcrash --prepared data/processed/vzcrash.parquet \
  --model artifacts/vehicle_model.joblib --speed max --output artifacts/vzcrash_predictions.csv
```

Preparation writes provenance and inspection summaries alongside standardized Parquet.
Training compares Logistic Regression, Random Forest, and CPU XGBoost using validation
PR-AUC (average precision for configured classes). Only the selected model is evaluated on
the test partition. Preprocessing imputation, scaling, and weights are fitted using the
training partition. Class weights default to balanced and can be a mapping of label to weight.

Artifacts include models, JSON metadata, feature lists, resolved configuration snapshots,
validation comparisons, test metrics, held-out features, and a split-membership CSV. Use a
different `paths.artifacts` per experiment/dataset to avoid overwriting a previous experiment.
Only load trusted joblib artifacts. The environment actually tested is recorded separately
in `docs/VERIFICATION.md`; dependency ranges are declared rather than an exact portable lockfile.

Group splitting prioritizes participant, driver, vehicle, trip, and event. It conservatively
links **all** shared known identifiers, including streams, so a shared event cannot cross
partitions even when higher identifiers conflict. If too few independent groups remain, or
validation lacks a class needed for PR-AUC, training fails clearly; it never falls back to
random neighboring-window splits. Collect more independent data or specify a justified
research protocol rather than repeatedly searching seeds for favorable results.

For POLIDriving:

```bash
python scripts/prepare_polidriving.py --config configs/polidriving.yaml
python scripts/train_vehicle.py --config configs/polidriving.yaml --prepared data/processed/polidriving.parquet
```

For a future independently labeled physiology dataset, register its adapter or supply
common-format prepared data. Configure its real driver-state labels and score classes:

```bash
python scripts/train_physiology.py --config configs/physiology.yaml --prepared data/processed/physiology.parquet
```

Do not use crash clip labels as physiological diagnoses. Do not join unrelated datasets
row-by-row. Physiology training is optional; HR-only data is supported.

## Prediction versus event classification

The default horizon is zero: features ending at `t` classify the available event annotation.
A whole-clip label does not establish when within the clip an event happened. For an
explicit future-event experiment, map verified `event_timestamp_column`, set
`pipeline.prediction_horizon_seconds: 5` and `training.score_labels: ['1']`, and choose a
score target appropriate to that task. Confirm `training.future_annotations_complete: true`
only after reviewing annotation coverage. `training.max_followup_gap_seconds` bounds allowed
gaps in observed follow-up. Targets use `(t, t+5]`; features stop at `t`.
Right-censored targets and windows containing an event onset are excluded. Complete follow-up
and event annotation are required; mere lack of an annotation is not proof of a negative.

Classification metrics include per-class precision/recall/F1, confusion matrix, ROC-AUC,
and PR-AUC. Undefined metrics are null. `training.evaluate.event_metrics` additionally
accepts real onset timestamps and explicit driving-exposure intervals for warning episodes,
false alarms/hour, event detection rate, and lead time. These are not fabricated from clip
durations; training leaves temporal metrics null until suitable exposure is supplied.

## Replay and live ingestion

### Friendly live dashboard

Start the server and open `http://127.0.0.1:8000/`:

```bash
python scripts/serve.py
```

The home page is a plain-language dashboard with one continuous hypothetical drive. It sends
one randomized synthetic SensorFrame per second through the real `/ingest` API. A moving marker
and route are embedded in an actual OpenStreetMap street map of Dubai. The route points are
illustrative; they are not map-matched GPS tracks or route guidance. Map tiles and the Leaflet
library require internet access in the viewing browser. The screen shows current GPS, speed,
accelerometer, gyroscope, steering/swerving, RPM, throttle, engine load, four tire pressures,
heart rate, SpO2, and signal quality. Use **Pause drive** or **Restart trip** at the top.

The demonstration index automatically wanders through LOW, MEDIUM, HIGH, and VERY HIGH,
with random order and per-second variation. Its conditions include traffic, lane movement,
hard braking, and a low-tire-pressure example. The index is scripted solely for presentation;
it is not a trained-model output, validated safety category, or crash probability. Incoming
frames still pass through the actual API and preprocessing engine. Tire pressure, heading,
and steering are typed optional SensorFrame fields, but the current trained baseline feature
extractor does not use them. They are displayed as source telemetry only.

The developer interface remains available at `http://127.0.0.1:8000/docs`. The dashboard
continues to show its labeled demonstration index even if a model is loaded; use the API
response to inspect actual model outputs separately.
Future phone, OBD-II and wearable gateways will send the same SensorFrame JSON to `/ingest`;
the dashboard itself does not need to change.

Replay supports `--speed 1`, `--speed 2`, `--speed 100`, and `--speed max`. Measurements are
unchanged. Timing uses timestamp differences against a monotonic clock. CSV/Parquet outputs
have `.stats.json` companions with processed frames, elapsed time, simulated duration,
mean latency and p95 latency. Simulated duration is the timeline span, **not driving exposure**.

```bash
# Start without models to check ingestion; risk scores remain null.
python scripts/serve.py
# Or load your real trained model:
ACCIDENT_RISK_VEHICLE_MODEL=artifacts/vehicle_model.joblib python scripts/serve.py
```

Models carry their preprocessing settings into serving; conflicting model configurations
are rejected. `.env.example` documents environment overrides. The service defaults to
`127.0.0.1:8000`, one worker and one in-memory session containing independent streams.

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/ingest -H 'Content-Type: application/json' \
  -d '{"timestamp":1700000000,"stream_id":"trip-1","source":"android","accel_x":0.2,"accel_y":0.1,"accel_z":9.81}'
curl -X POST http://127.0.0.1:8000/ingest -H 'Content-Type: application/json' \
  -d '{"timestamp":1700000001,"stream_id":"trip-1","source":"obd","vehicle_speed":12.0,"rpm":1800}'
curl 'http://127.0.0.1:8000/prediction/latest?stream_id=trip-1'
```

All wire timestamps are **Unix seconds UTC**, all speeds m/s, acceleration m/s², gyroscopes
rad/s, HR bpm, SpO2/throttle/load percent, RR/HRV milliseconds, and signal quality 0–1.
Optional fields may be null. `sensor_timestamps` records original observation times when
a gateway carries values forward; it cannot contain future readings. A packet need contain
only its updated sensor. Slow HR or SpO2 updates never hold up IMU processing.

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | Process, session and model readiness |
| `POST /ingest` | One SensorFrame; latest prediction or null during warmup |
| `POST /ingest/batch` | `{"frames": [...]}` up to 1000; per-item temporal errors |
| `GET /prediction/latest` | Optional `stream_id`; required when multiple streams have predictions |
| `POST /reset` | Clear buffers and predictions |
| `POST /session/start` | Fresh active session |
| `POST /session/stop` | Stop ingestion; retain latest result |

Malformed batches are rejected by schema validation before processing. Temporal failures
inside a structurally valid batch are reported per item; prior accepted items remain.
Out-of-order packets are rejected, not retroactively inserted into prior predictions.
`/replay/start` is intentionally omitted: use the CLI to avoid background replay/session
ownership complexity. This is a local research API, without vendor Bluetooth drivers,
authentication, persistent sessions, or a production alerting service.

## Repository guide

```text
configs/                 Explicit timing, model, dataset and fusion settings
data/raw/                Read-only licensed originals (ignored)
data/interim/            Verified conversions (ignored)
data/processed/          Common prepared data (ignored)
artifacts/               Models, metrics, predictions (ignored)
src/accident_risk/
  datasets/              Inspection, registry, confirmed mappings
  sources/               Common DataSource, replay, live queue
  processing/            Validation, causal synchronization, windows, features
  models/                Classical models, fusion, optional CNN
  training/              Group splits, experiments, evaluation
  inference/             Shared engine and source runner
  api/                   FastAPI ingestion service
scripts/                 Runnable workflows and synthetic smoke test
tests/                   Offline synthetic unit/integration tests
docs/                    Architecture, datasets, hardware, limitations, verification
```

The complete file tree is in [REPOSITORY_TREE.md](docs/REPOSITORY_TREE.md).
See [ARCHITECTURE.md](docs/ARCHITECTURE.md), [LIVE_HARDWARE.md](docs/LIVE_HARDWARE.md),
and [MODEL_LIMITATIONS.md](docs/MODEL_LIMITATIONS.md) for contracts and research boundaries.
