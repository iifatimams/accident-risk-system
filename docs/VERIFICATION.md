# Implementation verification — 2026-09-16

Verified locally on macOS 14.5 arm64 using Python 3.11.16. Package installation completed
with `python -m pip install -e ".[dev]"` in the existing `.venv`. Homebrew `libomp` 23.1.1
was installed to satisfy CPU XGBoost's native dependency.

| Check | Observed result |
| --- | --- |
| Editable installation | Passed |
| `python -m pip check` | No broken requirements |
| `ruff check .` | Passed |
| `ruff format --check .` | Passed |
| `python -m pytest -q` | 46 passed; two upstream TestClient deprecation warnings |
| `python scripts/smoke_test.py` | Passed all three classical baselines, save/load, replay, API ingestion |
| Synthetic CLI inspection | Actual fixture schema/missingness/label counts and cadence printed |
| Synthetic CLI preparation | 160 standardized records plus provenance saved |
| Synthetic CLI evaluation | Saved model evaluated without refitting; JSON artifact written |
| Synthetic CLI replay | 160 frames processed; 140 predictions saved to CSV |
| Real Uvicorn startup | Bound localhost port 8765; `/health` returned HTTP 200 |
| Server cleanup | Verification process stopped gracefully |
| Raw data | VZCrash and POLIDriving folders absent; no files written to `data/raw` |
| `git diff --check` | Passed |

The replay smoke run measured mean processing latency approximately 13.46 ms and p95
15.89 ms across 160 synthetic packets. This is a small software smoke measurement, not
a throughput guarantee or scientific model-performance result. Both warmup packets and
prediction packets enter these latency summaries. The 917-second synthetic timeline span
includes gaps between events and is not driving exposure.

Important installed package versions: NumPy 2.4.6, pandas 3.0.5, scikit-learn 1.9.1,
XGBoost 3.2.0, PyTorch 2.14.0, FastAPI 0.141.1, Pydantic 2.13.5, pytest 9.1.1,
Ruff 0.16.7. The two warnings concern Starlette's HTTPX/AnyIO compatibility paths; tests
were not skipped or warning-filtered. Dependencies are declared with ranges, not an exact lockfile.

Loading PyTorch and XGBoost together reproduced a native segmentation fault consistent
with [the upstream macOS OpenMP conflict](https://github.com/dmlc/xgboost/issues/11500).
The optional temporal backend now uses a separate test process; package entry points guard
mixed loading on macOS. Tests still perform CNN forward/backward computation, device-priority
checks, and a forward pass on the device available to that process. Classical paths never
import PyTorch. The full suite passed after this change.

Generated smoke data, models and predictions are under `artifacts/synthetic_smoke/` and
remain ignored by Git. Synthetic model metadata explicitly identifies `SYNTHETIC_TEST_ONLY`.
These artifacts demonstrate software operation only; their classification metrics are not
reported as real-data research results.

Still dependent on supplied real data: native release schema/representation inspection,
confirmed columns/units/labels and identifiers, any nested-sequence reader, real model
training and evaluation, temporal exposure-based evaluation, and calibration. Real sensor
integration is also untested. Vendor Bluetooth code and the optional `/replay/start` API
endpoint were intentionally not implemented; CLI replay is complete.

The exact post-download workflow is in [README.md](../README.md#exact-next-commands-after-obtaining-vzcrash).
