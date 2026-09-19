# Architecture

```text
Local licensed dataset                Android / OBD-II / wearable gateway
         |                                          |
   DatasetAdapter                              SensorFrame JSON
         |                                          |
    ReplaySource                                 LiveBuffer
         +---------------- DataSource ---------------+
                              |
                         SensorFrame
                              |
                 schema validation / Synchronizer
                 per-stream, per-sensor timestamp + age
                              |
                  FeaturePipeline rolling histories
                 /                             \
       vehicle window (10 s)          physiology window (60 s)
                 |                             |
       identical feature definitions and fitted train-only preprocessing
                 |                             |
     event / vehicle classifiers       independent driver-state classifier
                 \                             /
                  available uncalibrated scores
                              |
                   configurable weighted index
                              |
                PredictionResult + quality + latency
```

`InferenceEngine` receives only SensorFrames. It has no dataset names, hardware protocols,
or replay/live branch. `run_source` consumes either acquisition source. FastAPI drains its
live queue immediately; it never awaits a particular sensor. State is isolated by stream.
The engine and API protect state with locks. In-memory queues are bounded, with explicit
overflow; inactive sessions reject ingestion. Use one API worker.

## Time and causality

Wire time is Unix seconds UTC. All durations, strides, limits, horizons and ages use seconds.
Clocks must be aligned upstream. Equal-time complementary fields are accepted; conflicting
same-sensor same-time readings are rejected. A live out-of-order frame is rejected rather
than revising historical predictions. Replay stable-sorts its input; preparation sorts by
stream/time and records provenance. Null means no new measurement, not a sensor reset.

Each channel preserves its observation timestamp under zero-order hold. A stale channel
becomes null in current snapshots and is listed in diagnostics. Optional `resample` produces
a configurable fixed-rate stream using only prior observations. It is not enabled by default:
native acquisition timestamps feed both training and inference. If a project chooses
resampling, apply the same explicit acquisition policy to both paths. IMU frequency never
depends on SpO2 cadence.

Rolling history is retained for the largest configured window. Emitted feature windows are
closed `[t-window, t]`. `stride_seconds` sets inference cadence independently of sensor input
rate; the default is one second, configurable down to the chosen fast cadence. Warmup
requires `min_samples` distinct frame times; it permits partial windows rather than waiting
a full minute for physiology. Complementary packets sharing a timestamp contribute when
observed; inference never waits for another packet at that timestamp.

Feature extraction deduplicates carried values by original channel timestamp, preventing
100 Hz IMU packets from inventing 100 Hz heart-rate samples. Summaries use unweighted unique
observations and population standard deviation unless documented otherwise. Vectors require
simultaneous axis observations. Physical longitudinal/lateral axes are only used after
mounting configuration. Gravity removal and orientation estimation are not silently applied.

Known low-quality physiology is excluded from snapshots; unknown quality is flagged.
Physiology scores require at least one current nonstale HR/SpO2/RR/RMSSD reading. Vehicle
scores require fresh kinematic or OBD evidence. Historical features alone do not bypass
that gate. Availability is checked identically in training and inference.

## Training boundary

```text
DatasetRecord(frame, label, identifiers, onset, context)
        |                   labels never enter SensorFrame
  FeaturePipeline
        |
  cutoff features + separate supervision
        |
  connected participant/driver/vehicle/trip/event/stream groups
        |
  train        validation        test
    |              |              |
 imputer/scaler  select family  evaluate selected family once
 classifier fit
```

Window generation is online within independent streams; no learned transform sees validation
or test rows. Group connections prevent overlap at every supplied identity level. Saved
models include the preprocessing configuration, label mapping and feature order. Inference
rejects incompatible settings. Context columns are preserved separately but are not baseline
features, avoiding accidental inclusion of outcome-derived context. Fusion is not trained:
its configured nonnegative weights are renormalized over available components. If none are
available, the index and presentation category are null.

`data_quality` is configured sensor coverage in the current snapshot, not an uncertainty or
reliability probability. LOW/MEDIUM/HIGH/CRITICAL thresholds are configurable presentation
categories. Prediction JSON includes staleness flags and original observation ages.

## Operational scope

Per-frame structured logs are DEBUG; session messages are INFO. Logging records session,
source, event timestamp, stream, latency, version and quality flags without raw payloads.
Replay reports all-packet processing latency separately from wall-clock playback delay.
The implementation is a simple in-memory research baseline: adapters currently materialize
tables and records, and rolling feature extraction scans windows. Large VZCrash studies need
file/partition selection and eventually streamed preparation or incremental statistics.
Long-running services should reset completed sessions to release per-stream state. No
throughput guarantee or production monitoring is claimed by synthetic latency measurements.
