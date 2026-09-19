# Scientific and operational limitations

This is a research implementation. No real-data training, external validation, prospective
evaluation, or crash-probability calibration was possible without actual dataset files.
Synthetic tests verify software behavior only.

- **Rare events:** crash prevalence differs between curated clips and continuous driving.
  High accuracy can hide poor crash recall and excessive false alarms. Report PR-AUC, recall,
  exposure-based false alarms, subgroup performance and uncertainty when real data permit.
- **Distribution shift:** vehicles, roads, weather, driver populations, phone mounts and
  sensor hardware differ. An event classifier trained on one dataset does not establish
  reliable performance on a different fleet or hardware stream.
- **Sensor orientation and gravity:** raw acceleration may contain gravity. Vector magnitude
  features are not gravity compensation. Longitudinal/lateral values are disabled unless
  mounting axes are explicitly configured. Device z gyro is not assumed to be physical yaw.
- **Physiology:** wearable noise, motion artifacts, skin contact, missing packets and device
  estimation algorithms affect HR/SpO2. Thresholds and model classes are not medical diagnoses.
  HRV requires genuine RR/IBI intervals with a suitable recording protocol; isolated or
  artifact-corrupted intervals need further quality control. HR-only input yields no invented HRV.
- **Missing sensors:** models use train-fitted imputation and missingness indicators. Some
  missingness patterns may never have appeared in training. Availability and age are reported;
  sensor coverage is not a confidence estimate. Unknown signal quality is flagged, not proved good.
- **Event labels:** clip-level crashes, near misses, ordinal risk annotations and future
  onsets have different meanings. Whole-clip labels may mislabel individual windows as current
  events. A future task needs verified timing, annotation completeness and pre-event features.
- **Follow-up:** future labels use `(t, t+h]` and exclude right-censored examples. The prototype
  assumes a supplied stream's endpoint represents completed observation; audit recording gaps
  and annotation coverage before future-event research. Lack of a recorded event alone is not
  a validated negative target.
- **Risk index:** fusion is a heuristic weighted index, renormalized over available models.
  Correlated components can double-count evidence. Missing components change its meaning.
  LOW/MEDIUM/HIGH/CRITICAL are configurable display bins, not validated safety categories.
- **Calibration:** estimator `predict_proba` values are class scores. No score is the exact
  probability that a person will crash. Calibration would require held-out reliability
  analysis under appropriate prevalence and deployment conditions, separately from model selection.
- **Grouping:** all shared known identifiers are linked conservatively. Small datasets may
  have insufficient independent components or classes to evaluate; random window splitting
  is not an acceptable fallback. Missing identities still limit the strength of leakage claims.
- **Timing and concurrency:** the API uses one process, locks, and in-memory state. Window
  scanning cost increases with sample rate/window length. Synthetic latency is not a production
  throughput guarantee. Delayed packets are rejected; clients must manage clocks and reconnects.
- **Dataset compatibility:** native archive/nested sequence layouts must be inspected before
  adding a reader. Current confirmed scalar mappings are tested; no unavailable real release
  schema has been claimed as validated. Adapters materialize records, so full-scale preparation
  needs selected partitions or future streaming work to keep memory bounded.
- **Scope:** no vendor drivers, deployed monitoring, production authentication, online model
  adaptation, calibrated alerts, or vehicle actuation. The optional CNN has architecture/device
  tests, not a proven advantage over classical baselines.
- **macOS native runtimes:** loading PyTorch and XGBoost in one process reproduced an
  OpenMP-related segmentation fault on the target environment. These experiment backends
  are isolated; package entry points reject mixed loading on macOS. Do not bypass this
  with duplicate-runtime environment flags. See [the upstream report](https://github.com/dmlc/xgboost/issues/11500).

Temporal metrics require actual events and driving-exposure intervals. Warning episodes are
threshold crossings with an explicit cooldown; each warning matches at most one event. Mean
lead time is calculated only for detected eligible events and is null when none were detected.
Always report missed events alongside lead time. Do not use the wall-clock span of unrelated
clips as the denominator for false alarms per driving hour.
