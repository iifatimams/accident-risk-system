# Dataset contracts and task interpretation

No dataset files were present during implementation. Source references below inform the
research task, not automatic local mappings. All column mappings must be checked against
the supplied release. No datasets are downloaded by installation, tests, or scripts.

## VZCrash

Place authorized files at `data/raw/vzcrash`. Access-condition acceptance is performed by
the user at the [official dataset card](https://huggingface.co/datasets/vzc-research-chapter/VZCrash),
which lists a CC BY-NC 4.0 license. The card describes 16-second events, 100 Hz accelerometer
and gyroscope signals, and 1 Hz GPS speed. Source units are g, degrees/second, and km/hour;
mapping converts these to m/s², radians/second and m/s. Review alignment quality before
assigning physical axes.

The card documents category values `0` crash, `1` near_miss, `2` normal_driving. After verifying
the actual local label column and its serialization, map these conceptually to `crash`,
`near_miss`, `normal`. No near-miss labels are manufactured from negative examples.
The [paper](https://arxiv.org/abs/2606.06074) benchmarks crash detection; its negatives include
near misses. A three-class experiment and the paper's binary task are different protocols.

Keep event IDs; also retain vehicle/participant identifiers where available. Never split
neighboring windows from an event across partitions. Whole-clip event classification can
be supported without a precise onset; future-event prediction cannot. A clip with a crash
label may include substantial normal movement before/after the crash, so even causal
window labels require careful interpretation.

The initial adapter handles confirmed scalar tables in CSV, Parquet or JSONL. If the
release contains nested sequences or separate signal/metadata tables, inspect the local
schema before adding its flattening/join reader. Preserve each sensor's true sample times
and event identity; do not upsample 1 Hz speed into falsely new 100 Hz observations. A
documented relative-to-epoch reference is needed if absolute time is unavailable; that
reference must never be used to claim measured driving hours or real calendar dates.

## POLIDriving

Place selected files at `data/raw/polidriving`. The authors' [repository](https://github.com/laboratorioAI/polidriving)
describes vehicle/driver/context measurements and ordinal risk annotations. It contains
real driver directories and a synthetic `furious` directory, as well as raw, consolidated,
preprocessed, expert-verified and semi-supervised variants. Explicitly select a consistent
real-data version; do not silently combine those variants or count them as independent trips.

Map OBD-II speed, RPM, throttle, engine load, HR, GPS, and context only where actual columns
exist and units are verified. Unavailable SpO2, HRV, or other sensors remain null. Context is
preserved as annotation, not automatically included in baseline features. Outcome-derived
accident context must be excluded from future-prediction features unless its availability
at prediction time is established.

Risk annotations support a vehicle-risk classification experiment. They do not establish
a calibrated future crash probability. Configure `training.target: vehicle_risk_score`
and specify which verified classes contribute to the score. Never repurpose an ordinal
risk category as an observed crash onset or medical diagnosis.

## Future physiology datasets

Register a new adapter through `datasets.registry.register`, or supply prepared records
using `save_prepared`. Keep participant/trip identifiers and genuine driver-state labels.
No row-level joins with unrelated VZCrash/POLIDriving participants are permitted. A separate
model can learn from HR alone or HR plus SpO2. RR/IBI intervals or device-supplied RMSSD are
required for HRV: BPM samples alone do not establish RMSSD or SDNN.

| Task | Appropriate supervision | Unsupported shortcut |
| --- | --- | --- |
| Crash detection | Verified current/past crash event labels | Calling a clip score a five-second warning |
| Near-miss classification | Verified near-miss annotations | Relabeling every negative as a near miss |
| Future-risk prediction | Known onsets, pre-onset windows, complete follow-up | Using future measurements or clip labels as onset |
| Driver-state estimation | Independently validated physiological/state labels | Treating high HR or low SpO2 as a diagnosed condition |

Prepared Parquet stores validated `frame_json`, label, group JSON, optional event timestamp,
and context JSON. All annotations remain separate from model features. Preparation fails
with file/row context on impossible values or unmapped labels. Inspection reports missingness;
missing table cells become null explicitly. Selected data and prepared outputs remain git-ignored.
