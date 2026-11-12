# RoadSense Synthetic Driving Test Data — v1

**SYNTHETIC SOFTWARE TEST DATA ONLY. NOT REAL ACCIDENTS, NOT REAL UAE ROUTES, NOT A TRAINED MODEL, AND NOT EVIDENCE OF COLLISION-PREDICTION ACCURACY.**

This package was generated from a deterministic Python script to exercise RoadSense's schema, sensor timestamp handling, replay logic, collision event labels and hard-negative cases. The data deliberately contains no identifiable participants, actual WHOOP readings, phone video, or physical vehicle telemetry.

## Contents

- `generate.py`: Python standard-library-only generator; seed `20261113`. Run `python generate.py` from this folder to recreate the dataset.
- `manifest.json`: scenario catalog, cadence, units, scenario labels and assumptions.
- `data/<scenario>.jsonl`: one **canonical** `accident_risk.schemas.SensorFrame` JSON object per line. Not raw iOS `SensorFrame.toDictionary()` payloads.
- `data/<scenario>_labels.csv`: **separate synthetic ground truth**. These labels do not belong in model input frames.
- `validate.py`: basic standalone checks for counts, timestamps, label consistency, and no invented physiology values; can optionally run repository Pydantic validation.

## Ten scenarios (40.0 seconds each)

| Scenario | Outcome | Purpose |
| --- | --- | --- |
| `normal_cruise` | normal | Steady nonhazardous driving |
| `gentle_brake` | normal | Ordinary braking: hard negative |
| `normal_roundabout` | normal | Ordinary turning/yaw: hard negative |
| `stopped_pedestrian` | normal | Pedestrian visible while ego stopped: hard negative |
| `lead_vehicle_safe` | normal | Follow and brake with ample spacing |
| `lead_vehicle_nearmiss` | near_miss | Fast approach and strong braking; no simulated contact |
| `lead_vehicle_collision` | collision | Simulated contact at t=28.0s |
| `pedestrian_crossing_nearmiss` | near_miss | Crossing-pedestrian conflict, ego vehicle brakes |
| `pedestrian_crossing_collision` | collision | Simulated contact at t=28.0s |
| `gps_obd_dropout` | normal | GPS/OBD intermittent dropout and recovery |

There are **401 time points per scenario at 10 Hz**, inclusive of t=0 and t=40.0, for **4,010 rows** in total. IMU fields are supplied at every time point. GPS fields occur at 1 Hz and synthetic OBD-like fields at 2 Hz, and are **omitted/null** at other times. When a sensor is unavailable, no zeros are invented. Every included value has an original observation time in `sensor_timestamps`.

### Units and axis convention

- `timestamp`: Unix seconds, UTC (arbitrary simulated timestamps, different for every event stream).
- `accel_x/y/z`: **user acceleration excluding gravity**, in m/s², using a simplified hypothetical fixed phone mounting where +x = longitudinal acceleration and +y = lateral acceleration. This is not an iPhone-verified frame. Gravity-inclusive magnitude must be reconstructed with mounting/gravity information; do **not** treat these three channels as raw gravity-inclusive sensor readings.
- `gyro_x/y/z`: rad/s; yaw is simplified, with left turn treated as positive around z.
- `vehicle_speed`, `gps_speed`: **m/s** (note: iOS `SensorFrame` stores its internal OBD speed in km/h before serializing; this package follows the **Python canonical schema**, which is m/s).
- `gps_heading`: radians clockwise from true north.
- GPS coordinates: approximately located within general UAE urban bounding areas, derived from small local synthetic displacements. **They do not follow verified physical roads or speed limits.** This is an UAE-*inspired* schematic prototype, not map-matched OSM routing.
- `heart_rate`, `rr_interval_ms`, `hrv_rmssd`, `spo2`: absent intentionally. No synthetic WHOOP/Helio measurements are impersonated as real observations.

### Ground-truth columns

`event_in_next_5s` is true only strictly **before** the synthetic collision timestamp, within five seconds. `contact_now` is true only at the precise synthetic collision tick. `lead_distance_m_truth` and `pedestrian_distance_m_truth` are simplified simulator-state metadata and **not measured by any sensor or camera**. They must not be inserted into model input features. The synthetic 'contact' acceleration pulse is schematic (a one-tick 29 m/s² bump) and is NOT validated vehicle crash physics.

## Safe ways to use this dataset

**Basic checks:**

```bash
python validate.py
python generate.py    # deterministic sensor content; this updates manifest generation timestamp
```

**Inside the RoadSense clone:**

```python
from pathlib import Path
from accident_risk.schemas import SensorFrame
from accident_risk.sources.replay import ReplaySource
import json
path = Path('path/to/roadsense_synthetic_v1/data/normal_roundabout.jsonl')
frames = [SensorFrame.model_validate(json.loads(line)) for line in path.read_text().splitlines()]
for frame in ReplaySource(frames, maximum_speed=True):
    # Run your isolated test engine against the frame. No HTTP traffic.
    pass
```

The package **does not send HTTP requests**. Do not POST it to a running production or public-ngrok RoadSense backend; the impact pulses could create incident records and notification events. If you run an integration test, launch a separate isolated database, mocked notification provider, no real accounts, and no real notification credentials. Never send simulated accident alerts to real family contacts.

## Limitations

This package does **not** constitute SUMO or CARLA simulation. It does not render camera frames, use validated real UAE road topology, simulate true tire-road dynamics, model head pose, or simulate physiology. Collision timestamps are stipulated for test scenarios. Hard-negative tests are designed to catch obvious software mistakes, not measure real-world false alarms. Do not train or evaluate production accident-prediction accuracy on these scenarios. For realistic OSM routes or video, build an independently sourced simulator dataset and keep it separately labeled.
