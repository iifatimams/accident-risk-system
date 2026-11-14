# Synthetic driving-scenario generator

**SYNTHETIC SOFTWARE TEST FIXTURE. Not evidence of real-world model
performance** (AGENTS.md: "Synthetic fixtures are software tests, never
evidence of model performance"). Images come from the MetaDrive simulator,
not a real camera; telemetry and physiology are hand-tuned heuristics, not
measured sensors or a calibrated vehicle/cardiovascular model.

## What this is for

Produces balanced crash / safe-near-miss multimodal episodes (front-camera
RGB frames + synchronized OBD-II-style telemetry + a synthesized heart-rate
trace + driver-attentiveness state) for bootstrapping/pretraining a
multimodal accident-risk model before enough real labeled data exists.
It does **not** feed `src/accident_risk/datasets/` or any real dataset
mapping, and it must never be used as a substitute for real sensor data
when reporting or evaluating that model's actual performance.

## Setup

```bash
pip install -e ".[simulation]"
```

Installs `metadrive-simulator` and `opencv-python-headless`. On Apple
Silicon macOS, `synthetic_data/macos_metadrive_patch.py` works around three
MetaDrive/Panda3D issues that otherwise prevent headless RGB rendering
here; see that module's docstring for the root causes. It is applied
automatically by `generate_scenarios.py` and is a no-op on other platforms
(it is only imported, never called, there).

## Usage

```bash
python -m synthetic_data.generate_scenarios --n-crash 5 --n-safe 5 --out-dir data/synthetic
```

Writes:

- `data/synthetic/images/ep_{episode_id}_step_{step}.jpg` -- front-camera RGB frames
- `data/synthetic/telemetry_log.csv` -- one row per saved frame, columns:

  | column | meaning | unit |
  |---|---|---|
  | `episode_id` | per-episode integer id | - |
  | `scenario_type` | `"crash"` or `"safe"` | - |
  | `step` | frame index within the episode | - |
  | `timestamp_s` | time since episode start | seconds |
  | `image_path` | path to the saved frame, relative to `--out-dir` | - |
  | `speed_mps` | ego vehicle speed | m/s |
  | `rpm` | synthesized engine speed | rev/min |
  | `throttle_frac` | synthesized throttle pedal | 0.0-1.0 |
  | `brake_frac` | synthesized brake pedal | 0.0-1.0 |
  | `engine_load_pct` | synthesized engine load | 0-100 % |
  | `heart_rate_bpm` | synthesized driver heart rate | bpm |
  | `driver_state` | `alert` / `distracted_phone` / `drowsy_asleep` | - |
  | `min_obstacle_dist_m` | nearest in-lane obstacle ahead, NaN if none in range | meters |
  | `collision_occurred` | 1 on the frame a collision was recorded, else 0 | - |
  | `imminent_accident_3s` | 1 for the 3.0s leading up to a collision, else 0 | - |

`episode_id` is the group key for any participant/trip-aware split of this
data, per AGENTS.md ("Use... event-aware splits; keep every event in one
partition").

`throttle_frac`/`brake_frac` are intentionally named and scaled differently
from `SensorFrame.throttle_position`/`engine_load` in
`src/accident_risk/schemas.py` (which are 0-100 percent, and have no brake
field at all) so this synthetic CSV is never mistaken for, or auto-mapped
onto, that real sensor schema.

Never write generator output into `data/raw/` (enforced by `main()`) and
never run the generator from a test (AGENTS.md: "Never modify files inside
`data/raw` or download datasets during tests").
