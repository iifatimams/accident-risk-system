# RoadSense -- capstone delivery document

**This is the single entry point for examining/demonstrating RoadSense.**
It consolidates what already exists across this repo's other docs (linked
below, not duplicated) with one thing none of them had: an honest,
per-phase maturity statement distinguishing *implemented* from
*integrated* from *tested* from *validated* -- the exact distinction the
project's own master directive insists on, and the one most capstone
write-ups blur.

**Nothing in this document claims road-safety certification, guaranteed
collision prediction, or a measured result that was not actually
measured.** Where a phase was not validated on real data or a physical
device, that is stated directly, not implied away.

## 1. System architecture

RoadSense is a native iOS driver-monitoring research app talking to a
local FastAPI/SQLite backend. High-level data flow:

```text
iPhone (front+rear camera, IMU/GPS, BLE OBD-II, BLE wearable)
   -> BackgroundCaptureService.swift (on-device Vision/CoreML, sensor fusion)
   -> HTTPS POST /ingest  (ngrok tunnel, or LAN)
   -> FastAPI backend (src/accident_risk/api/main.py)
        -> SQLite (data/app.sqlite3): users, sessions, alerts, consent, family contacts
        -> InferenceEngine (Python, for non-edge/legacy sources only)
        -> web dashboard (static/app/, served from the same backend)
   -> Mac-local operator console (127.0.0.1 only, never tunneled) for sign-in codes
```

The **on-device risk index is experimental and uncalibrated** -- see
section 11. Independent post-impact collision detection (section 10)
deliberately does not depend on it.

For the Python inference pipeline's own internals (causality rules,
feature windows, train/val/test group separation), see
[ARCHITECTURE.md](ARCHITECTURE.md) -- it predates this session's iOS/
camera/collision work and covers the lower-level data-pipeline contract
specifically, not the whole app.

## 2. Hardware inventory

- iPhone (front + rear camera, IMU, GPS, Bluetooth) -- primary platform.
- A BLE OBD-II adapter -- already working, pre-existing, preserved as-is
  this session (not rebuilt).
- A WHOOP wearable -- already working, pre-existing, preserved as-is.
- An Amazfit Helio Strap -- added this session (section 9); **UNVERIFIED
  on physical hardware**, logic-complete and unit-tested against mocked
  BLE packets only.
- A Mac, for the backend, operator console, and `xcodebuild`/`swift test`.
- Fitbit is explicitly out of scope (per the directive) and was not
  touched.

## 3. Phone mounting and calibration

Camera rotation was hardcoded to portrait-only before this session (a
single `videoRotationAngle = 90` for both cameras, regardless of how the
phone was actually held). Fixed to respond live to device orientation via
a pure, unit-tested mapping
(`ios/RoadSense/Services/CameraOrientation.swift`). **The landscape
front/back divergence is UNVERIFIED on physical hardware** -- it follows
Apple's documented sample convention from memory, not a measurement; see
`docs/DEVELOPMENT_PROGRESS.md`'s Phase 1 entry for exactly what to check
and the one-line fix if it's backward. No further calibration UI
(Phase 1.3's "lightweight configuration/check procedure") was built.

## 4. Sensor acquisition and units

Acceleration is m/s² (gravity-inclusive combined magnitude is
reconstructed separately from gravity-excluded per-axis values -- see
`BackgroundCaptureService.swift`'s own comments on why both are needed).
Gyroscope is rad/s. Vehicle speed from OBD is converted km/h -> m/s at the
point of use; GPS speed is m/s natively. All timestamps are Unix seconds
UTC. Missing sensors read as `null`, never a fabricated zero (enforced
throughout `SensorFrame`'s schema, `extra="forbid"`).

## 5. Front-camera perception (driver attention)

Replaced a single `eyesOnRoad` boolean with 8 discrete states
(`DriverAttentionState.swift`): attentive / looking left/right/down/up /
eyes closed / face not detected / **uncertain** (new -- a low-confidence
reading now reads as "I don't know," never silently "attentive").

Two real bugs found and fixed:
- No threshold existed for looking **up** at all (only down) -- a vanity-
  mirror glance could never be flagged.
- A single noisy "attentive" frame could instantly clear a pending/
  committed distraction with zero debounce, while entering a distraction
  required 1.5s of sustained evidence -- an asymmetric hysteresis bug,
  likely the cause of the originally reported "large head turns ... app
  still appears to report eyes on the road." Fixed with a symmetric
  dwell-time state machine (`AttentionHysteresis`).

DMD (Driver Monitoring Dataset) was used for a from-scratch driver-state
baseline, NOT to validate the rule-based system above -- see section 7.
Thresholds here are starting points, not tuned against labeled real
driving footage (none available in this environment).

## 6. Rear-camera object detection

The deployed rear-camera "car" detection is still a crude aspect-ratio
rectangle heuristic (documented as unreliable since before this session)
-- **no validated detector has been exported to Core ML or integrated**.
Pedestrian detection, by contrast, uses Apple's own trained
`VNDetectHumanRectanglesRequest`, a real detector.

Two candidate detector architectures were benchmarked on real BDD100K
val images/labels (not synthetic): a COCO-pretrained Faster R-CNN and an
SSDLite320-MobileNetV3 (the latter more Core-ML-export-appropriate).
Neither is currently good enough to deploy as-is for `traffic light`/
`traffic sign` classes; see
[benchmarks/bdd100k_pretrained_coco_v1.md](benchmarks/bdd100k_pretrained_coco_v1.md)
and
[benchmarks/bdd100k_detector_comparison_v1.md](benchmarks/bdd100k_detector_comparison_v1.md).
**n=40 images is a feasibility smoke test, not a statistically powered
benchmark** -- stated in both docs, not just here.

## 7. DMD experimental training

`scripts/extract_dmd_frames.py`/`train_driver_state_baseline.py` existed
before this session but had never actually been run (hardcoded paths from
another machine). Fixed and run for real:

- **Single-session baseline**: 85.1% held-out accuracy,
  `not_looking_road` recall 0.896 -- see
  [benchmarks/dmd_driver_state_baseline_v1.md](benchmarks/dmd_driver_state_baseline_v1.md).
- **Cross-driver (subject-separated) follow-up**: trained on 4 drivers,
  evaluated on a 5th held out ENTIRELY. `not_looking_road` recall
  **collapsed to 0.301** -- real evidence the single-session result was
  substantially optimistic. See
  [benchmarks/dmd_driver_state_multisession_v1.md](benchmarks/dmd_driver_state_multisession_v1.md).

**Honest conclusion, stated in that doc**: this baseline is not yet
evidence for replacing the Phase 2 rule-based system with a DMD-trained
model. Only the `gaze_on_road` label (confirmed to exist in the real
annotation data) was used -- no `phone_use` or up/down label was
fabricated; those do not exist in DMD's real taxonomy (confirmed by
reading the actual annotation JSONs, not assumed).

## 8. BDD100K benchmark

See section 6 above and the two linked benchmark docs. 82 of 96 DMD
archives (~104GB) and the vast majority of BDD100K's 10,000 val images
remain unextracted/unused -- this environment extracted only what was
needed for each specific, stated benchmark.

## 9. VZCrash, POLIDriving, and future-collision research

`RoadSenseOnDeviceRiskEngine.swift` combines two bundled Core ML models
(VZCrash-trained, POLIDriving-trained) with ~6 hand-authored heuristic
adjustments, previously inline and inseparable. Extracted the heuristic
half into `RiskFusionHeuristics.swift` (now independently unit-tested --
the CoreML half still cannot be, since it needs an Xcode-only generated
model class) -- see
[the progress doc's Phase 9.1 entry](DEVELOPMENT_PROGRESS.md) for the
full component breakdown and the specific confirmed-correct behaviors
(e.g. a 3-second eyes-closed hard override still wins even inside a
roundabout's false-positive suppression).

A real, previously undiscovered training-parity bug was found and fixed
along the way: `designSpeed` (one of POLIDriving's 14 live features)
defaulted to 80.0 km/h when no map speed-limit tag was found, but
`scripts/train_polidriving.py`'s own missing-value convention is 70.0 --
every such live prediction was silently 10 km/h off from what the model
expects for "unknown." Fixed; see section 12.

**Future-collision (5-second-ahead) research**: already thoroughly
documented in
[FUTURE_RISK_RESEARCH_SUMMARY.md](FUTURE_RISK_RESEARCH_SUMMARY.md)
(rounds 1-3, pre-existing). That document's own stated decision --
"do not indefinitely optimize against VZCrash... no model from this work
is deployed" -- was read and respected this session, not re-litigated;
continuing to tune against the same dataset without new data would
contradict the directive's own "do not endlessly tune" instruction. **No
future-risk model is deployed on iOS.**

## 10. Independent post-impact collision detection

A pure `|acceleration| >= 2.5g` threshold
(`CollisionDetector.swift`/`collision_detector.py`), deliberately
independent of the risk-fusion model above (reads none of its output).
Checkpoint resume items #1-#4 (a main-thread UI deadlock on
confirm/dismiss, a missing-column SQLite migration, a non-atomic/
non-persistent incident-dedup race, and missing lifecycle enforcement on
confirm/cancel) were all found, fixed, and regression-tested this
session -- see `docs/DEVELOPMENT_PROGRESS.md`'s checkpoint sections.

A further bug, a direct consequence of fixing #4, was found and fixed:
once the server correctly started rejecting an invalid confirm/cancel
(HTTP 409), the local retry timer had no way to know that and would have
retried the same rejected request every 10 seconds forever. Fixed with a
new `syncRejected` flag.

**Not fixed, investigated and left honestly documented instead**:
distinguishing a dropped/bumped phone from a real vehicle impact. A
single-sample acceleration threshold cannot currently tell these apart,
and no labeled data exists in this environment to validate a
discriminating heuristic -- adding an unvalidated one risks suppressing
real detections, which is worse than the stated limitation.

## 11. Risk fusion

See section 9. The displayed index is **experimental, not a calibrated
crash probability** -- stated in the engine's own code, in the dashboard
UI text, and here. Explainable breakdown (component-by-component) is now
exposed both in code (`RiskBreakdown` struct) and on the dashboard (an
"experimental" labeled panel).

## 12. OBD / WHOOP / Amazfit data

OBD and WHOOP were already working before this session and were
deliberately NOT rebuilt, per the directive's own instruction -- only
verified where touched incidentally (map/weather confidence flags,
section below). Amazfit Helio Strap support was added this session using
the standard Bluetooth Heart Rate Service, mirroring WHOOP's own
mechanism; see `docs/DEVELOPMENT_PROGRESS.md`'s Phase 15 entry for exactly
what remains unverified on physical hardware (whether the strap exposes
that service by default, and whether both wearables can stream
simultaneously).

Map/GPS/weather context (Phase 10) was audited: found and fixed the
design-speed training-parity bug above, and added confidence flags
(`isDesignSpeedFromMapData`/`isWeatherLiveData`) so a guessed fallback is
no longer indistinguishable from a real observation -- see
`docs/DEVELOPMENT_PROGRESS.md`'s Phase 10 entry. Also documented (not
fixed) that there are two independent map-context implementations (one
Python, mostly dead code; one iOS, the actually-live one) and which is
authoritative.

## 13. UAE synthetic driving simulation

**Not built.** SUMO/CARLA are not installed in this environment, and
installing a simulator is a real environment decision the user has not
yet made (Phase 8.1 explicitly asks for a hardware-requirements check
before installing one). What WAS done instead: the user supplied a
separate, already-schema-validated synthetic fixture package
(`RoadSense_Synthetic_Driving_Scenarios_v1`, 10 scenarios covering normal
driving, near-misses, and simulated collisions), integrated as
`tests/fixtures/roadsense_synthetic_v1/` with 35 tests covering schema
validity, collision-detector accuracy on every scenario, and a
notification-safety guard (a synthetic collision can never reach a real
family contact). This is explicitly a software-test fixture, not a UAE
road-network simulator, and is labeled as such throughout.

## 14. Privacy, consent, and family safety

Research consent/export was audited against the full Phase 17 checklist
and found already well-built (consent gates export regardless of role,
pseudonymous ids, GPS blanked by default, missing sensors stay null,
revocation preserves past data) -- see
`docs/DEVELOPMENT_PROGRESS.md`'s checkpoint #7 entry for the one real gap
found and fixed (newer Phase 2/6/7 signals weren't in the export) and the
dataset-versioning manifest added.

Family monitoring delivery remains **mock-only** -- no real SMS/push/
email provider is wired up, by explicit user instruction this session
("the family notification thingy can be added later"). Nothing in this
codebase can send a real notification to a real person today.

## 15. Dashboard and iPhone integration

The driver dashboard (`static/app/app.js`) was missing every signal added
this session until it was explicitly audited and wired in (attention
state, forward/pedestrian hazard states, the risk breakdown) -- verified
in an actual browser against a throwaway, isolated server (never
`data/app.sqlite3`), not just by reading the diff. See
`docs/DEVELOPMENT_PROGRESS.md`'s Phase 16 entry, including a real mistake
caught that way (an invalid shared `grid-area` that would have silently
overlapped two cards) and fixed before commit.

## 16. Reproducible training and evaluation

Every benchmark/training run referenced above has its exact reproduction
commands in its own doc (linked in sections 6-7). Nothing here needs
committed model artifacts or datasets to reproduce -- all synthetic/
benchmark outputs are regenerated from `data/raw/` (untouched throughout
this session) or the committed synthetic fixture.

## 17. Real-device tests

**Not performed.** No physical iPhone, OBD adapter, WHOOP strap, or
Amazfit strap was available in this environment. Every verification this
session used `xcodebuild` (compile-only, iOS Simulator destination),
`swift test` (unit tests, macOS host), and `pytest` (Python, isolated/
temp databases only) -- never a physical device, and never the real
`data/app.sqlite3`. A real test matrix (portrait/landscape mounting,
daylight/night, both cameras under thermal load, OBD+WHOOP+Amazfit
connected simultaneously, GPS quality variation, extended sessions) still
needs to be run on actual hardware before any of this session's fixes can
be called device-validated, not just logic-validated.

## 18. Limitations and false-positive analysis

Summarized from the sections above, in one place:

- The on-device risk index is experimental/uncalibrated (section 11).
- Rear-camera vehicle detection is still a crude heuristic, not a
  validated detector (section 6).
- DMD-trained driver-state models do not yet generalize across drivers
  (0.301 recall on a held-out driver vs 0.896 same-session) -- the
  existing rule-based system (section 5) is the better-justified choice
  today (section 7).
- Forward-collision and pedestrian-hazard estimation (optical
  time-to-contact, position/motion classification) are implemented and
  unit-tested but have never been evaluated against real labeled
  lead-vehicle-approach or pedestrian-conflict footage.
- The post-impact collision detector cannot distinguish a dropped phone
  from a real impact (section 10).
- Future-collision (5-second-ahead) prediction research concluded that no
  model from that line is ready to deploy (section 9).
- Family notification delivery is mock-only (section 14).
- UAE road-network simulation does not exist; only a software-test
  synthetic fixture does (section 13).
- Nothing in this list has been validated on a physical device (section
  17).

## 19. Final demonstration instructions

```sh
# Terminal 1: backend + Mac-local operator console
.venv/bin/python scripts/serve.py

# Terminal 2: expose the backend to a phone
ngrok http 8000

# Browser (same Mac only): generate a sign-in code
open http://127.0.0.1:8787

# Build and run the iOS app in Xcode, or just verify it compiles:
xcodebuild -project ios/RoadSense.xcodeproj -scheme RoadSense \
  -configuration Debug -sdk iphonesimulator \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath /private/tmp/roadsense-demo-build CODE_SIGNING_ALLOWED=NO build

# Dashboard: open the backend's own URL (http://127.0.0.1:8000/app) and sign in
# as an account manager to see the live driver view, alerts, and research export.

# Run every automated test:
.venv/bin/python -m pytest -q
cd ios && swift test

# Reproduce a specific benchmark: see that benchmark's own doc (sections 6-7)
# for its exact commands.
```

See [IPHONE_LOGIN_TESTING.md](IPHONE_LOGIN_TESTING.md) for the full
sign-in walkthrough and failure diagnosis, and
[DEVELOPMENT_PROGRESS.md](DEVELOPMENT_PROGRESS.md) for the complete,
phase-by-phase session log this document summarizes.

## 20. What counts as "100%" here

Per the directive's own distinctions, applied honestly per phase above:

| Phase | Implemented | Integrated | Tested (synthetic) | Tested (physical device) | Validated (real data) |
|---|---|---|---|---|---|
| 1 Camera orientation | yes | yes | yes (unit) | **no** | n/a |
| 2 Driver attention | yes | yes | yes (unit) | **no** | **no** |
| 3 DMD training | yes | n/a (research) | yes (real DMD data, not synthetic) | n/a | **no** (cross-driver recall 0.301) |
| 4 BDD100K detector | yes (benchmark only) | **no** | yes (real data, n=40) | **no** | **no** (n too small) |
| 5/6/7 Tracking + hazards | yes | yes | yes (unit) | **no** | **no** |
| 9 Risk fusion breakdown | yes | yes | yes (unit, heuristic half only) | **no** | n/a (not a new model) |
| 10 Map/weather | yes | yes | yes (unit) | **no** | n/a |
| 12 Post-impact detection | yes | yes | yes (unit + synthetic fixture) | **no** | **no** |
| 15 Amazfit | yes | yes | yes (mocked BLE only) | **no** | **no** |
| 16 Dashboard | yes | yes | yes (manual browser check) | n/a | n/a |
| 17 Research export | yes | yes | yes | n/a | n/a |
| 8 UAE simulation | **no** | **no** | n/a | n/a | n/a |
| 11 Future-risk prediction | research only | **no** | n/a | n/a | inconclusive (see doc) |
| 13 Offline sync | yes (checkpoint #3) | yes | yes | **no** | n/a |
| 14 Family notifications | partial (consent/UI) | partial | yes (mock sender) | **no** | n/a (no real provider) |
| 18 Device test matrix | n/a | n/a | n/a | **not performed** | n/a |

No row in this table claims more than what was actually done.
