# RoadSense development progress

This tracks phase-by-phase status against the master development directive.
It is a resumption checkpoint, not a completion record: a phase is only
"done" when code is written AND integrated AND tested AND (where hardware is
required) verified on the physical device -- those are separate claims, and
this file keeps them separate. See `docs/DEVELOPMENT_CHECKPOINT_2026-11-12.md`
for the state immediately before this entry.

## Phase 1 -- Dual-camera capture: PARTIALLY COMPLETE

Scope actually completed this session (everything that does NOT require the
physical iPhone or a real driving session):

1. **Portrait/landscape rotation is no longer hardcoded.** Previously both the
   front and rear `AVCaptureConnection`s were hardcoded to
   `videoRotationAngle = 90`, which is only correct in portrait. Added
   `ios/RoadSense/Services/CameraOrientation.swift`: a pure, UIKit-free
   mapping from phone orientation + camera position to the correct rotation
   angle (front and back diverge in landscape because the two sensors are
   mounted rotated 180 degrees relative to each other).
   `BackgroundCaptureService.swift` now observes
   `UIDevice.orientationDidChangeNotification` and re-applies the correct
   angle to both connections live, instead of only setting it once at
   startup.
   - **UNVERIFIED on physical hardware.** The landscape front/back divergence
     follows Apple's documented AVCam sample convention from memory, not a
     measurement. Before trusting it: mount the phone in landscape (both
     orientations) and confirm the front-camera debug feed and rear bounding
     boxes are upright, not rotated or mirrored. If wrong, only the two
     `case` lines in `CameraOrientation.swift` need to change -- the dynamic
     update plumbing is already correct either way.
2. **Camera session lifecycle handling.** Added observers for
   `AVCaptureSession.wasInterruptedNotification`,
   `.interruptionEndedNotification`, and `.runtimeErrorNotification`. A
   runtime error now restarts the session instead of silently leaving stale
   `eyesOnRoad`/`trafficLevel` values on the dashboard forever. Previously
   there was no handling for phone calls, Siri, another app taking the
   camera, or a hardware fault at all.
3. **Real FPS/latency instrumentation**, published for the dashboard:
   `frontDeliveredFps`, `backDeliveredFps`, `frontProcessingLatencyMs`,
   `backProcessingLatencyMs`. These measure actual sample-buffer arrivals and
   actual Vision-request completion time, not the configured
   `activeVideoMinFrameDuration`. **No real numbers are reported here** --
   the point of this work was to make it *possible* to see them on a real
   device; nothing in this session ran on an iPhone.
4. **Fixed previously-broken Swift test infrastructure.** `ios/Package.swift`
   declared a `swift test`/`swift build`-able "RoadSense" target that could
   never actually compile (it depends on UIKit, which doesn't exist on
   macOS, plus a stray `Info.plist` resource declaration that failed
   immediately regardless). This meant `ios/Tests/AuthTests.swift` (12 tests,
   already written, already tracked in git) had never actually been run by
   anyone -- `swift test` failed before reaching it. Restructured
   `Package.swift` into a `RoadSenseAuth` library target containing only the
   UIKit/AVFoundation-free files (`AuthManager.swift`, `RoadSenseAPI.swift`,
   `CameraOrientation.swift`) plus a matching test target. This does **not**
   change what the real app builds -- `ios/RoadSense.xcodeproj` has no SPM
   package dependency at all and was already compiling every file directly;
   confirmed this by grepping the `.pbxproj` for any package reference
   (none) before making the change.
   - `swift test`: **17 passed** (12 pre-existing `AuthTests` that could not
     previously run, plus 5 new `CameraOrientationTests`).

What Phase 1 still needs, and why it couldn't happen this session:

- **Actual measured FPS/latency/thermal/battery numbers.** Requires a
  physical iPhone running a real or simulated drive. The directive is
  explicit that a build or simulator run does not establish this.
- **Confirming the landscape rotation mapping is actually correct**, not
  just internally consistent (see point 1 above).
- **Front-camera mirroring behavior** was not touched this session --
  `isVideoMirrored` was left at its existing (unset/default) value. This
  only matters once Phase 2 implements LEFT/RIGHT attention states, where
  mirroring determines which physical direction "left" corresponds to; it
  does not affect pitch, yaw, roll, or EAR math used today.
- **On-screen mounting/calibration guidance (Phase 1.3)** was not built.
  Recording *which* orientation a session used is now possible
  (`currentOrientation` is published) but nothing persists it into
  `SensorFrame`/the research export yet -- that's a small follow-up, not
  done here to keep this commit reviewable as one coherent change.
- Frame rate was **left at 3 fps** (the pre-existing
  `activeVideoMinFrameDuration`/`activeVideoMaxFrameDuration` of
  `CMTime(1, 3)`) for both cameras. The directive explicitly says not to
  raise capture rate without measurements, and no on-device measurement was
  possible this session. This is very likely part of why Phase 2's "looking
  sideways takes too long to detect" complaint exists, but changing it
  without evidence would violate the same rule it's trying to satisfy.

Verification performed this session:

- `xcodebuild -project ios/RoadSense.xcodeproj -scheme RoadSense -configuration Debug -sdk iphonesimulator -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build`: **BUILD SUCCEEDED**. Compilation only, not a runtime or device test.
- `cd ios && swift test`: **17 passed, 0 failed.**
- `.venv/bin/python -m pytest -q`: **128 passed** (unchanged by this
  session's changes; run to confirm no regression).
- No Android files touched. No files under `data/raw` touched. No dataset
  downloads performed. `data/app.sqlite3` was already modified in the
  working tree at session start (unrelated to this work) and was
  deliberately left out of this commit.

## Phase 2 -- Front-camera driver attention detection: PARTIALLY COMPLETE

Scope completed this session (logic/state-machine fixes that don't require
hardware; see "what's not done" below for what does):

1. **Added the missing "looking up" case.** The previous classifier only
   checked `pitch < -0.4` (down); there was no upward threshold at all, so
   no amount of tilting the head back (e.g. to check a vanity mirror) could
   ever be flagged. New `classifyGaze` in
   `ios/RoadSense/Services/DriverAttentionState.swift` checks both
   directions (`pitchDown`/`pitchUp`, symmetric by default).
2. **Fixed an asymmetric-hysteresis flicker bug.** The old code required
   1.5s of *sustained* evidence to flag a distraction, but a single frame
   reading "attentive" -- which pose-estimation noise can produce even
   mid-turn -- instantly cleared it with zero debounce in the other
   direction. This is almost certainly why "large head turns sometimes
   cause face loss while app still appears to report eyes on the road":
   one noisy frame during a real sustained turn could cancel the pending
   distraction. `AttentionHysteresis` now requires sustained evidence in
   BOTH directions (`enterDwell` = 1.5s to become distracted, `exitDwell` =
   0.5s to return to attentive) -- same logic now governs face-loss and
   head-turn/eyes-closed detection uniformly, replacing the old separate,
   inconsistent `noFaceStartTime`/`distractedStartTime` timers.
3. **Added a confidence gate -> `.uncertain`.** Per "do not turn unknown
   into attentive": when Vision's face-detection confidence is below
   `minConfidence` (0.4 default), the frame now reads `.uncertain` instead
   of trusting whatever noisy pitch/yaw/EAR values a degraded reading
   (darkness, glare, partial occlusion) happens to produce. This is the
   mechanism for Phase 2.5 (nighttime): there is no way to validate real
   nighttime accuracy without a physical device, but the system now has an
   honest "I don't know" state for when detection degrades, rather than
   defaulting to "safe."
4. **Added 8 discrete `DriverAttentionState` cases** (`attentive`,
   `lookingLeft/Right/Down/Up`, `eyesClosed`, `faceNotDetected`,
   `uncertain`), published as `attentionState`, round-tripped through
   `SensorFrame.attention_state` to the backend's live dashboard
   (`cv_attention_state` in `/api/v1/predict/live`) and `ui_state`. The
   existing `eyes_on_road` boolean is preserved and now derived from this
   state (`DriverAttentionState.isEyesOnRoad`), not tracked separately --
   satisfies "preserve compatibility with the existing eyes_on_road field."
5. **Documented, not fixed, two real limitations (Phase 2.4):** this is
   head pose, not eye gaze -- a driver who moves only their eyes (not their
   head) toward a phone reads as attentive, and there is no pupil-tracking
   implementation. There is also no visual phone-use detector -- looking
   down is inferred from head pitch only, never a visually confirmed phone
   in frame. Both are stated directly in `DriverAttentionState.swift`'s
   doc comments rather than silently implied to work.

What's explicitly NOT done, and why:

- **The yaw-sign convention for lookingLeft/lookingRight is UNVERIFIED.**
  `classifyGaze` picks positive yaw = looking right as a starting
  convention; this cannot be confirmed without turning a real head in
  front of the actual (non-mirrored) front camera and checking which way
  `debugYaw` moves. Documented inline with the exact one-line fix if
  backward.
- **None of the dwell times or thresholds are empirically tuned.** `1.5s`
  enter / `0.5s` exit / pitch `±0.4` / yaw `0.45` / EAR `0.18` / confidence
  `0.4` are reasonable starting points, not validated against labeled
  real-driving footage -- no such dataset exists in this environment. The
  directive's own instruction ("tune based on labeled scenarios rather
  than arbitrary guesses") cannot be followed without either real labeled
  clips or a physical device test session.
- **Visible phone-use detection, true eye-gaze estimation**: not
  attempted. Documented as unimplemented rather than faked (Phase 2.4).
- **Nighttime accuracy**: not measurable here. The confidence-gate
  mechanism (#3 above) is the only thing this session could do toward
  Phase 2.5 without a physical device and real low-light footage.
- Still running at the Phase-1-documented 3fps capture rate, which bounds
  how fast any of this can possibly react regardless of threshold tuning.

Verification performed this session:

- `xcodebuild` (Debug/iphonesimulator): **BUILD SUCCEEDED.**
- `cd ios && swift test`: **32 passed** (17 from Phase 1, plus 8
  `ClassifyGazeTests` and 7 `AttentionHysteresisTests` -- the latter
  includes a test that directly reproduces and asserts against the
  flicker bug described in #2 above).
- `.venv/bin/python -m pytest -q`: **129 passed** (128 from before Phase 2,
  plus one new test asserting `attention_state`/`cv_attention_state`
  actually round-trips through `/ingest` -> `/api/v1/predict/live` instead
  of being silently dropped).
- No Android, `data/raw`, or dataset changes.

## Synthetic driving fixture integration (RoadSense_Synthetic_Driving_Scenarios_v1)

Not a numbered phase in the directive, but directly serves Phase 8's "keep
real and synthetic data separate" requirement and gives Phases 6/7/11/12
something concrete to test against before SUMO/CARLA exists. Done this
session, ahead of Phase 3, at the user's request.

The user supplied `RoadSense_Synthetic_Driving_Scenarios_v1.zip`: 10
deterministic 40-second/401-frame scenarios (normal cruise, gentle brake,
roundabout, stopped pedestrian, safe/near-miss/collision lead-vehicle
following, pedestrian-crossing near-miss/collision, GPS/OBD dropout) with
separate ground-truth CSVs. Its own README and manifest already state the
same constraints as this project's AGENTS.md (synthetic-only, no real
routes, not evidence of model accuracy, not for training without
independent validation, not for real alert delivery) -- it was generated
for exactly this project.

What was done:

1. **Validated independently before trusting it.** Running the package's own
   bundled `validate.py` was not permitted in this sandboxed session (it is
   external, unreviewed code); instead wrote a separate script that loads
   every one of the 4,010 frames through the REAL
   `accident_risk.schemas.SensorFrame` pydantic model (not a copy, not a
   mock) and confirmed: all validate cleanly, no ground-truth column
   (`event_in_next_5s`, `contact_now`, the `*_truth` distances) exists in
   the schema at all, no physiology fields are fabricated, timestamps are
   strictly 0.1s apart, and the two collision scenarios have exactly one
   `contact_now` tick (at 28.0s) and exactly 50 `event_in_next_5s` positives
   (the preceding 5 seconds at 10Hz) -- matching the package's own claims.
2. **Committed the fixture into `tests/fixtures/roadsense_synthetic_v1/`**
   verbatim (unmodified from the supplied zip), clearly separated from
   `data/raw` (real participant data, never touched) and from
   `data/processed`/`data/interim`.
3. **Added `tests/synthetic_fixtures.py`**: a small loader (`load_scenario`,
   `iter_scenarios`) that returns `SensorFrame` objects and ground-truth
   label dicts as two SEPARATE collections -- mirroring the package's own
   design intent that labels must never leak into model input.
4. **Added `tests/test_synthetic_fixtures.py`** (35 tests), covering every
   point the user asked for:
   - Schema validity + no leaked ground truth (all 10 scenarios).
   - Hard negatives (normal cruise, gentle brake, roundabout, stopped
     pedestrian, safe following, GPS/OBD dropout) AND both near-miss
     scenarios never trigger the independent kinematic collision detector.
   - Both collision scenarios trigger the detector at EXACTLY the labeled
     28.0s contact tick and nowhere else in the other 400 frames.
   - The 5-second event-label window is exactly the 50 ticks before
     impact, and is False at and after the impact tick itself.
   - All 10 scenarios replay through the real `ReplaySource` +
     `InferenceEngine` pipeline without crashing, including
     `gps_obd_dropout`'s missing-GPS/OBD stretches (AGENTS.md: "missing
     sensors must not crash inference").
   - A synthetic collision frame posted through the real `/ingest` HTTP
     path, against an isolated `tmp_path` database (never the real
     `data/app.sqlite3`, never a running production/ngrok instance),
     confirms the app's notification sender stays the existing
     `MockNotificationSender` -- there is currently no real notification
     provider wired in at all (per the 2026-11-12 checkpoint), so no
     simulated event can reach a real family contact today, and this test
     guards against that silently changing later.

What this does NOT establish, stated per AGENTS.md's instruction that
synthetic fixtures are software tests, never evidence of model
performance: no trained production model was loaded or evaluated against
this data (the pipeline test uses a bare `InferenceEngine()`, same as this
repo's existing convention, so `overall_risk_index` stays `None`
throughout); the "risk-score false positive" requirement is satisfied here
as a pipeline-crash/abstention regression guard, not a claim about a real
risk model's false-positive rate, since no trained model exists to
evaluate.

Verification: `.venv/bin/python -m pytest -q`: **164 passed** (129 before
this work, 35 new). No Android, `data/raw`, or real-dataset changes.

## Phase 4 -- Rear-camera object detection: audit + first real benchmark run

### 4.1 Audit of existing BDD100K work

Read-only audit, no files modified: `data/raw/bdd100k/` contains the full
100k-images zip (5.3GB) and two label zips, but `data/interim/bdd100k_frames/`
was empty and no benchmark output existed anywhere in the repo (checked
`artifacts/`, `models/`) before this session. `scripts/benchmark_detector_bdd100k.py`
existed but had evidently never actually been run here -- confirms the
directive's own caution not to assume a script means a result exists.

`data/raw/dmd/` similarly contains 108GB of compressed drowsiness/gaze/
distraction archives, fully unextracted, with no prior training output.
DMD audit stops here this session (Phase 3) -- 108GB is a much larger
extraction than BDD100K's sample, and the user chose to prioritize BDD100K
first. SUMO/CARLA (Phase 8.1) were also checked and are not installed.

### 4.2/4.3 First real benchmark run

Extracted a deterministic 40-image sample of the real BDD100K val set
(every 250th of 10,000 val filenames, sorted) plus matching real
ground-truth label JSONs into `data/interim/bdd100k_frames/` (gitignored,
not committed, fully reproducible from `data/raw/bdd100k/*.zip` -- exact
commands in the results doc below). Fixed the script's hardcoded `/tmp/...`
paths to the repo's own `data/interim/` so this is reproducible by anyone
who clones the repo and has the raw zips, rather than depending on a path
that only existed on one machine.

Ran the existing COCO-pretrained `fasterrcnn_mobilenet_v3_large_320_fpn`
against it. Full results, methodology, and honest caveats:
[docs/benchmarks/bdd100k_pretrained_coco_v1.md](benchmarks/bdd100k_pretrained_coco_v1.md).

Headline (n=40 images, score>=0.5, IoU>=0.5): car precision 0.736/recall
0.176, person recall 0.056, traffic light 0/116 recall, traffic sign
0/150 recall (partially a taxonomy gap, not just a miss -- see doc). This
is a genuine measured result, not fabricated, and is explicitly flagged as
a feasibility smoke test on n=40, not a statistically powered benchmark --
per the directive's own instruction, this is NOT yet enough evidence for a
fine-tuning decision (Phase 4.3's actual ask).

### 4.2 continued -- SSDLite vs Faster R-CNN comparison

Benchmarked a second, mobile-appropriate detector (SSDLite320-
MobileNetV3-Large) on the exact same 40-image sample. Full results:
[docs/benchmarks/bdd100k_detector_comparison_v1.md](benchmarks/bdd100k_detector_comparison_v1.md).

Real, somewhat counterintuitive result: SSDLite has ~5.6x fewer
parameters but was ~3x SLOWER per image on this Mac's PyTorch/MPS
backend, and clearly less accurate (lower recall on every class with any
true positives: car 0.072 vs 0.176, person 0.028 vs 0.056, truck 0.167 vs
0.500, bus 0.200 vs 0.400). Explicitly flagged why this does NOT settle
the architecture question: this Mac/MPS latency number says nothing
about actual Core ML/iPhone Neural Engine performance (a different
execution path entirely), and SSDLite remains the architecturally better
Core ML export target regardless (no RPN/ROI-align stage, unlike the
two-stage Faster R-CNN). Evidence-gated recommendation: evaluate on a
much larger sample before choosing on accuracy grounds, and treat
on-device latency as a separate, unmeasured question (Phase 4.4/4.5).

What's NOT done:

- DMD audit/extraction (Phase 3) -- now done (see Phase 3 section above),
  including a cross-driver generalization result.
- A larger (ideally full 10,000-image) BDD100K val run, and a day/night
  breakdown (the script doesn't currently read BDD100K's `timeofday`
  attribute) -- both explicitly named as follow-ups in the results docs.
- SUMO/CARLA UAE simulation (Phase 8.1-8.3): neither is installed; not
  attempted this session.
- Any Core ML export or on-device benchmark (Phase 4.4/4.5): both runs so
  far used the full PyTorch model on this Mac, not an exported/quantized
  mobile model -- the SSDLite-vs-Faster-R-CNN latency numbers above are
  explicitly NOT a substitute for this.

Verification: the benchmark script itself ran to completion with real
data (see results doc); `.venv/bin/python -m pytest -q`: still **164
passed** (unaffected -- this work didn't touch any tested code path).

## Phase 3 -- DMD driver-attention model: audit + first real baseline run

### 3.1 Audit of actual local DMD data

Read-only audit, no files modified. `data/raw/dmd/` holds 108GB across
three annotation types (drowsiness 13GB/13 sessions, gaze 19GB/14
sessions, distraction 77GB/14 sessions), each a real participant/session
with an ASAM OpenLABEL JSON of genuine per-frame labels -- confirmed by
reading actual annotation files, not assumed. `data/interim/dmd_frames/`
was empty and no prior checkpoint/evaluation output existed anywhere
before this session, despite `scripts/extract_dmd_frames.py` and
`scripts/train_driver_state_baseline.py` already existing (same pattern as
the BDD100K scripts: written, never actually run here -- both had
hardcoded `/tmp/dmd_audit/...` paths from whatever machine last ran them).

**Real label taxonomies confirmed present**: drowsiness
(`eyes_state`/`blinks`/`yawning`), distraction (`gaze_on_road` binary,
`talking`, `hands_using_wheel`, `driver_actions` incl. `radio`/`drinking`/
`reach_side`/`talking_to_passenger`), gaze (`gaze_zone` 9-way incl.
mirrors/infotainment/steering_wheel). **No `phone_use` or `looking_up`/
`looking_down` label exists anywhere in this data** -- confirmed by
reading the actual JSONs. Per AGENTS.md and the directive, nothing in this
project should claim training/validation against those labels from DMD.

### 3.2/3.3 First real baseline run

Fixed both scripts' hardcoded `/tmp` paths to the repo's own `data/raw`/
`data/interim` (matching the BDD100K fix pattern). Extracted only the one
needed video+annotation pair (416MB) from the `gC-14` distraction archive
(not the full 108GB corpus), then ran the existing extraction + MobileNetV3
baseline exactly as already designed.

Full results, methodology, and honest caveats:
[docs/benchmarks/dmd_driver_state_baseline_v1.md](benchmarks/dmd_driver_state_baseline_v1.md).

Headline: single-session held-out accuracy 85.1% (not_looking_road recall
0.896, precision 0.551 -- over-predicts distraction, the safer failure
direction but a real measured weakness). The causal majority-vote
smoothing evidence-gate test made results WORSE (75.0% accuracy), a real
negative result: naive temporal smoothing should not be assumed to help
without testing it, at least not on this single short session.

### 3.2/3.3 continued -- cross-driver (subject-separated) result

Extended to 4 more drivers (gA-1, gB-6, gE-28 added to training; gF-23
held out entirely for validation -- zero frames from that driver in
training, a true subject-separated split per Phase 3.2's actual
instruction). Full results:
[docs/benchmarks/dmd_driver_state_multisession_v1.md](benchmarks/dmd_driver_state_multisession_v1.md).

**Important, sobering result**: `not_looking_road` recall collapsed from
0.896 (single-session) to **0.301** on the held-out driver, even though
raw accuracy looked similar (0.851 -> 0.870, misleading due to class
imbalance). This is real evidence the single-session baseline was
substantially optimistic and did not generalize -- reported honestly
rather than re-run or re-tuned until it looked better. Current honest
conclusion: this frozen-backbone MobileNetV3 baseline is NOT yet
evidence for deploying a DMD-trained model over the existing Apple Vision
rule-based pipeline (Phase 2); if anything it's evidence to keep relying
on Phase 2's rules until a properly cross-validated model clears a much
higher bar.

What's still NOT done:

- n=1 held-out driver is not a statistically powered generalization
  claim -- a real evaluation needs leave-one-driver-out across most/all
  of DMD's 13-14 distraction participants. 82 more DMD archives (~100GB
  across drowsiness/gaze/distraction) remain unextracted.
- Any comparison against the existing Apple Vision rule-based system from
  Phase 2 (Phase 3.3's actual ask) -- needs DMD video through an
  iOS-equivalent Vision pipeline, not attempted.
- Core ML export / on-device benchmark (Phase 3.4) -- correctly not
  pursued given the cross-driver result above.
- drowsiness and gaze annotation types were audited (taxonomy confirmed)
  but no baseline was trained on them this session.

Verification: scripts ran to completion with real data (see results doc);
`.venv/bin/python -m pytest -q`: still **164 passed** (unaffected).

## Checkpoint resume tasks (docs/DEVELOPMENT_CHECKPOINT_2026-11-12.md): #1 and #2

The 2026-11-12 checkpoint listed 8 resume tasks in priority order, written
before this session's Phase 1-4 work started. Picked back up now that
those phases have tested milestones of their own.

### #1: `syncDecision` main-thread deadlock -- FIXED

Confirmed the exact bug the checkpoint described: `syncDecision` read
`AuthManager.shared.token` (an `@MainActor`-isolated property) via
`DispatchQueue.main.sync { ... }`. `confirmSuspectedCollision` and
`dismissSuspectedCollision` call `syncDecision` directly (not via
`syncQueue.async`), and `ContentView.swift`'s `onConfirm`/`onDismiss`
closures call those directly from SwiftUI button actions -- i.e. from the
main thread. `DispatchQueue.main.sync` from the main thread is a
guaranteed GCD deadlock: tapping Confirm or Dismiss on an incident that
had already uploaded (`serverAlertId` set) would hang the entire UI
forever, every time, not intermittently.

Fixed by wrapping the network call in `Task { await AuthManager.shared.token ... }`
instead -- `Task` never blocks the calling thread regardless of which
thread calls `syncDecision`, so this is safe both from the UI (confirm/
dismiss) and from the background retry timer.

**Not covered by an automated regression test**: `BackgroundCaptureService`
imports `UIKit`/`AVFoundation` and cannot be part of the `swift test`-able
`RoadSenseAuth` target (see Phase 1's notes on why), and a real deadlock
test would need to actually run the UI on a device/simulator. Verified by
code inspection (confirmed the exact call chain above) and a successful
`xcodebuild` compile. A device-level regression test belongs in Phase 18
(final native iOS test matrix: "tap Confirm on an incident, confirm the
UI does not freeze").

### #2: SQLite `alerts.detail_json` migration -- FIXED

Confirmed the bug: `CREATE TABLE IF NOT EXISTS alerts (...)` in `db.py`'s
`SCHEMA` does nothing to a database that already has an `alerts` table
without `detail_json` (added later, after the independent
collision-detector lifecycle needed somewhere to store its fields) --
such a database would hit "no such column: detail_json" on every alert
read/write. Added `Database._migrate()`, run once at startup right after
`executescript(SCHEMA)`: checks `PRAGMA table_info(alerts)` and runs
`ALTER TABLE alerts ADD COLUMN detail_json TEXT NOT NULL DEFAULT '{}'`
only if the column is actually missing -- idempotent, and a true no-op on
an already-current schema (both are tested, see below).

Tested against a synthetic pre-migration database built by hand in
`tests/test_db_migration.py` (manually `CREATE TABLE`, no `detail_json`,
two rows already in it) -- NEVER the real `data/app.sqlite3`, per the
checkpoint's own instruction. Confirms: the column gets added, both
pre-existing rows survive with `detail_json='{}'` (not NULL, not an
error), the migrated database still accepts new writes through the normal
`create_alert` path, re-opening an already-migrated database doesn't
raise "duplicate column," and a brand-new database needs no migration at
all.

Verification: `.venv/bin/python -m pytest -q`: **167 passed** (164
before, 3 new migration tests). `xcodebuild`: BUILD SUCCEEDED (confirms
the Swift fix compiles). No Android, `data/raw`, or real-database changes
-- the real `data/app.sqlite3` was never opened by anything in this
session; only synthetic/tmp_path databases were used for testing.

### #3: incident deduplication is now persistent and atomic -- FIXED

Confirmed the bug: `SessionRecorder._incident_ids_seen` was a plain
in-process `dict[str, int]`, and its check (`.get(...)` under one lock
acquisition) and create (`db.create_alert(...)` then a LATER lock
acquisition to record the id) were two separate steps, not one atomic
operation. Two concrete failure modes, both now fixed:

- **Lost on restart**: a retried/offline-queued upload of the same
  `client_incident_id` arriving after the server process restarted would
  find an empty cache and create a SECOND alert for the same real event.
- **Not atomic**: two concurrent requests for the same `client_incident_id`
  (e.g. the immediate on-device report and a retry-timer upload landing
  before either response returns) could both pass the "not seen yet"
  check before either one recorded it, each creating its own alert.

Fixed by moving the dedup key into the database itself: added a
`client_incident_id TEXT` column + `UNIQUE` index on `alerts` (idempotent
migration in `Database._migrate`, same pattern as #2 above -- SQLite
treats every `NULL` as distinct under `UNIQUE`, so the many `high_risk`/
manual-report alerts with no `client_incident_id` never collide with each
other). Added `Database.get_or_create_collision_alert()`: a single
`INSERT ... ON CONFLICT(client_incident_id) DO NOTHING`, which SQLite's
own writer serialization makes atomic regardless of how many concurrent
callers race on it, falling back to a plain `SELECT` only when nothing
was inserted. `SessionRecorder._check_collision` now uses this (and a new
read-only `get_alert_by_client_incident_id` for the case where there's no
active session to attach a brand-new incident to, e.g. a retry arriving
after the session already ended) instead of the removed in-memory dict.

Four new tests in `tests/test_app_api.py`, matching the checkpoint's own
list: retry after the session ended (stream released from the recorder's
in-memory state, must still resolve via the database), dedup surviving an
actual simulated server restart (two separate `create_app()`/`Database`
instances sharing one sqlite file, no in-memory state carried over at
all), 16 real concurrent threads hitting
`get_or_create_collision_alert` directly with the same id (asserts
exactly one row is ever created and every caller agrees on its id -- run
5x locally to rule out flakiness), and a sanity check that two different
drivers' distinct incident ids never get merged.

Verification: `.venv/bin/python -m pytest -q`: **171 passed** (167
before, 4 new). No Android, `data/raw`, or real-database changes.

Remaining checkpoint resume tasks (#4 lifecycle enforcement for
confirm/cancel, #5 iOS lifecycle tests, #6 family monitoring UI/real
provider, #7 research consent/export verification) are not started.

## Phase 5 -- Temporal object tracking: IMPLEMENTED (logic-complete, pending a real detector)

Added `ios/RoadSense/Services/TemporalTracker.swift`: a pure, unit-tested
greedy IoU tracker (`ObjectTracker`) that assigns persistent identity to
per-frame detections across time -- same greedy per-class IoU matching
strategy already used (and already documented as a simplification) in the
BDD100K benchmark scripts, applied across time instead of across a
prediction/ground-truth pair. Handles the directive's explicit asks:
object persistence (a track survives up to `maxMissedFrames` consecutive
missed detections without losing its identity -- directly fixes what
would otherwise be "a 1-frame miss creates a brand-new car"), bounded
history per track (for Phase 6's size-over-time calculation), and uses
real per-frame timestamps throughout, never an assumed fixed FPS.

Wired into `BackgroundCaptureService.processRoadTraffic`: the existing
car-like-rectangle heuristic's detections now feed the tracker every
frame, and the largest car-class track is published as `leadTrack`.

**Explicitly NOT fixed**: the input is still the same crude aspect-ratio
rectangle heuristic documented as unreliable since the Phase 1 checkpoint
("Rectangle detection cannot reliably identify cars"). Tracking gives that
heuristic *persistent identity*, not *correctness* -- a consistently
wrong detection is now a consistently wrong TRACK. This is why Phase 6
below stays deliberately conservative rather than trusting this input.

8 new Swift tests (`TemporalTrackerTests`): new-track creation, identity
preserved across a matching frame, different classes never merge into one
track, a single missed frame does not fragment a track, a track IS
dropped after exceeding `maxMissedFrames`, and history is bounded.

## Phase 6 -- Forward collision-risk estimation: IMPLEMENTED (experimental)

Added `ios/RoadSense/Services/ForwardCollisionHazard.swift`. Per the
directive's own instruction ("a monocular camera cannot automatically
provide trustworthy absolute distance... do not fabricate depth or claim
precise metres"), this never attempts to estimate metric distance.
Instead it uses **optical time-to-contact** (apparent-size growth rate,
sometimes called Lee's tau in the perception literature): while a tracked
object is genuinely growing larger in the frame, dividing its current
apparent size by its own growth rate gives a time-until-contact estimate
that needs no absolute size or distance calibration at all -- a real,
established technique, not invented for this project. The real
assumptions behind it (roughly constant physical size, roughly constant
closing velocity, needs the object tracked face-on rather than mostly
turning) are stated directly in the module's doc comments, not hidden.

Produces exactly the hazard states the directive asked for:
`noRelevantLeadObject`, `leadObjectTracked`, `possibleClosingMotion`,
`increasingClosingHazard`, `insufficientEvidence`. Deliberately
conservative: below a minimum ego speed, or without enough tracked
history, it never escalates past `leadObjectTracked` -- it does not
guess "closing" from an ambiguous signal, and does not assume a lead
object is safe just because speed/history data happens to be missing.
Classification happens in `BackgroundCaptureService.sendCurrentSnapshot`
(not inside `VisionManager`), since it needs ego speed, which only that
layer has. The pre-existing `leadClosingFast` boolean (previously always
hardcoded `false`, with a comment saying why) is now actually derived
from this -- true only for the strongest (`increasingClosingHazard`)
state -- and the new discrete `forwardHazardState` rounds-trips to the
backend's live dashboard (`cv_forward_hazard_state`), same pattern as
Phase 2's `attention_state`.

13 new Swift tests: `OpticalExpansionTests` (5 -- including a known-value
check: size growing 0.1->0.2 over 1s must yield exactly a 2.0s TTC, a
monotonicity check that faster growth yields shorter TTC, and confirms
static/shrinking objects and too-little history never produce a number)
and `ClassifyForwardHazardTests` (8 -- corridor filtering, insufficient
evidence, low/missing ego speed never escalating, and the two closing
states firing only with sufficient speed AND sufficient apparent growth).
Plus 1 new Python test confirming `forward_hazard_state` actually
round-trips through `/ingest` -> `/api/v1/predict/live` rather than being
silently dropped (same pattern as the existing `attention_state` test).

**What this is NOT**: a validated collision-prediction feature. It is
explicitly gated on Phase 4.4 (a real exported detector) actually
happening before its output should be trusted for anything beyond
internal research/dashboard display. The directive's own Phase 11
distinction applies here too: this is "implemented and integrated," not
"validated on held-out real data" or "demonstrated reliably" -- those
need actual driving data with labeled lead-vehicle approach events, which
does not exist in this environment. The synthetic fixture integrated
earlier this session (`lead_vehicle_nearmiss`/`lead_vehicle_collision`
scenarios) has `lead_distance_m_truth` ground truth that could evaluate
this honestly in a follow-up -- not attempted yet.

Verification: `xcodebuild` (Debug/iphonesimulator): **BUILD SUCCEEDED**.
`cd ios && swift test`: **53 passed** (21 new: 8 tracker + 13 hazard/TTC).
`.venv/bin/python -m pytest -q`: **172 passed** (1 new). No Android,
`data/raw`, or real-database changes.

## Phase 7 -- Pedestrian collision-risk estimation: IMPLEMENTED

Added `ios/RoadSense/Services/PedestrianHazard.swift`, reusing the Phase 5
tracker but over the REAL `VNDetectHumanRectanglesRequest` detections
(unlike the vehicle rectangle heuristic feeding Phase 6, this one IS a
validated Apple-trained detector -- stated explicitly in the module's doc
comment so the two are never confused). Classifies each tracked
pedestrian from position + lateral motion alone (deliberately NOT gated
on ego speed, unlike Phase 6 -- there is no equivalent image-growth
artifact to guard against here) into exactly the distinctions the
directive asked for: `roadsideVisible` (stationary, outside the forward
corridor), `inOrNearPath` (currently within it), `movingAcrossScene`
(moving, but away from the corridor), `enteringConflictZone` (moving
toward it), `insufficientEvidence` (too little tracked history). Per
"visibility alone is not an imminent collision" and "do not treat every
visible pedestrian as a major risk spike": the two common, unalarming
states are the default outcome, and the two conflict-relevant states
require actual directional motion evidence, never firing just because
someone is in frame.

Wired into `BackgroundCaptureService.processRoadTraffic` alongside the
existing pedestrian detection: when multiple pedestrians are tracked at
once, the published `pedestrianHazardState` is the MOST SEVERE of them
(ranked insufficientEvidence < roadsideVisible < movingAcrossScene <
inOrNearPath < enteringConflictZone), so one person entering the path
is never masked by several others standing safely at the roadside.
Rounds-trips to the backend dashboard (`cv_pedestrian_hazard_state`),
same pattern as Phases 2 and 6.

7 new Swift tests: insufficient history, a stationary roadside
pedestrian, a pedestrian already inside the corridor, walking toward the
corridor from BOTH sides (left-to-right and right-to-left), walking away
from it, and the severity ordering used for aggregation.

What's NOT done: no real driving footage with labeled pedestrian-conflict
events exists in this environment to validate this against (same honest
limitation as Phase 6); "vehicle stopped vs moving" context (directive's
own text) was deliberately NOT added as a gate, since pedestrian path
conflict is a real risk regardless of whether the ego vehicle is
currently moving -- unlike Phase 6's speed gate, which exists specifically
to suppress an image-growth artifact that only appears while driving.

Verification: `xcodebuild`: **BUILD SUCCEEDED**. `swift test`: **60
passed** (7 new). `pytest`: **173 passed** (1 new). No Android, `data/raw`,
or real-database changes.

## Phase 9 -- Explainable risk fusion: IMPLEMENTED (9.1 only)

### 9.1 Explainable risk breakdown

`RoadSenseOnDeviceRiskEngine.predictRisk` was a single opaque function
returning one `Double`, with the raw CoreML model output and ~6 different
hand-authored heuristic adjustments (jerk boost, HR-velocity boost,
traffic/pedestrian context, an eyes-closed ramp, an eyes-off-road
penalty, a false-positive suppressor, and a hard override) all summed
together inline, in one function, with no way to ask "why is the score
80 right now?" This file ALSO could not be unit tested at all in this
environment -- it imports CoreML and generated model classes that only
exist inside an Xcode build.

Extracted the heuristic half (steps 5-10 of the original function) into
`ios/RoadSense/Services/RiskFusionHeuristics.swift`: a pure function
(`applyRiskHeuristics`) taking the raw ML fusion value as a plain
`Double` input and returning a `RiskBreakdown` struct with every
component named separately (`mlFusionBase`, `jerkBoost`,
`hrVelocityBoost`, `trafficContext`, `pedestrianContext`,
`eyesClosedPenalty`, `attentionPenalty`, `leadClosingHardBoost`, plus
`falsePositiveSuppressionApplied`/`hardOverrideApplied` flags and the
final clamped `total`) -- this is a line-for-line port, same constants,
same order of operations, not a re-tuning. `RoadSenseOnDeviceRiskEngine`
was refactored to compute the ML components exactly once (new private
`mlComponents`/`breakdown` helpers, no duplicated CoreML inference) and
delegate to this function; `predictRisk`'s existing signature and
behavior are completely unchanged for its one existing caller. A new
`predictRiskBreakdown` method returns the full struct.

Wired into `BackgroundCaptureService.sendCurrentSnapshot`: the payload
now includes a `risk_breakdown` object (every named component) alongside
the existing `risk_score`, dashboard-only like `camera_status`/`debug_*`
-- never part of the validated `SensorFrame` schema, and the index itself
remains exactly as experimental/uncalibrated as before regardless of how
it's broken down. Rounds-trips to the backend (`risk_breakdown` in
`/api/v1/predict/live`).

13 new Swift tests (`RiskFusionHeuristicsTests`) -- the first tests this
heuristic logic has ever had, now that it's separated from the untestable
CoreML half. Pins down exactly what the original code's own comments
claimed: jerk/HR-velocity boosts only above their thresholds and capped,
the continuous (not step-function) eyes-closed ramp, the attention
penalty's traffic/pedestrian escalation, the false-positive suppressor
correctly capping roundabout/swerve cases, and -- critically -- that the
eyes-closed-3-seconds hard override still wins even inside a roundabout
(a real safety condition must never be suppressed by the false-positive
guard meant for ordinary turning noise). Plus 1 new Python round-trip
test.

What's NOT done (9.2 false-alarm investigation, 9.3 sensor-fusion
integrity audit): not attempted this session -- 9.1 alone was a
substantial, self-contained unit of work, and false-alarm tuning
specifically needs real driving data this environment doesn't have.

Verification: `xcodebuild`: **BUILD SUCCEEDED**. `swift test`: **73
passed** (13 new). `pytest`: **174 passed** (1 new). No Android,
`data/raw`, or real-database changes.

## Phases 10-19

Not started.
