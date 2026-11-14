# RoadSense final multimodal architecture -- results

**Post-sprint update**: after reviewing section 3 below, the fake
"car"-classification + traffic-density counting was REMOVED from the
live code (not just left undeployed), the BDD100K dataset and every
BDD100K-only script were deleted, and the pretrained-COCO-detector
Core-ML-export line was also dropped in favor of a different candidate
(YOLO/Ultralytics) still pending a package-install approval -- see the
bottom of section 3 for exactly what changed and why. Section 3's
original text is left intact below as the honest record of what was
tried first and why it didn't work.

Written at the end of an urgent, time-boxed sprint implementing the
two-pipeline multimodal architecture (Pipeline A: 5-second predictive
risk fusion from POLIDriving + DMD + BDD100K; Pipeline B: independent
VZCrash-supported collision detection). Every number below is from a
command that was actually run this session; every "not completed" item
is stated as such, not glossed over. See
[DEVELOPMENT_PROGRESS.md](DEVELOPMENT_PROGRESS.md) and
[CAPSTONE_DELIVERY.md](CAPSTONE_DELIVERY.md) for the rest of the
project's history -- this document covers only this sprint's work.

## 1. Architecture description

```text
Pipeline A (5-second predictive risk fusion)
  Front camera  -> DMDAttentionModel.mlpackage + Apple Vision -> DriverAttentionState
  Rear camera   -> (still the pre-existing crude rectangle heuristic -- see section 5)
  OBD + IMU/GPS -> FusionModel.mlpackage (current) + PolidrivingFiveSecondModel.mlpackage (5s-ahead)
  All three     -> MultimodalRiskFusion.swift (confidence-weighted evidence-based late fusion)
                -> MultimodalRiskEstimate (current risk, 5s-ahead risk, risk level,
                   vehicle/attention/road contributions, evidence quality, dominant factor)

Pipeline B (independent collision detection)
  IMU (accelMag) -> CollisionDetector.swift (pure 2.5g threshold, UNCHANGED, still
                     the sole AUTHORITATIVE trigger -- never gated by Pipeline A)
                 -> CollisionIncidentStore (local-first persistence)
                 -> VZCrashModel.mlpackage (classifyCollisionEvidence, SUPPORTING evidence only)
                 -> confirm/dismiss UI, backend sync, family-notification path (UNCHANGED)
```

VZCrash was moved OUT of Pipeline A's magnitude entirely this sprint
(was previously 50% of the predictive score -- see section 4). Pipeline
A and Pipeline B share no gating dependency in either direction: a
collision is detected from the IMU threshold alone, regardless of what
the predictive risk score says, and the predictive score never reads
`suspectedCollision`.

## 2. What was fully completed, verified, and deployed

### 2.1 POLIDriving 5-second-ahead forecasting (NEW)

- Built a genuine 5-second-ahead target: `hazardous_in_5s = risk_level(t+5 rows) in {high, very_high}`, shifting within each session's own `_lab.csv` only (never across sessions).
- Sampling-rate justification, not assumed: furious/andres sessions verified directly against their own `_col.csv` timestamps (~0.97-0.99 rows/sec, confirmed per-file); alonso's two sessions have no `_col.csv` (they fed the separate `labeled/` semi-supervised pipeline, evidenced by their `_nor/_ovs/_res/_sel/_tst/_vot` files) so they inherit the rate verified on the other sessions -- stated as an inherited assumption, not independently re-measured, in both the script's own comments and here.
- **Driver-held-out evaluation** (train: furious + andres, 3 sessions, 20,963 rows; test: alonso, 2 sessions, 40,585 rows, a driver never seen in training):

| model | ROC-AUC | hazardous_in_5s precision | hazardous_in_5s recall |
|---|---:|---:|---:|
| logistic_regression | 0.777 | 0.488 | 0.866 |
| hist_gradient_boosting | 0.834 | 0.558 | 0.861 |
| **random_forest (deployed)** | **0.918** | **0.606** | **0.930** |

- Exported to `ios/RoadSense/Models/PolidrivingFiveSecondModel.mlpackage` -- refit WITHOUT the `StandardScaler` first used for model selection (tree splits are scale-invariant, so the metrics above hold for the exported model too; this also keeps the Swift-side contract identical to the existing `FusionModel.mlpackage`: 14 raw features in, no scaler to keep in parity on the live side).
- Reproduce: `.venv/bin/python scripts/train_polidriving_5s_ahead.py`

### 2.2 VZCrash reorganized out of the predictive score (FIX)

- **Before this sprint**, `RoadSenseOnDeviceRiskEngine`'s `mlFusionBase` was `0.5*poliRisk + 0.5*vzRisk` -- an EQUAL, i.e. PRINCIPAL, weight, directly contradicting the directive's explicit requirement.
- **Now**: `mlFusionBase = poliRisk * weatherMult` only. VZCrash's `classPred` remains inside the existing false-positive-suppression heuristic (a supporting gate, not a magnitude contributor) -- consistent with "supporting the post-impact safety system," not predicting.
- A new `classifyCollisionEvidence(effectiveAccel:gyroMag:speedKmh:accelObd:)` method runs VZCrash specifically at the moment `CollisionDetector`'s threshold fires, to help discriminate a real collision from harsh braking/potholes/drops/turns. Attached to the incident as new `collisionEvidenceClass`/`collisionEvidenceConfidence` fields on `StoredIncident` (Optional, Codable-backward-compatible, same pattern as the existing `syncRejected` field) -- **supporting evidence only; it never gates whether the incident exists.**
- VZCrash itself was NOT retrained -- the existing `VZCrashModel.mlpackage` checkpoint was reused, per the directive's own "reuse existing trained artifacts" instruction.

### 2.3 Multimodal risk fusion engine (NEW)

- `ios/RoadSense/Services/MultimodalRiskFusion.swift`: an evidence-based temporal late-fusion engine, since POLIDriving/DMD/BDD100K have no synchronized multimodal labels (three unrelated datasets, never concatenated into a fake joint training set here).
- Combines confidence-weighted vehicle/attention/road-hazard evidence into one `MultimodalRiskEstimate`: current risk, a SEPARATE 5-second-ahead risk (from 2.1's model, never blended into current risk), risk level, per-dimension contributions, contextual adjustment, evidence quality (mean confidence across the 3 modalities -- 0 when none have fresh evidence, never silently read as "safe"), and a dominant-factor explanation.
- Wired into `BackgroundCaptureService.sendCurrentSnapshot`, adding a `multimodal_risk` field to the dashboard/backend payload alongside the existing `risk_breakdown`.
- **13 new unit tests**, all passing -- covering missing-modality handling, confidence weighting, the "stationary pedestrian alone is not high risk" requirement, and 5s-ahead-reported-separately-from-current.

### 2.4 DMD driver-attention model, v2 (IMPROVED, REAL NUMBER)

The directive calls out the v1 single-session-trained, cross-driver-evaluated recall of 0.301 as insufficient. v2 (`scripts/train_driver_state_multisession_v2.py`) differs from v1 in three concrete ways: fine-tunes the last 2 MobileNetV3-Small inverted-residual blocks (not just the classifier head -- head-only training was a plausible cause of v1's collapse, since it never adapted any visual features to this task at all), adds label-safe augmentation (ColorJitter for illumination, small RandomRotation for camera angle -- deliberately NOT horizontal flip, which would silently invert gaze-direction semantics), and actually persists a checkpoint (v1 never saved weights or exported anything).

Same held-out-driver split as v1 (train: gC-14/gA-1/gB-6/gE-28, 5,423 frames; val: gF-23, 1,233 frames, NEVER seen in training):

| | v1 (head-only) | v2 (this sprint) |
|---|---:|---:|
| not_looking_road recall | 0.301 | **0.419** |
| not_looking_road precision | (not reported) | 0.634 |
| looking_road recall | (not reported) | 0.957 |
| overall accuracy | (not reported) | 0.876 |

This is a real, measured improvement (not fabricated, not cherry-picked -- the full classification report and confusion matrix are in `artifacts/dmd_attention/metrics.json`), but still far from a reliable production detector: 108 of 186 not-looking-road frames are still missed on the held-out driver. **This is reported honestly as a genuine improvement, not a solved problem.**

- Exported to `ios/RoadSense/Models/DMDAttentionModel.mlpackage` (an image classifier; ImageNet normalization baked into the traced model itself, not approximated via `ct.ImageType`'s single-scalar scale/bias, which cannot express per-channel stats correctly -- see `scripts/export_dmd_attention_coreml.py`'s `NormalizedModel` wrapper).
- **Integrated into the live front-camera pipeline**: `VisionManager.processDriverFace` now runs this model on every frame a face is detected, and combines its binary output with Apple Vision's own (richer, 8-state) classification via a new, unit-tested `combineDMDWithVision` function (`DriverAttentionState.swift`). DMD is coarser than Vision (2 classes vs. 8) so it is used ONLY to escalate -- when DMD strongly disagrees with Vision's "attentive" (probability >= 0.65), the state escalates to `.uncertain`; it can NEVER clear an already-flagged Vision distraction state, since resolving a disagreement toward the safer-looking label is the dangerous direction to be wrong in. When the model is unavailable, Vision alone is used -- the directive's own "preserve Apple Vision as a fallback" requirement.
- Reproduce training: `.venv/bin/python scripts/train_driver_state_multisession_v2.py`. Export: `.venv/bin/python scripts/export_dmd_attention_coreml.py`.
- A real bug found and fixed during export: running the export script standalone segfaulted (exit 139) inside `coremltools.convert`, while the identical logic run inline with `torch` imported BEFORE `coremltools` worked -- a library init-order issue between the two packages on this machine, not a logic bug. Fixed by reordering the script's imports (now documented in the script itself).

### 2.5 Build and test verification

- `cd ios && swift test` -> **118 passed** (98 existing + 20 new: 13 `MultimodalRiskFusionTests`, 5 `CombineDMDWithVisionTests`, 2 `CollisionIncidentStoreTests`).
- `xcodebuild -scheme RoadSense -sdk iphonesimulator ... build` -> **BUILD SUCCEEDED**, confirming all 4 bundled Core ML packages (FusionModel, VZCrashModel, PolidrivingFiveSecondModel, DMDAttentionModel) and every Swift change compile and link in the real Xcode target -- not just the SPM test target.
- `.venv/bin/python -m pytest -q` -> **177 passed** (unaffected by this sprint; no Python backend code changed).
- **No physical iPhone was available this session.** The build above is a simulator-destination compile check, confirming the app builds and the models load into the bundle correctly -- it does NOT confirm on-device camera/inference latency, which remains unverified (see `DEVELOPMENT_PROGRESS.md`'s Phase 18 test matrix for exactly what that would still require).

## 3. What was attempted and HONESTLY did not succeed

### 3.1 BDD100K detector fine-tuning -- regressed, NOT deployed

Fine-tuned a Faster R-CNN MobileNetV3-320-FPN (COCO-pretrained backbone, frozen; new 11-class BDD100K head, FPN+RPN fine-tuned) on 600 extracted BDD100K train images, 4 epochs (`scripts/finetune_bdd100k_detector.py`), then re-evaluated on the SAME 40 held-out val images the earlier pretrained-only benchmark used (`scripts/eval_bdd100k_finetuned.py`) -- a fair, apples-to-apples before/after comparison.

| class | pretrained recall (prior benchmark) | fine-tuned recall (this sprint) |
|---|---:|---:|
| car | 0.176 | 0.189 |
| person | 0.056 | **0.000** |
| bus | 0.400 | 0.000 |
| truck | 0.500 | 0.000 |
| traffic light | 0.000 | 0.000 |

**The fine-tuned detector is WORSE than the pretrained baseline on every class with any signal**, catastrophically so on `person`. This is a real, measured negative result: 600 images and 4 epochs is not enough for a freshly-replaced detection head (new class vocabulary) to recover competent behavior, especially on classes that are rare or need many more epochs to learn from a cold head. **Per the project's own "retain a working fallback" principle, the fine-tuned checkpoint was NOT deployed.**

### 3.2 BDD100K Core ML export -- confirmed infeasible this session

Separately, exporting the two-stage Faster R-CNN to Core ML was attempted and failed immediately and reproducibly: `torch.jit.trace` on the full detection model raises `RuntimeError: Only tensors, lists, tuples of tensors, or dictionary of tensors can be output from traced functions` -- detection models return a list of per-image result dicts, which `torch.jit.trace` fundamentally cannot capture. This is a well-documented limitation of multi-stage detectors with PyTorch tracing/`coremltools`, not something more time alone (within this session's remaining budget) would resolve without substantial custom export engineering (splitting backbone/RPN/ROI-head export, a hand-written NMS layer, etc.).

**Conclusion at the time this was written: the rear-camera pipeline is UNCHANGED this sprint.** It still uses the pre-existing crude rectangle-aspect-ratio heuristic for vehicle density, and Apple's own `VNDetectHumanRectanglesRequest` for pedestrians (both pre-existing, documented in `CAPSTONE_DELIVERY.md` section 6). No BDD100K-trained or BDD100K-benchmarked model is deployed to the live rear-camera pipeline. `TemporalTracker`/`ForwardCollisionHazard`/`PedestrianHazard` still run on top of that existing heuristic's output, unchanged.

### 3.3 Post-sprint: the fake part removed, the real part kept, dataset deleted

After reviewing 3.1/3.2, decided the "is this rectangle specifically a car" classification and the traffic-density bucket it fed (empty/light/moderate/crowded) were fabricating a signal -- a generic rectangle finder has no actual evidence a box is a vehicle rather than a sign, a shadow, or a building. That classification and the traffic-density counter were **removed from `BackgroundCaptureService.processRoadTraffic`** (the `switch carRects.count` bucketing block, and the `classLabel: "car"` label on tracked boxes, now `"object"`).

What was KEPT, because it is a genuinely different and more defensible claim: tracking a detected rectangle over time and checking whether it is growing in the frame (`ForwardCollisionHazard.swift`'s optical-expansion/time-to-contact math) does not need to know the object's class -- only that something ahead is getting bigger, i.e. closing in. That logic, and the `TemporalTracker`-based lead-object tracking feeding it, is unchanged and still live. `trafficLevel`/`traffic_level` is left in the `SensorFrame` schema (for backward wire-protocol compatibility) but is no longer computed from anything -- it simply stays at its default (`"empty"`), which is now an honest "no signal," not a fabricated one.

Also deleted: the downloaded BDD100K dataset (`data/raw/bdd100k/`, ~5.7GB of zips), its extracted frame samples (`data/interim/bdd100k_frames/`, `data/interim/bdd100k_frames_train_sample/`), the failed fine-tune checkpoint (`artifacts/bdd100k_detector/`), and every BDD100K-only script (`download_bdd100k.sh`, `resume_bdd_images.sh`, `benchmark_detector_bdd100k.py`, `benchmark_detector_bdd100k_ssdlite.py`, `extract_bdd100k_train_sample.py`, `finetune_bdd100k_detector.py`, `eval_bdd100k_finetuned.py`). The two benchmark docs (`docs/benchmarks/bdd100k_*.md`) are kept as the historical record of what was measured and why it wasn't deployed; they are no longer reproducible since the scripts and dataset are gone.

### 3.4 Post-sprint: a real detector, genuinely deployed (DONE, once approved)

With approval to install `ultralytics`, a pretrained YOLOv8n (COCO) was exported to Core ML (`scripts/export_yolo_road_detector.py`) and bundled as `ios/RoadSense/Models/RoadObjectDetector.mlpackage`. (yolo11n, the newer architecture, was tried first and hit a real coremltools bug in its attention blocks -- `TypeError: only 0-dimensional arrays can be converted to Python scalars` -- and was abandoned in favor of yolov8n, which converts cleanly.)

**Verified before integration**, not just assumed to work: ran the exported `.mlpackage` directly against Ultralytics' own test photo (a bus with several people) -- it returned class 5/"bus" at 0.92 confidence and four class-0/"person" detections at 0.37-0.85 confidence, correctly matching what's actually in that image.

**Integrated into the live pipeline**, replacing the old generic-rectangle "car" guess entirely: `BackgroundCaptureService.processRoadTraffic` now runs this model via `VNCoreMLModel`/`VNCoreMLRequest` (Vision handles the image resize and NMS thresholds automatically -- the model's own `confidence`/`coordinates` output format is Vision's standard object-detector convention). Only vehicle-class detections (car/truck/bus/motorcycle/bicycle/train) feed `TemporalTracker`'s lead-object tracking; pedestrians remain on the separate, already-validated `VNDetectHumanRectanglesRequest` path. Detections now carry their REAL class label (not a fabricated "car"), so `TemporalTracker`'s same-class matching is now meaningful across frames in a way it genuinely wasn't with the old generic "object" label.

This is a pretrained-COCO model, not fine-tuned on driving footage -- `ForwardCollisionHazard`'s classification still stays conservative for exactly that reason, unchanged from before. It is a real improvement in WHAT is detected (actual trained vehicle recognition vs. a shape guess), not a claim that distance/closing-speed estimation is now precise.

Verification: `xcodebuild` (iphonesimulator) -> BUILD SUCCEEDED. `swift test` -> 118 passed (unaffected; this is CoreML/Vision code, outside the SPM-testable scope, same limitation as the rest of `BackgroundCaptureService.swift`). No physical-device test -- the CoreML inference itself was verified in Python before bundling, not on-device.

## 4. Genuinely unfinished items (no fabricated completion claims)

- **BDD100K road-object detector**: not deployed (section 3.1/3.2). The rear-camera "road-hazard" contribution to `MultimodalRiskFusion` is therefore still driven by the pre-existing heuristic detector, not a BDD100K-trained model.
- **Scenario-level integration testing** (section 10's 15 listed scenarios -- harsh braking, approaching a slowing vehicle, combined distraction+hazard, etc.): NOT run as live/replayed integration tests this sprint. What WAS verified is unit-level: the fusion math itself (13 tests), the DMD/Vision combiner (5 tests), and the existing synthetic-fixture collision/offline-sync tests (unchanged, still passing). Running the actual synthetic driving scenarios through the new fusion code end-to-end, and a quantitative baseline comparison (POLIDriving-only vs. attention-only vs. road-only vs. combined) against labeled scenarios, was not completed -- doing this honestly, rather than fabricating comparative numbers under time pressure, was judged the correct tradeoff.
- **Physical-device testing**: none performed (no iPhone available). The Xcode build above is simulator-only.
- **DMD multi-driver scale**: still only 5 of DMD's ~96 driver/session archives are used (4 train, 1 held-out) -- recall 0.419 is a real improvement, not a claim of population-level reliability.
- **Family notification pathway**: untouched this sprint, per the standing instruction to defer it ("the family notification thingy can be added later").
- **UAE SUMO simulation**: not attempted this sprint (pre-existing, documented limitation; unrelated to this sprint's scope).

## 5. Dataset-to-feature mapping (section 11 of the directive)

| Dataset | Trained artifact | Feeds into |
|---|---|---|
| POLIDriving (14 features: speed, rpm, acceleration, throttle, engine temp/load, heart rate, weather, visibility, precipitation, accident history, design speed, hour) | `FusionModel.mlpackage` (current risk, pre-existing) + `PolidrivingFiveSecondModel.mlpackage` (5s-ahead, NEW this sprint) | `RoadSenseOnDeviceRiskEngine` -> `MultimodalRiskFusion`'s vehicle dimension + separate 5s-ahead output |
| DMD GAZE category (dedicated in-cabin `rgb_face` camera frames, `gaze_zone/*` OpenLABEL annotations: road/front_right/mirrors/left/right/infotainment/steering_wheel) | `DMDAttentionModel.mlpackage` (REPLACED post-sprint -- see section 7; was previously trained on the DISTRACTION category's coarser binary `gaze_on_road` label) | `VisionManager.processDriverFace` -> `combineDMDWithVision` (off_road probability only) -> `AttentionHysteresis` -> `MultimodalRiskFusion`'s attention dimension |
| BDD100K (10,000 val / 70,000 train dashcam images, box2d annotations) | benchmarked (pretrained COCO detector, n=40) but NOT exported/deployed this sprint | not currently in the live pipeline -- the pre-existing crude heuristic still feeds `TemporalTracker`/road-hazard evidence |
| VZCrash (g-sensor/gyro event clips, crash/near_miss/normal labels) | `VZCrashModel.mlpackage` (pre-existing, reused) | `classifyCollisionEvidence` -> `CollisionIncidentStore` (collision-support evidence only) + a supporting gate inside `RiskFusionHeuristics`' false-positive suppression (never the predictive score's magnitude) |

## 7. Post-sprint: real eye-gaze estimation (not just head pose)

The directive's own worked example: a driver can hold their head level
and only move their eyes -- e.g. a phone held below the steering wheel
-- which pure head-pose classification (pitch/yaw/roll) cannot see at
all. The project's own code had already documented this as a known,
unfixed limitation before this pass.

### 7.1 Pupil-based eye-gaze offset (DONE, real, unit-tested)

Added `GazeOffset`/`gazeOffset(...)`/`averageGazeOffset(...)` to
`DriverAttentionState.swift`: computes where the pupil sits inside its
own eye's bounding box, using Vision's EXISTING `leftPupil`/`rightPupil`
+ `leftEye`/`rightEye` landmarks (no new Vision request -- these were
already being requested for the eye-closure/EAR calculation). Returns
nil, never a fabricated reading, when either eye's landmarks aren't
reliably found that frame.

`classifyGaze` now checks this eye-gaze signal BEFORE head pitch: a
confident downward pupil offset overrides an otherwise-"attentive" head
pose, which is exactly the below-the-wheel-phone case. 9 new unit tests
cover this directly, including `testEyesCastDownWithLevelHeadIsCaughtByGazeAlone`
-- the literal worked example, written as a test. Wired into
`BackgroundCaptureService.processDriverFace` via a `gazeOffsetFromLandmarks`
bridge. Horizontal pupil offset is computed but not yet used to override
yaw (left/right head turns were already covered by yaw alone, and there
is no labeled dataset in this environment to validate a horizontal
override threshold the way DMD's gaze-zone data at least approximates
for vertical).

### 7.2 DMD gaze-zone model: a real failure, then a real success (DONE)

Inspected DMD's actual GAZE category (not the DISTRACTION category the
earlier model used) and found a genuinely richer 9-zone taxonomy per
OpenLABEL annotation: `gaze_zone/front`, `/front_right`, `/left_mirror`,
`/center_mirror`, `/right_mirror`, `/left`, `/right`, `/infotainment`,
`/steering_wheel`, `/not_valid` -- where `steering_wheel` is DMD's own
real ground truth for almost exactly the directive's phone-below-the-
wheel example. Built a 3-class mapping (road / brief_glance [mirrors +
front_right] / off_road [left, right, infotainment, steering_wheel]),
extracted from each driver's "s6" calibration session
(`scripts/extract_dmd_gaze_frames.py`).

**First attempt (fine-tune the last 2 backbone blocks, the same recipe
that improved the earlier binary model from 0.301->0.419 recall)
OVERFIT badly**: held-out-driver accuracy 0.345, off_road recall
**0.023** -- barely above random for 3 classes. Each driver's gaze
session is one short, scripted calibration clip with little visual
diversity, so fine-tuning real conv weights on only 4 such clips
memorized driver-specific appearance, not the general gaze concept.

**Second attempt (head-only: freeze the backbone entirely, train only
the final linear layer) generalized far better**: held-out accuracy
**0.560**, off_road recall **0.452**, off_road precision 0.824. This is
the OPPOSITE lesson from the binary-model improvement (there, deeper
fine-tuning helped) -- a real, measured, dataset-specific finding, not
a general rule. The head-only model is what's deployed
(`ios/RoadSense/Models/DMDAttentionModel.mlpackage`, replacing the
earlier binary model), feeding its `off_road` probability -- deliberately
NOT `brief_glance`, which is legitimate driving behavior, not distraction
evidence -- into the existing `combineDMDWithVision` combiner.

### 7.3 Off-road-gaze duration and severity (DONE)

Added `offRoadGazeSec` (mirrors the existing `eyesClosedSec` pattern):
counts only while the COMMITTED, post-hysteresis attention state is a
CONFIRMED off-road state, excluding `faceNotDetected`/`uncertain` (a
confidence problem, never escalated into distraction severity).
`MultimodalRiskFusion`'s attention-dimension value now scales with a
duration-based multiplier (capped at 2x over 5 seconds), so a
distraction sustained for several seconds reads as more severe than one
that just started. Added to `SensorFrame`'s wire format and the
dashboard's `risk_breakdown` payload.

### 7.4 Portrait/landscape orientation -- re-examined, not a bug

Checked whether Vision's `orientation: .up` in the face-processing
requests was wrong for landscape mounting. It is NOT: the AVCapture
connection's `videoRotationAngle` (Phase 1 work) already rotates the
delivered pixel buffer to upright BEFORE it reaches Vision, so `.up` is
the correct assumption given that pre-rotation. The genuinely unverified
part -- whether `CameraOrientation.swift`'s rotation-angle math itself
is correct for both cameras in landscape -- was already flagged as
unverified-on-hardware in `CAPSTONE_DELIVERY.md` and remains so; nothing
new was found or fixed here, and no false "fix" was made.

### 7.5 Not done this pass (no fabricated completion claims)

- Individual driver/phone-mount calibration (a dedicated calibration UI
  or flow) -- not implemented.
- Sunglasses-specific handling -- no new logic; the existing confidence
  threshold (`.uncertain` below a minimum detection confidence) is the
  only defense, not something specific to sunglasses.
- Horizontal pupil-offset override for yaw -- computed, not yet used
  (7.1).
- `brief_glance` probability from the new DMD model -- real model
  output, not yet consumed anywhere in the fusion logic.
- No physical-device verification of any of the above -- confirmed via
  `xcodebuild` (compile/link) and the Python-side held-out-driver
  metrics only.

## 8. Reproducibility commands

```sh
# POLIDriving 5-second-ahead training
.venv/bin/python scripts/train_polidriving_5s_ahead.py

# DMD gaze-zone model (current, deployed)
.venv/bin/python scripts/extract_dmd_gaze_frames.py
.venv/bin/python scripts/train_dmd_gaze_v3.py
.venv/bin/python scripts/export_dmd_gaze_coreml.py

# Road-object detector (YOLOv8n)
.venv/bin/python scripts/export_yolo_road_detector.py

# Full verification
cd ios && swift test
xcodebuild -project ios/RoadSense.xcodeproj -scheme RoadSense \
  -configuration Debug -sdk iphonesimulator \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath /private/tmp/roadsense-build CODE_SIGNING_ALLOWED=NO build
.venv/bin/python -m pytest -q
```
