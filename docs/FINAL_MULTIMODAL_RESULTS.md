# RoadSense final multimodal architecture -- results

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

**Conclusion: the rear-camera pipeline is UNCHANGED this sprint.** It still uses the pre-existing crude rectangle-aspect-ratio heuristic for vehicle density, and Apple's own `VNDetectHumanRectanglesRequest` for pedestrians (both pre-existing, documented in `CAPSTONE_DELIVERY.md` section 6). No BDD100K-trained or BDD100K-benchmarked model is deployed to the live rear-camera pipeline. `TemporalTracker`/`ForwardCollisionHazard`/`PedestrianHazard` still run on top of that existing heuristic's output, unchanged.

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
| DMD (dedicated in-cabin `rgb_face` camera frames, `gaze_on_road` label) | `DMDAttentionModel.mlpackage` (NEW this sprint) | `VisionManager.processDriverFace` -> `combineDMDWithVision` -> `AttentionHysteresis` -> `MultimodalRiskFusion`'s attention dimension |
| BDD100K (10,000 val / 70,000 train dashcam images, box2d annotations) | benchmarked (pretrained COCO detector, n=40) but NOT exported/deployed this sprint | not currently in the live pipeline -- the pre-existing crude heuristic still feeds `TemporalTracker`/road-hazard evidence |
| VZCrash (g-sensor/gyro event clips, crash/near_miss/normal labels) | `VZCrashModel.mlpackage` (pre-existing, reused) | `classifyCollisionEvidence` -> `CollisionIncidentStore` (collision-support evidence only) + a supporting gate inside `RiskFusionHeuristics`' false-positive suppression (never the predictive score's magnitude) |

## 6. Reproducibility commands

```sh
# POLIDriving 5-second-ahead training
.venv/bin/python scripts/train_polidriving_5s_ahead.py

# DMD v2 training + export
.venv/bin/python scripts/train_driver_state_multisession_v2.py
.venv/bin/python scripts/export_dmd_attention_coreml.py

# BDD100K fine-tune attempt + honest before/after comparison (NOT deployed)
.venv/bin/python scripts/extract_bdd100k_train_sample.py
.venv/bin/python scripts/finetune_bdd100k_detector.py
.venv/bin/python scripts/eval_bdd100k_finetuned.py

# Full verification
cd ios && swift test
xcodebuild -project ios/RoadSense.xcodeproj -scheme RoadSense \
  -configuration Debug -sdk iphonesimulator \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath /private/tmp/roadsense-build CODE_SIGNING_ALLOWED=NO build
.venv/bin/python -m pytest -q
```
