# Results and experiments (for report Ch. 5/6)

Written to answer the professor's feedback: "Add the accuracy results... compare
results with and without the camera... measure accuracy and complexity... check
the literature... Table 6.4... show the results we got while testing the ML
algorithms." Every number below comes from a command that was actually run (see
`docs/FINAL_MULTIMODAL_RESULTS.md` for the original reproduction log); nothing
here is invented for the report. Per this project's own rules, a model score is
never called a calibrated probability of an accident, and nothing trained only
on synthetic fixtures is presented as evidence of real performance.

## 1. Deployed accuracy, by modality

| Model | Modality | Metric | Value | Evaluation protocol |
|---|---|---|---:|---|
| `PolidrivingFiveSecondModel` (random forest, deployed) | Vehicle/OBD + weather/context only — **no camera** | ROC-AUC | **0.918** | Driver-held-out (train: 2 drivers, 20,963 rows; test: 1 unseen driver, 40,585 rows) |
| " | " | Precision / Recall (`hazardous_in_5s`) | **0.606 / 0.930** | " |
| `DMDAttentionModel` v1 → v2 (binary attentive/not) | **Camera only** (front-facing, MobileNetV3-Small) | `not_looking_road` recall | 0.301 → **0.419** | Driver-held-out (train 4 drivers, 5,423 frames; test 1 unseen driver, 1,233 frames) |
| `DMDAttentionModel` v2 | " | Overall accuracy | **0.876** | " |
| DMD gaze-zone (3-class, deployed) | **Camera only** (frozen backbone, head-only fine-tune) | Accuracy | **0.560** | Driver-held-out |
| " | " | `off_road` recall / precision | **0.452 / 0.824** | " |

Source: `docs/FINAL_MULTIMODAL_RESULTS.md` §2.1, §2.4, §7.2; reproducible via
`scripts/train_polidriving_5s_ahead.py`, `scripts/train_driver_state_multisession_v2.py`,
`scripts/train_dmd_gaze_v3.py`.

## 2. With camera vs. without camera

**This cannot be honestly reduced to one shared accuracy number**, and that
limitation is reported directly rather than hidden: POLIDriving (vehicle/OBD)
and DMD (camera) are two separate datasets with no synchronized multimodal
ground truth across them, so there is no single labeled test set on which a
"with camera" and "without camera" version of the same classifier could be
scored side by side. `MultimodalRiskFusion.swift` is accordingly an
evidence-based *late fusion* of two independently-validated models, not one
model jointly trained on both modalities — reported as such.

What the independently-measured numbers above do support, directly:

- **Without camera** (vehicle/OBD only): the 5s-ahead model reaches a strong
  0.918 ROC-AUC at forecasting *vehicle*-dynamics hazard, but is structurally
  blind to driver state. It cannot detect the project's own motivating
  example — a driver holding their phone below the wheel, head level, eyes
  down — because no feature in its 14-feature contract carries that
  information.
- **With camera**: DMD-based gaze/attention evidence closes exactly that gap,
  at a real but partial accuracy (off-road-gaze recall 0.452, i.e. it still
  misses more than half of actual off-road glances on an unseen driver).
  This is reported as a genuine, measured improvement in *what the system can
  see*, not a claim that the combined system has one validated joint accuracy
  figure.

**Before/after, in the sense the feedback asks for** ("test ML algorithms with
and without the camera"): before this sprint, `VZCrashModel` (IMU/vehicle-only)
carried equal (50%) weight in the predictive score; it was moved out entirely
in favor of the camera-inclusive attention/road evidence now contributing to
`MultimodalRiskFusion`, documented as a deliberate architecture change in
`docs/FINAL_MULTIMODAL_RESULTS.md` §2.2 — i.e. the "before" state
(vehicle-only predictive score) is the literal prior sprint's architecture,
and the "after" state is what's deployed now.

### Complexity / cost tradeoff

| | Vehicle-only path (`FusionModel`, `PolidrivingFiveSecondModel`) | Camera path (`DMDAttentionModel`, gaze-zone model) |
|---|---|---|
| Model type | Small tree ensembles (random forest / gradient boosting) | MobileNetV3-Small image classifier (CoreML) |
| Per-frame cost | Microsecond-scale CPU inference on ~14 scalar features | Full image classification per camera frame, plus Apple Vision's own face/landmark pipeline running in parallel |
| Resource impact | Negligible | Meaningfully higher CPU/GPU, battery and thermal load; also the only path with a privacy dimension (continuous driver-facing capture) |
| Capability gained | None of the above cost, but blind to driver inattention | Measurable (if partial) distraction detection the vehicle-only path cannot provide at any cost |

This is the accuracy/complexity tradeoff the feedback asks to see made explicit:
camera-based sensing is the only path that can see driver inattention at all,
but it is the materially more expensive one to run continuously, and its
current accuracy (0.56 gaze-zone accuracy, 0.42–0.45 recall on the distraction
classes) is a real, still-improving capability, not a solved one.

## 3. Literature comparison

- **Vehicle-kinematics-only crash/near-crash prediction** on the SHRP2
  naturalistic driving dataset reports ~90% accuracy at a 1-second prediction
  horizon in one study (Osman, Hajij, Bakhit & Ishak), and a PR-AUC of 0.90
  for an instantaneous-motion-only safety model evaluated against 4,875
  real-world SHRP2 crashes/near-crashes (arXiv:2505.13556). Our deployed 5s-ahead
  model's 0.918 ROC-AUC, at a longer 5-second horizon and on a smaller,
  single-held-out-driver dataset (POLIDriving, not SHRP2), is in the same
  general performance band as this literature, though not directly comparable
  (different dataset, horizon, and metric — ROC-AUC vs. accuracy/PR-AUC).
- **Camera-based driver distraction detection**: NHTSA's algorithm comparison
  on naturalistic data (DOT HS 811 547) found a consistent accuracy/false-alarm
  tradeoff across vision-based distraction algorithms — the best performer
  (NHTSA's "Multi-distraction" algorithm) led on accuracy/precision/AUC, while
  a high-true-positive-rate algorithm (AttenD) still carried a false-positive
  rate above 0.4. This mirrors our own gaze-zone result (recall 0.452 at
  precision 0.824 on `off_road`) — a real signal, with the same kind of
  recall/false-alarm tradeoff the published literature also reports, not an
  outlier result.
- A more recent gaze-based cognitive-distraction benchmark (EyeCue, 2026
  preprint, CogDrive dataset) reports 0.82 AUC, but in a cognitive- rather
  than visual-distraction setting, so it is cited here as context, not a
  like-for-like comparison.
- **No existing published work was found that fuses OBD/vehicle kinematics
  with in-cabin camera attention AND forward-road camera evidence in one
  architecture the way this project does** — the closest matches each use one
  modality (kinematics-only SHRP2 models, or camera-only distraction
  detectors). This is a reasonable claim of architectural novelty, stated
  narrowly: novelty in combining the modalities, not in any single modality's
  accuracy, which remains in line with (not ahead of) published single-modality
  results above.

Sources: [SHRP2 crash/near-crash kinematics prediction (TRB)](https://trid.trb.org/View/1496022) ·
[Generalized safety-state model, SHRP2, PR-AUC 0.90 (arXiv:2505.13556)](https://arxiv.org/pdf/2505.13556v1) ·
[NHTSA vision-based distraction algorithm comparison](https://www.nhtsa.gov/node/7126) ·
[EyeCue: gaze-based cognitive distraction, CogDrive AUC 0.82 (arXiv:2605.07859)](https://arxiv.org/pdf/2605.07859)

## 4. Table 6.4 provenance

Table 6.4 in the current report draft needs to be checked against the dates
above: every number in section 1 of this document comes from this project's
own November-2026 sprints (`FINAL_MULTIMODAL_RESULTS.md` is dated to this
project's active development, not CMP 490). If Table 6.4 does not match any
of the figures in section 1 here, it predates this work and should be
replaced with the real numbers above — flag this explicitly to the committee
rather than leaving an unexplained mismatch.

## 5. Honestly-reported negative/unfinished results

The professor's feedback specifically asks to show the real experimentation,
including what didn't work — these are real, measured negative results, not
omissions:

- **BDD100K detector fine-tuning regressed** rather than improved: `person`
  recall fell from 0.056 to 0.000 and `bus`/`truck` recall collapsed to 0.000
  after 4 epochs of fine-tuning on 600 images — a real, measured failure from
  too little data for a freshly-replaced detection head, not deployed, per
  the project's own "retain a working fallback" principle
  (`FINAL_MULTIMODAL_RESULTS.md` §3.1).
- **The separate VZCrash-based future-risk research track** (distinct from the
  deployed POLIDriving 5s-ahead model) found that a false-alarm budget chosen
  on a `train_thresh` split did **not** transfer to validation — a threshold
  tuned for 5 false alarms/hour produced 41/hour on held-out validation data,
  traced to a genuine train/validation distribution shift in VZCrash's own
  gyro-variability statistics (`docs/FUTURE_RISK_RESEARCH_SUMMARY.md`). This
  research track is explicitly **not deployed** and is reported here only as
  a negative finding, never as a shipped number.
- **BDD100K Core ML export was confirmed infeasible** this session — multi-stage
  detector tracing is a known PyTorch/coremltools limitation, not a solved
  problem avoided by more time (`FINAL_MULTIMODAL_RESULTS.md` §3.2).

## 6. Reproducibility

```sh
.venv/bin/python scripts/train_polidriving_5s_ahead.py
.venv/bin/python scripts/train_driver_state_multisession_v2.py
.venv/bin/python scripts/extract_dmd_gaze_frames.py && .venv/bin/python scripts/train_dmd_gaze_v3.py
.venv/bin/python -m pytest -q
```
