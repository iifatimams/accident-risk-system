# DMD: single-session gaze-on-road baseline -- v1

Real run against real DMD (Driver Monitoring Dataset) annotations, actually
executed in this environment. Previous mentions of this baseline in git
history/scripts were never confirmed to have been run -- this is the first
measured result.

## Dataset audit (read-only; `data/raw/dmd/` was not modified)

`data/raw/dmd/` contains 108GB of compressed per-session archives across
three annotation types, never previously extracted (`data/interim/dmd_frames/`
was empty before this session):

| split        | archives | size |
|--------------|---------:|-----:|
| drowsiness   | 13       | 13GB |
| gaze         | 14       | 19GB |
| distraction  | 14       | 77GB |

Each archive is one participant/session (e.g. `gA-1`, `gC-14`) containing
one or more sub-sessions (`s1`, `s2`, `s3`...), each with `rgb_mosaic.avi`
(combined view), `rgb_face.mp4`, `rgb_body.mp4`, optionally `rgb_hands.mp4`,
and an ASAM OpenLABEL-format JSON annotation (`rgb_ann_<type>.json`) with
real per-frame driver labels -- not placeholder or inferred labels.

**Real label taxonomies found** (confirmed by reading actual annotation
JSONs, not assumed from documentation):

- **drowsiness**: `eyes_state/{open,close,opening,closing}`,
  `blinks/blinking`, `yawning/{with_hand,without_hand}`.
- **distraction**: `gaze_on_road/{looking_road,not_looking_road}`,
  `talking/talking`, `hands_using_wheel/{both,only_left,only_right}`,
  `driver_actions/{safe_drive,radio,drinking,reach_side,talking_to_passenger,unclassified}`.
- **gaze**: `gaze_zone/{left_mirror,left,front,center_mirror,front_right,
  right_mirror,right,infotainment,steering_wheel,not_valid}`,
  `blinks/blinking`.

**No `phone_use` or `looking_up`/`looking_down` label exists anywhere in
this data.** Per AGENTS.md ("never fabricate dataset columns") and the
project directive ("do not fabricate labels... if the dataset does not
support them"), no RoadSense feature should claim to be trained or
validated against DMD-labeled phone use or up/down head pose -- only
`gaze_on_road` (binary) and `gaze_zone` (9-way) are real, available labels.

## What was run

Exactly the plan already described in `scripts/extract_dmd_frames.py` and
`scripts/train_driver_state_baseline.py`'s own docstrings (both scripts
existed before this session but had hardcoded `/tmp/dmd_audit/...` paths
pointing at no longer-present local state -- fixed to the repo's own
`data/interim/` and `data/raw/`, then actually run):

```bash
mkdir -p data/interim/dmd_frames/gC_14_s1
tar -xzf data/raw/dmd/distraction/dmd-dataset-distraction-gC-14.tar.gz \
  -C data/interim/dmd_frames/gC_14_s1 --strip-components=4 \
  "dmd/gC/14/s1/gC_14_s1_2019-03-04T11;56;20+01;00_rgb_face.mp4" \
  "dmd/gC/14/s1/gC_14_s1_2019-03-04T11;56;20+01;00_rgb_ann_distraction.json"
.venv/bin/python scripts/extract_dmd_frames.py     # -> 1337 labeled frames
.venv/bin/python scripts/train_driver_state_baseline.py
```

Only the one needed video+annotation pair was extracted from the 4.3GB
`gC-14` archive (416MB), not the full 108GB DMD corpus. This is one
driver, one session, one sub-session (`gC/14/s1`) -- see limitations.

## Results

**Frozen-backbone MobileNetV3-small + linear head, 8 epochs, held-out val
split from the same session's later time window (causal, not shuffled):**

| class            | precision | recall | f1    | support |
|------------------|----------:|-------:|------:|--------:|
| looking_road     |     0.974 |  0.841 | 0.902 |     220 |
| not_looking_road |     0.551 |  0.896 | 0.683 |      48 |
| accuracy         |           |        | 0.851 |     268 |

Confusion matrix (rows=true, cols=pred): `[[185, 35], [5, 43]]`.

**Causal rolling majority-vote smoothing (window=5 sampled frames, ~0.84s
historical lookback) -- the evidence-gate test for whether temporal
smoothing helps before justifying a GRU:**

| class            | precision | recall | f1    | support |
|------------------|----------:|-------:|------:|--------:|
| looking_road     |     0.896 |  0.786 | 0.838 |     220 |
| not_looking_road |     0.373 |  0.583 | 0.455 |      48 |
| accuracy         |           |        | 0.750 |     268 |

## Reading this honestly

- **The single-frame classifier (85.1% accuracy, 0.896 recall on
  not_looking_road) clearly fits this one session.** `not_looking_road`
  precision is weak (0.551) -- the model over-predicts distraction (35
  false positives vs 5 false negatives), which for a safety feature is the
  less dangerous failure direction, but it is a real measured weakness,
  not a strength.
- **Majority-vote smoothing made results WORSE here (85.1% -> 75.0%
  accuracy), not better.** This is itself a real, useful negative result
  per the evidence-gating purpose of this script: on this single short
  session, naive smoothing does not help and should not be assumed to
  help elsewhere without separately testing it. A plausible reason is
  that `not_looking_road` intervals in this session are sometimes shorter
  than the 5-sample (~0.84s) window, so smoothing drags short real events
  toward the majority class -- not confirmed, just the likely mechanism.
- **This is single-driver, single-session.** Train and val frames come
  from the same person, camera, lighting, and seat position -- different
  time windows of the SAME ~4-minute recording. This measures "can a
  linear head fit this one session," not generalization across drivers,
  lighting, or DMD's other 13+ participants. A real baseline needs frames
  from multiple DMD subjects, which was not attempted this session (87
  more archives, ~104GB more, remain unextracted).
- n=1337 total frames (1069 train / 268 val), ~85%/15% class imbalance,
  no hyperparameter search. A first feasibility pass, not a tuned model.

## What this does NOT establish

- Not evidence this would work on a different driver, camera angle, or
  lighting condition.
- Not a comparison against the existing Apple Vision rule-based system
  running in `BackgroundCaptureService.swift` (Phase 2) -- that
  comparison (Phase 3.3's actual ask) requires DMD video run through an
  iOS-equivalent Vision pipeline, not attempted here.
- Not a Core ML export or on-device benchmark (Phase 3.4).
- Not evidence for `phone_use`, `looking_up`, or any label this dataset
  does not actually contain (see taxonomy audit above).
- `data/raw/dmd/*.tar.gz` was only read from, never modified.
