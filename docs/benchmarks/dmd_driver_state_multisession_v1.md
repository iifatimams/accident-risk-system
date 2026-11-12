# DMD: cross-driver (subject-separated) gaze-on-road baseline -- v1

Direct follow-up to `dmd_driver_state_baseline_v1.md`'s single-session
result, addressing that doc's own stated limitation: "this measures 'can
it fit this one session,' not generalization across drivers." This run
trains on 4 drivers and evaluates on a 5th, held out entirely.

## What was run

```bash
# Extracted the same way as the single-session baseline (one face.mp4 +
# one rgb_ann_distraction.json per driver, from data/raw/dmd/distraction/):
#   gC-14/s1, gA-1/s1, gB-6/s1, gE-28/s1  -> data/interim/dmd_frames/<session>/
#   gF-23/s1                              -> data/interim/dmd_frames/gF_23_s1/
.venv/bin/python scripts/extract_dmd_frames_multisession.py
.venv/bin/python scripts/train_driver_state_multisession.py
```

Train: 5,423 frames across 4 drivers (gC-14, gA-1, gB-6, gE-28). Val:
1,233 frames from driver gF-23 -- **zero frames from this driver appear in
training.** Same model (frozen-backbone MobileNetV3-small + linear head,
8 epochs, inverse-frequency class weighting) and same binary task
(`gaze_on_road`: looking_road / not_looking_road) as the single-session
run, so this is a controlled comparison of ONE variable: subject overlap
between train and val.

## Results

| class            | precision | recall | f1    | support |
|------------------|----------:|-------:|------:|--------:|
| looking_road     |     0.887 |  0.971 | 0.927 |    1047 |
| not_looking_road |     0.651 |  0.301 | 0.412 |     186 |
| accuracy         |           |        | 0.870 |    1233 |

Confusion matrix (rows=true, cols=pred): `[[1017, 30], [130, 56]]`.

## Reading this honestly, against the single-session result

| metric                          | single-session (same driver) | cross-driver (held-out driver) |
|----------------------------------|------------------------------:|--------------------------------:|
| overall accuracy                 |                         0.851 |                           0.870 |
| not_looking_road recall          |                         0.896 |                           0.301 |
| not_looking_road precision       |                         0.551 |                           0.651 |

**Overall accuracy looks similar or even slightly better, but that number
is misleading here** -- it's dominated by the majority `looking_road`
class. The metric that actually mattered for a safety feature
(`not_looking_road` recall: "when the driver really isn't looking at the
road, does the model notice?") **collapsed from 0.896 to 0.301** on a
driver the model never trained on. In plain terms: on this held-out
driver, the model misses roughly 7 out of 10 real not-looking-road
moments (130 false negatives vs 56 true positives).

**This is real evidence that the single-session result was substantially
optimistic and should not be trusted as a generalization claim** -- which
is exactly what that doc's limitations section already said, now with a
number attached instead of just a caveat. A plausible (not confirmed)
cause: the frozen ImageNet backbone + a single linear head may be
learning something closer to "this specific person's face/lighting/seat
position correlates with not_looking_road" than a genuinely pose-general
signal, which a single held-out driver can't distinguish from real
generalization -- and apparently doesn't, here.

## What this does NOT establish

- **n=1 held-out driver.** This is one generalization data point, not a
  statistically powered cross-validation. A real evaluation needs
  leave-one-driver-out across most/all of DMD's 13-14 distraction
  participants, not one held-out split.
- Not a claim that MobileNetV3 + a linear head is the wrong architecture
  -- a frozen backbone is a deliberately cheap feasibility probe (per the
  original script's own framing). Fine-tuning more of the backbone,
  more training data per driver, or a different architecture could change
  this; none of that was tried here.
- Not a comparison against the Apple Vision rule-based system in
  `BackgroundCaptureService.swift` (Phase 2) -- still not attempted.
- Not evidence for or against deploying ANY driver-state model to iOS --
  if anything, this result argues for NOT deploying a single-driver-fit
  model and instead continuing to rely on the rule-based Vision pipeline
  (Phase 2) until a properly cross-validated DMD model clears a
  meaningfully higher bar than 0.301 recall on unseen drivers.
- `data/raw/dmd/*.tar.gz` was only read from, never modified.
