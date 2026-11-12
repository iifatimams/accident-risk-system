# Future-collision-risk prediction: research summary (rounds 1-3)

Status: **research milestone, not a validated or deployed predictor.** Nothing here
has been exported to Core ML or wired into the iOS app. This document is the
reproducible record of what was tried, what the evidence actually shows, and what
is still unresolved -- read it before extending or citing any number from this work.

## What problem this addresses

RoadSense's on-device risk engine (`RoadSenseOnDeviceRiskEngine.swift`) combines two
bundled Core ML models, but neither was trained or validated as "predict whether an
accident happens in the next 5 seconds." This research line asks: using VZCrash (the
dataset one of those models, `VZCrashModel.mlpackage`, was trained on), can a
genuinely causal, future-looking predictor be built and honestly evaluated?

## Reproducing each round

```bash
# Round 1/2 dataset + baseline (2-4 train shards, original/expanded features)
.venv/bin/python scripts/build_future_risk_dataset.py --out-dir data/processed/future_risk_vzcrash        # round 1 (2 shards)
.venv/bin/python scripts/build_future_risk_dataset.py --out-dir data/processed/future_risk_vzcrash_v2      # round 2 (4 shards, expanded features)
.venv/bin/python scripts/train_future_risk_baseline.py --dataset-dir data/processed/future_risk_vzcrash_v2 --protocol hazardous_broad --feature-set expanded

# Round 3 dataset + evaluation (adds speed-relative features, proper train/thresh/val split, cooldown FA/hr, speed-band analysis)
.venv/bin/python scripts/build_future_risk_dataset.py --out-dir data/processed/future_risk_vzcrash_v3
.venv/bin/python scripts/train_future_risk_baseline.py --dataset-dir data/processed/future_risk_vzcrash_v3 --protocol collision_only --feature-set expanded
.venv/bin/python scripts/train_future_risk_baseline.py --dataset-dir data/processed/future_risk_vzcrash_v3 --protocol collision_only --feature-set speed_only
.venv/bin/python scripts/train_future_risk_baseline.py --dataset-dir data/processed/future_risk_vzcrash_v3 --protocol collision_only --feature-set no_speed
.venv/bin/python scripts/train_future_risk_baseline.py --dataset-dir data/processed/future_risk_vzcrash_v3 --protocol collision_only --feature-set speed_relative
```

Requires a Hugging Face account that has accepted VZCrash's gated terms
(`huggingface-cli login` once; cached thereafter). All outputs are git-ignored
(`data/processed/*`, `artifacts/*`) -- re-run the commands above to regenerate them;
nothing here needs to be committed.

## Round-by-round

**Round 1** (2 train shards, 9 original features, single sparse hard-negative sample
per clip): first feasibility pass. Random forest PR-AUC 0.562, but n=126 hard
negatives -- too small to trust the ~71% hard-negative false-positive rate it
reported.

**Round 2** (4 train shards, dense causal negative sampling, +11 temporal/escalation
features, collision_only vs hazardous_broad protocols): found that VZCrash's
"near_miss" label has no kinematic signature distinguishable from normal driving
(peak accel/gyro statistics nearly identical) -- near_miss onset estimates are
unreliable and near_miss should not be treated as a confirmed hazardous event.
Diagnosed that crash clips run ~68 km/h on average even in their "safe" early
seconds vs ~37 km/h for normal clips -- a likely confound (the model may partly key
off "fast context" rather than genuine escalation).

**Round 3** (+3 causal speed-relative features, proper train_fit/train_thresh/
validation separation, cooldown-based false-warning counting, fixed false-alarm-
budget operating points, speed-band breakdown): the speed-confound story turned out
more nuanced than round 2 suggested --

- Removing all speed features barely changes overall PR-AUC (no_speed 0.542 vs
  full 0.548, gradient_boosting) -- absolute speed is *not* the dominant driver of
  overall accuracy.
- For logistic regression specifically, removing speed did cut the hard-negative
  false-positive rate meaningfully (0.306 -> 0.118). For the better-performing
  random forest and gradient boosting, it did not (0.579->0.541, 0.764->0.739) --
  something else, not yet isolated, still drives a high rate there.
- A genuine speed-matched evaluation (same model, broken out by speed band) shows
  PR-AUC well above each band's own chance level in every band (e.g. 40-60 km/h:
  0.594 vs 0.152 base rate) -- real discriminative signal survives even when speed
  context is held roughly constant.

### Post-round-3 audit (this document records it; do not re-derive without re-reading)

Two claims from the round-3 verbal report were **wrong as originally stated** and
are corrected here:

1. **False-alarm budgets chosen on `train_thresh` do not transfer to `validation`.**
   A gradient_boosting threshold selected for "5 false alarms/hour" actually
   produces **41/hour** on validation (8x over). Measured cause: validation's
   normal-driving clips have ~3x higher average gyroscope variability than train's
   (std 26.2 vs 4.7) -- a real distribution shift between the dataset's own
   train/validation partition, not a code bug. Any operating-point claim from this
   line of work must report the number **actually achieved on validation**, never
   the target the threshold was selected for.
2. **Reported "lead time" (~4s mean) is a sampling artifact, not evidence of 4
   seconds of genuine advance warning.** Positive examples are only ever generated
   at 5 fixed lead times (1, 2, 3, 4, 4.5s before the estimated onset) -- 4.5s is a
   hard ceiling imposed by the 5s horizon definition, not a measured limit. Hit
   rate is nearly flat across those 5 points (67-74%), and 54% of detected events
   fire at *all five* simultaneously, which mechanically pushes the reported
   "earliest lead" toward the 4.5s ceiling whenever a detection happens at all. This
   does not establish how far in advance a continuously-scored live system would
   actually warn -- that was never tested.

Two things were checked and found correct (no bug): event-level recall and lead-time
both filter to `target == 1` rows before aggregating, so a hit on an out-of-horizon
negative window from the same clip can never be counted as "catching" that event
(`train_future_risk_baseline.py`, `event_level_recall` / `earliest_detection_lead_time`).

## Standing limitations (true as of round 3, restated plainly)

- **Onset is an estimated proxy** (peak |accel| in the 100Hz signal), not a
  verified ground-truth collision timestamp. For crash clips the four proxy methods
  tested agree to within ~0.03-0.06s (not a first-order uncertainty); for near_miss
  they disagree by over a second median (why near_miss is excluded from the primary
  `collision_only` protocol).
- **Recording-context confounding is real but only partially explained.** Crash and
  normal clips differ in average speed and (per the round-3 audit) in validation's
  gyroscope variability; speed-band evaluation shows genuine signal survives this,
  but the gap for tree-based models' hard-negative rate is not fully closed.
- **Train/validation class imbalance**: crash rate is ~4.9% of clips in the train
  shards used vs ~28% in the validation shards used -- not the same distribution,
  so absolute validation metrics should not be read as real-world base-rate
  performance.
- **Operating-point numbers do not transfer across the dataset's own splits** (see
  audit above) -- a live system would need re-calibration against its actual
  deployment distribution, not a number taken from this research.
- Not evaluated beyond VZCrash; no real-vehicle or real-iPhone data involved.

## Decision

Per the 2026-11-12 development plan: do not indefinitely optimize against VZCrash.
No quick, scientifically justified fix for the tree-model hard-negative gap was
available within this round's scope. This baseline is preserved as-is (rounds 1-3,
all outputs kept on disk) and development moves on to the other RoadSense pillars
(research recording/export, collision detection, family monitoring, wearables).
**No model from this work is deployed, exported to Core ML, or used by the iOS app.**
