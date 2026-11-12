# BDD100K: Faster R-CNN vs SSDLite -- detector comparison v1

Direct follow-up to `bdd100k_pretrained_coco_v1.md`, adding a second
COCO-pretrained detector on the exact SAME 40-image real BDD100K sample
and labels, to inform Phase 4.2/4.3's actual decision ("evaluate suitable
lightweight object detectors... consider Core ML compatibility, runtime
cost"). Same limitations apply as the first benchmark: n=40 is a
feasibility smoke test, not a statistically powered benchmark.

## What was run

```bash
.venv/bin/python scripts/benchmark_detector_bdd100k.py          # Faster R-CNN (updated: now also reports latency/params)
.venv/bin/python scripts/benchmark_detector_bdd100k_ssdlite.py  # SSDLite320-MobileNetV3-Large (new)
```

Both against the identical 40-image sample extracted for the first
benchmark (see that doc for exact extraction commands) -- the only
variable that changes between the two runs is the detector architecture.

## Results

| class          | Faster R-CNN precision/recall | SSDLite precision/recall |
|----------------|-------------------------------:|---------------------------:|
| bus            | 0.667 / 0.400                  | 1.000 / 0.200               |
| car            | 0.736 / 0.176                  | 0.842 / 0.072                |
| motor          | n/a / 0.000                     | n/a / 0.000                  |
| person         | 0.333 / 0.056                   | 1.000 / 0.028                |
| traffic light  | 0.000 / 0.000                   | n/a / 0.000                   |
| traffic sign   | n/a / 0.000                     | n/a / 0.000                   |
| train          | 0.000 / n/a (3 FP, 0 GT)        | 0.000 / n/a (1 FP, 0 GT)      |
| truck          | 0.400 / 0.500                   | 0.500 / 0.167                 |

| footprint (this Mac, MPS, batch 1) | Faster R-CNN       | SSDLite             |
|-------------------------------------|-------------------:|---------------------:|
| parameters                          | 19,386,354          | 3,440,060             |
| mean inference latency              | 30.5 ms/image       | 90.8 ms/image          |
| max inference latency               | 33.0 ms/image       | 104.6 ms/image         |

(Both figures exclude a warmup inference, which otherwise shows up as a
multi-second one-time JIT-compilation outlier on first use.)

## Reading this honestly

- **SSDLite has ~5.6x fewer parameters but was ~3x SLOWER per image on
  this Mac's MPS backend, and clearly less accurate** (lower recall on
  every class that had any true positives at all: car 0.072 vs 0.176,
  person 0.028 vs 0.056, truck 0.167 vs 0.500, bus 0.200 vs 0.400).
  SSDLite's higher precision numbers (1.000 on person/bus) come from
  firing far fewer predictions overall, not from being more discriminating.
- **This is a real, measured result, and it runs against the common
  intuition that a smaller single-stage detector is automatically
  faster.** It is NOT evidence that SSDLite would be slower on an actual
  iPhone. PyTorch's MPS backend on a Mac does not execute the same code
  path as a Core ML model running on the iPhone's Neural Engine -- this
  latency comparison says something about these two models under PyTorch/
  MPS on this machine, and nothing directly about on-device Core ML
  performance. Treating this Mac-based number as a stand-in for iPhone
  latency would be a mistake Phase 4.5 explicitly warns against ("do not
  claim on-device validation from a simulator-only build" extends
  naturally to "a training-machine benchmark is not an on-device one").
- **Architecturally, SSDLite remains the more appropriate Core ML export
  candidate** regardless of this result: it has no RPN/ROI-align stage
  (Faster R-CNN's two-stage design is a documented poor fit for mobile
  export), and `coremltools` has a well-supported conversion path for it.
  The accuracy gap above is the actual concern for choosing it as-is --
  not speed, which this benchmark cannot honestly speak to for iPhone.
- **Neither detector is currently good enough on this sample to deploy
  as-is** for `traffic light`/`traffic sign` (both near 0 recall on both
  models) or for `person` (0.056/0.028) -- the classes that matter most
  for pedestrian-hazard work (Phase 7).

## Recommendation (evidence-gated, not a final decision)

Given the data available in this environment: **evaluate SSDLite on a
much larger BDD100K sample (ideally the full 10,000-image val set) before
deciding between the two architectures on accuracy grounds**, and treat
on-device Core ML latency as a SEPARATE, not-yet-measured question (Phase
4.4/4.5) -- this Mac-based latency number must not be used to decide
"SSDLite is slower" for the iPhone. Given current evidence, if forced to
choose today: SSDLite's export-friendliness makes it the better long-term
target, but its accuracy gap on this sample means either a larger
evaluation or BDD100K fine-tuning (Phase 4.3) is needed before shipping it
over the status quo (the existing generic-rectangle heuristic in
`BackgroundCaptureService.swift`, which has no real accuracy numbers at
all and is known to be cruder than either model here).

## What this does NOT establish

- Not on-device (Core ML/iPhone) latency for either model -- see above.
- Not a statistically powered accuracy comparison (n=40).
- Not a decision to fine-tune, export, or deploy either model -- Phase
  4.3/4.4 remain open.
- `data/raw/bdd100k/*.zip` was only read from, never modified.
