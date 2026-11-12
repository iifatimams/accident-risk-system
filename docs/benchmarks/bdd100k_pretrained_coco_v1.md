# BDD100K: off-the-shelf COCO detector, real labels -- v1

Real run, real BDD100K ground truth, actually executed in this environment
(previous mentions of this benchmark in git history/scripts were never
confirmed to have been run -- see `docs/DEVELOPMENT_PROGRESS.md` Phase 4).
This is the first measured result. It is a feasibility smoke test, not a
statistically powered benchmark, and not evidence for or against fine-tuning
without a larger sample.

## What was run

`scripts/benchmark_detector_bdd100k.py`: a COCO-pretrained
`fasterrcnn_mobilenet_v3_large_320_fpn` (torchvision default weights),
score threshold 0.5, IoU threshold 0.5, greedy per-class matching, against
**40 real BDD100K val images and their real ground-truth boxes** (never
synthetic, never fabricated).

### How the 40 images were selected (exactly reproducible)

`data/raw/bdd100k/bdd100k_images_100k.zip` contains all 10,000 val images;
none were extracted in bulk. Only this sample's specific files were pulled
out of the zip, into `data/interim/bdd100k_frames/` (gitignored, like the
rest of `data/interim/`, so it is not committed -- regenerate with the
commands below):

```bash
unzip -l data/raw/bdd100k/bdd100k_images_100k.zip \
  | grep -oE "100k/val/[a-z0-9-]+\.jpg" | sort > /tmp/bdd_val_images.txt
awk 'NR % 250 == 1' /tmp/bdd_val_images.txt > /tmp/bdd_val_sample.txt   # 40 of 10000
mkdir -p data/interim/bdd100k_frames/images data/interim/bdd100k_frames/labels
unzip -o -j data/raw/bdd100k/bdd100k_images_100k.zip $(cat /tmp/bdd_val_sample.txt) \
  -d data/interim/bdd100k_frames/images
sed 's#100k/val/\(.*\)\.jpg#100k/val/\1.json#' /tmp/bdd_val_sample.txt > /tmp/bdd_val_sample_labels.txt
unzip -o -j data/raw/bdd100k/bdd100k_labels.zip $(cat /tmp/bdd_val_sample_labels.txt) \
  -d data/interim/bdd100k_frames/labels
.venv/bin/python scripts/benchmark_detector_bdd100k.py
```

This selects every 250th filename from the alphabetically sorted val list
(deterministic, not random, and not cherry-picked for this result).

## Results (n=40 images, IoU>=0.5, score>=0.5)

| class          |  TP |  FP |  FN | precision | recall |
|----------------|----:|----:|----:|----------:|-------:|
| bike           |   0 |   0 |   0 |       n/a |    n/a |
| bus            |   2 |   1 |   3 |     0.667 |  0.400 |
| car            |  78 |  28 | 366 |     0.736 |  0.176 |
| motor          |   0 |   0 |   2 |       n/a |  0.000 |
| person         |   2 |   4 |  34 |     0.333 |  0.056 |
| traffic light  |   0 |   1 | 116 |     0.000 |  0.000 |
| traffic sign   |   0 |   0 | 150 |       n/a |  0.000 |
| train          |   0 |   3 |   0 |     0.000 |    n/a |
| truck          |   6 |   9 |   6 |     0.400 |  0.500 |

Taxonomy gap (BDD100K ground-truth class with no COCO equivalent at all,
isolated out rather than silently scored as a miss): **`rider`**, 1
ground-truth box in this sample.

## Reading this honestly

- **Recall is low across the board**, most strikingly for `traffic light`
  (0/116) and `traffic sign` (0/150, though `traffic sign` only maps to
  COCO's much narrower `stop sign` -- see script docstring, this is a
  real taxonomy gap as much as a detection failure) and `person` (2/34).
  Dashcam BDD100K scenes contain many small, distant objects; a
  COCO-pretrained detector at a 0.5 score threshold is evidently missing
  most of them in this sample.
- **`car` has reasonable precision (0.736) but low recall (0.176)**: when
  it says "car" it is usually right, but it is missing roughly 5 out of 6
  ground-truth cars in this sample (366 FN vs 78 TP) -- very likely small/
  distant/partially-occluded vehicles below the 0.5 score threshold.
- **`train`: 3 false positives with 0 ground-truth trains in this sample**
  at all -- the model is hallucinating a class that never appears here.
- n=40 is small. These numbers should NOT be read as "the detector has
  17.6% car recall" in general -- only that on this specific, deterministic
  40-image sample, that is what was measured. A larger (ideally full
  10,000-image val) run is needed before drawing a fine-tuning decision,
  per Phase 4.3 ("benchmark first").

## What this does NOT establish

- Not a claim about on-device (Core ML / iPhone) performance -- this ran
  the full PyTorch model on this Mac (`mps` backend), not an exported,
  quantized, or otherwise mobile-ready version.
- Not evidence for or against fine-tuning on BDD100K -- that decision
  (Phase 4.3) needs either a larger sample of this same benchmark or a
  cost/benefit read of these numbers, not a prediction model dependency.
- Not a day/night or lighting breakdown (Phase 4.3 asks for this; the
  current sample is too small to split meaningfully and the script does
  not currently read BDD100K's `attributes.timeofday` field -- a
  reasonable next addition, not done here).
- `data/raw/bdd100k/*.zip` was never modified, only read from, consistent
  with AGENTS.md.
