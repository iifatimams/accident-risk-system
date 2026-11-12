"""Benchmark an off-the-shelf, COCO-pretrained detector on a small BDD100K
val sample, with REAL ground-truth boxes, before committing to any BDD100K
YOLO fine-tune.

Evidence-gating step (spec's reconciled ML plan): fine-tuning a detector on
BDD100K is expensive (labeling format work, training time, iOS export). The
cheap first question is "how far off is an already-trained general detector
on this specific domain (dashcam, US roads)?" If it is already close on the
classes that matter (car/person/bus/truck), fine-tuning may only be needed
for BDD-specific classes a COCO detector cannot express at all (rider,
generic traffic sign, lane markings) rather than from scratch.

Known, stated limitations:
  - n=40 images (sampled every 250th of the 10,000 val images, not random)
    -- a feasibility smoke test, not a statistically powered benchmark.
  - COCO and BDD100K category taxonomies are NOT 1:1. BDD's "rider" (person
    actively riding a bike/motorcycle) has no COCO equivalent -- we report
    it as a structural gap, not a missed detection. BDD's "traffic sign" is
    broader than COCO's "stop sign" -- same caveat, reported, not silently
    dropped or faked as a match.
  - IoU-based matching at a single threshold (0.5), greedy per-class
    matching. This is standard but simplified vs. full COCO mAP (no
    multi-threshold averaging, no area-size breakdown).
"""

import json
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.transforms.functional import to_tensor

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = REPO_ROOT / "data/interim/bdd100k_frames/images"
LABELS_DIR = REPO_ROOT / "data/interim/bdd100k_frames/labels"
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# COCO category name -> BDD100K category name, for classes that have a
# genuine semantic equivalent. Classes left out (rider, lane, drivable
# area, generic traffic sign) are reported separately as taxonomy gaps.
COCO_TO_BDD = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "person": "person",
    "bicycle": "bike",
    "motorcycle": "motor",
    "traffic light": "traffic light",
    "stop sign": "traffic sign",  # partial overlap only, see module docstring
}


def iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(label_path: Path) -> list[tuple[str, list[float]]]:
    data = json.loads(label_path.read_text())
    boxes = []
    for frame in data["frames"]:
        for obj in frame["objects"]:
            box = obj.get("box2d")
            if box is None:
                continue
            boxes.append((obj["category"], [box["x1"], box["y1"], box["x2"], box["y2"]]))
    return boxes


def main() -> None:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).to(device).eval()
    categories = weights.meta["categories"]

    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"Evaluating on {len(image_paths)} BDD100K val images (device={device})")

    # Per-BDD-class running totals for precision/recall at IoU>=0.5.
    stats = {name: {"tp": 0, "fp": 0, "fn": 0} for name in set(COCO_TO_BDD.values())}
    taxonomy_gap_gt_counts: dict[str, int] = {}
    n_images = 0
    latencies_ms: list[float] = []

    with torch.no_grad():
        # Untimed warmup: MPS/CUDA JIT-compiles kernels on first use, which
        # would otherwise show up as a multi-second outlier in the per-image
        # latency numbers below.
        model([to_tensor(Image.open(image_paths[0]).convert("RGB")).to(device)])
        for img_path in image_paths:
            label_path = LABELS_DIR / (img_path.stem + ".json")
            if not label_path.exists():
                continue
            n_images += 1
            gt_boxes = load_ground_truth(label_path)

            image = Image.open(img_path).convert("RGB")
            tensor = to_tensor(image).to(device)
            start = perf_counter()
            output = model([tensor])[0]
            latencies_ms.append((perf_counter() - start) * 1000)

            preds = []
            for score, label_idx, box in zip(
                output["scores"].cpu().tolist(),
                output["labels"].cpu().tolist(),
                output["boxes"].cpu().tolist(),
            ):
                if score < SCORE_THRESHOLD:
                    continue
                coco_name = categories[label_idx]
                bdd_name = COCO_TO_BDD.get(coco_name)
                if bdd_name is not None:
                    preds.append((bdd_name, box))

            # Greedy per-class IoU matching.
            for bdd_class in stats:
                gt_this_class = [b for cat, b in gt_boxes if cat == bdd_class]
                pred_this_class = [b for cls, b in preds if cls == bdd_class]
                matched_gt = set()
                for pbox in pred_this_class:
                    best_iou, best_idx = 0.0, -1
                    for i, gbox in enumerate(gt_this_class):
                        if i in matched_gt:
                            continue
                        cur = iou(pbox, gbox)
                        if cur > best_iou:
                            best_iou, best_idx = cur, i
                    if best_iou >= IOU_THRESHOLD:
                        matched_gt.add(best_idx)
                        stats[bdd_class]["tp"] += 1
                    else:
                        stats[bdd_class]["fp"] += 1
                stats[bdd_class]["fn"] += len(gt_this_class) - len(matched_gt)

            for cat, _ in gt_boxes:
                if cat not in COCO_TO_BDD.values() and cat not in ("lane", "drivable area"):
                    taxonomy_gap_gt_counts[cat] = taxonomy_gap_gt_counts.get(cat, 0) + 1

    print(f"\n=== Pretrained COCO detector vs real BDD100K val labels (n={n_images} images) ===")
    print(f"{'class':<16}{'TP':>6}{'FP':>6}{'FN':>6}{'precision':>12}{'recall':>10}")
    for name, s in sorted(stats.items()):
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        print(f"{name:<16}{tp:>6}{fp:>6}{fn:>6}{precision:>12.3f}{recall:>10.3f}")

    print("\n=== BDD100K ground-truth categories this COCO detector CANNOT express at all ===")
    print("(these count toward FN implicitly if ever matched against stats above incorrectly;")
    print(" here they are isolated out as a genuine taxonomy gap, not a detector failure)")
    for cat, count in sorted(taxonomy_gap_gt_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<20} {count} ground-truth boxes in this sample")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== Model footprint (this Mac, device={device}, batch size 1) ===")
    print(f"parameters: {n_params:,}")
    print(f"mean inference latency: {sum(latencies_ms) / len(latencies_ms):.1f} ms/image")
    print(f"max inference latency: {max(latencies_ms):.1f} ms/image")


if __name__ == "__main__":
    main()
