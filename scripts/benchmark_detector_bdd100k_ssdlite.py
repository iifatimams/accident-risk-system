"""Second detector benchmark on the SAME real BDD100K sample as
benchmark_detector_bdd100k.py, this time with a mobile-appropriate
single-stage detector (SSDLite320-MobileNetV3-Large) instead of the
two-stage Faster R-CNN used there.

Why this one specifically (Phase 4.2's own criteria): SSDLite is designed
for mobile/edge inference (much lower latency and parameter count than
Faster R-CNN), has a documented, well-supported Core ML export path
(unlike a two-stage R-CNN with its RPN/ROI-align, which is a poor fit for
on-device export), and ships with COCO-pretrained weights so this is a
like-for-like comparison on the exact same 40 images/labels -- the only
variable that changes is the detector architecture.

Same known limitations as the first benchmark (n=40 feasibility smoke
test, not statistically powered; COCO/BDD100K taxonomy gaps isolated out,
not silently counted as misses) -- see that script's docstring.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.transforms.functional import to_tensor

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = REPO_ROOT / "data/interim/bdd100k_frames/images"
LABELS_DIR = REPO_ROOT / "data/interim/bdd100k_frames/labels"
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

COCO_TO_BDD = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "person": "person",
    "bicycle": "bike",
    "motorcycle": "motor",
    "traffic light": "traffic light",
    "stop sign": "traffic sign",
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
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights).to(device).eval()
    categories = weights.meta["categories"]

    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"Evaluating SSDLite320-MobileNetV3-Large on {len(image_paths)} BDD100K val images (device={device})")

    stats = {name: {"tp": 0, "fp": 0, "fn": 0} for name in set(COCO_TO_BDD.values())}
    taxonomy_gap_gt_counts: dict[str, int] = {}
    n_images = 0

    import time
    latencies_ms = []

    with torch.no_grad():
        # MPS/CUDA JIT-compiles kernels on first use; an untimed warmup
        # pass on a throwaway tensor avoids that one-time cost polluting
        # the latency numbers below with a multi-second outlier.
        model([to_tensor(Image.open(image_paths[0]).convert("RGB")).to(device)])
        for img_path in image_paths:
            label_path = LABELS_DIR / (img_path.stem + ".json")
            if not label_path.exists():
                continue
            n_images += 1
            gt_boxes = load_ground_truth(label_path)

            image = Image.open(img_path).convert("RGB")
            tensor = to_tensor(image).to(device)
            start = time.perf_counter()
            output = model([tensor])[0]
            latencies_ms.append((time.perf_counter() - start) * 1000)

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

    print(f"\n=== SSDLite320-MobileNetV3-Large vs real BDD100K val labels (n={n_images} images) ===")
    print(f"{'class':<16}{'TP':>6}{'FP':>6}{'FN':>6}{'precision':>12}{'recall':>10}")
    for name, s in sorted(stats.items()):
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        print(f"{name:<16}{tp:>6}{fp:>6}{fn:>6}{precision:>12.3f}{recall:>10.3f}")

    print("\n=== BDD100K ground-truth categories this COCO detector CANNOT express at all ===")
    for cat, count in sorted(taxonomy_gap_gt_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<20} {count} ground-truth boxes in this sample")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== Model footprint (this Mac, device={device}, batch size 1) ===")
    print(f"parameters: {n_params:,}")
    print(f"mean inference latency: {sum(latencies_ms) / len(latencies_ms):.1f} ms/image")
    print(f"max inference latency: {max(latencies_ms):.1f} ms/image")


if __name__ == "__main__":
    main()
