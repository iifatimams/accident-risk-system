"""Evaluate the fine-tuned detector (scripts/finetune_bdd100k_detector.py's
checkpoint) on the SAME 40 held-out BDD100K val images the pretrained-only
benchmark (benchmark_detector_bdd100k.py) used, for a fair, apples-to-
apples before/after comparison -- these 40 images were never in the
fine-tuning sample (they come from data/interim/bdd100k_frames, the
fine-tune used data/interim/bdd100k_frames_train_sample instead).
"""

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = REPO_ROOT / "data/interim/bdd100k_frames/images"
LABELS_DIR = REPO_ROOT / "data/interim/bdd100k_frames/labels"
CHECKPOINT_DIR = REPO_ROOT / "artifacts/bdd100k_detector"
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(label_path: Path):
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
    classes = json.loads((CHECKPOINT_DIR / "classes.json").read_text())
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes))
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "model.pt", map_location=device))
    model.to(device).eval()

    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"Evaluating fine-tuned detector on {len(image_paths)} held-out BDD100K val images (device={device})")

    stats = {name: {"tp": 0, "fp": 0, "fn": 0} for name in classes if name != "background"}

    with torch.no_grad():
        for image_path in image_paths:
            label_path = LABELS_DIR / f"{image_path.stem}.json"
            gt_boxes = load_ground_truth(label_path)
            image = Image.open(image_path).convert("RGB")
            tensor = to_tensor(image).to(device)
            prediction = model([tensor])[0]

            pred_boxes = []
            for box, label, score in zip(prediction["boxes"].cpu().tolist(), prediction["labels"].cpu().tolist(), prediction["scores"].cpu().tolist()):
                if score < SCORE_THRESHOLD:
                    continue
                class_name = classes[label]
                if class_name in stats:
                    pred_boxes.append((class_name, box))

            matched_gt = set()
            for class_name, box in pred_boxes:
                best_iou, best_idx = 0.0, -1
                for idx, (gt_class, gt_box) in enumerate(gt_boxes):
                    if idx in matched_gt or gt_class != class_name:
                        continue
                    i = iou(box, gt_box)
                    if i > best_iou:
                        best_iou, best_idx = i, idx
                if best_iou >= IOU_THRESHOLD:
                    stats[class_name]["tp"] += 1
                    matched_gt.add(best_idx)
                else:
                    stats[class_name]["fp"] += 1

            for idx, (gt_class, _) in enumerate(gt_boxes):
                if idx not in matched_gt and gt_class in stats:
                    stats[gt_class]["fn"] += 1

    print(f"\n=== Fine-tuned Faster R-CNN vs real BDD100K val labels (n={len(image_paths)} images) ===")
    results = {}
    for class_name, counts in sorted(stats.items()):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        print(f"  {class_name:15s} precision={precision:.3f}  recall={recall:.3f}  (tp={tp} fp={fp} fn={fn})")
        results[class_name] = {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}

    with open(CHECKPOINT_DIR / "val_eval.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: None if x != x else x)
    print(f"\nSaved to {CHECKPOINT_DIR / 'val_eval.json'}")


if __name__ == "__main__":
    main()
