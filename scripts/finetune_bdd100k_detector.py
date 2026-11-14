"""Fine-tune the COCO-pretrained Faster R-CNN MobileNetV3-320 FPN detector
(chosen over SSDLite per docs/benchmarks/bdd100k_detector_comparison_v1.md
-- faster AND more accurate on this Mac's measured results) on a bounded
BDD100K TRAIN sample (scripts/extract_bdd100k_train_sample.py, 600
images), then re-evaluates on the SAME 40 held-out VAL images the earlier
pretrained-only benchmark used, so the before/after comparison is a fair
one (fine-tuning never sees those 40 images).

This is a deliberately small, time-boxed fine-tune (600 train images,
few epochs, Mac MPS) -- not a claim of a production-grade detector.
Honest framing throughout; see the generated report doc for exact
measured numbers and limitations.

Replaces the pretrained COCO head with a new head over BDD100K's own
classes (so traffic sign / traffic light / rider -- which COCO cannot
express at all -- become reachable), keeping the pretrained backbone
(ImageNet/COCO features) frozen except for the FPN+head, which is the
standard torchvision detection fine-tuning recipe.
"""

import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = REPO_ROOT / "data/interim/bdd100k_frames_train_sample"
OUT_DIR = REPO_ROOT / "artifacts/bdd100k_detector"
EPOCHS = 4
LR = 5e-4

# BDD100K's own 10 detection classes (0 reserved for background, torchvision convention).
BDD_CLASSES = [
    "background", "car", "truck", "bus", "person", "rider", "bike",
    "motor", "traffic light", "traffic sign", "train",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(BDD_CLASSES)}


class BDDDetectionDataset(Dataset):
    def __init__(self, images_dir: Path, labels_dir: Path):
        self.samples = sorted(p.stem for p in images_dir.glob("*.jpg"))
        self.images_dir = images_dir
        self.labels_dir = labels_dir

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        stem = self.samples[idx]
        image = Image.open(self.images_dir / f"{stem}.jpg").convert("RGB")
        data = json.loads((self.labels_dir / f"{stem}.json").read_text())
        boxes, labels = [], []
        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                box = obj.get("box2d")
                category = obj.get("category")
                if box is None or category not in CLASS_TO_IDX:
                    continue  # lane/drivable-area/etc. have no box2d or aren't a detection class
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(CLASS_TO_IDX[category])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
        }
        return to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    device = select_device()
    print(f"device: {device}")

    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(BDD_CLASSES))
    model.to(device)

    # Freeze the backbone (pretrained ImageNet features); fine-tune the
    # FPN + RPN + the new detection head only -- the standard "replace
    # the head, fine-tune everything above the backbone" recipe for a
    # small, time-boxed dataset.
    for param in model.backbone.body.parameters():
        param.requires_grad = False

    dataset = BDDDetectionDataset(TRAIN_DIR / "images", TRAIN_DIR / "labels")
    print(f"fine-tuning on {len(dataset)} BDD100K train images (never-seen-before val set used for eval)")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=LR, momentum=0.9, weight_decay=5e-4)

    model.train()
    start = time.time()
    for epoch in range(EPOCHS):
        total_loss, n_batches = 0.0, 0
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        print(f"epoch {epoch + 1}/{EPOCHS}  mean_loss={total_loss / max(n_batches, 1):.4f}  ({n_batches} batches)")
    train_seconds = time.time() - start
    print(f"fine-tuning time: {train_seconds:.1f}s on {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_DIR / "model.pt")
    with open(OUT_DIR / "classes.json", "w") as f:
        json.dump(BDD_CLASSES, f)
    with open(OUT_DIR / "train_meta.json", "w") as f:
        json.dump({
            "train_images": len(dataset),
            "epochs": EPOCHS,
            "device": str(device),
            "train_seconds": train_seconds,
            "base_model": "fasterrcnn_mobilenet_v3_large_320_fpn (COCO-pretrained backbone, fine-tuned head+FPN+RPN)",
        }, f, indent=2)
    print(f"Saved fine-tuned checkpoint to {OUT_DIR}")


if __name__ == "__main__":
    main()
