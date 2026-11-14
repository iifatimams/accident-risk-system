"""MobileNetV3-small gaze-zone model, v3 -- trained on DMD's dedicated
GAZE category annotations (scripts/extract_dmd_gaze_frames.py), not the
binary gaze_on_road label v1/v2 used from the DISTRACTION category.

3 classes instead of 2, specifically so brief, necessary glances (mirror
checks, instrument cluster) can be told apart from sustained distraction
(looking at the infotainment screen, the steering-wheel/lap area, or
turned fully left/right) -- a binary road/not-road label cannot express
that distinction at all, and the capstone directive explicitly asks for
it ("distinguish brief mirror and instrument-cluster checks from
sustained distraction").

Started with v2's recipe (fine-tune the last 2 inverted-residual
blocks, not just the classifier head). That OVERFIT badly on this
dataset: held-out-driver accuracy 0.345, off_road recall 0.023 --
barely above random for 3 classes. Each driver's gaze-category session
is a single short, scripted calibration clip (little within-class
visual diversity per driver), so fine-tuning real conv-layer weights
on only 4 such clips memorized driver-specific appearance rather than
the general gaze-direction concept.

HEAD-ONLY training (freeze the whole backbone, only train the final
linear layer -- the SAME recipe v1's binary model used) generalizes
far better on this specific dataset: held-out accuracy 0.552, off_road
recall 0.765. This is the opposite lesson from the v1->v2 improvement
on the DISTRACTION-category binary task (where fine-tuning deeper
layers helped) -- a real, measured, dataset-specific finding, not a
general rule; see docs/FINAL_MULTIMODAL_RESULTS.md for both numbers.
This script now trains head-only, matching what was actually measured
and deployed.

Same augmentation (ColorJitter + small RandomRotation) and
held-out-driver discipline (train: gC-14/gA-1/gB-6/gE-28, val: gF-23,
never seen in training) as before.
"""

import json
import time
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data/interim/dmd_frames/gaze_multisession_dataset"
OUT_DIR = REPO_ROOT / "artifacts/dmd_gaze_v3"
BATCH_SIZE = 32
EPOCHS = 8
HEAD_LR = 1e-3


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(n_classes: int):
    # Head-only (see module docstring: fine-tuning the backbone overfit
    # badly on this dataset's small, low-diversity-per-driver clips).
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, n_classes)
    return model, weights.transforms()


def main() -> None:
    device = select_device()
    print(f"device: {device}")

    model, base_tfm = build_model(n_classes=3)
    model.to(device)

    train_tfm = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15),
        transforms.RandomRotation(degrees=8),
        base_tfm,
    ])

    train_ds = datasets.ImageFolder(DATA_ROOT / "train", transform=train_tfm)
    val_ds = datasets.ImageFolder(DATA_ROOT / "val", transform=base_tfm)
    print("classes:", train_ds.classes)
    assert train_ds.classes == val_ds.classes, "train/val class sets must match"
    print(f"train: {len(train_ds)} frames from 4 drivers (gC-14, gA-1, gB-6, gE-28)")
    print(f"val:   {len(val_ds)} frames from 1 HELD-OUT driver (gF-23, never in training)")

    class_counts = torch.zeros(len(train_ds.classes))
    for _, label in train_ds.samples:
        class_counts[label] += 1
    sample_weights = [1.0 / class_counts[label] for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(model.classifier[-1].parameters(), lr=HEAD_LR)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss, n = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            n += len(labels)
        print(f"epoch {epoch + 1}/{EPOCHS}  train_loss={total_loss / n:.4f}")
    train_seconds = time.time() - start
    print(f"training time: {train_seconds:.1f}s on {device}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print("\n=== Held-out driver (gF-23, NEVER seen in training) ===")
    report_str = classification_report(all_labels, all_preds, target_names=train_ds.classes, digits=3)
    print(report_str)
    cm = confusion_matrix(all_labels, all_preds)
    print("confusion matrix (rows=true, cols=pred):")
    print(train_ds.classes)
    print(cm)

    report_dict = classification_report(
        all_labels, all_preds, target_names=train_ds.classes, digits=3, output_dict=True
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_DIR / "model.pt")
    with open(OUT_DIR / "classes.json", "w") as f:
        json.dump(train_ds.classes, f)
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({
            "classes": train_ds.classes,
            "report": report_dict,
            "confusion_matrix": cm.tolist(),
            "train_frames": len(train_ds),
            "val_frames": len(val_ds),
            "train_drivers": ["gC-14", "gA-1", "gB-6", "gE-28"],
            "val_driver_held_out": "gF-23",
            "epochs": EPOCHS,
            "device": str(device),
            "train_seconds": train_seconds,
            "source": "DMD gaze category (gaze_zone/* OpenLABEL annotations), not the binary gaze_on_road distraction-category label v1/v2 used",
        }, f, indent=2)
    print(f"\nSaved checkpoint + metrics to {OUT_DIR}")


if __name__ == "__main__":
    main()
