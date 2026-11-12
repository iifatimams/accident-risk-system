"""MobileNetV3-small baseline for driver gaze-on-road classification.

Evidence-gating step (spec Section 51/79, reconciled plan): before adding
any temporal model (GRU or otherwise) on top of per-frame driver-state
predictions, establish whether a *single-frame* classifier is even viable.
This trains a frozen-backbone MobileNetV3-small (ImageNet pretrained) with a
new linear head on the frame sample extracted by extract_dmd_frames.py, and
reports held-out metrics honestly.

Known, stated limitations of this run (do not oversell it):
  - Single driver, single session (gC-14/s1) -- train and val frames come
    from the SAME person/camera/lighting/seat position, just different time
    windows. This measures "can it fit this one session", not generalization
    across drivers. A real baseline needs frames from multiple DMD subjects.
  - Class imbalance (~85% looking_road) is handled with inverse-frequency
    sample weighting but n=1337 total is small.
  - No hyperparameter search; this is a first feasibility pass, not a tuned
    production model.
"""

import time
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

DATA_ROOT = Path(__file__).resolve().parent.parent / "data/interim/dmd_frames/driver_state_dataset"
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-3


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(n_classes: int) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, n_classes)  # trainable head only
    return model, weights.transforms()


def main() -> None:
    device = select_device()
    print(f"device: {device}")

    model, tfm = build_model(n_classes=2)
    model.to(device)

    train_ds = datasets.ImageFolder(DATA_ROOT / "train", transform=tfm)
    val_ds = datasets.ImageFolder(DATA_ROOT / "val", transform=tfm)
    print("classes:", train_ds.classes)
    assert train_ds.classes == val_ds.classes, "train/val class sets must match"

    # Inverse-frequency sampling to counter the ~85/15 class imbalance.
    class_counts = torch.zeros(len(train_ds.classes))
    for _, label in train_ds.samples:
        class_counts[label] += 1
    sample_weights = [1.0 / class_counts[label] for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(model.classifier[-1].parameters(), lr=LR)
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
    print(f"training time: {time.time() - start:.1f}s on {device}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print("\n=== Held-out validation (same session, later time window) ===")
    print(classification_report(all_labels, all_preds, target_names=train_ds.classes, digits=3))
    print("confusion matrix (rows=true, cols=pred):")
    print(train_ds.classes)
    print(confusion_matrix(all_labels, all_preds))

    # --- Temporal-aggregation evidence gate ---
    # Spec requires evaluating lightweight temporal aggregation BEFORE
    # justifying a GRU. Cheapest possible test: does a rolling majority
    # vote over the raw per-frame predictions (restored to actual time
    # order via the frame_NNNNNN.jpg filename, since val_loader's
    # shuffle=False still preserves ImageFolder's class-grouped order, not
    # chronological order) reduce error vs. the raw per-frame predictions
    # above? If yes, that is real evidence a GRU may be unnecessary for
    # this task; if no, it's evidence smoothing alone isn't enough.
    #
    # The window must be causal (historical-only): a live deployment can
    # only see frames up to "now", so voting must never use frames after i.
    frame_indices = [int(Path(p).stem.split("_")[1]) for p, _ in val_ds.samples]
    order = sorted(range(len(frame_indices)), key=lambda i: frame_indices[i])
    ordered_true = [all_labels[i] for i in order]
    ordered_pred = [all_preds[i] for i in order]

    window = 5
    smoothed_pred = []
    for i in range(len(ordered_pred)):
        lo = max(0, i - window + 1)
        window_votes = ordered_pred[lo:i + 1]
        smoothed_pred.append(max(set(window_votes), key=window_votes.count))

    # Extracted samples are every SAMPLE_STRIDE-th real video frame, so a
    # window of `window` extracted samples spans window*SAMPLE_STRIDE real
    # frames at ~29.76fps (confirmed via cv2 on this session's video).
    window_seconds = window * 5 / 29.76
    print(f"\n=== Causal rolling majority-vote smoothing (window={window} sampled frames, ~{window_seconds:.2f}s historical lookback) ===")
    print(classification_report(ordered_true, smoothed_pred, target_names=train_ds.classes, digits=3))
    print("confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(ordered_true, smoothed_pred))


if __name__ == "__main__":
    main()
