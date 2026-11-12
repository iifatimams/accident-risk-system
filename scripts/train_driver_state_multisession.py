"""MobileNetV3-small baseline for driver gaze-on-road, trained on 4 drivers
and evaluated on a 5th driver NEVER seen in training -- the actual
generalization test the single-session baseline (train_driver_state_baseline.py)
explicitly could not run. Same model/training recipe as that script;
see its docstring for the shared limitations (no hyperparameter search,
class imbalance via inverse-frequency sampling).

Run scripts/extract_dmd_frames_multisession.py first.
"""

import time
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

DATA_ROOT = Path(__file__).resolve().parent.parent / "data/interim/dmd_frames/multisession_dataset"
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-3


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(n_classes: int):
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

    model, tfm = build_model(n_classes=2)
    model.to(device)

    train_ds = datasets.ImageFolder(DATA_ROOT / "train", transform=tfm)
    val_ds = datasets.ImageFolder(DATA_ROOT / "val", transform=tfm)
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

    print("\n=== Held-out driver (gF-23, NEVER seen in training) ===")
    print(classification_report(all_labels, all_preds, target_names=train_ds.classes, digits=3))
    print("confusion matrix (rows=true, cols=pred):")
    print(train_ds.classes)
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
