"""Exports the DMD attention model trained by
scripts/train_driver_state_multisession_v2.py (artifacts/dmd_attention/model.pt)
to Core ML, as an image-input classifier matching the training
preprocessing (MobileNetV3_Small_Weights.DEFAULT's own transforms:
resize/center-crop to 224x224, ImageNet normalization) so the live app
feeds the model exactly what it was trained on.

Run AFTER scripts/train_driver_state_multisession_v2.py has produced
artifacts/dmd_attention/model.pt + classes.json.
"""

import json
from pathlib import Path

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

# Imported AFTER torch/torchvision deliberately -- importing coremltools
# first caused a reproducible segfault (exit 139) during ct.convert when
# this script ran standalone, which did NOT reproduce when the identical
# logic ran as an inline `python -c` snippet with torch imported first.
# A known class of macOS/Accelerate-framework library-init-order issue
# between coremltools and torch; import order is the fix, not the logic.
import coremltools as ct

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "artifacts/dmd_attention"
OUT_PATH = REPO_ROOT / "ios/RoadSense/Models/DMDAttentionModel.mlpackage"


class NormalizedModel(nn.Module):
    """Wraps the trained classifier with the EXACT per-channel ImageNet
    normalization MobileNet_V3_Small_Weights.DEFAULT.transforms() applies
    at train/eval time, baked into the traced graph itself -- ct.ImageType's
    `scale`/`bias` are single scalars (one value for all 3 channels), which
    cannot express ImageNet's per-channel mean/std correctly, and silently
    using one channel's values for all three would be exactly the kind of
    training/live feature-mismatch bug this project explicitly guards
    against elsewhere (see RoadContext.swift's design_speed fix). Baking
    per-channel normalization into the model itself sidesteps that
    limitation entirely: the exported Core ML model takes a plain 0-1 RGB
    image (scale=1/255, bias=0) and normalizes internally.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.base(x)


def main() -> None:
    classes = json.loads((CHECKPOINT_DIR / "classes.json").read_text())
    base_model = mobilenet_v3_small(weights=None)
    in_features = base_model.classifier[-1].in_features
    base_model.classifier[-1] = nn.Linear(in_features, len(classes))
    base_model.load_state_dict(torch.load(CHECKPOINT_DIR / "model.pt", map_location="cpu"))
    base_model.eval()

    model = NormalizedModel(base_model).eval()
    example_input = torch.rand(1, 3, 224, 224)
    traced = torch.jit.trace(model, example_input)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=example_input.shape, scale=1.0 / 255.0, bias=[0.0, 0.0, 0.0], color_layout="RGB")],
        classifier_config=ct.ClassifierConfig(class_labels=classes),
        source="pytorch",
    )
    mlmodel.save(str(OUT_PATH))
    print(f"Saved {OUT_PATH}")
    print(f"classes: {classes}")


if __name__ == "__main__":
    main()
