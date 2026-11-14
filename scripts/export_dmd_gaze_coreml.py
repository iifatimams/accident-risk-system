"""Exports the DMD gaze-zone model (scripts/train_dmd_gaze_v3.py,
artifacts/dmd_gaze_v3/model.pt) to Core ML, replacing the earlier
binary DMDAttentionModel.mlpackage (trained on the DISTRACTION
category's coarser gaze_on_road label) with this 3-class model trained
on DMD's dedicated GAZE category annotations (road/brief_glance/
off_road).

Same export approach as scripts/export_dmd_attention_coreml.py
(ImageNet normalization baked into the traced model itself, not
approximated via Core ML's single-scalar ImageType scale/bias).
"""

import json
from pathlib import Path

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small

import coremltools as ct

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "artifacts/dmd_gaze_v3"
OUT_PATH = REPO_ROOT / "ios/RoadSense/Models/DMDAttentionModel.mlpackage"


class NormalizedModel(nn.Module):
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
