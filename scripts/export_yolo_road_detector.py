"""Exports a pretrained YOLOv8n (COCO) detector to Core ML via the
`ultralytics` package's own export path, which has first-class Core ML
support (unlike torchvision's research-oriented detection models, whose
list-of-dicts output `torch.jit.trace` cannot capture at all -- see
docs/FINAL_MULTIMODAL_RESULTS.md section 3.2).

yolo11n (the newer architecture) was tried first and failed: its
attention blocks hit a real coremltools bug (`TypeError: only
0-dimensional arrays can be converted to Python scalars`, inside
coremltools' own `int` op conversion). yolov8n (no attention blocks,
the more mature/commonly-exported architecture) converts cleanly.

Output is Vision-compatible out of the box (image input, `confidence`
[N,80] and `coordinates` [N,4] outputs, NMS already applied inside the
model) -- BackgroundCaptureService.swift consumes it directly via
VNCoreMLModel/VNCoreMLRequest, no custom post-processing needed.

This is a pretrained-COCO model -- NOT fine-tuned on driving footage.
The earlier BDD100K fine-tune attempt (on a different architecture)
regressed versus its own pretrained baseline and was abandoned; this
script does not attempt that again.
"""

from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "ios/RoadSense/Models/RoadObjectDetector.mlpackage"


def main() -> None:
    model = YOLO("yolov8n.pt")
    exported_path = model.export(format="coreml", imgsz=320, nms=True, half=False)
    exported_path = Path(exported_path)

    if OUT_PATH.exists():
        import shutil

        shutil.rmtree(OUT_PATH)
    exported_path.rename(OUT_PATH) if exported_path != OUT_PATH else None
    print(f"Saved {OUT_PATH}")
    print(f"classes: {model.names}")


if __name__ == "__main__":
    main()
