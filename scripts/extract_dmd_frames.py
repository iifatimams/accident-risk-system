"""Extract a capped, labeled frame sample from ONE already-downloaded DMD session.

Evidence-gating step for the vision baseline (spec Section 51/79): before any
temporal model (GRU or otherwise) is justified, we need to know whether a
single-frame classifier can separate the driver-state classes at all. This
script turns one DMD face-camera video + its ASAM OpenLabel annotation JSON
into a small on-disk image dataset, split causally (by time, not shuffled)
into train/val so adjacent near-duplicate frames don't leak across the split.

Deliberately NOT a general-purpose DMD loader: it hardcodes the one session
we extracted for the audit (distraction/gC-14/s1) and the one binary task
that session's labels actually support cleanly (gaze_on_road: looking_road
vs not_looking_road). Extending to more sessions/tasks is future work, gated
on this baseline actually being informative.
"""

import json
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSION_DIR = REPO_ROOT / "data/interim/dmd_frames/gC_14_s1"
VIDEO_PATH = SESSION_DIR / "gC_14_s1_2019-03-04T11;56;20+01;00_rgb_face.mp4"
ANNOTATION_PATH = SESSION_DIR / "gC_14_s1_2019-03-04T11;56;20+01;00_rgb_ann_distraction.json"
OUT_ROOT = REPO_ROOT / "data/interim/dmd_frames/driver_state_dataset"

SAMPLE_STRIDE = 5  # keep every 5th frame -> ~1336 frames instead of 6681
VAL_FRACTION = 0.2  # last 20% of session time -> val (causal split, no shuffle)
LABELS = {
    "gaze_on_road/looking_road": "looking_road",
    "gaze_on_road/not_looking_road": "not_looking_road",
}


def build_frame_label_map() -> dict[int, str]:
    """Expand each action's frame_intervals into a flat {frame_index: label} map."""
    with open(ANNOTATION_PATH) as f:
        ol = json.load(f)["openlabel"]
    frame_label: dict[int, str] = {}
    for action in ol["actions"].values():
        label = LABELS.get(action["type"])
        if label is None:
            continue
        for interval in action["frame_intervals"]:
            for idx in range(interval["frame_start"], interval["frame_end"] + 1):
                frame_label[idx] = label
    return frame_label


def main() -> None:
    frame_label = build_frame_label_map()
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {VIDEO_PATH}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    split_boundary = int(total_frames * (1 - VAL_FRACTION))

    for label in set(LABELS.values()):
        for split in ("train", "val"):
            (OUT_ROOT / split / label).mkdir(parents=True, exist_ok=True)

    counts = {"train": {}, "val": {}}
    idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % SAMPLE_STRIDE == 0 and idx in frame_label:
            label = frame_label[idx]
            split = "val" if idx >= split_boundary else "train"
            out_path = OUT_ROOT / split / label / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            counts[split][label] = counts[split].get(label, 0) + 1
            written += 1
        idx += 1
    cap.release()

    print(f"Session total frames: {total_frames}, stride: {SAMPLE_STRIDE}")
    print(f"Written: {written} labeled frames to {OUT_ROOT}")
    for split in ("train", "val"):
        print(f"  {split}: {counts[split]}")


if __name__ == "__main__":
    main()
