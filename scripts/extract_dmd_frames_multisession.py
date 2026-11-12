"""Extract labeled frames from MULTIPLE DMD sessions/drivers, with a
driver-level (subject-separated) train/val split -- unlike
extract_dmd_frames.py (single session, time-based split), this is the
actual Phase 3.2 ask: "Split by driver/participant where identities
permit, rather than putting nearby frames from the same driver into both
training and validation."

Five sessions are available locally after extraction (see
docs/benchmarks/dmd_driver_state_baseline_v1.md for how): gC-14, gA-1,
gB-6, gE-28 go to TRAIN; gF-23 is held out ENTIRELY for VAL -- that driver
never appears in a single training frame. This directly tests
generalization to an unseen driver, which the single-session baseline
explicitly could not.

Same binary task as extract_dmd_frames.py (gaze_on_road: looking_road vs
not_looking_road) -- the only label this data cleanly supports; see that
script and docs/benchmarks/dmd_driver_state_baseline_v1.md for why no
other label (phone_use, looking_up/down) is used.
"""

import json
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
FRAMES_ROOT = REPO_ROOT / "data/interim/dmd_frames"
OUT_ROOT = REPO_ROOT / "data/interim/dmd_frames/multisession_dataset"

SAMPLE_STRIDE = 5
LABELS = {
    "gaze_on_road/looking_road": "looking_road",
    "gaze_on_road/not_looking_road": "not_looking_road",
}

# (driver_tag, session_dir_name, video/annotation basename prefix, split)
SESSIONS = [
    ("gC-14", "gC_14_s1", "gC_14_s1_2019-03-04T11;56;20+01;00", "train"),
    ("gA-1", "gA_1_s1", "gA_1_s1_2019-03-08T09;31;15+01;00", "train"),
    ("gB-6", "gB_6_s1", "gB_6_s1_2019-03-11T13;55;14+01;00", "train"),
    ("gE-28", "gE_28_s1", "gE_28_s1_2019-03-15T10;23;30+01;00", "train"),
    ("gF-23", "gF_23_s1", "gF_23_s1_2019-03-05T09;23;45+01;00", "val"),
]


def build_frame_label_map(annotation_path: Path) -> dict[int, str]:
    with open(annotation_path) as f:
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
    counts: dict[str, dict[str, int]] = {"train": {}, "val": {}}
    for label in set(LABELS.values()):
        for split in ("train", "val"):
            (OUT_ROOT / split / label).mkdir(parents=True, exist_ok=True)

    for driver_tag, session_dir, prefix, split in SESSIONS:
        video_path = FRAMES_ROOT / session_dir / f"{prefix}_rgb_face.mp4"
        annotation_path = FRAMES_ROOT / session_dir / f"{prefix}_rgb_ann_distraction.json"
        if not video_path.exists() or not annotation_path.exists():
            raise FileNotFoundError(
                f"{driver_tag}: missing {video_path} or {annotation_path} -- "
                "extract this session first (see docs/benchmarks/dmd_driver_state_baseline_v1.md)"
            )
        frame_label = build_frame_label_map(annotation_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        written = 0
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % SAMPLE_STRIDE == 0 and idx in frame_label:
                label = frame_label[idx]
                # driver_tag prefix on the filename: frames from different
                # drivers must never collide/overwrite each other in the
                # same class folder, and it keeps provenance visible on disk.
                out_path = OUT_ROOT / split / label / f"{driver_tag}_frame_{idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
                counts[split][label] = counts[split].get(label, 0) + 1
                written += 1
            idx += 1
        cap.release()
        print(f"{driver_tag} ({split}): {total_frames} total frames, {written} labeled samples written")

    print(f"\nWritten to {OUT_ROOT}")
    for split in ("train", "val"):
        print(f"  {split}: {counts[split]}")


if __name__ == "__main__":
    main()
