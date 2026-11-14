"""Extracts frames + a richer 3-class gaze label from DMD's dedicated
GAZE category (data/raw/dmd/gaze/), not the binary gaze_on_road label
the earlier `extract_dmd_frames_multisession.py` used from the
DISTRACTION category.

Why this exists: inspected a real gaze annotation file
(dmd/gA/1/s6/..._rgb_ann_gaze.json, OpenLABEL format) and found DMD
actually records a 9-zone gaze taxonomy per frame interval --
`gaze_zone/front`, `/front_right`, `/left_mirror`, `/center_mirror`,
`/right_mirror`, `/left`, `/right`, `/infotainment`, `/steering_wheel`,
plus `/not_valid` and a separate `blinks/blinking` action -- not just a
binary on/off-road label. Each "s6" session is a scripted gaze-
calibration clip: one clean, contiguous interval per zone (roughly
150-430 frames each), which is exactly the kind of clean ground truth a
frame classifier needs.

`gaze_zone/steering_wheel` is the DMD label that matches the capstone
directive's own example almost exactly: "a driver may hold a phone
below the steering wheel... eyes directed downward, head relatively
straight" -- this is real, existing, available ground truth for
precisely that scenario, not a guess.

Class mapping (3 classes, chosen to match the directive's explicit
"distinguish brief mirror/instrument-cluster checks from sustained
distraction" requirement -- a 2-class road/not-road label cannot
express that distinction at all):
  - road:          gaze_zone/front
  - brief_glance:  gaze_zone/left_mirror, center_mirror, right_mirror,
                    front_right (treated as instrument-cluster-adjacent,
                    near-forward)
  - off_road:      gaze_zone/left, right, infotainment, steering_wheel
  - EXCLUDED:       gaze_zone/not_valid (ambiguous/unannotated tail)

Same train/held-out-driver split as the existing DMD work (train:
gC-14, gA-1, gB-6, gE-28; held-out: gF-23) for direct comparability.
"""

import json
import tarfile
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
GAZE_RAW_DIR = REPO_ROOT / "data/raw/dmd/gaze"
OUT_ROOT = REPO_ROOT / "data/interim/dmd_frames/gaze_multisession_dataset"
FRAME_STRIDE = 3  # sample every 3rd frame within a labeled interval

ZONE_TO_CLASS = {
    "gaze_zone/front": "road",
    "gaze_zone/left_mirror": "brief_glance",
    "gaze_zone/center_mirror": "brief_glance",
    "gaze_zone/right_mirror": "brief_glance",
    "gaze_zone/front_right": "brief_glance",
    "gaze_zone/left": "off_road",
    "gaze_zone/right": "off_road",
    "gaze_zone/infotainment": "off_road",
    "gaze_zone/steering_wheel": "off_road",
}

# driver -> (archive stem, session dir) for the one "s6" calibration
# session each driver has in the gaze category.
SESSIONS = {
    "gC-14": "gC/14/s6",
    "gA-1": "gA/1/s6",
    "gB-6": "gB/6/s6",
    "gE-28": "gE/28/s6",
    "gF-23": "gF/23/s6",
}
TRAIN_DRIVERS = ["gC-14", "gA-1", "gB-6", "gE-28"]
HELD_OUT_DRIVER = "gF-23"


def find_session_members(archive_path: Path, session_dir: str) -> tuple[str, str]:
    """Returns (face_mp4_member, gaze_json_member) inside the tar for this session."""
    with tarfile.open(archive_path, "r:gz") as tf:
        names = tf.getnames()
    face_candidates = [n for n in names if f"/{session_dir}/" in n and n.endswith("_rgb_face.mp4")]
    gaze_candidates = [n for n in names if f"/{session_dir}/" in n and n.endswith("_rgb_ann_gaze.json")]
    if not face_candidates or not gaze_candidates:
        raise FileNotFoundError(f"missing face video or gaze annotation for {session_dir} in {archive_path}")
    return face_candidates[0], gaze_candidates[0]


def extract_session(driver: str, out_split: str) -> int:
    archive_path = GAZE_RAW_DIR / f"dmd-dataset-gaze-{driver}.tar.gz"
    session_dir = SESSIONS[driver]
    face_member, gaze_member = find_session_members(archive_path, session_dir)

    extract_dir = Path(f"/tmp/dmd_gaze_extract_{driver}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extract(face_member, path=extract_dir)
        tf.extract(gaze_member, path=extract_dir)

    gaze_json_path = extract_dir / gaze_member
    face_video_path = extract_dir / face_member
    data = json.loads(gaze_json_path.read_text())
    actions = data["openlabel"]["actions"]

    frame_to_class: dict[int, str] = {}
    for action in actions.values():
        zone_type = action.get("type")
        class_name = ZONE_TO_CLASS.get(zone_type)
        if class_name is None:
            continue  # not_valid, blinks/blinking, or any other non-gaze-zone action
        for interval in action.get("frame_intervals", []):
            for frame_idx in range(interval["frame_start"], interval["frame_end"] + 1, FRAME_STRIDE):
                frame_to_class[frame_idx] = class_name

    cap = cv2.VideoCapture(str(face_video_path))
    written = 0
    for frame_idx, class_name in sorted(frame_to_class.items()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        out_dir = OUT_ROOT / out_split / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{driver}_frame_{frame_idx:06d}.jpg"), frame)
        written += 1
    cap.release()

    # Free the extracted video/json immediately -- these are regenerable
    # from the raw archive and we don't want 1GB+ face videos lingering.
    face_video_path.unlink(missing_ok=True)
    gaze_json_path.unlink(missing_ok=True)

    print(f"  {driver} ({session_dir}): wrote {written} frames to {out_split}/")
    return written


def main() -> None:
    print("Extracting TRAIN drivers...")
    total_train = sum(extract_session(d, "train") for d in TRAIN_DRIVERS)
    print(f"\nExtracting HELD-OUT driver '{HELD_OUT_DRIVER}'...")
    total_val = extract_session(HELD_OUT_DRIVER, "val")
    print(f"\nTotal: {total_train} train frames, {total_val} held-out-driver val frames")
    print(f"Written to {OUT_ROOT}")


if __name__ == "__main__":
    main()
