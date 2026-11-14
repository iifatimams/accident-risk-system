"""Extract a bounded sample of BDD100K TRAIN images + labels for a
feasible fine-tuning run, by filename-stem match between
bdd100k_images_100k.zip and bdd100k_labels.zip -- same pairing the
existing 40-image val benchmark sample already relies on. Reads the
zips under data/raw/ (never modifies or re-downloads them, per
AGENTS.md); writes only to data/interim/.

Deliberately NOT a random sample across all 70,000 train images (that
would mean decompressing the whole zip's central directory sequentially
regardless); takes every Nth entry instead, bounded by N_IMAGES, which
is a fast, deterministic, order-based sample -- stated here as a real
limitation, not hidden.
"""

import json
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_ZIP = REPO_ROOT / "data/raw/bdd100k/bdd100k_images_100k.zip"
LABELS_ZIP = REPO_ROOT / "data/raw/bdd100k/bdd100k_labels.zip"
OUT_DIR = REPO_ROOT / "data/interim/bdd100k_frames_train_sample"
N_IMAGES = 600
STRIDE = 70000 // N_IMAGES


def main() -> None:
    out_images = OUT_DIR / "images"
    out_labels = OUT_DIR / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(IMAGES_ZIP) as izf, zipfile.ZipFile(LABELS_ZIP) as lzf:
        train_images = sorted(n for n in izf.namelist() if n.startswith("100k/train/") and n.endswith(".jpg"))
        print(f"{len(train_images)} train images available in zip; sampling every {STRIDE}th -> up to {N_IMAGES}")

        label_names = {n.split("/")[-1][:-5]: n for n in lzf.namelist() if n.startswith("100k/train/") and n.endswith(".json")}

        written = 0
        for img_name in train_images[::STRIDE]:
            stem = img_name.split("/")[-1][:-4]
            label_name = label_names.get(stem)
            if label_name is None:
                continue  # no matching label -- skip, don't fabricate one
            label_bytes = lzf.read(label_name)
            data = json.loads(label_bytes)
            if not data.get("frames") or not data["frames"][0].get("objects"):
                continue  # no annotated objects -- not useful for training
            (out_images / f"{stem}.jpg").write_bytes(izf.read(img_name))
            (out_labels / f"{stem}.json").write_bytes(label_bytes)
            written += 1
            if written >= N_IMAGES:
                break

    print(f"Wrote {written} image+label pairs to {OUT_DIR}")


if __name__ == "__main__":
    main()
