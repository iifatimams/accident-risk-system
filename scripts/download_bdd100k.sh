#!/bin/bash
# Downloads BDD100K 100K-Images + Detection-2020-Labels (single-frame
# detector training set) from the direct HTTP mirror, into data/raw/bdd100k/.
# Plain direct links, no auth/gating -- confirmed reachable 2026-11-12.
set -uo pipefail

DEST="data/raw/bdd100k"
mkdir -p "$DEST"

echo "Starting Detection 2020 Labels (220MB)..."
curl -sS --retry 5 --retry-delay 5 -C - \
  -o "$DEST/bdd100k_det_20_labels.zip" \
  "http://128.32.162.150/bdd100k/bdd100k_det_20_labels.zip" \
  && echo "[OK] det_20_labels" || echo "[FAIL] det_20_labels"

echo "Starting 100K Images (5.3GB)..."
curl -sS --retry 5 --retry-delay 5 -C - \
  -o "$DEST/bdd100k_images_100k.zip" \
  "http://128.32.162.150/bdd100k/bdd100k_images_100k.zip" \
  && echo "[OK] images_100k" || echo "[FAIL] images_100k"

echo "ALL_BDD_DOWNLOADS_DONE"
