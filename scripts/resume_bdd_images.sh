#!/bin/bash
# Robust resume loop for bdd100k_images_100k.zip -- the plain curl attempt
# dropped mid-transfer (connection reset), so retry with -C - resume until
# it actually completes, capped at 30 attempts.
set -uo pipefail
OUT="data/raw/bdd100k/bdd100k_images_100k.zip"
URL="http://128.32.162.150/bdd100k/bdd100k_images_100k.zip"
TARGET_SIZE=5669071832

attempt=0
while [ "$attempt" -lt 30 ]; do
  attempt=$((attempt + 1))
  cur=$(stat -f%z "$OUT" 2>/dev/null || echo 0)
  if [ "$cur" -ge "$TARGET_SIZE" ]; then
    echo "[OK] images_100k already complete ($cur bytes)"
    break
  fi
  echo "[attempt $attempt] resuming from $cur / $TARGET_SIZE bytes..."
  curl -sS --retry 3 --retry-delay 5 -C - -o "$OUT" "$URL"
  rc=$?
  cur=$(stat -f%z "$OUT" 2>/dev/null || echo 0)
  if [ "$rc" -eq 0 ] && [ "$cur" -ge "$TARGET_SIZE" ]; then
    echo "[OK] images_100k complete ($cur bytes)"
    break
  fi
  echo "[retry] curl exited $rc, size now $cur, retrying in 3s..."
  sleep 3
done
echo "RESUME_SCRIPT_DONE"
