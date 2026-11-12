#!/bin/bash
# Downloads the DMD (Driver Monitoring Dataset) presigned-URL archives
# the user obtained from Vicomtech, into data/raw/dmd/<category>/.
# URLs are S3 presigned links with a 48h expiry (generated 2026-11-11
# ~19:53-20:05 UTC) -- run this soon after generation.
# NOTE: written for plain bash 3.2 (macOS default) -- no associative
# arrays, parallel indexed arrays instead.
set -uo pipefail

DEST_ROOT="data/raw/dmd"
mkdir -p "$DEST_ROOT/gaze" "$DEST_ROOT/drowsiness" "$DEST_ROOT/distraction"

MAX_PARALLEL=4

throttle() {
  while [ "$(jobs -r | wc -l | tr -d ' ')" -ge "$MAX_PARALLEL" ]; do
    sleep 2
  done
}

download() {
  local url="$1"
  local out="$2"
  throttle
  (
    curl -sS --retry 5 --retry-delay 5 -C - -o "$out" "$url" \
      && echo "[OK] $out" \
      || echo "[FAIL] $out"
  ) &
}

# ---- gaze ----
GAZE_BASE="https://datasets.vicomtech.org/di21-dmd-dataset-gaze"
GAZE_Q="X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=Cvu9WzjJuNCiu0Ve%2F20261111%2Feu-vicom-cluster03-ovh%2Fs3%2Faws4_request&X-Amz-Date=20261111T195333Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host"
GAZE_KEYS=(gA-1 gA-5 gB-6 gB-7 gB-9 gB-10 gC-13 gC-14 gE-28 gE-29 gF-23 gZ-33 gZ-36 gZ-37)
GAZE_SIGS=(
  3510657ae45e3848601ee9644d8b64827f24260b8c1ca846309ca385dfaf32c2
  8e9b7727b018955e5323b46c10364393bbf773b032c8b036900f29f86f3ffcc3
  3551830868a89c7bef7f6eb404717bfa80d57e7716448990aff1fea400c1c5f2
  42acfe74e2fe29fb2213163623ccd9d132b3f354e51bcaf597c839b937f4a79a
  11a7d766ea18598102d2622d5b5a4944ba090c260f0435b1c113560c5a4a5712
  b5e44932dab5f8f7687eb3e80241ef9dc473694ea2eaebed8fe228d92b420728
  4017899f66e9030585cedc3fe3523213255a19e0de64a6147ceeeb9a6d1978ae
  71afe97223a6e6ba49d6cd362adbebf0fd6fddfd2578737e94d405fb45f67c30
  1d96f166c82c31b39f5257582d6773d15326fb670f3cab0a8eb00157481b7f17
  1d7cfe0cc45cd3f063e7e5208d70f5a7aa5aedfee7d8d53b239c99c6a959aa05
  d4a72cf97b4d69577bf6d87e16eff7f5c37955a6c853a858867c571b0b4f500b
  e379d111b810fb65455227191135bd059ed538b9773a5cec2f4e43bd8530e626
  9499f0a154db26b0728a9009fb47def99060c6a28dcf68b7673d7081b2bd5ecb
  6c5407cbab6bbd052e406167e79da7bf7b3115ace6f446fee16413f525cb7840
)
for i in "${!GAZE_KEYS[@]}"; do
  key="${GAZE_KEYS[$i]}"
  sig="${GAZE_SIGS[$i]}"
  download "${GAZE_BASE}/dmd-dataset-gaze-${key}.tar.gz?${GAZE_Q}&X-Amz-Signature=${sig}" \
    "$DEST_ROOT/gaze/dmd-dataset-gaze-${key}.tar.gz"
done

# ---- drowsiness ----
DROWSY_BASE="https://datasets.vicomtech.org/di21-dmd-dataset-drowsiness"
DROWSY_Q="X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=Cvu9WzjJuNCiu0Ve%2F20261111%2Feu-vicom-cluster03-ovh%2Fs3%2Faws4_request&X-Amz-Date=20261111T200521Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host"
DROWSY_KEYS=(gA-1 gA-5 gB-6 gB-7 gB-9 gB-10 gC-13 gC-14 gE-29 gF-23 gZ-33 gZ-36 gZ-37)
DROWSY_SIGS=(
  940ca29acec9afaba3fb1b0442d2cba530f2ef88b89ce34ffc448d2cc1b17d91
  7aaa132c2e0549cef76e7a9a6e762da34ca8f14891e104200f54ffbaa47e6884
  2591e79d869e533ea181c5b609179393779cbab680fe950abcbffa14a31cb686
  82379f86843be28e352f39ad6807d0a4bff74d0733c0c2aa72cf6739599e2ca7
  8f3fc8f647170d1430b30f9db9c3a76b4da3f5f293db305e9b7af118f49ce902
  31b72a48f9025021a48373d59592d18f18d03ee646f4d650b95877c201892558
  05d62629f7c9a7c6424889595ecb0b3e2774498e7d00e502274bbaf8bbae0480
  6ddfd5a45b085024cba18c35d43b754560ee74bb1b5bb099ee74deaa0b33f938
  876ec6ca16590133781879fba4478b0b4e5e4f7ec32c73f503f9408f289ea4e7
  392ccbe8f70a4a0d87ed1be62e2f5118b31718f72504074122f23fccbddfd559
  a270a28de21a559e2b3c85d44c175f340bd7b31ec47fe949793c14b9f0361972
  6ca0bff6ef6bf754404f9d107e84ff7795f15a2892ef9f7d0fbc22cd41bd589d
  052a9732b8c1abe6c1daaff4e46363b8a4a7adcc16563a2219b82c9b1a49810f
)
for i in "${!DROWSY_KEYS[@]}"; do
  key="${DROWSY_KEYS[$i]}"
  sig="${DROWSY_SIGS[$i]}"
  download "${DROWSY_BASE}/dmd-dataset-drowsiness-${key}.tar.gz?${DROWSY_Q}&X-Amz-Signature=${sig}" \
    "$DEST_ROOT/drowsiness/dmd-dataset-drowsiness-${key}.tar.gz"
done

# ---- distraction (rgb-ir) ----
DISTR_BASE="https://datasets.vicomtech.org/di21-dmd-dataset-distraction-rgb-ir"
DISTR_Q="X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=Cvu9WzjJuNCiu0Ve%2F20261111%2Feu-vicom-cluster03-ovh%2Fs3%2Faws4_request&X-Amz-Date=20261111T200539Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host"
DISTR_KEYS=(gA-1 gA-5 gB-6 gB-7 gB-9 gB-10 gC-13 gC-14 gE-28 gE-29 gF-23 gZ-33 gZ-36 gZ-37)
DISTR_SIGS=(
  3b7ebd17da5da67f7394e2ce4958c034c54cc9e914a581f16342c9130ecb6ca0
  04bafee8737cda9636d443edfcdf6a3147d6aae55bf0a0321f49de5d5318eaef
  6144f81c3efa5f2365cdb0c609cb9332d2be07f33cd1950e9fb07a3d61a00565
  3bdce385a864addb19edfb3624586c305bb4b662757746f448f774b89984d833
  9c5f93d88d57562dffcd071e2c2e890a9b52a426f0b76c4b672d43e015b7c250
  12bed1f702f4691b7adde98be3ef9911743b9fefc93b964331d954695a8e537f
  b30a95a295df4c075bdf76d517bc9b582e6cee2e9920909c80b52718e443dd1d
  57155eb373477aaf5e4a6347485484368ba4c45514a6e460f8915624b85d41c0
  daa23a84f608cfb8ba3883681bed02251990b3b61f2a80427451bdddb807b47b
  289f2492c7cd893eb2309b153a7918c1171171244e08842daf7eb990bf338e7e
  728fbfb44d70fbd715533f260d58c094d39c84c416097e0e36e769abc736e66b
  55d8940274da7ce2be7c1284c962279e85ce0b43faae86735b9261ee99573074
  c3859db50fd8934db42c42c25bcb1a208e1c04a961a59b1dfded206588774095
  014d26018cb5584a2567aee1b1ea5f6b97f25acf45fe25d0c10370588f663954
)
for i in "${!DISTR_KEYS[@]}"; do
  key="${DISTR_KEYS[$i]}"
  sig="${DISTR_SIGS[$i]}"
  download "${DISTR_BASE}/dmd-dataset-distraction-${key}.tar.gz?${DISTR_Q}&X-Amz-Signature=${sig}" \
    "$DEST_ROOT/distraction/dmd-dataset-distraction-${key}.tar.gz"
done

wait
echo "ALL_DOWNLOADS_DONE"
