"""Build a strictly-causal "accident within the next N seconds" dataset from VZCrash.

RoadSense's core unvalidated claim is a 5-second-ahead collision risk score. VZCrash
gives whole-clip labels (crash / near_miss / normal_driving) over a 16-second window,
not a human-annotated accident-onset timestamp. Turning clip labels into a genuine
future-prediction target requires an onset estimate and a hard causal rule.

Onset estimate (IMPORTANT, read before trusting any number downstream):
  There is no verified ground-truth onset in this dataset. Four proxy definitions
  are computed and compared (see --onset-method and the onset_sensitivity.csv
  this script writes):
    peak_accel         argmax(|accel|) -- peak impact/deceleration magnitude.
    peak_jerk          argmax(|d(accel)/dt|) -- the sharpest rate-of-change, which
                        can precede the peak magnitude itself (a jerk spike often
                        marks the start of an impact, not its climax).
    threshold_crossing first sample where |accel| exceeds ACCEL_THRESHOLD_G -- the
                        earliest kinematically-detectable moment, if the clip ever
                        crosses the threshold (None otherwise -- not every clip does).
    peak_gyro          argmax(|gyro|) -- peak rotational rate, relevant for
                        swerve/evasive near-miss events rather than impacts.
  None of these is "ground truth." They are different, independently defensible
  guesses at the same unknown onset time, and they disagree with each other by an
  amount this script measures and reports, not merely asserts.

  A full-shard inspection (see docs/ or the printed summary) found that VZCrash's
  "near_miss" clips have peak acceleration and peak gyroscope statistics that are
  NOT distinguishable from "normal_driving" clips (crash: ~3.9g peak vs near_miss:
  ~1.5g vs normal: ~1.5g; gyro shows the same pattern). That means a kinematic peak
  in a near_miss clip is likely just ordinary driving noise, not a real evasive
  event -- so onset estimation, and therefore future-prediction labeling, is on much
  shakier ground for near_miss than for crash. This is exactly why this script keeps
  source_label on every row: it lets training/evaluation treat "collision_only"
  (crash vs normal) and "hazardous_broad" (crash+near_miss vs normal) as separate,
  separately-trustworthy experiments instead of one blended number.

Causal rule: every feature for a decision time t is computed ONLY from samples with
timestamp <= t. A positive example at lead time L means "the estimated onset occurs
L seconds after t" (L in (0, horizon]); features never see the onset or anything
after it. No feature uses the clip's own boundaries (e.g. "seconds until clip ends")
since a live deployment has no such boundary -- that would be a leak specific to how
this dataset happens to be clipped, not a real signal.

Negative examples, densely sampled (not single random points) so false-warning
rate can be measured per unit of observed time, and kept distinguishable by kind:
  - normal_clip: every WINDOW_SEC-respecting step through a normal_driving clip.
  - pre_event_same_clip: every step through the SAFE, pre-danger part of a crash/
    near_miss clip itself (more than horizon+safety_margin before the estimated
    onset) -- a harder, more realistic negative than a normal clip, since it's the
    same driver/session/sensor just earlier in time.

Leakage: VZCrash's schema (event_id, label, vehiclesize, gyro_is_hd, gsensor, gyro,
gps_speed) has no participant, session, or trip identifier beyond event_id -- each
row is already one independent recording. This script verifies zero event_id
overlap between the splits it builds; it cannot check for leakage at any coarser
grouping because no such identifier exists in the source data. That absence is
reported explicitly, not silently assumed away.

Output: <out-dir>/{train,validation}.parquet, a provenance.json, and
onset_sensitivity.csv (all four onset estimates per crash/near_miss clip actually
used, for the sensitivity analysis above).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

HORIZON_SEC = 5.0
LONG_WINDOW_SEC = 3.0
SHORT_WINDOW_SEC = 1.0
LEAD_TIMES_SEC = (1.0, 2.0, 3.0, 4.0, 4.5)
SAFETY_MARGIN_SEC = 1.0  # extra buffer beyond horizon before a point counts as a safe hard negative
HARD_NEG_STEP_SEC = 1.0
NORMAL_STEP_SEC = 1.5
SAMPLE_RATE_HZ = 100.0
ACCEL_THRESHOLD_G = 2.0  # chosen from observed data: crash peaks avg ~3.9g, normal/near_miss ~1.5g

LABEL_NAMES = {0: "crash", 1: "near_miss", 2: "normal"}
ONSET_METHODS = ("peak_accel", "peak_jerk", "threshold_crossing", "peak_gyro")


def accel_magnitude(gsensor) -> np.ndarray:
    arr = np.asarray(list(gsensor), dtype=float)
    return np.sqrt((arr**2).sum(axis=1))


def gyro_magnitude(gyro) -> np.ndarray:
    arr = np.asarray(list(gyro), dtype=float)
    return np.sqrt((arr**2).sum(axis=1))


def compute_onset_candidates(accel_mag: np.ndarray, gyro_mag: np.ndarray) -> dict:
    """All four onset proxies, in seconds. threshold_crossing is None if never crossed."""
    jerk = np.abs(np.diff(accel_mag)) * SAMPLE_RATE_HZ
    above = np.nonzero(accel_mag > ACCEL_THRESHOLD_G)[0]
    return {
        "peak_accel": float(np.argmax(accel_mag)) / SAMPLE_RATE_HZ,
        "peak_jerk": (float(np.argmax(jerk)) / SAMPLE_RATE_HZ) if len(jerk) else None,
        "threshold_crossing": (float(above[0]) / SAMPLE_RATE_HZ) if len(above) else None,
        "peak_gyro": float(np.argmax(gyro_mag)) / SAMPLE_RATE_HZ,
    }


def linear_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    t = np.arange(len(values)) / SAMPLE_RATE_HZ
    return float(np.polyfit(t, values, 1)[0])


def extract_causal_features(accel_mag: np.ndarray, gyro_mag: np.ndarray, gps_speed: np.ndarray, t: float):
    """Features from samples with timestamp <= t only. None if not enough history."""
    idx_hi = int(round(t * SAMPLE_RATE_HZ))
    idx_lo_long = idx_hi - int(LONG_WINDOW_SEC * SAMPLE_RATE_HZ)
    idx_lo_short = idx_hi - int(SHORT_WINDOW_SEC * SAMPLE_RATE_HZ)
    if idx_lo_long < 0 or idx_hi > len(accel_mag) or idx_hi <= idx_lo_long:
        return None

    a_long, g_long = accel_mag[idx_lo_long:idx_hi], gyro_mag[idx_lo_long:idx_hi]
    a_short, g_short = accel_mag[idx_lo_short:idx_hi], gyro_mag[idx_lo_short:idx_hi]
    accel_jerk = np.abs(np.diff(a_long)) * SAMPLE_RATE_HZ
    gyro_jerk = np.abs(np.diff(g_long)) * SAMPLE_RATE_HZ

    speed_idx_hi = min(int(t), len(gps_speed) - 1)
    speed_idx_lo = max(0, speed_idx_hi - int(LONG_WINDOW_SEC))
    speed_window = gps_speed[speed_idx_lo:speed_idx_hi + 1]
    speed_now = float(gps_speed[speed_idx_hi])
    speed_delta = speed_now - float(gps_speed[speed_idx_lo])

    accel_mean_long = float(a_long.mean())
    gyro_mean_long = float(g_long.mean())
    accel_mean_short = float(a_short.mean())
    gyro_mean_short = float(g_short.mean())
    jerk_max_val = float(accel_jerk.max()) if len(accel_jerk) else 0.0

    # Speed-relative features: a round-2 investigation found crash clips run at
    # ~68 km/h on average even in their "safe" early seconds vs ~37 km/h for
    # normal clips -- the model was partly keying off absolute speed context
    # (constant through a whole clip) rather than genuine escalation. These
    # ratios use ONLY the current window's speed_now (itself already computed
    # causally above from samples <= t) -- never a whole-clip average -- so a
    # live deployment could compute them with no knowledge of the future or of
    # the recording's overall character.
    speed_divisor = max(speed_now, SPEED_FLOOR_KMH)

    return {
        # long-window (3s) features -- the original 9-feature set lives in here.
        "accel_mean": accel_mean_long,
        "accel_std": float(a_long.std()),
        "accel_max": float(a_long.max()),
        "jerk_max": jerk_max_val,
        "gyro_mean": gyro_mean_long,
        "gyro_std": float(g_long.std()),
        "gyro_max": float(g_long.max()),
        "speed_kmh": speed_now,
        "speed_delta_kmh": speed_delta,
        # expanded causal temporal features -- all strictly from samples <= t.
        "jerk_mean": float(accel_jerk.mean()) if len(accel_jerk) else 0.0,
        "accel_trend_slope": linear_slope(a_long),
        "gyro_jerk_max": float(gyro_jerk.max()) if len(gyro_jerk) else 0.0,
        "gyro_trend_slope": linear_slope(g_long),
        "accel_mean_short": accel_mean_short,
        "accel_max_short": float(a_short.max()),
        "gyro_mean_short": gyro_mean_short,
        "gyro_max_short": float(g_short.max()),
        "accel_escalation": accel_mean_short - accel_mean_long,
        "gyro_escalation": gyro_mean_short - gyro_mean_long,
        "speed_std": float(speed_window.std()) if len(speed_window) else 0.0,
        # causal speed-relative features (current-window ratios, not whole-clip stats).
        "accel_per_speed": accel_mean_long / speed_divisor,
        "gyro_per_speed": gyro_mean_long / speed_divisor,
        "jerk_per_speed": jerk_max_val / speed_divisor,
    }


SPEED_FLOOR_KMH = 5.0  # avoids exploding ratios at near-zero speed (stationary/idling)

ORIGINAL_FEATURE_COLS = [
    "accel_mean", "accel_std", "accel_max", "jerk_max",
    "gyro_mean", "gyro_std", "gyro_max", "speed_kmh", "speed_delta_kmh",
]
TEMPORAL_FEATURE_COLS = [
    "jerk_mean", "accel_trend_slope", "gyro_jerk_max", "gyro_trend_slope",
    "accel_mean_short", "accel_max_short", "gyro_mean_short", "gyro_max_short",
    "accel_escalation", "gyro_escalation",
]
ABSOLUTE_SPEED_COLS = ["speed_kmh", "speed_delta_kmh", "speed_std"]
SPEED_RELATIVE_COLS = ["accel_per_speed", "gyro_per_speed", "jerk_per_speed"]
EXPANDED_FEATURE_COLS = ORIGINAL_FEATURE_COLS + TEMPORAL_FEATURE_COLS + ["speed_std"] + SPEED_RELATIVE_COLS


def build_rows_for_clip(event_id: int, label: int, gsensor, gyro, gps_speed, onset_method: str):
    """Returns (rows, sensitivity_record_or_None, skipped_reason_or_None)."""
    accel_mag = accel_magnitude(gsensor)
    gyro_mag_arr = gyro_magnitude(gyro)
    gps_speed = np.asarray(gps_speed, dtype=float)
    clip_duration = len(accel_mag) / SAMPLE_RATE_HZ
    rows: list[dict] = []

    if label in (0, 1):
        candidates = compute_onset_candidates(accel_mag, gyro_mag_arr)
        sensitivity_record = {"event_id": event_id, "source_label": LABEL_NAMES[label], **candidates}
        onset_time = candidates.get(onset_method)
        if onset_time is None:
            return rows, sensitivity_record, "onset_method_not_applicable"

        for lead in LEAD_TIMES_SEC:
            t = onset_time - lead
            if t < LONG_WINDOW_SEC:
                continue
            feats = extract_causal_features(accel_mag, gyro_mag_arr, gps_speed, t)
            if feats is None:
                continue
            rows.append({
                "event_id": event_id, "source_label": LABEL_NAMES[label],
                "decision_time_sec": t, "lead_time_sec": lead,
                "onset_time_sec": onset_time, "target": 1,
                "negative_kind": None, **feats,
            })

        safe_cutoff = onset_time - (HORIZON_SEC + SAFETY_MARGIN_SEC)
        t = LONG_WINDOW_SEC
        while t <= safe_cutoff:
            feats = extract_causal_features(accel_mag, gyro_mag_arr, gps_speed, t)
            if feats is not None:
                rows.append({
                    "event_id": event_id, "source_label": LABEL_NAMES[label],
                    "decision_time_sec": t, "lead_time_sec": None,
                    "onset_time_sec": onset_time, "target": 0,
                    "negative_kind": "pre_event_same_clip", **feats,
                })
            t += HARD_NEG_STEP_SEC
        return rows, sensitivity_record, None

    if clip_duration <= LONG_WINDOW_SEC:
        return rows, None, "clip_too_short"
    t = LONG_WINDOW_SEC
    while t <= clip_duration:
        feats = extract_causal_features(accel_mag, gyro_mag_arr, gps_speed, t)
        if feats is not None:
            rows.append({
                "event_id": event_id, "source_label": LABEL_NAMES[label],
                "decision_time_sec": t, "lead_time_sec": None,
                "onset_time_sec": None, "target": 0,
                "negative_kind": "normal_clip", **feats,
            })
        t += NORMAL_STEP_SEC
    return rows, None, None


def build_split(files: list[str], split_name: str, onset_method: str):
    all_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    skipped: dict[str, int] = {}
    for filename in files:
        path = hf_hub_download(repo_id="vzc-research-chapter/VZCrash", repo_type="dataset", filename=filename)
        df = pd.read_parquet(path)
        print(f"  {filename}: {len(df)} clips")
        for _, row in df.iterrows():
            rows, sensitivity, skip_reason = build_rows_for_clip(
                int(row["event_id"]), int(row["label"]), row["gsensor"], row["gyro"], row["gps_speed"], onset_method,
            )
            all_rows.extend(rows)
            if sensitivity is not None:
                sensitivity_rows.append(sensitivity)
            if skip_reason is not None:
                skipped[skip_reason] = skipped.get(skip_reason, 0) + 1
    out = pd.DataFrame(all_rows)
    out["split"] = split_name
    return out, pd.DataFrame(sensitivity_rows), skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--train-shards", nargs="+",
        default=[f"data/train-{i:05d}-of-00011.parquet" for i in range(4)],
    )
    parser.add_argument(
        "--validation-shards", nargs="+",
        default=[f"data/validation-{i:05d}-of-00003.parquet" for i in range(2)],
    )
    parser.add_argument("--onset-method", choices=ONSET_METHODS, default="peak_accel")
    parser.add_argument("--out-dir", default="data/processed/future_risk_vzcrash_v3")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    print(f"Onset method: {args.onset_method}")
    print("Building train split...")
    train_df, train_sens, train_skipped = build_split(args.train_shards, "train", args.onset_method)
    print("Building validation split...")
    val_df, val_sens, val_skipped = build_split(args.validation_shards, "validation", args.onset_method)

    overlap = set(train_df["event_id"]) & set(val_df["event_id"])
    if overlap:
        raise RuntimeError(f"Train/validation event_id overlap detected ({len(overlap)} ids) -- leakage")

    train_df.to_parquet(out_dir / "train.parquet", index=False)
    val_df.to_parquet(out_dir / "validation.parquet", index=False)
    sensitivity_all = pd.concat(
        [train_sens.assign(split="train"), val_sens.assign(split="validation")], ignore_index=True
    )
    sensitivity_all.to_csv(out_dir / "onset_sensitivity.csv", index=False)

    provenance = {
        "built_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "source_dataset": "vzc-research-chapter/VZCrash",
        "train_shards": args.train_shards,
        "validation_shards": args.validation_shards,
        "onset_method_used_for_labels": args.onset_method,
        "onset_proxy_caveat": (
            "No verified ground-truth onset exists in this dataset. Four proxy methods "
            "are computed for every crash/near_miss clip (see onset_sensitivity.csv); "
            "this run's labels use only one of them. near_miss onset estimates are "
            "especially uncertain -- full-shard inspection found near_miss peak "
            "acceleration/gyro statistics indistinguishable from normal_driving."
        ),
        "leakage_check": (
            "Verified zero event_id overlap between train/validation. VZCrash's schema "
            "has no participant/session/trip identifier beyond event_id, so leakage at "
            "any coarser grouping cannot be checked from available metadata."
        ),
        "horizon_sec": HORIZON_SEC, "long_window_sec": LONG_WINDOW_SEC, "short_window_sec": SHORT_WINDOW_SEC,
        "lead_times_sec": list(LEAD_TIMES_SEC), "safety_margin_sec": SAFETY_MARGIN_SEC,
        "hard_negative_step_sec": HARD_NEG_STEP_SEC, "normal_step_sec": NORMAL_STEP_SEC,
        "train_rows": len(train_df), "validation_rows": len(val_df),
        "train_target_counts": train_df["target"].value_counts().to_dict(),
        "validation_target_counts": val_df["target"].value_counts().to_dict(),
        "train_source_label_counts": train_df["source_label"].value_counts().to_dict(),
        "validation_source_label_counts": val_df["source_label"].value_counts().to_dict(),
        "clips_skipped": {"train": train_skipped, "validation": val_skipped},
        "build_seconds": round(time.time() - start, 1),
    }
    with open(out_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    print(f"\nWrote {out_dir}/train.parquet ({len(train_df)} rows), validation.parquet ({len(val_df)} rows)")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
