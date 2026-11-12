"""Evidence-gating baseline for genuine 5-second-ahead collision risk prediction.

Round 3: directly investigates the round-2 finding that the model partly keys
off absolute speed context (crash clips run faster throughout, even in their
"safe" early seconds) rather than genuine pre-event escalation. Fixes two
methodology issues along the way: thresholds were previously chosen on the
same split they were reported on (optimistic), and false-warning counting did
not model a real alert-suppression policy.

--feature-set options (see build_future_risk_dataset.py for exact columns):
  original9        the original 9 features (comparability with prior rounds).
  expanded          original9 + causal temporal features + speed-relative
                    ratios + speed_std -- the "full model."
  speed_only        ONLY speed_kmh/speed_delta_kmh/speed_std. A trivial
                    "is this a fast context" baseline -- if this does nearly as
                    well as "expanded", the full model isn't adding much beyond
                    speed.
  no_speed          expanded with every speed-derived feature removed (no
                    absolute speed, no speed-relative ratios) -- pure
                    accel/gyro kinematics. Measures how much remains once
                    speed is removed entirely.
  speed_relative    no_speed + the speed-relative ratios (accel_per_speed etc,
                    current-window ratios, never whole-clip stats) but NOT the
                    absolute speed_kmh/speed_delta_kmh/speed_std. Tests whether
                    normalizing by current speed (instead of dropping it)
                    recovers useful signal without the confound.

--protocol collision_only (default, primary) drops near_miss entirely -- see
round-2 finding that near_miss has no kinematic signature distinguishable from
normal driving. --protocol hazardous_broad is kept only as an explicitly
exploratory, separately-reported comparison.

Evaluation methodology:
  1. train.parquet is split by event_id (grouped, no leakage) into train_fit
     (80%) and train_thresh (20%). Models fit on train_fit only.
  2. ALL thresholds (both the max-F1 point and the fixed false-warning-budget
     operating points) are chosen on train_thresh -- never on validation.
     validation.parquet is touched only once, to report final numbers at
     thresholds that were already fixed before it was touched.
  3. False warnings are counted with a COOLDOWN policy, not a bare per-window
     rate: once a warning fires in a clip, further warnings in that same clip
     are suppressed for COOLDOWN_SEC seconds regardless of score fluctuation.
     This is a simple, explicit, stated alerting policy, not the only one that
     could be chosen.
  4. Event-level recall and lead time are reported AT each fixed false-warning
     budget, not only at the threshold that maximizes window-level F1.
  5. Speed-band breakdown and a speed-matched evaluation test whether the full
     model has power within a single speed band (comparable context), not just
     across the whole range where speed alone would separate crash from normal.

No-warning baseline (never fires): 0 false warnings/hour, 0% event recall, by
construction -- the floor every model here is compared against.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

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
NON_SPEED_KINEMATIC_COLS = [c for c in ORIGINAL_FEATURE_COLS if c not in ABSOLUTE_SPEED_COLS] + TEMPORAL_FEATURE_COLS

FEATURE_SETS = {
    "original9": ORIGINAL_FEATURE_COLS,
    "expanded": ORIGINAL_FEATURE_COLS + TEMPORAL_FEATURE_COLS + ["speed_std"] + SPEED_RELATIVE_COLS,
    "speed_only": ABSOLUTE_SPEED_COLS,
    "no_speed": NON_SPEED_KINEMATIC_COLS,
    "speed_relative": NON_SPEED_KINEMATIC_COLS + SPEED_RELATIVE_COLS,
}

STEP_SEC_BY_KIND = {"normal_clip": 1.5, "pre_event_same_clip": 1.0}
COOLDOWN_SEC = 5.0  # once a warning fires, suppress further warnings in the same clip for this long
FA_BUDGETS_PER_HOUR = (1.0, 5.0, 10.0, 20.0, 50.0)
SPEED_BANDS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 250)]
MIN_BAND_SUPPORT = 150


def apply_protocol(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    if protocol == "collision_only":
        return df[df["source_label"] != "near_miss"].copy()
    return df.copy()


def grouped_split(df: pd.DataFrame, fit_frac: float = 0.8, seed: int = 123):
    """Split by event_id so no clip contributes rows to both halves."""
    ids = df["event_id"].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ids)
    cut = int(len(shuffled) * fit_frac)
    fit_ids, thresh_ids = set(shuffled[:cut]), set(shuffled[cut:])
    return df[df["event_id"].isin(fit_ids)].copy(), df[df["event_id"].isin(thresh_ids)].copy()


class CooldownCounter:
    """Counts cooldown-suppressed warnings for many different hit patterns over
    the SAME fixed set of clips, without redoing sort/groupby per call -- the
    expensive per-row work (sort, group boundaries) happens once in __init__;
    each call() is then just a numpy-indexed scan."""

    def __init__(self, df_kind: pd.DataFrame, cooldown_sec: float):
        order = np.lexsort((df_kind["decision_time_sec"].to_numpy(), df_kind["event_id"].to_numpy()))
        self.order = order
        self.sorted_event_id = df_kind["event_id"].to_numpy()[order]
        self.sorted_time = df_kind["decision_time_sec"].to_numpy()[order]
        change = np.nonzero(np.diff(self.sorted_event_id) != 0)[0] + 1
        self.bounds = np.concatenate(([0], change, [len(self.sorted_event_id)]))
        self.cooldown_sec = cooldown_sec
        self.n = len(df_kind)

    def count(self, hit: np.ndarray) -> int:
        sorted_hit = hit[self.order]
        warnings = 0
        for s, e in zip(self.bounds[:-1], self.bounds[1:], strict=True):
            last_t = -1e18
            for i in range(s, e):
                if sorted_hit[i] == 1 and (self.sorted_time[i] - last_t) >= self.cooldown_sec:
                    warnings += 1
                    last_t = self.sorted_time[i]
        return warnings


def false_warnings_per_hour(df: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict:
    hit = (scores >= threshold).astype(int)
    tmp = df.assign(_hit=hit)
    out = {}
    for kind, group in tmp[tmp["target"] == 0].groupby("negative_kind", dropna=False):
        step = STEP_SEC_BY_KIND.get(str(kind))
        if step is None:
            continue
        warnings = CooldownCounter(group, COOLDOWN_SEC).count(group["_hit"].to_numpy())
        observed_seconds = len(group) * step
        out[str(kind)] = {
            "n_windows": int(len(group)),
            "observed_hours": round(observed_seconds / 3600, 4),
            "warning_events_with_cooldown": warnings,
            "false_warnings_per_hour": (warnings / observed_seconds * 3600) if observed_seconds > 0 else None,
        }
    return out


def choose_thresholds_on_holdout(thresh_df: pd.DataFrame, thresh_scores: np.ndarray) -> dict:
    """All threshold selection happens here, on train_thresh only -- never on validation."""
    y = thresh_df["target"].to_numpy()
    grid = np.unique(np.quantile(thresh_scores, np.linspace(0.0, 1.0, 300)))

    best_f1, best_f1_thr = -1.0, float(grid[len(grid) // 2]) if len(grid) else 0.5
    for thr in grid:
        hit = (thresh_scores >= thr).astype(int)
        tp = int(((hit == 1) & (y == 1)).sum())
        fp = int(((hit == 1) & (y == 0)).sum())
        fn = int(((hit == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_f1, best_f1_thr = f1, float(thr)

    normal_mask = (thresh_df["negative_kind"] == "normal_clip").to_numpy()
    normal_thresh = thresh_df[normal_mask]
    normal_scores = thresh_scores[normal_mask]
    normal_seconds = len(normal_thresh) * STEP_SEC_BY_KIND["normal_clip"]
    counter = CooldownCounter(normal_thresh, COOLDOWN_SEC)

    # Compute each threshold's false-alarm rate exactly ONCE, reuse for every budget.
    fa_by_threshold = []
    for thr in grid:
        hit = (normal_scores >= thr).astype(int)
        warnings = counter.count(hit)
        fa_rate = (warnings / normal_seconds * 3600) if normal_seconds > 0 else float("inf")
        fa_by_threshold.append((float(thr), fa_rate))

    budget_thresholds = {}
    for target in FA_BUDGETS_PER_HOUR:
        candidates = [thr for thr, fa in fa_by_threshold if fa <= target]
        budget_thresholds[target] = min(candidates) if candidates else None

    return {"max_f1_threshold": best_f1_thr, "max_f1_value": best_f1, "budget_thresholds_per_hour": budget_thresholds}


def event_level_recall(df: pd.DataFrame, hit: np.ndarray) -> float:
    tmp = df.assign(_hit=hit)
    positives = tmp[tmp["target"] == 1]
    if positives["event_id"].nunique() == 0:
        return float("nan")
    return float(positives.groupby("event_id")["_hit"].max().mean())


def false_positive_rate_by_kind(df: pd.DataFrame, hit: np.ndarray) -> dict:
    tmp = df.assign(_hit=hit)
    out = {}
    for kind, group in tmp[tmp["target"] == 0].groupby("negative_kind", dropna=False):
        out[str(kind)] = {"n": int(len(group)), "false_positive_rate": float(group["_hit"].mean())}
    return out


def earliest_detection_lead_time(df: pd.DataFrame, hit: np.ndarray) -> dict:
    tmp = df.assign(_hit=hit)
    positives = tmp[tmp["target"] == 1].copy()
    if len(positives) == 0:
        return {"n_positive_events": 0}
    detected = positives[positives["_hit"] == 1]
    per_event_best_lead = detected.groupby("event_id")["lead_time_sec"].max()
    n_positive_events = positives["event_id"].nunique()
    if len(per_event_best_lead) == 0:
        return {"n_positive_events": int(n_positive_events), "n_detected_events": 0, "detection_rate": 0.0}
    return {
        "n_positive_events": int(n_positive_events),
        "n_detected_events": int(len(per_event_best_lead)),
        "detection_rate": float(len(per_event_best_lead) / n_positive_events),
        "mean_sec": float(per_event_best_lead.mean()),
        "median_sec": float(per_event_best_lead.median()),
        "min_sec": float(per_event_best_lead.min()),
        "max_sec": float(per_event_best_lead.max()),
    }


def speed_band_breakdown(df: pd.DataFrame, scores: np.ndarray, threshold: float) -> list[dict]:
    out = []
    speed = df["speed_kmh"].to_numpy()
    for lo, hi in SPEED_BANDS:
        mask = (speed >= lo) & (speed < hi)
        if mask.sum() == 0:
            continue
        band_df, band_scores = df[mask], scores[mask]
        y = band_df["target"].to_numpy()
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        entry = {"speed_band_kmh": f"{lo}-{hi}", "n_pos": n_pos, "n_neg": n_neg}
        if n_pos >= MIN_BAND_SUPPORT and n_neg >= MIN_BAND_SUPPORT:
            entry["sufficient_support"] = True
            entry["pr_auc"] = float(average_precision_score(y, band_scores)) if len(set(y)) > 1 else None
            hit = (band_scores >= threshold).astype(int)
            tp = int(((hit == 1) & (y == 1)).sum())
            fp = int(((hit == 1) & (y == 0)).sum())
            entry["precision"] = tp / (tp + fp) if (tp + fp) else 0.0
            entry["recall"] = tp / n_pos if n_pos else 0.0
        else:
            entry["sufficient_support"] = False
        out.append(entry)
    return out


def run_at_operating_points(df: pd.DataFrame, scores: np.ndarray, thresholds: dict, label: str) -> dict:
    y = df["target"].to_numpy()
    pr_auc = float(average_precision_score(y, scores))
    result = {"operating_point_label": label, "n_rows": int(len(df)), "pr_auc": pr_auc, "by_threshold": {}}

    points = {"max_f1": thresholds["max_f1_threshold"]}
    for target, thr in thresholds["budget_thresholds_per_hour"].items():
        points[f"fa_budget_{target}_per_hour"] = thr

    for name, thr in points.items():
        if thr is None:
            result["by_threshold"][name] = {"threshold": None, "note": "budget unreachable on train_thresh"}
            continue
        hit = (scores >= thr).astype(int)
        report = classification_report(
            y, hit, target_names=["no_event_5s", "event_within_5s"], output_dict=True, zero_division=0
        )
        result["by_threshold"][name] = {
            "threshold": float(thr),
            "confusion_matrix": confusion_matrix(y, hit).tolist(),
            "window_level_report": report,
            "event_level_recall": event_level_recall(df, hit),
            "false_positive_rate_by_negative_kind": false_positive_rate_by_kind(df, hit),
            "false_warnings_per_hour_by_negative_kind": false_warnings_per_hour(df, scores, thr),
            "earliest_detection_lead_time_sec": earliest_detection_lead_time(df, hit),
        }
    result["speed_band_breakdown"] = speed_band_breakdown(df, scores, thresholds["max_f1_threshold"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default="data/processed/future_risk_vzcrash_v3")
    parser.add_argument("--protocol", choices=["collision_only", "hazardous_broad"], default="collision_only")
    parser.add_argument("--feature-set", choices=list(FEATURE_SETS), default="expanded")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--models", nargs="+", default=["logistic_regression", "random_forest", "gradient_boosting"])
    args = parser.parse_args()

    feature_cols = FEATURE_SETS[args.feature_set]
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir or f"artifacts/future_risk_baseline_v3/{args.protocol}_{args.feature_set}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = apply_protocol(pd.read_parquet(dataset_dir / "train.parquet"), args.protocol)
    val_df = apply_protocol(pd.read_parquet(dataset_dir / "validation.parquet"), args.protocol)
    train_fit, train_thresh = grouped_split(train_df)

    print(f"protocol={args.protocol} feature_set={args.feature_set} ({len(feature_cols)} features: {feature_cols})")
    print(f"train_fit: {len(train_fit)} rows, train_thresh: {len(train_thresh)} rows, validation: {len(val_df)} rows")
    print(f"train_fit target counts: {train_fit['target'].value_counts().to_dict()}")
    print(f"validation target counts: {val_df['target'].value_counts().to_dict()}")

    scaler = StandardScaler()
    X_fit = scaler.fit_transform(train_fit[feature_cols])
    y_fit = train_fit["target"].to_numpy()
    X_thresh = scaler.transform(train_thresh[feature_cols])
    X_val = scaler.transform(val_df[feature_cols])

    available_models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=42),
    }

    all_results = []
    for name in args.models:
        model = available_models[name]
        print(f"\n=== {name} ===")
        model.fit(X_fit, y_fit)

        thresh_scores = model.predict_proba(X_thresh)[:, 1]
        thresholds = choose_thresholds_on_holdout(train_thresh, thresh_scores)
        print("thresholds chosen on train_thresh (NOT validation):", json.dumps(thresholds, indent=2))

        val_scores = model.predict_proba(X_val)[:, 1]
        result = run_at_operating_points(val_df, val_scores, thresholds, label=f"{name}_validation")
        result["model"] = name
        result["thresholds_source"] = "train_thresh (20% of train, grouped by event_id, held out from fit)"
        all_results.append(result)
        print(json.dumps(result, indent=2)[:3000], "... (truncated in stdout, full JSON on disk)")
        joblib.dump(model, out_dir / f"{name}.joblib")

    joblib.dump(scaler, out_dir / "scaler.joblib")
    summary = {
        "protocol": args.protocol, "feature_set": args.feature_set, "feature_cols": feature_cols,
        "dataset_dir": str(dataset_dir),
        "train_fit_rows": len(train_fit), "train_thresh_rows": len(train_thresh), "validation_rows": len(val_df),
        "cooldown_sec": COOLDOWN_SEC, "fa_budgets_per_hour": list(FA_BUDGETS_PER_HOUR),
        "no_warning_baseline": "0 false warnings/hour, 0% event recall, by construction",
        "results": all_results,
    }
    with open(out_dir / "validation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved models and validation_results.json to {out_dir}")


if __name__ == "__main__":
    main()
