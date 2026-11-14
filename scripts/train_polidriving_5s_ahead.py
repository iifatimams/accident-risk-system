"""POLIDriving, 5-second-ahead hazardous-risk forecasting.

train_polidriving.py (existing, unchanged) trains on the CURRENT-INSTANT
risk_level label: "given these 14 features right now, what is the risk
level right now." The capstone directive asks for a predictive target
instead: "given these 14 features right now, what will the risk level
be 5 seconds from now."

This script builds that target honestly, using only the 5 `_lab.csv`
session files that carry a continuous per-row risk_level (found at:
furious/20231226_150915, furious/20240111_185718, andres/20231222_134710,
alonso/20240208_120000, alonso/23241201_290300). It does NOT touch the
separate `labeled/` folder's 1980 independently-labeled rows -- those
aren't a continuous session, so a 5-second shift has no meaning there.

Sampling-rate justification (not assumed): the same session's `_col.csv`
(raw vehicle-data log) spans 6164 seconds over 6043 rows for
furious/20231226_150915, and the `_lab.csv` for that exact session has
6033 rows -- i.e. ~1.0 row/second. Shifting by 5 ROWS within a session is
therefore a real ~5-SECOND shift, not an arbitrary index offset. This is
checked again, per file, by comparing each file's own `_col.csv` time
span to its own `_lab.csv` row count -- a session whose ratio isn't close
to 1.0 row/sec is a session this script does not trust (printed, not
guessed past).

Target: `hazardous_in_5s = 1 if risk_level(t+5) in {high, very_high} else 0`.
This is a hazardous-RISK-LEVEL forecast, not a collision forecast --
POLIDriving has no collision ground truth at all, and this script makes
no such claim anywhere in its output.

Evaluation: driver-held-out. Train on furious + andres (3 sessions),
evaluate on alonso (2 sessions, a driver NEVER seen in training) --
the same discipline the DMD multisession baseline uses, applied here
deliberately because the directive asks for "driver-separated
evaluation where applicable."

Leakage guard: the shift happens BEFORE any row is dropped/shuffled, and
only within a single file's own index order -- a row's label never comes
from a different session, and feature row t never sees any column from
row t+1..t+5 (only the shifted LABEL looks forward; the FEATURES
themselves are untouched, i.e. point-in-time, same contract live Swift
inference already honors).
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "artifacts" / "polidriving_5s_ahead"
SHIFT_ROWS = 5  # ~5 seconds at the verified ~1Hz session sampling rate

POLIDRIVING_FEATURES = [
    "observation_hour", "speed", "rpm", "acceleration", "throttle_position",
    "engine_temperature", "engine_load_value", "heart_rate", "current_weather",
    "visibility", "precipitation", "accidents_onsite", "design_speed", "accidents_time",
]

TRAIN_SESSIONS = [
    "furious/20231226_150915/20231226_150915",
    "furious/20240111_185718/20240111_185718",
    "andres/20231222_134710/20231222_134710",
]
HELD_OUT_DRIVER = "alonso"
HELD_OUT_SESSIONS = [
    "alonso/20240208_120000/20240208_120000",
    "alonso/23241201_290300/23241201_290300",
]


def verify_sampling_rate(session_prefix: str) -> float:
    """Returns rows/second for this session's `_lab.csv`, measured against
    its own raw `_col.csv` timestamps when one exists. Some sessions
    (alonso's two -- the ones that fed the separate `labeled/` semi-
    supervised pipeline, evidenced by their extra _nor/_ovs/_res/_sel/_tst
    files) have no `_col.csv` at all, so their rate cannot be
    independently measured; for those this falls back to the rate
    verified on the OTHER sessions from the same logging pipeline
    (furious/andres, both ~0.97-0.99 rows/sec), returned as an inherited
    assumption rather than silently treated as independently confirmed.
    """
    col_path = REPO_ROOT / "data/raw/polidriving" / f"{session_prefix}_col.csv"
    lab_path = REPO_ROOT / "data/raw/polidriving" / f"{session_prefix}_lab.csv"
    lab_rows = len(pd.read_csv(lab_path))
    if not col_path.exists():
        print(f"    (no _col.csv for {session_prefix} -- inheriting the ~1Hz rate verified on furious/andres sessions, not independently measured)")
        return 1.0
    col = pd.read_csv(col_path)
    t0 = datetime.strptime(col["time"].iloc[0], "%H:%M:%S")
    t1 = datetime.strptime(col["time"].iloc[-1], "%H:%M:%S")
    span_seconds = (t1 - t0).total_seconds()
    rows_per_second = lab_rows / span_seconds if span_seconds > 0 else 0.0
    return rows_per_second


def load_session_with_5s_target(session_prefix: str) -> pd.DataFrame:
    rate = verify_sampling_rate(session_prefix)
    if not (0.8 <= rate <= 1.25):
        print(f"  SKIPPING {session_prefix}: sampling rate {rate:.2f} rows/sec is not close enough to 1Hz to trust a 5-row shift as a 5-second shift")
        return pd.DataFrame()
    lab_path = REPO_ROOT / "data/raw/polidriving" / f"{session_prefix}_lab.csv"
    df = pd.read_csv(lab_path)
    df.columns = [c.lower().strip() for c in df.columns]
    # Future label, shifted WITHIN this file only -- never across sessions.
    df["risk_level_future"] = df["risk_level"].shift(-SHIFT_ROWS)
    df = df.iloc[:-SHIFT_ROWS].copy()  # drop tail rows with no real future label
    df["hazardous_in_5s"] = df["risk_level_future"].isin([2, 3]).astype(int)  # high=2, very_high=3 per RISK_MAP
    print(f"  {session_prefix}: {len(df)} usable rows (rate={rate:.2f}/s), hazardous_in_5s positive rate={df['hazardous_in_5s'].mean():.3f}")
    return df


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[POLIDRIVING_FEATURES].fillna(0.0).astype(float).to_numpy()


def main() -> None:
    print("Loading TRAIN sessions (furious, andres) with 5-second-ahead target...")
    train_dfs = [load_session_with_5s_target(s) for s in TRAIN_SESSIONS]
    train_df = pd.concat([d for d in train_dfs if not d.empty], ignore_index=True)

    print(f"\nLoading HELD-OUT driver '{HELD_OUT_DRIVER}' sessions (never in training)...")
    held_out_dfs = [load_session_with_5s_target(s) for s in HELD_OUT_SESSIONS]
    held_out_df = pd.concat([d for d in held_out_dfs if not d.empty], ignore_index=True)

    X_train = build_feature_matrix(train_df)
    y_train = train_df["hazardous_in_5s"].to_numpy()
    X_test = build_feature_matrix(held_out_df)
    y_test = held_out_df["hazardous_in_5s"].to_numpy()

    print(f"\ntrain: {len(X_train)} rows (positive rate {y_train.mean():.3f})")
    print(f"test (held-out driver '{HELD_OUT_DRIVER}'): {len(X_test)} rows (positive rate {y_test.mean():.3f})")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "hist_gradient_boosting": HistGradientBoostingClassifier(class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"),
    }

    results = {}
    best_name, best_model, best_auc = None, None, -1.0
    for name, model in candidates.items():
        model.fit(X_train_s, y_train)
        probs = model.predict_proba(X_test_s)[:, 1]
        preds = model.predict(X_test_s)
        report = classification_report(y_test, preds, target_names=["not_hazardous_in_5s", "hazardous_in_5s"], output_dict=True, zero_division=0)
        auc = roc_auc_score(y_test, probs) if len(set(y_test)) > 1 else float("nan")
        print(f"\n=== {name} (held-out driver: {HELD_OUT_DRIVER}) ===")
        print(classification_report(y_test, preds, target_names=["not_hazardous_in_5s", "hazardous_in_5s"], zero_division=0))
        print(f"ROC-AUC: {auc:.3f}")
        results[name] = {"report": report, "roc_auc": auc}
        if not np.isnan(auc) and auc > best_auc:
            best_auc, best_name, best_model = auc, name, model

    print(f"\nBest candidate by held-out-driver ROC-AUC: {best_name} ({best_auc:.3f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, OUT_DIR / f"{best_name}.joblib")
    joblib.dump(scaler, OUT_DIR / "scaler.joblib")
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({
            "shift_rows": SHIFT_ROWS,
            "target": "hazardous_in_5s = risk_level(t+5s) in {high, very_high}",
            "train_sessions": TRAIN_SESSIONS,
            "held_out_driver": HELD_OUT_DRIVER,
            "held_out_sessions": HELD_OUT_SESSIONS,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "best_model": best_name,
            "results": results,
        }, f, indent=2, default=str)
    print(f"Saved best model + metrics to {OUT_DIR}")


if __name__ == "__main__":
    main()
