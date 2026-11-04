"""
scripts/train_vzcrash.py
End-to-End Training & Evaluation for VZCrash Accident & Distractor Detection
using the official Hugging Face dataset: vzc-research-chapter/VZCrash.
"""

# pylint: disable=invalid-name

import os
from datasets import load_dataset
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BENCHMARK_COLS = ["effective_accel", "gyro_mag", "speed_kmh", "abs_accel_obd"]


def extract_signal_features(row):
    """
    Extracts peak acceleration and gyro dynamics from VZCrash time-series signals.
    """
    # 1. Parse G-Sensor (Acceleration)
    g_data = row.get("gsensor")
    peak_accel = 9.80665
    if g_data is not None:
        try:
            arr = np.array(g_data, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                mags = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
                peak_accel = float(np.max(mags))
            elif arr.ndim == 1 and len(arr) > 0:
                peak_accel = float(np.max(np.abs(arr)))
        except (ValueError, TypeError, IndexError):
            pass

    # 2. Parse Gyroscope
    gyro_data = row.get("gyro")
    peak_gyro = 0.0
    if gyro_data is not None:
        try:
            arr = np.array(gyro_data, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                mags = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
                peak_gyro = float(np.max(mags))
            elif arr.ndim == 1 and len(arr) > 0:
                peak_gyro = float(np.max(np.abs(arr)))
        except (ValueError, TypeError, IndexError):
            pass

    # 3. Parse Speed
    speed_raw = row.get("gps_speed")
    speed_val = 0.0
    speed_delta = 0.0
    if speed_raw is not None:
        try:
            arr = np.array(speed_raw, dtype=float)
            if arr.ndim == 1 and len(arr) > 0:
                speed_val = float(np.mean(arr))
                speed_delta = float(np.max(arr) - np.min(arr))
            else:
                speed_val = float(speed_raw)
        except (ValueError, TypeError, IndexError):
            pass

    return peak_accel, peak_gyro, speed_val, speed_delta


def fetch_and_prepare_vzcrash():
    """
    Fetches VZCrash dataset from Hugging Face and extracts peak kinematic features.
    """
    token = os.environ.get("HF_TOKEN")
    print("  Fetching 'vzc-research-chapter/VZCrash' from Hugging Face Hub...")

    ds = load_dataset("vzc-research-chapter/VZCrash", token=token)
    splits = [ds[split].to_pandas() for split in ds.keys()]
    df = pd.concat(splits, ignore_index=True)
    print(f"  Loaded {len(df)} records. Extracting signal features...")

    df = df.rename(columns={col: col.lower().strip() for col in df.columns})

    sample_df = df.sample(n=min(len(df), 40000), random_state=42)
    y_sample = sample_df["label"].to_numpy().astype(int)

    records = []
    for _, row in sample_df.iterrows():
        p_acc, p_gyr, spd, spd_d = extract_signal_features(row)
        records.append({
            "effective_accel": p_acc,
            "gyro_mag": p_gyr,
            "speed_kmh": spd,
            "abs_accel_obd": spd_d,
        })

    X = pd.DataFrame(records)[BENCHMARK_COLS].fillna(0.0)
    return X, y_sample


def main():
    """Main training execution entry point for VZCrash model."""
    print("=" * 70)
    print("  TRAINING VZCRASH 3-CLASS DETECTOR (REAL TIME-SERIES EXTRACTION)")
    print("=" * 70)

    X, y = fetch_and_prepare_vzcrash()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n  Fitting Classifier with Class Balancing...")
    clf = HistGradientBoostingClassifier(
        max_iter=150,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  MODEL EVALUATION RESULTS (Test Set - Accuracy: {acc * 100:.2f}%):")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Crash (0)", "Near-Miss (1)", "Normal (2)"],
            zero_division=0,
        )
    )
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n  Fitting Isolation Forest on Normal Driving Baselines...")
    normal_data_scaled = X_train_scaled[y_train == 2]
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(normal_data_scaled)

    out_dir = "artifacts/benchmark_models"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "vzcrash_xgb.joblib")
    scaler_path = os.path.join(out_dir, "benchmark_scaler.joblib")
    iso_path = os.path.join(out_dir, "unsupervised_iso.joblib")

    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(iso, iso_path)

    print(f"\n  Balanced Artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
