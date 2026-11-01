"""
scripts/train_vzcrash.py
End-to-End Training & Evaluation for VZCrash Accident & Distractor Detection.
Generates:
  - artifacts/benchmark_models/vzcrash_xgb.joblib
  - artifacts/benchmark_models/benchmark_scaler.joblib
  - artifacts/benchmark_models/unsupervised_iso.joblib
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Features matching run_advanced_terminal.py:
# np.array([[effective_accel, gyro_mag, speed_kmh, abs(accel_obd)]])
BENCHMARK_COLS = ["effective_accel", "gyro_mag", "speed_kmh", "abs_accel_obd"]

def load_or_synthesize_vzcrash(raw_path="data/raw"):
    """
    Loads raw VZCrash telematics CSV or synthesizes crash vs near-miss vs normal events.
    """
    candidates = glob.glob(f"{raw_path}/**/*vzcrash*.csv", recursive=True) + \
                 glob.glob(f"{raw_path}/**/*crash*.csv", recursive=True)

    vz_file = candidates[0] if candidates else None

    if vz_file and os.path.exists(vz_file):
        print(f"📖 Loading VZCrash dataset from: {vz_file}")
        df = pd.read_csv(vz_file)
        # Adapt column mappings if present
        if all(c in df.columns for c in BENCHMARK_COLS) and "target" in df.columns:
            return df[BENCHMARK_COLS], df["target"]

    print("⚠️ VZCrash CSV not found in data/raw. Synthesizing crash telematics dataset...")
    np.random.seed(42)
    n_samples = 12000

    # 1. Normal Driving & Potholes (Class 2)
    # High vertical bump/accel occasionally, but zero/low longitudinal G and stable gyro
    n_normal = int(n_samples * 0.70)
    accel_normal = np.random.normal(9.8, 1.2, n_normal).clip(7.0, 14.0)
    gyro_normal = np.random.exponential(0.15, n_normal).clip(0.0, 0.8)
    speed_normal = np.random.normal(55.0, 18.0, n_normal).clip(0.0, 120.0)
    obd_accel_normal = np.random.normal(0.0, 0.5, n_normal).clip(-1.8, 1.8)
    y_normal = np.full(n_normal, 2, dtype=int)

    # 2. Near-Miss / Hard Braking Distractor (Class 1)
    # High deceleration, sharp gyro deviation, speed dropping fast
    n_near = int(n_samples * 0.20)
    accel_near = np.random.normal(16.0, 2.5, n_near).clip(12.0, 21.5)
    gyro_near = np.random.normal(0.9, 0.3, n_near).clip(0.4, 1.8)
    speed_near = np.random.normal(45.0, 15.0, n_near).clip(10.0, 90.0)
    obd_accel_near = np.random.normal(3.8, 0.8, n_near).clip(2.5, 6.0)
    y_near = np.full(n_near, 1, dtype=int)

    # 3. True High-G Crash (Class 0)
    # Violent acceleration spike (>25 m/s²), severe angular rate, sudden speed collapse
    n_crash = int(n_samples * 0.10)
    accel_crash = np.random.normal(38.0, 8.0, n_crash).clip(24.0, 65.0)
    gyro_crash = np.random.normal(2.5, 0.8, n_crash).clip(1.2, 5.0)
    speed_crash = np.random.normal(60.0, 25.0, n_crash).clip(15.0, 140.0)
    obd_accel_crash = np.random.normal(7.5, 2.0, n_crash).clip(5.0, 14.0)
    y_crash = np.full(n_crash, 0, dtype=int)

    X = pd.DataFrame({
        "effective_accel": np.concatenate([accel_normal, accel_near, accel_crash]),
        "gyro_mag": np.concatenate([gyro_normal, gyro_near, gyro_crash]),
        "speed_kmh": np.concatenate([speed_normal, speed_near, speed_crash]),
        "abs_accel_obd": np.concatenate([obd_accel_normal, obd_accel_near, obd_accel_crash]),
    })[BENCHMARK_COLS]
    y = np.concatenate([y_normal, y_near, y_crash])
    return X, y

def main():
    print("=" * 70)
    print("🚀 TRAINING VZCRASH 3-CLASS DETECTOR & ANOMALY DETECTOR")
    print("=" * 70)

    X, y = load_or_synthesize_vzcrash()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3-Class Gradient Boosted Trees (Accident / Near-Miss / Normal)
    print("🌲 Fitting Classifier (Crash=0, Near-Miss=1, Normal=2)...")
    clf = GradientBoostingClassifier(
        n_estimators=160,
        learning_rate=0.08,
        max_depth=6,
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print("\n📊 MODEL EVALUATION RESULTS (Test Set):")
    print(f"Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Crash (0)", "Near-Miss (1)", "Normal (2)"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Unsupervised Anomaly Model (Isolation Forest for out-of-distribution events)
    print("\n🔍 Fitting Isolation Forest for telemetry anomaly scoring...")
    normal_data_scaled = X_train_scaled[y_train == 2]
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(normal_data_scaled)

    # Save artifacts into benchmark_models directory
    out_dir = "artifacts/benchmark_models"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "vzcrash_xgb.joblib")
    scaler_path = os.path.join(out_dir, "benchmark_scaler.joblib")
    iso_path = os.path.join(out_dir, "unsupervised_iso.joblib")

    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(iso, iso_path)

    print(f"\n✅ Benchmark Artifacts saved:")
    print(f"   - {model_path}")
    print(f"   - {scaler_path}")
    print(f"   - {iso_path}")

if __name__ == "__main__":
    main()