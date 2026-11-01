"""
scripts/train_polidriving.py
End-to-End Training & Evaluation for POLIDriving Driver Risk Assessment.
Generates:
  - artifacts/fusion_model/fusion_model.joblib
  - artifacts/fusion_model/scaler.joblib
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

FEATURE_COLS = [
    "speed",
    "rpm",
    "throttle_position",
    "engine_load_value",
    "acceleration",
    "mean_hr_30s",
    "rmssd_30s",
    "sdnn_30s",
    "pnn50_30s",
]

def load_or_synthesize_polidriving(raw_path="data/raw"):
    """
    Looks for POLIDriving CSV files under data/raw or data/processed.
    If not found, generates a statistically representative training distribution
    matching real-world OBD + Whoop telemetry dynamics.
    """
    candidates = glob.glob(f"{raw_path}/**/*poli*.csv", recursive=True) + \
                 glob.glob(f"{raw_path}/**/*.csv", recursive=True)

    polidriving_file = None
    for f in candidates:
        if "vzcrash" not in f.lower() and "recording" not in f.lower():
            polidriving_file = f
            break

    if polidriving_file and os.path.exists(polidriving_file):
        print(f"📖 Loading POLIDriving dataset from: {polidriving_file}")
        df = pd.read_csv(polidriving_file)

        # Normalize column aliases
        col_map = {
            "vehicle_speed": "speed",
            "Speed": "speed",
            "Engine_RPM": "rpm",
            "RPM": "rpm",
            "throttle": "throttle_position",
            "Throttle": "throttle_position",
            "load": "engine_load_value",
            "Engine_Load": "engine_load_value",
            "accel": "acceleration",
            "accel_obd": "acceleration",
            "hr": "mean_hr_30s",
            "heart_rate": "mean_hr_30s",
        }
        df = df.rename(columns=col_map)

        # Impute missing columns with baseline driving medians
        for col in FEATURE_COLS:
            if col not in df.columns:
                if "hr" in col:
                    df[col] = 75.0
                elif col in ["rmssd_30s", "sdnn_30s"]:
                    df[col] = 40.0
                elif col == "pnn50_30s":
                    df[col] = 15.0
                else:
                    df[col] = 0.0

        if "risk_label" not in df.columns:
            # Derived risk target based on kinematics & stress if unlabelled
            # 0: Low Risk, 1: High/Elevated Risk
            df["risk_label"] = (
                (df["speed"] > 110.0) |
                (df["throttle_position"] > 75.0) |
                (df["acceleration"].abs() > 3.0) |
                (df["mean_hr_30s"] > 105.0)
            ).astype(int)

        X = df[FEATURE_COLS]
        y = df["risk_label"]
        return X, y

    print("⚠️ POLIDriving CSV not found in data/raw. Synthesizing naturalistic driving training dataset...")
    np.random.seed(42)
    n_samples = 15000

    # Normal / Low-Risk Driving Profile
    speed_low = np.random.normal(60.0, 15.0, int(n_samples * 0.70)).clip(0, 100)
    rpm_low = (speed_low * 30 + np.random.normal(800, 150, len(speed_low))).clip(700, 3500)
    throttle_low = np.random.normal(20.0, 8.0, len(speed_low)).clip(5, 45)
    load_low = np.random.normal(30.0, 10.0, len(speed_low)).clip(10, 60)
    accel_low = np.random.normal(0.0, 0.6, len(speed_low)).clip(-2.0, 2.0)
    hr_low = np.random.normal(72.0, 6.0, len(speed_low)).clip(55, 90)
    rmssd_low = np.random.normal(48.0, 8.0, len(speed_low)).clip(30, 80)
    sdnn_low = np.random.normal(45.0, 7.0, len(speed_low)).clip(25, 75)
    pnn50_low = np.random.normal(18.0, 4.0, len(speed_low)).clip(5, 40)
    y_low = np.zeros(len(speed_low), dtype=int)

    # Aggressive / Elevated-Risk Driving Profile
    speed_high = np.random.normal(110.0, 20.0, int(n_samples * 0.30)).clip(40, 180)
    rpm_high = (speed_high * 38 + np.random.normal(1200, 300, len(speed_high))).clip(2000, 6500)
    throttle_high = np.random.normal(70.0, 15.0, len(speed_high)).clip(40, 100)
    load_high = np.random.normal(75.0, 12.0, len(speed_high)).clip(40, 100)
    accel_high = np.random.normal(-1.2, 1.8, len(speed_high)).clip(-5.5, 4.0)
    hr_high = np.random.normal(105.0, 14.0, len(speed_high)).clip(85, 155)
    rmssd_high = np.random.normal(22.0, 5.0, len(speed_high)).clip(8, 35)
    sdnn_high = np.random.normal(24.0, 6.0, len(speed_high)).clip(10, 38)
    pnn50_high = np.random.normal(6.0, 2.5, len(speed_high)).clip(1, 15)
    y_high = np.ones(len(speed_high), dtype=int)

    data = {
        "speed": np.concatenate([speed_low, speed_high]),
        "rpm": np.concatenate([rpm_low, rpm_high]),
        "throttle_position": np.concatenate([throttle_low, throttle_high]),
        "engine_load_value": np.concatenate([load_low, load_high]),
        "acceleration": np.concatenate([accel_low, accel_high]),
        "mean_hr_30s": np.concatenate([hr_low, hr_high]),
        "rmssd_30s": np.concatenate([rmssd_low, rmssd_high]),
        "sdnn_30s": np.concatenate([sdnn_low, sdnn_high]),
        "pnn50_30s": np.concatenate([pnn50_low, pnn50_high]),
    }
    X = pd.DataFrame(data)[FEATURE_COLS]
    y = np.concatenate([y_low, y_high])
    return X, y

def main():
    print("=" * 70)
    print("🚀 TRAINING POLIDRIVING RISK FUSION MODEL")
    print("=" * 70)

    X, y = load_or_synthesize_polidriving()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest / Histogram GBDT for robust probabilistic risk calibration
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print("\n📊 MODEL EVALUATION RESULTS (Test Set):")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Low Risk (0)", "High Risk (1)"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save artifacts into the exact directory needed by run_advanced_terminal.py
    out_dir = "artifacts/fusion_model"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "fusion_model.joblib")
    scaler_path = os.path.join(out_dir, "scaler.joblib")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n✅ Artifacts successfully written:")
    print(f"   - {model_path}")
    print(f"   - {scaler_path}")

if __name__ == "__main__":
    main()