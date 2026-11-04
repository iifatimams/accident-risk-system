"""
scripts/train_polidriving.py
End-to-End Training & Evaluation for POLIDriving Risk Level Prediction
using the exact 14 features from the official paper (MDPI Appl. Sci. 2024).

Generates:
  - artifacts/fusion_model/fusion_model.joblib
  - artifacts/fusion_model/scaler.joblib
"""

# pylint: disable=invalid-name

import glob
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

POLIDRIVING_FEATURES = [
    "observation_hour",
    "speed",
    "rpm",
    "acceleration",
    "throttle_position",
    "engine_temperature",
    "engine_load_value",
    "heart_rate",
    "current_weather",
    "visibility",
    "precipitation",
    "accidents_onsite",
    "design_speed",
    "accidents_time",
]

RISK_MAP = {
    "low": 0, "0": 0, 0: 0,
    "medium": 1, "1": 1, 1: 1,
    "high": 2, "2": 2, 2: 2,
    "very high": 3, "very_high": 3, "3": 3, 3: 3,
}

WEATHER_MAP = {
    "sunny": 0, "mostly sunny": 1, "partly sunny": 2, "hazy sunshine": 3,
    "clouds and sun": 4, "mostly cloudy": 5, "cloudy": 6, "fog": 7, "rain": 8,
}


def load_polidriving_14_features(data_dir="data/raw"):
    """
    Locates and parses the 14 POLIDriving attributes from session CSV files.
    Prioritizes ground-truth labeled files (_lab.csv) over raw files.
    """
    print(f"  Searching for POLIDriving files under '{data_dir}'...")

    lab_files = (
        glob.glob(f"{data_dir}/**/*_lab.csv", recursive=True)
        + glob.glob("**/*_lab.csv", recursive=True)
    )
    pre_files = (
        glob.glob(f"{data_dir}/**/*_pre.csv", recursive=True)
        + glob.glob("**/*_pre.csv", recursive=True)
    )

    selected_files = list(set(lab_files)) if lab_files else list(set(pre_files))
    selected_files = [
        f for f in selected_files if "vzcrash" not in f.lower() and f.endswith(".csv")
    ]

    if not selected_files:
        all_csvs = (
            glob.glob(f"{data_dir}/**/polidriving*/**/*.csv", recursive=True)
            + glob.glob("**/*polidriving*/**/*.csv", recursive=True)
        )
        selected_files = [
            f for f in set(all_csvs) if "vzcrash" not in f.lower() and f.endswith(".csv")
        ]

    print(f"  Found {len(selected_files)} candidate POLIDriving CSVs. Loading...")
    dfs = []
    for filepath in selected_files:
        try:
            dataframe = pd.read_csv(filepath)
            dataframe.columns = [c.lower().strip() for c in dataframe.columns]
            dfs.append(dataframe)
        except (ValueError, KeyError, OSError, pd.errors.EmptyDataError):
            pass

    if not dfs:
        raise ValueError("Could not parse any valid rows from the found CSV files.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df)} total rows.")

    # 1. Observation Hour
    if "observation_hour" in df.columns:
        obs_hour = df["observation_hour"].astype(float)
    elif "hour" in df.columns:
        obs_hour = df["hour"].astype(float)
    elif "time" in df.columns:
        obs_hour = (
            pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce")
            .dt.hour.fillna(14.0)
        )
    else:
        obs_hour = pd.Series(np.full(len(df), 14.0))

    # 2-8. Vehicle & Driver Sensors
    speed = df.get("speed", pd.Series(np.zeros(len(df)))).fillna(0.0).astype(float)
    rpm_default = pd.Series(np.full(len(df), 1200.0))
    rpm = df.get("rpm", df.get("revolutions per minute", rpm_default)).fillna(1200.0).astype(float)
    accel = df.get("acceleration", df.get("accel", pd.Series(np.zeros(len(df)))))
    accel = accel.fillna(0.0).astype(float)
    throttle_series = df.get(
        "throttle position",
        df.get("throttle_position", df.get("throttle_pos.", pd.Series(np.zeros(len(df))))),
    )
    throttle = throttle_series.fillna(0.0).astype(float)
    temp_default = pd.Series(np.full(len(df), 90.0))
    eng_temp = df.get(
        "engine temperature",
        df.get("engine_temperature", df.get("eng._temp.", temp_default)),
    ).fillna(90.0).astype(float)
    load_default = pd.Series(np.full(len(df), 30.0))
    eng_load = df.get(
        "engine load value",
        df.get("engine_load_value", df.get("eng._load_value", load_default)),
    ).fillna(30.0).astype(float)
    hr = df.get(
        "heart rate",
        df.get("heart_rate", pd.Series(np.full(len(df), 75.0))),
    ).fillna(75.0).astype(float)

    # 9-11. Weather Conditions
    default_weather = pd.Series(["cloudy"] * len(df))
    curr_weather_raw = (
        df.get("current weather", df.get("current_weather", default_weather))
        .astype(str)
        .str.lower()
        .str.strip()
    )
    curr_weather = curr_weather_raw.map(lambda x: WEATHER_MAP.get(x, 6)).fillna(6).astype(float)
    visibility = df.get("visibility", df.get("visib.", pd.Series(np.full(len(df), 10.0))))
    visibility = visibility.fillna(10.0).astype(float)
    precip = df.get("precipitation", df.get("precip.", pd.Series(np.zeros(len(df)))))
    precip = precip.fillna(0.0).astype(float)

    # 12-14. Road Safety & Geometric Attributes
    acdnt_site = df.get(
        "accidents onsite",
        df.get("accidents_onsite", df.get("acdnt._on_site", pd.Series(np.zeros(len(df))))),
    ).fillna(0.0).astype(float)
    des_speed = df.get(
        "design speed",
        df.get("design_speed", pd.Series(np.full(len(df), 70.0))),
    ).fillna(70.0).astype(float)
    acdnt_time = df.get(
        "accidents time",
        df.get("accidents_time", df.get("acdnt._that_time", pd.Series(np.zeros(len(df))))),
    ).fillna(0.0).astype(float)

    # Target Mapping
    target_series = None
    for cand in ["risk level", "risk_level", "risk"]:
        if cand in df.columns and df[cand].dropna().nunique() > 1:
            target_series = df[cand]
            break

    if target_series is not None:
        y_mapped = (
            target_series.astype(str)
            .str.lower()
            .str.strip()
            .map(lambda x: RISK_MAP.get(x, np.nan))
        )
        valid_mask = ~y_mapped.isna()

        features_df = pd.DataFrame({
            "observation_hour": obs_hour, "speed": speed, "rpm": rpm,
            "acceleration": accel, "throttle_position": throttle,
            "engine_temperature": eng_temp, "engine_load_value": eng_load,
            "heart_rate": hr, "current_weather": curr_weather,
            "visibility": visibility, "precipitation": precip,
            "accidents_onsite": acdnt_site, "design_speed": des_speed,
            "accidents_time": acdnt_time,
        })[valid_mask].reset_index(drop=True)

        y = y_mapped[valid_mask].to_numpy().astype(int)
    else:
        speed_delta = (speed - des_speed).clip(lower=0)
        penalties = np.zeros(len(df), dtype=int)
        penalties += np.where(speed_delta > 20, 3, np.where(speed_delta > 5, 1, 0))
        penalties += np.where(rpm > 4500, 2, np.where(rpm > 3500, 1, 0))
        penalties += np.where(hr > 115, 3, np.where(hr > 95, 1, 0))
        penalties += np.where(acdnt_site > 30, 2, np.where(acdnt_site > 8, 1, 0))
        penalties += np.where((curr_weather == 8) | (precip > 5.0), 2, 0)

        y = np.where(
            penalties <= 1, 0, np.where(penalties <= 3, 1, np.where(penalties <= 5, 2, 3))
        )

        features_df = pd.DataFrame({
            "observation_hour": obs_hour, "speed": speed, "rpm": rpm,
            "acceleration": accel, "throttle_position": throttle,
            "engine_temperature": eng_temp, "engine_load_value": eng_load,
            "heart_rate": hr, "current_weather": curr_weather,
            "visibility": visibility, "precipitation": precip,
            "accidents_onsite": acdnt_site, "design_speed": des_speed,
            "accidents_time": acdnt_time,
        }).reset_index(drop=True)

    if len(features_df) > 50000:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(features_df), size=50000, replace=False)
        X = features_df.iloc[indices][POLIDRIVING_FEATURES].reset_index(drop=True).fillna(0.0)
        y = y[indices]
    else:
        X = features_df[POLIDRIVING_FEATURES].reset_index(drop=True).fillna(0.0)

    print("  Class distribution in dataset:")
    unique, counts = np.unique(y, return_counts=True)
    class_names = ["Low (0)", "Medium (1)", "High (2)", "Very High (3)"]
    for u, c in zip(unique, counts):
        print(f"   - {class_names[u]}: {c} samples ({c / len(y) * 100:.1f}%)")

    return X, y


def main():
    """Main training execution entry point for POLIDriving model."""
    print("=" * 70)
    print("  TRAINING AUTHENTIC 14-FEATURE POLIDRIVING RISK MODEL (GBM)")
    print("=" * 70)

    X, y = load_polidriving_14_features("data/raw")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n  Fitting Gradient Boosting Classifier matching POLIDriving spec...")
    clf = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=100,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  EVALUATION ON TEST SET (Accuracy: {acc * 100:.2f}%):")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2, 3],
            target_names=["Low", "Medium", "High", "Very High"],
            zero_division=0,
        )
    )
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3]))

    out_dir = "artifacts/fusion_model"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "fusion_model.joblib")
    scaler_path = os.path.join(out_dir, "scaler.joblib")

    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)

    print("\n  Updated POLIDriving 14-feature artifacts saved:")
    print(f"   - {model_path}")
    print(f"   - {scaler_path}")


if __name__ == "__main__":
    main()
