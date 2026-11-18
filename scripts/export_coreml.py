import coremltools as ct
from sklearn.ensemble import RandomForestClassifier

# Import your real data functions from your existing scripts
from train_vzcrash import fetch_and_prepare_vzcrash

# POLIDriving's current-instant model (FusionModel.mlpackage) used to be
# trained and exported here too. It was retired: scripts/train_polidriving.py
# split train/test with a plain random train_test_split rather than a
# driver-held-out one, so its accuracy could never be honestly compared to
# scripts/train_polidriving_5s_ahead.py's driver-held-out 0.918 ROC-AUC.
# PolidrivingFiveSecondModel.mlpackage is now the sole POLIDriving model
# deployed on-device (see RoadSenseOnDeviceRiskEngine.swift).


def export_real_models():
    print("1. Training VZCrash Model on Hugging Face Data...")
    X_vz, y_vz = fetch_and_prepare_vzcrash()

    # FIX: Removed Pipeline and StandardScaler. Random Forests evaluate raw thresholds perfectly.
    vz_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
    vz_rf.fit(X_vz, y_vz)

    # Convert to CoreML (Input must be named 'features' for Swift)
    vz_mlmodel = ct.converters.sklearn.convert(vz_rf, "features")
    vz_mlmodel.save("VZCrashModel.mlpackage")
    print("   Saved VZCrashModel.mlpackage for Xcode")


if __name__ == "__main__":
    export_real_models()