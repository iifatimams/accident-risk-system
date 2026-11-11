import coremltools as ct
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Import your real data functions from your existing scripts
from train_vzcrash import fetch_and_prepare_vzcrash
from train_polidriving import load_polidriving_14_features

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

    print("\n2. Training POLIDriving Model on local dataset...")
    X_poli, y_poli = load_polidriving_14_features("data/raw")
    
    # FIX: Removed Pipeline and StandardScaler.
    poli_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
    poli_rf.fit(X_poli, y_poli)
    
    # Convert to CoreML
    poli_mlmodel = ct.converters.sklearn.convert(poli_rf, "features")
    poli_mlmodel.save("FusionModel.mlpackage")
    print("   Saved FusionModel.mlpackage for Xcode")

if __name__ == "__main__":
    export_real_models()