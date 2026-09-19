"""Small CPU baseline pipelines; imputation/scaling are fitted only on training data."""

import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_baseline(family: str, seed: int = 42) -> Pipeline:
    """Build an sklearn-compatible estimator; all families accept sample weights."""
    if family == "logistic_regression":
        estimator = LogisticRegression(max_iter=1500, random_state=seed)
    elif family == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=150, min_samples_leaf=2, random_state=seed, n_jobs=2
        )
    elif family == "xgboost":
        if sys.platform == "darwin" and "torch" in sys.modules:
            raise RuntimeError(
                "Run XGBoost and optional PyTorch experiments in separate processes on macOS: their OpenMP runtimes can conflict"
            )
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            tree_method="hist",
            device="cpu",
            n_jobs=2,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown baseline: {family}")
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )
