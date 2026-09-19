"""Driver-state summaries. HRV requires actual RR/IBI or a device-reported RMSSD."""

import numpy as np

from accident_risk.processing.vehicle_features import series, statistics
from accident_risk.processing.windows import Window


def physiology_features(window: Window, drop_threshold: float = 3) -> dict[str, float]:
    """Compute unique-reading summaries, preserving missingness and age in seconds."""
    features: dict[str, float] = {}
    for name in ("heart_rate", "spo2", "physiology_signal_quality"):
        times, values = series(window, name)
        features.update(statistics(name, times, values))
        features[f"{name}_age_seconds"] = (
            float(window.end - times[-1]) if len(times) else float("nan")
        )
    _, rr = series(window, "rr_interval_ms")
    _, device_rmssd = series(window, "hrv_rmssd")
    features["rmssd_ms"] = (
        float(np.sqrt(np.mean(np.diff(rr) ** 2)))
        if len(rr) >= 2
        else (float(device_rmssd[-1]) if len(device_rmssd) else float("nan"))
    )
    features["sdnn_ms"] = float(np.std(rr, ddof=1)) if len(rr) >= 2 else float("nan")
    _, spo2 = series(window, "spo2")
    features["spo2_drop_count"] = (
        float(np.sum(np.diff(spo2) <= -drop_threshold)) if len(spo2) else float("nan")
    )
    return features
