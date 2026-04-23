from __future__ import annotations

import numpy as np
import pandas as pd

from .data_prep import ensure_feature_frame


def risk_level(probability: float) -> str:
    if probability >= 0.8:
        return "Critical"
    if probability >= 0.6:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


def predict_single_startup(
    row: pd.DataFrame,
    priority_bundle: dict,
    effort_bundle: dict,
    cluster_bundle: dict,
) -> dict:
    priority_features = ensure_feature_frame(row, priority_bundle["features"])
    effort_features = ensure_feature_frame(row, effort_bundle["features"])
    cluster_features = ensure_feature_frame(row, cluster_bundle["features"])

    priority_scaled = priority_bundle["scaler"].transform(priority_features)
    prob = float(priority_bundle["model"].predict_proba(priority_scaled)[0, 1])
    label = "Acquired" if prob >= 0.5 else "Closed"
    confidence = prob if label == "Acquired" else (1.0 - prob)

    effort_value = float(effort_bundle["model"].predict(effort_features)[0])

    cluster_scaled = cluster_bundle["scaler"].transform(cluster_features)
    segment = int(cluster_bundle["model"].predict(cluster_scaled)[0])

    return {
        "predicted_status": label,
        "confidence": round(confidence, 4),
        "risk_level": risk_level(1.0 - prob),
        "effort_estimate": round(max(0.0, effort_value), 2),
        "segment": segment,
    }


def predict_bulk(
    frame: pd.DataFrame,
    priority_bundle: dict,
    effort_bundle: dict,
    cluster_bundle: dict,
) -> pd.DataFrame:
    p_features = ensure_feature_frame(frame, priority_bundle["features"])
    e_features = ensure_feature_frame(frame, effort_bundle["features"])
    c_features = ensure_feature_frame(frame, cluster_bundle["features"])

    p_scaled = priority_bundle["scaler"].transform(p_features)
    prob = priority_bundle["model"].predict_proba(p_scaled)[:, 1]

    status = np.where(prob >= 0.5, "Acquired", "Closed")
    confidence = np.where(status == "Acquired", prob, 1.0 - prob)
    effort = effort_bundle["model"].predict(e_features)

    c_scaled = cluster_bundle["scaler"].transform(c_features)
    seg = cluster_bundle["model"].predict(c_scaled)

    out = frame.copy()
    out["predicted_status"] = status
    out["confidence"] = confidence.round(4)
    out["risk_level"] = [risk_level(1.0 - p) for p in prob]
    out["effort_estimate"] = np.maximum(0.0, effort).round(2)
    out["segment"] = seg
    return out
