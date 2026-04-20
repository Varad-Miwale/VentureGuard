from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC


def train_priority_model(x: pd.DataFrame, y: pd.Series, random_state: int = 42) -> dict[str, Any]:
    class_counts = y.value_counts()
    can_stratify = len(class_counts) > 1 and int(class_counts.min()) >= 2

    split_kwargs = {
        "test_size": 0.2,
        "random_state": random_state,
    }
    if can_stratify:
        split_kwargs["stratify"] = y

    x_train, x_test, y_train, y_test = train_test_split(x, y, **split_kwargs)

    if y_train.nunique() < 2:
        raise ValueError("Training data must include at least two outcome classes.")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)
    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)
    y_prob = model.predict_proba(x_test_scaled)[:, 1]

    roc_auc = 0.0
    if y_test.nunique() > 1:
        roc_auc = float(roc_auc_score(y_test, y_prob))

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
    }

    return {
        "scaler": scaler,
        "model": model,
        "features": list(x.columns),
        "metrics": metrics,
    }


def train_effort_model(x: pd.DataFrame, y: pd.Series, random_state: int = 42) -> dict[str, Any]:
    if len(x) < 5:
        raise ValueError("At least 5 rows are required to train the effort model reliably.")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state
    )

    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("reg", LinearRegression()),
        ]
    )
    model.fit(x_train, y_train)

    pred = model.predict(x_test)

    r2 = float(r2_score(y_test, pred)) if len(y_test) > 1 else 0.0

    metrics = {
        "r2": r2,
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
    }

    return {
        "model": model,
        "features": list(x.columns),
        "metrics": metrics,
    }


def _best_kmeans_k(x_scaled: np.ndarray, min_k: int = 2, max_k: int = 6, random_state: int = 42) -> tuple[int, float]:
    if len(x_scaled) < 10:
        return 2, 0.0

    best_k = min_k
    best_score = -1.0

    upper = min(max_k, len(x_scaled) - 1)
    for k in range(min_k, max(min_k, upper) + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(x_scaled)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(x_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k

    if best_score < 0:
        return 2, 0.0
    return best_k, float(best_score)


def train_cluster_model(x: pd.DataFrame, random_state: int = 42) -> dict[str, Any]:
    if len(x) < 5:
        raise ValueError("At least 5 rows are required to train clustering reliably.")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    best_k, best_silhouette = _best_kmeans_k(x_scaled, random_state=random_state)
    model = KMeans(n_clusters=best_k, n_init=20, random_state=random_state)
    labels = model.fit_predict(x_scaled)

    metrics = {
        "clusters": int(best_k),
        "silhouette": float(best_silhouette),
        "samples": int(len(x)),
    }

    return {
        "scaler": scaler,
        "model": model,
        "features": list(x.columns),
        "labels": labels.tolist(),
        "metrics": metrics,
    }
