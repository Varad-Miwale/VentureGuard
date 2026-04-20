from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd

from .config import ACQUIRED, CLOSED, DROP_COLUMNS, FEATURE_COLUMNS, STATUS_COL


def load_csv(file_or_path) -> pd.DataFrame:
    return pd.read_csv(file_or_path)


def _parse_numeric_column(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"[$,]", "", regex=True).str.strip(),
        errors="coerce",
    )


def clean_dataset(df: pd.DataFrame, keep_status_filter: bool = True) -> pd.DataFrame:
    frame = df.copy()

    to_drop = [col for col in DROP_COLUMNS if col in frame.columns]
    if to_drop:
        frame = frame.drop(columns=to_drop)

    if STATUS_COL in frame.columns:
        frame[STATUS_COL] = frame[STATUS_COL].astype(str).str.strip().str.lower()

    for col in FEATURE_COLUMNS:
        if col in frame.columns:
            frame[col] = _parse_numeric_column(frame[col])

    if keep_status_filter and STATUS_COL in frame.columns:
        frame = frame[frame[STATUS_COL].isin([ACQUIRED, CLOSED])]

    required = [col for col in FEATURE_COLUMNS if col in frame.columns]
    if STATUS_COL in frame.columns:
        required.append(STATUS_COL)

    if required:
        frame = frame.dropna(subset=required)

    return frame.reset_index(drop=True)


def available_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in FEATURE_COLUMNS if col in df.columns]


def prepare_classification_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
    features = available_feature_columns(df)
    if not features:
        raise ValueError("No supported feature columns found.")
    if STATUS_COL not in df.columns:
        raise ValueError("Missing required status column.")

    x = df[features].fillna(0)
    y = (df[STATUS_COL] == ACQUIRED).astype(int)
    return x, y, features


def prepare_regression_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
    # Predict funding_total_usd as a proxy for expected execution effort.
    if "funding_total_usd" not in df.columns:
        raise ValueError("Missing funding_total_usd required for regression target.")

    features = [col for col in available_feature_columns(df) if col != "funding_total_usd"]
    if not features:
        raise ValueError("No supported regression feature columns found.")

    x = df[features].fillna(0)
    y = df["funding_total_usd"].astype(float)
    return x, y, features


def prepare_cluster_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    features = available_feature_columns(df)
    if not features:
        raise ValueError("No supported clustering feature columns found.")
    x = df[features].fillna(0)
    return x, features


def ensure_feature_frame(frame: pd.DataFrame, ordered_features: Iterable[str]) -> pd.DataFrame:
    ordered = list(ordered_features)
    aligned = frame.copy()

    for col in ordered:
        if col not in aligned.columns:
            aligned[col] = 0.0

    aligned = aligned[ordered]

    for col in ordered:
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(0.0)

    return aligned
