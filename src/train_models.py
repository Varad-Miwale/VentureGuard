from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    CLEAN_DATA_PATH,
    CLUSTER_MODEL_PATH,
    EFFORT_MODEL_PATH,
    METRICS_PATH,
    PRIORITY_MODEL_PATH,
)
from .data_prep import (
    clean_dataset,
    load_csv,
    prepare_classification_data,
    prepare_cluster_data,
    prepare_regression_data,
)
from .model_store import save_joblib, save_json
from .modeling import train_cluster_model, train_effort_model, train_priority_model


def run_training(input_path: Path) -> dict:
    df = load_csv(input_path)
    clean = clean_dataset(df, keep_status_filter=True)

    x_cls, y_cls, _ = prepare_classification_data(clean)
    x_reg, y_reg, _ = prepare_regression_data(clean)
    x_cluster, _ = prepare_cluster_data(clean)

    priority_bundle = train_priority_model(x_cls, y_cls)
    effort_bundle = train_effort_model(x_reg, y_reg)
    cluster_bundle = train_cluster_model(x_cluster)

    save_joblib(PRIORITY_MODEL_PATH, priority_bundle)
    save_joblib(EFFORT_MODEL_PATH, effort_bundle)
    save_joblib(CLUSTER_MODEL_PATH, cluster_bundle)

    clean.to_csv(CLEAN_DATA_PATH, index=False)

    metrics = {
        "priority": priority_bundle["metrics"],
        "effort": effort_bundle["metrics"],
        "cluster": cluster_bundle["metrics"],
        "rows_used": int(len(clean)),
    }
    save_json(METRICS_PATH, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and persist Startup Predictor models")
    parser.add_argument(
        "--input",
        type=Path,
        default=CLEAN_DATA_PATH,
        help="Path to source CSV. Defaults to data/processed/startup_clean.csv",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input dataset not found at {args.input}. Provide a valid --input path."
        )

    metrics = run_training(args.input)
    print("Training complete.")
    print(metrics)


if __name__ == "__main__":
    main()
