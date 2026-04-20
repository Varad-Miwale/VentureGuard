# VentureGuard Pro

VentureGuard Pro is a production-style Streamlit application for startup intelligence. It combines three ML workflows (classification, regression, clustering) with persistent artifacts, operational checks, and prediction audit history.

## Why This Is Production-Ready

- Modular codebase with clear separation of concerns across data prep, modeling, inference, storage, and UI layers
- Persistent model artifact lifecycle (train, save, load, evaluate)
- Batch and single inference flows with schema alignment
- Built-in metrics and artifact health checks through the Admin and Model Center pages
- Prediction logging for traceability and post-hoc analysis
- Streamlit Cloud compatible project structure and Python runtime pinning

## Core Features

- Startup Analyzer for one-record predictions with confidence and risk labels
- Bulk Upload scoring for CSV files with downloadable scored output
- Dashboard and Analytics pages for interactive exploration
- Clusters page for startup segmentation insights
- Prediction History page for operational visibility
- Admin Panel for dataset, model, and runtime checks

## ML Stack

- Priority Classifier: SVC (RBF kernel) with calibrated probabilities
- Effort Estimator: Polynomial Regression pipeline
- Startup Segmentation: K-Means with silhouette-driven cluster count selection

## Architecture Overview

- app.py: Streamlit entrypoint, page routing, orchestration
- src/data_prep.py: cleaning, type coercion, feature alignment
- src/modeling.py: train/evaluate routines for all models
- src/inference.py: single and batch prediction logic + risk mapping
- src/model_store.py: joblib and metrics JSON persistence
- src/history_store.py: prediction history read/write
- src/ui.py: shared UI helpers and style utilities
- src/config.py: paths, constants, feature definitions
- src/train_models.py: CLI training entrypoint

## Quick Start (Local)

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run the app

```bash
streamlit run app.py
```

3. Open:

```text
http://localhost:8501
```

## Model Training Workflow

You can train models either from UI or CLI.

### Option A: In-App Training

1. Upload dataset from sidebar
2. Click Train / Retrain Models
3. Review metrics in Model Center
4. Validate artifact status in Admin Panel

### Option B: CLI Training

```bash
python -m src.train_models --input data/processed/startup_clean.csv
```

Any valid dataset path can be passed to --input.

## Dataset Contract

Expected target/status values include startup outcomes such as acquired and closed. Typical feature columns include:

- funding_rounds
- funding_total_usd
- milestones
- relationships
- avg_participants
- age_first_funding_year
- age_last_funding_year
- has_VC
- has_angel
- is_software
- is_web
- is_mobile

The pipeline performs numeric coercion and feature-frame alignment to reduce runtime schema failures.

## Project Structure

```text
Startup Predictor/
|-- app.py
|-- requirements.txt
|-- runtime.txt
|-- README.md
|-- .streamlit/
|   `-- config.toml
|-- data/
|   |-- raw/
|   `-- processed/
|       |-- startup_clean.csv
|       `-- prediction_history.csv
|-- models/
|   |-- priority_model.joblib
|   |-- effort_model.joblib
|   |-- cluster_model.joblib
|   `-- model_metrics.json
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_prep.py
|   |-- modeling.py
|   |-- inference.py
|   |-- model_store.py
|   |-- history_store.py
|   |-- ui.py
|   `-- train_models.py
|-- notebooks/
`-- assets/
```

## Deployment (Streamlit Cloud)

1. Push latest code to main branch
2. In Streamlit Cloud, set repository and app file path to app.py
3. Confirm runtime.txt uses Python 3.11
4. Ensure requirements.txt installs all dependencies
5. Trigger redeploy and verify health in app pages

## Operations Checklist

- Confirm model artifact files exist under models/
- Confirm metrics JSON is generated after training
- Verify prediction_history.csv is writable
- Run a sample single prediction and one bulk scoring file
- Check Admin Panel for missing artifacts or data issues

## Limitations

- CSV-based persistence is simple and transparent but not ideal for high-concurrency workloads
- Model performance depends on source dataset quality and label consistency
- Streamlit is suitable for rapid productization, but large-scale API serving may require a dedicated backend service

## Suggested Next Upgrades

- Replace CSV history with PostgreSQL + migrations
- Add automated tests (data prep, inference, schema contracts)
- Add model versioning and rollback metadata
- Introduce CI pipeline for lint, test, and artifact checks
- Add authentication and role-based admin access

## Version

- Version: 2.1.0
- Status: Production-style modular architecture, deployment-ready
