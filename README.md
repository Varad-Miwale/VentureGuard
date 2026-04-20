# VentureGuard Pro

A production-style Streamlit application for startup outcome intelligence with persisted ML models, batch scoring, analytics, and prediction audit history.

## Highlights

- Trained model workflow with persistent artifacts in models/
- Modular architecture (data prep, modeling, inference, storage, UI)
- Single startup analyzer with confidence and risk labels
- Bulk CSV scoring with downloadable outputs
- Model Center with metrics for classification, regression, and clustering
- Prediction history for audit and review
- Admin page for system and artifact health checks

## ML Components

- Priority Classifier: SVC (RBF kernel, calibrated probabilities)
- Effort Estimator: Polynomial Regression
- Startup Segmentation: K-Means with silhouette-based k selection

## Application Pages

- Dashboard
- Startup Analyzer
- Bulk Upload
- Analytics
- Model Center
- Clusters
- Prediction History
- Admin Panel

## Quick Start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run Streamlit app

```bash
streamlit run app.py
```

App URL: http://localhost:8501

## Training Workflow

You can train from inside the app, or from CLI.

### Option A: In-App

- Upload your startup CSV from the sidebar
- Click Train / Retrain Models
- Artifacts are saved automatically under models/

### Option B: CLI

```bash
python -m src.train_models --input data/processed/startup_clean.csv
```

You can point --input to any valid source CSV.

## Expected Dataset Notes

The dataset should include status values like acquired and closed, plus startup feature columns such as:

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

## Project Structure

```text
Startup Predictor/
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── data/
│   ├── raw/
│   └── processed/
│       ├── startup_clean.csv
│       └── prediction_history.csv
├── models/
│   ├── priority_model.joblib
│   ├── effort_model.joblib
│   ├── cluster_model.joblib
│   └── model_metrics.json
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_prep.py
│   ├── modeling.py
│   ├── inference.py
│   ├── model_store.py
│   ├── history_store.py
│   ├── ui.py
│   └── train_models.py
├── notebooks/
└── assets/
```

## Production Notes

- Model artifacts are versionable and can be retrained independently.
- The app can load from uploaded data or fallback processed data.
- Prediction history is persisted to CSV for traceability.

## Version

- Version: 2.0.0
- Status: Interview-ready modular architecture
