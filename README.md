# Startup Predictor

Startup Predictor is a Streamlit application that estimates startup outcomes using machine learning. It supports single startup analysis, bulk CSV scoring, clustering insights, and saved prediction history.

## What The Project Includes

- A modular codebase with separate files for data prep, model training, inference, storage, and UI helpers
- Three ML workflows:
	- Classification for likely outcome
	- Regression for estimated effort score
	- Clustering for startup segmentation
- Model artifacts saved to disk so the app can reuse trained models
- In-app training and a CLI training command
- Model health and artifact status checks in Model Center

## Tech Stack

- Python
- Streamlit
- pandas, numpy
- scikit-learn
- plotly
- joblib

## Quick Start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run the app

```bash
streamlit run app.py
```

3. Open http://localhost:8501

## Training Models

Option 1: Train from the app sidebar using an uploaded CSV.

Option 2: Train from CLI.

```bash
python -m src.train_models --input data/processed/startup_clean.csv
```

## Expected Dataset Columns

The project expects outcome labels (for example acquired or closed) and startup features such as:

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
|-- app.py
|-- README.md
|-- requirements.txt
|-- runtime.txt
|-- .streamlit/
|   `-- config.toml
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- src/
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

## Deployment Notes

For Streamlit Cloud:

1. Deploy from main branch
2. Set app file path to app.py
3. Keep runtime.txt pinned to a supported Python version
4. Make sure requirements.txt is complete

## Current Limitations

- Prediction history is stored in CSV, which is simple but not ideal for high concurrency
- Model quality depends on training data quality
- No automated test suite yet

## Practical Next Steps

- Add unit tests for data prep and inference
- Move history storage to a database
- Add CI checks for lint and tests
- Add authentication for admin actions

## Status

- Version: 2.1.1
- State: Clean modular structure, ready for demos and interviews
