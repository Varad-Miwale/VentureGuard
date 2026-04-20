from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

CLEAN_DATA_PATH = PROCESSED_DIR / "startup_clean.csv"
HISTORY_PATH = PROCESSED_DIR / "prediction_history.csv"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

PRIORITY_MODEL_PATH = MODELS_DIR / "priority_model.joblib"
EFFORT_MODEL_PATH = MODELS_DIR / "effort_model.joblib"
CLUSTER_MODEL_PATH = MODELS_DIR / "cluster_model.joblib"

STATUS_COL = "status"
ACQUIRED = "acquired"
CLOSED = "closed"

DROP_COLUMNS = [
    "Unnamed: 0",
    "name",
    "permalink",
    "homepage_url",
    "twitter_username",
    "logo_url",
    "short_description",
    "description",
    "founded_at",
    "closed_at",
    "first_funding_at",
    "last_funding_at",
    "age_first_milestone_year",
    "age_last_milestone_year",
]

FEATURE_COLUMNS = [
    "funding_rounds",
    "funding_total_usd",
    "milestones",
    "relationships",
    "avg_participants",
    "age_first_funding_year",
    "age_last_funding_year",
    "has_VC",
    "has_angel",
    "is_software",
    "is_web",
    "is_mobile",
]
