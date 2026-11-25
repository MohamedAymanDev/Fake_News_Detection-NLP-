"""All paths and settings in one place."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Data folders
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "merged_cleaned.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DIR / "cleaned_data.csv"

# Analysis outputs
ANALYSIS_DIR = PROCESSED_DIR / "analysis"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

# Saved models
MODELS_DIR = BASE_DIR / "models"
TFIDF_PATH = MODELS_DIR / "tfidf.joblib"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
RESULTS_PATH = MODELS_DIR / "results.csv"

# Column names in the dataset
TEXT_COLUMN = "News content"
LABEL_COLUMN = "Label"
PROCESSED_TEXT_COLUMN = "processed_text"

# Settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 5000
