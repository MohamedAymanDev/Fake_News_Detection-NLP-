"""Read and save the data."""
import pandas as pd

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def load_raw_data():
    """Read the original dataset (before cleaning)."""
    return pd.read_csv(RAW_DATA_PATH)


def save_processed_data(df):
    """Save the dataset after cleaning."""
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8-sig")


def load_processed_data():
    """Read the dataset after cleaning."""
    return pd.read_csv(PROCESSED_DATA_PATH)
