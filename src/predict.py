"""Use the saved model to classify new text."""
import joblib

from src.config import BEST_MODEL_PATH, TFIDF_PATH
from src.preprocessing import preprocess_text


def models_exist():
    return BEST_MODEL_PATH.exists() and TFIDF_PATH.exists()


def load_artifacts():
    """Load the TF-IDF vectorizer and the best model."""
    tfidf = joblib.load(TFIDF_PATH)
    saved = joblib.load(BEST_MODEL_PATH)
    return tfidf, saved["model"], saved["name"]


def predict_news(text, tfidf, model):
    """Return ('Real' or 'Fake', confidence in percent or None)."""
    features = tfidf.transform([preprocess_text(text)])
    label = int(model.predict(features)[0])   # 0 = fake, 1 = real
    confidence = None
    if hasattr(model, "predict_proba"):       # SVM does not have it
        confidence = round(float(model.predict_proba(features)[0][label]) * 100, 1)
    return ("Real" if label == 1 else "Fake"), confidence
