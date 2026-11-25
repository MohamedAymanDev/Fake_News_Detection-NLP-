"""Train the models, evaluate them and save the best one."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.config import (
    BEST_MODEL_PATH, FIGURES_DIR, LABEL_COLUMN, MAX_FEATURES, MODELS_DIR,
    PROCESSED_TEXT_COLUMN, RANDOM_STATE, RESULTS_PATH, TEST_SIZE, TFIDF_PATH,
)


def split_data(df):
    """80% train / 20% test. real = 1, fake = 0."""
    X = df[PROCESSED_TEXT_COLUMN].fillna("")
    y = df[LABEL_COLUMN].map({"real": 1, "fake": 0})
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE),
    }


def plot_confusion_matrices(y_test, predictions):
    fig, axes = plt.subplots(1, len(predictions), figsize=(15, 4))
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
                    cbar=False, xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"], ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=120)
    plt.close()


def plot_comparison(results):
    results.plot(kind="bar", figsize=(8, 4), rot=0, ylim=(0.5, 1))
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=120)
    plt.close()


def train_and_save(df):
    """Full training step. Returns the results table."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = split_data(df)

    tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    models = get_models()
    predictions = {}
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        predictions[name] = model.predict(X_test_tfidf)
        print(f"{name} trained")

    results = pd.DataFrame({
        "Accuracy": {n: accuracy_score(y_test, p) for n, p in predictions.items()},
        "Macro F1": {n: f1_score(y_test, p, average="macro") for n, p in predictions.items()},
    }).round(4)
    results.to_csv(RESULTS_PATH)

    plot_confusion_matrices(y_test, predictions)
    plot_comparison(results)

    best_name = results["Macro F1"].idxmax()
    joblib.dump(tfidf, TFIDF_PATH)
    joblib.dump({"name": best_name, "model": models[best_name]}, BEST_MODEL_PATH)
    print(f"\nBest model: {best_name}")
    return results
