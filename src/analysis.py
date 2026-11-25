"""Data analysis: tables are saved as CSV, charts as PNG."""
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # save charts without opening a window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    ANALYSIS_DIR, FIGURES_DIR, LABEL_COLUMN, PROCESSED_TEXT_COLUMN
)

COLORS = {"real": "#2ecc71", "fake": "#e74c3c"}
sns.set_style("whitegrid")


def _prepare_folders():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def analyze_labels(df):
    """Count real / fake articles and draw a bar + pie chart."""
    counts = df[LABEL_COLUMN].value_counts()
    percent = (df[LABEL_COLUMN].value_counts(normalize=True) * 100).round(2)
    table = pd.DataFrame({"count": counts, "percent": percent})
    table.to_csv(ANALYSIS_DIR / "label_counts.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(counts.index, counts.values, color=[COLORS[l] for l in counts.index])
    axes[0].set_title("Number of Articles")
    axes[1].pie(counts, labels=counts.index, autopct="%1.1f%%",
                colors=[COLORS[l] for l in counts.index])
    axes[1].set_title("Percentage of Articles")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "label_distribution.png", dpi=120)
    plt.close()
    return table


def analyze_length(df):
    """Article length (in words) for each label."""
    stats = df.groupby(LABEL_COLUMN)["word_count"].describe().round(1)
    stats.to_csv(ANALYSIS_DIR / "length_stats.csv")

    short = df[df["word_count"] < 1000]  # very long articles hide the chart shape
    plt.figure(figsize=(8, 4))
    sns.histplot(data=short, x="word_count", hue=LABEL_COLUMN, bins=40, palette=COLORS)
    plt.title("Article Length (words) by Label")
    plt.xlabel("Number of words")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "article_length.png", dpi=120)
    plt.close()
    return stats


def analyze_platforms(df):
    """Real vs fake count for the top 5 platforms."""
    top5 = df["platform"].value_counts().head(5).index
    table = pd.crosstab(df[df["platform"].isin(top5)]["platform"], df[LABEL_COLUMN])
    table.to_csv(ANALYSIS_DIR / "platform_label_counts.csv")
    return table


def analyze_top_words(df, n=10):
    """Most common words (after preprocessing) in each label."""
    result = {}
    for label in ["real", "fake"]:
        text = " ".join(df[df[LABEL_COLUMN] == label][PROCESSED_TEXT_COLUMN].fillna(""))
        result[label] = Counter(text.split()).most_common(n)
    table = pd.DataFrame(result)
    table.to_csv(ANALYSIS_DIR / "top_words.csv", index=False, encoding="utf-8-sig")
    return table


def run_basic_analysis(df):
    """Analysis that only needs the raw data (no text preprocessing)."""
    _prepare_folders()
    df = df.copy()
    df["word_count"] = df["News content"].str.split().str.len()
    analyze_labels(df)
    analyze_length(df)
    analyze_platforms(df)


def run_text_analysis(df):
    """Analysis that needs the processed text."""
    _prepare_folders()
    analyze_top_words(df)
