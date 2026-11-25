"""Arabic text cleaning."""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer

from src.config import TEXT_COLUMN, PROCESSED_TEXT_COLUMN

nltk.download("stopwords", quiet=True)


def clean_arabic_text(text):
    """Remove diacritics, normalize letters, remove punctuation and numbers."""
    text = re.sub(r"[\u064B-\u065F]", "", text)                      # diacritics
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")  # alef
    text = text.replace("ة", "ه")                                    # teh marbuta
    text = text.replace("ى", "ي")                                    # yeh
    text = re.sub(r"[^\w\s]", " ", text)                             # punctuation
    text = re.sub(r"\d+", " ", text)                                 # numbers
    text = re.sub(r"\s+", " ", text).strip()                         # extra spaces
    return text


# Clean the stop-words the same way so they match the cleaned text
STOP_WORDS = set(clean_arabic_text(w) for w in stopwords.words("arabic"))
STEMMER = ISRIStemmer()


def preprocess_text(text):
    """Full pipeline: clean -> remove stop-words -> stemming."""
    text = clean_arabic_text(str(text))
    words = text.split()
    words = [STEMMER.stem(w) for w in words if w not in STOP_WORDS]
    return " ".join(words)


def preprocess_dataframe(df):
    """Add the `word_count` and `processed_text` columns."""
    df = df.copy()
    df["word_count"] = df[TEXT_COLUMN].str.split().str.len()
    df[PROCESSED_TEXT_COLUMN] = df[TEXT_COLUMN].apply(preprocess_text)
    return df
