"""Streamlit app:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from src.config import ANALYSIS_DIR, FIGURES_DIR, PROCESSED_DATA_PATH, RAW_DATA_PATH, RESULTS_PATH
from src.predict import load_artifacts, models_exist, predict_news

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")


@st.cache_resource
def get_model():
    return load_artifacts()


@st.cache_data
def read_csv(path):
    return pd.read_csv(path)


st.title("📰 Arabic Fake News Detection")
st.caption("Paste an Arabic news text and the model will tell you if it looks real or fake.")

tab_predict, tab_data, tab_models = st.tabs(["🔍 Check News", "📊 Data Analysis", "🤖 Model Results"])

# ------------------------------------------------------------ Tab 1: Predict
with tab_predict:
    if not models_exist():
        st.warning("No trained model found. Run this command first:")
        st.code("python run_pipeline.py")
    else:
        tfidf, model, model_name = get_model()
        st.write(f"Model in use: **{model_name}**")

        text = st.text_area("News text", height=200, placeholder="اكتب أو الصق الخبر هنا...")

        if st.button("Check", type="primary"):
            if not text.strip():
                st.error("Please enter some text.")
            else:
                label, confidence = predict_news(text, tfidf, model)
                extra = f"  (confidence {confidence}%)" if confidence is not None else ""
                if label == "Real":
                    st.success(f"✅ Real news{extra}")
                else:
                    st.error(f"🚨 Fake news{extra}")
                st.caption("This is a machine-learning prediction, not a fact-check. Always verify important news.")

# ------------------------------------------------------------ Tab 2: Data
with tab_data:
    st.subheader("Before cleaning (raw data)")
    st.dataframe(read_csv(RAW_DATA_PATH).head(10), use_container_width=True)

    st.subheader("After cleaning (processed data)")
    if PROCESSED_DATA_PATH.exists():
        st.dataframe(read_csv(PROCESSED_DATA_PATH).head(10), use_container_width=True)
    else:
        st.info("Processed data not found yet. Run `python run_pipeline.py`.")

    st.subheader("Charts")
    col1, col2 = st.columns(2)
    if (FIGURES_DIR / "label_distribution.png").exists():
        col1.image(str(FIGURES_DIR / "label_distribution.png"), caption="Label distribution")
        col2.image(str(FIGURES_DIR / "article_length.png"), caption="Article length")

    st.subheader("Tables")
    col3, col4 = st.columns(2)
    if (ANALYSIS_DIR / "platform_label_counts.csv").exists():
        col3.write("Top platforms")
        col3.dataframe(read_csv(ANALYSIS_DIR / "platform_label_counts.csv"), use_container_width=True)
    if (ANALYSIS_DIR / "top_words.csv").exists():
        col4.write("Most common words (real / fake)")
        col4.dataframe(read_csv(ANALYSIS_DIR / "top_words.csv"), use_container_width=True)

# ------------------------------------------------------------ Tab 3: Models
with tab_models:
    if RESULTS_PATH.exists():
        st.subheader("Model comparison")
        st.dataframe(read_csv(RESULTS_PATH), use_container_width=True)
        st.image(str(FIGURES_DIR / "model_comparison.png"))
        st.subheader("Confusion matrices")
        st.image(str(FIGURES_DIR / "confusion_matrices.png"))
    else:
        st.info("No results yet. Run `python run_pipeline.py`.")
