import pandas as pd
import streamlit as st

from src.config import (
    ANALYSIS_DIR,
    FIGURES_DIR,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    RESULTS_PATH
)

from src.predict import (
    load_artifacts,
    models_exist,
    predict_news
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Arabic Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.html("""
<style>

/* GLOBAL */

.stApp {
    background:
        radial-gradient(circle at 10% 10%, #172554 0%, transparent 35%),
        radial-gradient(circle at 90% 90%, #312e81 0%, transparent 30%),
        #080b18;

    color: #f8fafc;
}

/* Hide only Streamlit menu and footer */
#MainMenu,
footer {
    visibility: hidden;
}

/* Keep the top header transparent (no black bar) */
[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
}

/* Keep the sidebar toggle arrow visible on the left */
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    position: fixed !important;
    top: 12px !important;
    left: 12px !important;
    z-index: 999999 !important;
}

[data-testid="stSidebarCollapsedControl"] button {
    background: rgba(30, 41, 90, 0.85) !important;
    border: 1px solid rgba(103, 232, 249, 0.4) !important;
    border-radius: 10px !important;
    color: #67e8f9 !important;
}

[data-testid="stSidebarCollapsedControl"] button:hover {
    background: #312e81 !important;
    border-color: #67e8f9 !important;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
    animation: pageEnter 0.8s ease;
}

/* ANIMATIONS */

@keyframes pageEnter {
    from {
        opacity: 0;
        transform: translateY(25px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes float {
    0%, 100% {
        transform: translateY(0);
    }

    50% {
        transform: translateY(-10px);
    }
}

@keyframes glow {
    from {
        box-shadow: 0 0 10px rgba(99,102,241,.1);
    }

    to {
        box-shadow: 0 0 30px rgba(34,211,238,.25);
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }

    50% {
        opacity: .55;
    }
}

@keyframes reveal {
    0% {
        opacity: 0;
        transform: scale(.9);
    }

    100% {
        opacity: 1;
        transform: scale(1);
    }
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10182d, #080d1b);
    border-right: 1px solid #273653;
}

[data-testid="stSidebar"] p {
    color: #a5b4cf;
}

/* SIDEBAR TITLE */

.sidebar-logo {
    text-align: center;
    padding: 20px 5px;
}

.sidebar-logo-icon {
    font-size: 48px;
    animation: float 3s ease-in-out infinite;
}

.sidebar-logo-title {
    font-size: 22px;
    font-weight: 800;
    color: #67e8f9;
    margin-top: 10px;
}

/* HERO */

.hero {
    position: relative;
    overflow: hidden;

    background: linear-gradient(
        120deg,
        rgba(30,41,90,.95),
        rgba(49,46,129,.8),
        rgba(8,47,73,.95)
    );

    border: 1px solid rgba(129,140,248,.4);
    border-radius: 24px;

    padding: 42px;

    margin-bottom: 35px;

    animation: pageEnter 1s ease;

    box-shadow: 0 15px 50px rgba(0,0,0,.3);
}

.hero::before {
    content: "";

    position: absolute;

    width: 350px;
    height: 350px;

    right: -100px;
    top: -150px;

    background: rgba(34,211,238,.2);

    filter: blur(90px);
    border-radius: 50%;

    animation: float 5s ease-in-out infinite;
}

.hero::after {
    content: "";

    position: absolute;

    width: 250px;
    height: 250px;

    left: -100px;
    bottom: -150px;

    background: rgba(129,140,248,.2);

    filter: blur(80px);
    border-radius: 50%;

    animation: float 6s ease-in-out infinite reverse;
}

.hero-badge {
    display: inline-block;

    padding: 9px 16px;

    background: rgba(34,211,238,.12);
    border: 1px solid rgba(34,211,238,.4);

    border-radius: 30px;

    color: #67e8f9;

    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;

    margin-bottom: 20px;
}

.hero-title {
    font-size: clamp(28px, 4vw, 43px);
    font-weight: 900;
    color: #ffffff;

    margin-bottom: 15px;

    position: relative;
    z-index: 1;
}

.hero-subtitle {
    font-size: 16px;
    line-height: 1.9;

    color: #cbd5e1;

    max-width: 800px;

    position: relative;
    z-index: 1;
}

/* HEADINGS */

h1, h2, h3 {
    color: #f8fafc !important;
    font-weight: 800 !important;
}

p {
    color: #aab8d0;
}

/* METRIC CARDS */

.metric-card {
    background: linear-gradient(145deg, #18243e, #10182b);

    border: 1px solid #2b3b5e;
    border-radius: 18px;

    padding: 24px;

    min-height: 125px;

    transition: all .35s ease;

    animation: pageEnter .8s ease;
}

.metric-card:hover {
    transform: translateY(-8px);
    border-color: #22d3ee;

    box-shadow: 0 12px 35px rgba(34,211,238,.12);
}

.metric-label {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 12px;
}

.metric-value {
    font-size: 27px;
    font-weight: 900;
    color: #67e8f9;
}

/* SECTION CARDS */

.section-card {
    background: rgba(20,30,53,.85);

    border: 1px solid #2b3b5e;
    border-radius: 20px;

    padding: 25px;

    margin: 20px 0;

    animation: pageEnter .7s ease;

    transition: all .3s ease;
}

.section-card:hover {
    border-color: #6366f1;
    box-shadow: 0 10px 35px rgba(99,102,241,.1);
}

/* BUTTONS */

.stButton > button {
    background: linear-gradient(90deg, #6366f1, #0891b2);

    color: white !important;

    border: none;
    border-radius: 12px;

    font-weight: 800;

    padding: 12px 20px;

    transition: all .3s ease;
}

.stButton > button:hover {
    transform: translateY(-4px);

    box-shadow: 0 8px 25px rgba(34,211,238,.3);
}

/* INPUT */

.stTextArea textarea {
    background: #101a30 !important;
    color: #ffffff !important;

    border: 1px solid #334766 !important;
    border-radius: 15px !important;

    font-size: 16px !important;
}

.stTextArea textarea:focus {
    border-color: #22d3ee !important;
    box-shadow: 0 0 15px rgba(34,211,238,.12);
}

/* TABS */

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    background: #151f37;
    border-radius: 10px;

    padding: 12px 20px;

    transition: all .3s ease;
}

.stTabs [aria-selected="true"] {
    color: #67e8f9 !important;
    border-bottom: 2px solid #22d3ee !important;
}

/* RESULTS */

.result-real,
.result-fake {
    padding: 30px;

    border-radius: 20px;

    text-align: center;

    animation: reveal .7s ease;
}

.result-real {
    background: linear-gradient(145deg, #052e25, #064e3b);
    border: 1px solid #22c55e;
}

.result-fake {
    background: linear-gradient(145deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444;
}

.result-title {
    font-size: 30px;
    font-weight: 900;
}

/* IMAGES */

[data-testid="stImage"] img {
    border-radius: 15px;
    transition: all .4s ease;
}

[data-testid="stImage"] img:hover {
    transform: scale(1.02);
}

/* DATAFRAME */

[data-testid="stDataFrame"] {
    border: 1px solid #293c5c;
    border-radius: 12px;
}

/* DIVIDER */

hr {
    border-color: #293653;
}

/* SCROLLBAR */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #080b18;
}

::-webkit-scrollbar-thumb {
    background: #475569;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #6366f1;
}

/* REDUCED MOTION */

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation: none !important;
        transition: none !important;
    }
}

</style>
""")


# =========================================================
# CACHED FUNCTIONS
# =========================================================

@st.cache_resource
def get_model():
    return load_artifacts()


@st.cache_data
def read_csv(path):
    return pd.read_csv(path)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def metric_card(label, value, icon):

    st.html(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                {icon} {label}
            </div>

            <div class="metric-value">
                {value}
            </div>

        </div>
        """,
    )


def display_image(path, caption):

    if path.exists():

        st.image(
            str(path),
            caption=caption,
            use_container_width=True
        )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.html(
        """
        <div class="sidebar-logo">

            <div class="sidebar-logo-icon">
                📰
            </div>

            <div class="sidebar-logo-title">
                Fake News AI
            </div>

            <p>Arabic News Intelligence</p>

        </div>
        """,
    )

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Navigate",
        [
            "🔍 Check News",
            "📊 Data Analysis",
            "🤖 Model Results",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### System Status")

    if models_exist():
        st.success("🟢 Model Available")
    else:
        st.warning("🟠 Model Not Trained")


# =========================================================
# HERO SECTION
# =========================================================

st.html(
    """
    <div class="hero">
        <div class="hero-badge">
            ✦ ARTIFICIAL INTELLIGENCE PROJECT
        </div>

        <div class="hero-title">
            Arabic Fake News Detection
        </div>

        <div class="hero-subtitle">
            An intelligent Machine Learning system designed
            to analyze Arabic news and detect potentially
            misleading or fake information.
            <br><br>
            Analyze news. Explore data. Evaluate intelligence.
        </div>
    </div>
    """,
)


# =========================================================
# PAGE 1: CHECK NEWS
# =========================================================

if page == "🔍 Check News":

    st.markdown("## 🔍 News Detection")

    st.write(
        "Enter an Arabic news article and let the AI model analyze it."
    )

    st.divider()

    if not models_exist():

        st.warning("No trained model was found.")

        st.code("python run_pipeline.py")

    else:

        tfidf, model, model_name = get_model()

        col1, col2 = st.columns(2)

        with col1:
            metric_card("Active Model", model_name, "🤖")

        with col2:
            metric_card("System Status", "Ready", "🟢")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### 📝 Enter News Article")

        st.write(
            "Paste the Arabic news content below for classification."
        )

        news_text = st.text_area(
            "News Content",
            height=220,
            placeholder="الصق نص الخبر العربي هنا...",
            label_visibility="collapsed"
        )

        check_button = st.button(
            "🔍 Analyze News",
            type="primary",
            use_container_width=True
        )

        if check_button:

            if not news_text.strip():

                st.error("Please enter a news article first.")

            else:

                with st.spinner("🤖 AI is analyzing the news..."):

                    label, confidence = predict_news(
                        news_text,
                        tfidf,
                        model
                    )

                st.divider()

                st.markdown("## Analysis Result")

                confidence_text = (
                    f"{confidence}%"
                    if confidence is not None
                    else "N/A"
                )

                if label == "Real":

                    st.html(
                        f"""
                        <div class="result-real">

                            <div style="font-size:55px;">✅</div>

                            <div class="result-title"
                                 style="color:#4ade80;">
                                REAL NEWS
                            </div>

                            <p style="color:#bbf7d0;">
                                The model classified this article
                                as potentially real.
                            </p>

                            <h2 style="color:#4ade80 !important;">
                                {confidence_text}
                            </h2>

                            <p style="color:#bbf7d0;">
                                Model Confidence
                            </p>

                        </div>
                        """,
                    )

                else:

                    st.html(
                        f"""
                        <div class="result-fake">

                            <div style="font-size:55px;">🚨</div>

                            <div class="result-title"
                                 style="color:#fca5a5;">
                                FAKE NEWS
                            </div>

                            <p style="color:#fecaca;">
                                The model classified this article
                                as potentially fake or misleading.
                            </p>

                            <h2 style="color:#fca5a5 !important;">
                                {confidence_text}
                            </h2>

                            <p style="color:#fecaca;">
                                Model Confidence
                            </p>

                        </div>
                        """,
                    )

                st.info(
                    "This is a machine-learning prediction, "
                    "not a verified fact-check. "
                    "Always verify important news using reliable sources."
                )


# =========================================================
# PAGE 2: DATA ANALYSIS
# =========================================================

elif page == "📊 Data Analysis":

    st.markdown("## 📊 Dataset Analysis")

    st.write(
        "Explore the dataset, preprocessing results, "
        "and statistical insights."
    )

    st.divider()

    if RAW_DATA_PATH.exists():

        raw_df = read_csv(str(RAW_DATA_PATH))

        col1, col2, col3 = st.columns(3)

        with col1:
            metric_card(
                "Total Articles",
                f"{len(raw_df):,}",
                "📰"
            )

        with col2:
            metric_card(
                "Total Features",
                len(raw_df.columns),
                "📋"
            )

        with col3:
            metric_card(
                "Missing Values",
                f"{raw_df.isnull().sum().sum():,}",
                "⚠️"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### 📁 Raw Dataset")

        st.dataframe(
            raw_df.head(10),
            use_container_width=True
        )

    else:

        st.warning("Raw dataset not found.")

    st.divider()

    st.markdown("### 🧹 Processed Dataset")

    if PROCESSED_DATA_PATH.exists():

        processed_df = read_csv(str(PROCESSED_DATA_PATH))

        st.metric(
            "Processed Articles",
            f"{len(processed_df):,}"
        )

        st.dataframe(
            processed_df.head(10),
            use_container_width=True
        )

    else:

        st.info("Processed dataset not found.")

    st.divider()

    st.markdown("### 📈 Data Visualizations")

    col1, col2 = st.columns(2)

    with col1:

        display_image(
            FIGURES_DIR / "label_distribution.png",
            "Label Distribution"
        )

    with col2:

        display_image(
            FIGURES_DIR / "article_length.png",
            "Article Length Distribution"
        )

    st.divider()

    st.markdown("### 🔎 Dataset Insights")

    col1, col2 = st.columns(2)

    with col1:

        platform_path = ANALYSIS_DIR / "platform_label_counts.csv"

        if platform_path.exists():

            st.markdown("#### Top Platforms")

            st.dataframe(
                read_csv(str(platform_path)),
                use_container_width=True
            )

    with col2:

        words_path = ANALYSIS_DIR / "top_words.csv"

        if words_path.exists():

            st.markdown("#### Most Common Words")

            st.dataframe(
                read_csv(str(words_path)),
                use_container_width=True
            )


# =========================================================
# PAGE 3: MODEL RESULTS
# =========================================================

elif page == "🤖 Model Results":

    st.markdown("## 🤖 Model Performance")

    st.write(
        "Evaluate and compare the performance of trained models."
    )

    st.divider()

    if RESULTS_PATH.exists():

        results_df = read_csv(str(RESULTS_PATH))

        st.markdown("### 🏆 Model Comparison")

        st.dataframe(
            results_df,
            use_container_width=True
        )

        st.divider()

        st.markdown("### 📊 Performance Visualization")

        display_image(
            FIGURES_DIR / "model_comparison.png",
            "Model Performance Comparison"
        )

        st.divider()

        st.markdown("### 🎯 Confusion Matrices")

        display_image(
            FIGURES_DIR / "confusion_matrices.png",
            "Confusion Matrices"
        )

    else:

        st.warning("Model evaluation results are not available.")

        st.code("python run_pipeline.py")


# =========================================================
# PAGE 4: ABOUT PROJECT
# =========================================================

elif page == "ℹ️ About Project":

    st.markdown("## ℹ️ About Project")

    st.html(
        """
        <div class="section-card">

            <h2>📰 Arabic Fake News Detection</h2>

            <p>
                This project uses Natural Language Processing
                and Machine Learning techniques to identify
                potentially fake Arabic news articles.
            </p>

            <br>

            <h3>🎯 Project Objectives</h3>

            <ul>
                <li>Analyze Arabic news articles.</li>
                <li>Apply NLP preprocessing techniques.</li>
                <li>Extract textual features using TF-IDF.</li>
                <li>Train and evaluate Machine Learning models.</li>
                <li>Classify news articles as Real or Fake.</li>
            </ul>

            <br>

            <h3>⚙️ Technologies Used</h3>

            <p>
                Python • Pandas • Scikit-learn • NLP •
                TF-IDF • Streamlit • Machine Learning
            </p>

            <br>

            <h3>🧠 System Workflow</h3>

            <p>
                Raw Data → Text Preprocessing →
                Feature Extraction → Model Training →
                Evaluation → Prediction
            </p>

        </div>
        """,
    )