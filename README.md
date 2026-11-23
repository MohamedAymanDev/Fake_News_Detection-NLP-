# 📰 Arabic Fake News Detection

A simple machine-learning project that classifies Arabic news articles as **real** or **fake**.
It includes data analysis, Arabic text preprocessing, three trained models, and a **Streamlit** web app to test new text.

---

## ✨ Features

- Arabic text cleaning (diacritics, letter normalization, stop-words, stemming)
- Data analysis with simple charts and tables
- Three models compared: **Logistic Regression**, **Random Forest**, **SVM**
- The best model is selected automatically (by Macro F1) and saved
- Streamlit app to check any news text

---

## 📂 Project Structure

```
fake_news_project/
├── app.py                  # Streamlit app
├── run_pipeline.py         # Runs everything: load -> analysis -> clean -> train
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                # BEFORE: original dataset
│   │   └── merged_cleaned.csv
│   └── processed/          # AFTER: cleaned data and analysis tables
│       ├── cleaned_data.csv
│       └── analysis/       # label_counts, length_stats, platform_label_counts, top_words
│
├── models/                 # Saved TF-IDF, best model, and results.csv
├── reports/
│   └── figures/            # Charts (PNG)
│
└── src/
    ├── config.py           # Paths and settings
    ├── data_loader.py      # Read and save data
    ├── preprocessing.py    # Arabic text cleaning
    ├── analysis.py         # Tables and charts
    ├── train.py            # Train, evaluate, and save models
    └── predict.py          # Predict on new text
```

---

## 📊 Dataset

| Item | Value |
|------|-------|
| Articles | 5,352 |
| Real | 3,913 (73.1%) |
| Fake | 1,439 (26.9%) |

**Columns:** `Id`, `date`, `platform`, `title`, `News content`, `Label`

> The data is imbalanced (more real than fake), so besides accuracy we also use **Macro F1**,
> which treats both classes equally.

---

## ⚙️ How It Works

1. **Load** the raw data from `data/raw/`
2. **Analyze** the data (label distribution, article length, platforms)
3. **Clean** the text:
   - remove diacritics
   - normalize letters (`أ إ آ` → `ا`, `ة` → `ه`, `ى` → `ي`)
   - remove punctuation and numbers
   - remove Arabic stop-words
   - stemming with `ISRIStemmer`
4. **Convert** text to numbers with TF-IDF (5000 features)
5. **Train** 3 models on 80% of the data (stratified split)
6. **Evaluate** on the other 20% and save the best model

---

## 🚀 Installation

```bash
# 1. Go to the project folder
cd fake_news_project

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install the requirements
pip install -r requirements.txt
```

---

## ▶️ Usage

### Step 1: Run the pipeline

```bash
python run_pipeline.py
```

This will:
- analyze the raw data and save tables and charts
- clean the text and save `data/processed/cleaned_data.csv`
- train the models and save them in `models/`

> The first run downloads the NLTK Arabic stop-words, so an internet connection is needed once.

### Step 2: Run the app

```bash
streamlit run app.py
```

Then open the link shown in the terminal (usually http://localhost:8501).

---

## 🖥️ The App

| Tab | What it shows |
|-----|---------------|
| 🔍 **Check News** | Paste a news text and get **Real** or **Fake** (with confidence when available) |
| 📊 **Data Analysis** | Data before and after cleaning, charts, and tables |
| 🤖 **Model Results** | Model comparison table, bar chart, and confusion matrices |

---

## 🧪 Results

Results are saved in `models/results.csv` after running the pipeline.
In our experiments, the models reach around **88–90% accuracy**, and the Fake class is usually harder to detect than the Real class because the data is imbalanced.

---

## 🔧 Settings

All paths and settings are in `src/config.py`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `TEST_SIZE` | 0.2 | Percentage of data used for testing |
| `RANDOM_STATE` | 42 | Makes results repeatable |
| `MAX_FEATURES` | 5000 | Number of words used by TF-IDF |

---

## 💡 Ideas to Improve

- Add more fake news examples, or use `class_weight='balanced'`
- Use word pairs with `TfidfVectorizer(ngram_range=(1, 2))`
- Try an Arabic pre-trained model such as **AraBERT**

---

## ⚠️ Disclaimer

This project is for learning purposes. The model gives a **prediction**, not a fact-check.
Always verify important news from trusted sources.